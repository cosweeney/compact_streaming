"""
Shared library for the 11-parameter-model / Gaussian-model fits (nb/fits/).

Differences from the earlier fit scripts (fit_*_mocks.py, fit_*_DR1_windowed_test.py):
  * CAMB background is computed ONCE per likelihood evaluation and reused for every tracer-bin
    (q_perp, q_par for all z_eff come from the same background object). The old code re-ran CAMB inside
    model_multipoles for each tracer, i.e. 5x per joint likelihood call.
  * Emulators come from the avg_M models trained for the 11-parameter ("compact") and Gaussian ("gauss") models.
    Model file names follow the training scripts' zdir(z) = 'z' + str(z).replace('.', 'p'), so z_eff = 0.510
    is 'z0p51' (common.py's loader formats it '0p510' and would not find the new files).
  * s_min is applied by bin centre (s_center >= s_min), the same convention as the existing SMAX_FIT cut, to both
    the data vector and the matching block of the covariance.
  * Sampling stops on the EFT-GSM convergence rule: every CHECK_EVERY steps compute the integrated autocorrelation
    time tau; stop when (i) chain length > TAU_FACTOR * tau for every parameter and (ii) tau changed by less than
    TAU_TOL since the previous check. Chains are stored with emcee's HDF5 backend, so a fit can be resumed.
  * Walkers scale with the number of parameters (WALKERS_PER_PARAM * ndim, rounded up to even).

Fit definitions (build_specs) follow the fit list in the project notes. Run one fit with run_fit.py, all of them
with launch_fits.py.
"""
import os
for _k, _v in (('OMP_NUM_THREADS', '1'), ('OPENBLAS_NUM_THREADS', '1'), ('MKL_NUM_THREADS', '1'),   # forced, not setdefault:
               ('TF_CPP_MIN_LOG_LEVEL', '2'), ('TF_NUM_INTRAOP_THREADS', '1'), ('TF_NUM_INTEROP_THREADS', '1'),
               ('HDF5_USE_FILE_LOCKING', 'FALSE')):                                                   # CAMB/TF must stay 1-thread
    os.environ[_k] = _v

import json
import math
import sys
import time
import zlib
import multiprocessing
from pathlib import Path

import numpy as np
import emcee
import psutil

sys.path.insert(1, str(Path(__file__).resolve().parent.parent))   # finds nb/fits/common.py when run from a subfolder
from common import (COSMO_FID, LNAS_FID, NS_FIXED, NS_MEAN, NS_SIG, OMBH2_MEAN, OMBH2_SIG,
                    TRACERS, ZBINS, Z_EFF, SMAX_FIT, NBINS_COV, ELL_ALL, ELLS_FIT,
                    get_background, RESULTS_FID, ap_multipoles, load_tracer_arrays, data_vec_from,
                    poles_path_for, cov_path_for, save_getdist_chain, PARAM_NAMES_7, PARAM_LABELS_7, Emulator)

NICE = 19


def lower_priority():
    """Set this process to niceness NICE (inherited by its emcee pool workers). Called at the start of each fit process, NOT at
    import time: common.py sets niceness 10 when imported, and a process that inherited a higher value (e.g. 19 from a niced
    launcher) cannot lower it back to 10, which crashed every fit at startup."""
    psutil.Process().nice(NICE)

# common.py excludes BGS (it predates the z = 0.295 emulator node). Add it here so all six DESI tracer-bins are fit:
# BGS_BRIGHT-21.5 (0.1 < z < 0.4, z_eff = 0.295), LRG1-3, ELG2, QSO. BGS mocks live in the 'bright' programme.
BGS = 'BGS_BRIGHT-21.5'
TRACERS = {BGS: 'bright', **TRACERS}
ZBINS = {BGS: [(0.1, 0.4)], **ZBINS}
Z_EFF = {(BGS, 0.1, 0.4): 0.295, **Z_EFF}

# ------------------------------------------------------------------ settings
CHAIN_ROOT = Path('/spiffball/cosweeney/data/chains/fits_11par')
MODELS = {   # model -> (emulator directory, file-name suffix)
    'compact': ('/spiffball/cosweeney/data/models_11par_avg_M/', '_11par'),
    'gauss':   ('/spiffball/cosweeney/data/models_gauss_avg_M/', '_gauss'),
    'compact_lowM':  ('/spiffball/cosweeney/data/models_11par_low_M/',  '_11par'),   # mass-scheme comparison (baseline joint fits only)
    'compact_highM': ('/spiffball/cosweeney/data/models_11par_high_M/', '_11par'),
}
EXTRACT_DIR = Path('/spiffball/cosweeney/DESI/DR1/CorrFuncs/gccomb/gccomb_extracted')
DR1_COV_DIR = Path('/spiffball/cosweeney/DESI/DR1/covariances')
NRAN = {BGS: 1, 'LRG': 8, 'ELG_LOPnotqso': 10, 'QSO': 4}

SMINS_SINGLE_MOCK = (20, 25, 30, 35, 40)     # one group of MOCKS_PER_SMIN mocks per s_min
MOCKS_PER_SMIN = 5
CORES_JOINT, CORES_INDIV = 3, 1              # cores per fit (emcee can use at most ~nwalkers/2)
WALKERS_PER_PARAM = 3

CHECK_EVERY, TAU_FACTOR, TAU_TOL = 100, 100, 0.01     # EFT-GSM convergence rule
MAX_STEPS = 300_000                                   # safety cap; a fit that hits it is flagged converged=false
BURN_TAUS, THIN_TAUS = 5, 0.5                         # saved chain: discard 5 tau_max, thin by tau_max/2
LOG_EVERY = 1000

FIT_KEYS = [(tracer, zmin, zmax) for tracer in TRACERS for zmin, zmax in ZBINS[tracer]]   # 6 tracer-bins: BGS, LRG1-3, ELG2, QSO


def kname(key):
    return f'{key[0]}_{key[1]}-{key[2]}'


SHORT_NAMES = dict(zip((kname(k) for k in FIT_KEYS), ['BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG2', 'QSO']))   # FIT_KEYS order


def zdir(z):
    return f"z{str(z).replace('.', 'p')}"


# ------------------------------------------------------------------ fit list
def mock_counts():
    return {key: np.load(poles_path_for(key[0], TRACERS[key[0]], key[1], key[2]))['poles'].shape[0]
            for key in FIT_KEYS}


def build_specs():
    specs = []

    def add(name, model, data, mode, keys, smin, mock_idx=None, priority=0, **extra):
        specs.append(dict(name=name, model=model, data=data, mode=mode, keys=keys, smin=smin,
                          mock_idx=mock_idx, priority=priority,
                          ncores=CORES_JOINT if mode == 'joint' else CORES_INDIV, **extra))

    for model in ('compact', 'gauss'):
        if model == 'compact':
            for key in FIT_KEYS:
                add(f'compact_mean_indiv_{kname(key)}_smin20', 'compact', 'mock_mean', 'indiv', [key], 20)
        for smin in (20, 40):
            add(f'{model}_mean_joint_smin{smin}', model, 'mock_mean', 'joint', FIT_KEYS, smin)
        if model == 'compact':
            for key in FIT_KEYS:
                add(f'compact_dr1_indiv_{kname(key)}_smin20', 'compact', 'dr1', 'indiv', [key], 20)
        for smin in (20, 40):
            add(f'{model}_dr1_joint_smin{smin}', model, 'dr1', 'joint', FIT_KEYS, smin)

    # mass-scheme comparison: baseline joint fits (s_min = 20) with the compact low_M / high_M emulators
    for model in ('compact_lowM', 'compact_highM'):
        for data, tag in (('mock_mean', 'mean'), ('dr1', 'dr1')):
            add(f'{model}_{tag}_joint_smin20', model, data, 'joint', FIT_KEYS, 20)

    # (a) mean-of-mocks joint fit with the AVERAGED covariance (single-mock covariance / n_mock, per tracer) instead of the single-mock one
    add('compact_mean_joint_avgcov_smin20', 'compact', 'mock_mean', 'joint', FIT_KEYS, 20, cov_by_nmock=True)

    # (b) DR1 leave-one-out: joint fits of all tracers but one (s_min = 20, compact avg_M)
    for left_out in FIT_KEYS:
        add(f'compact_dr1_joint_no{SHORT_NAMES[kname(left_out)]}_smin20', 'compact', 'dr1', 'joint',
            [k for k in FIT_KEYS if k != left_out], 20)

    # single-mock joint fits (compact only): the first n_common mocks, split into groups of MOCKS_PER_SMIN, one group per s_min.
    # Mock i is assumed to be row i of every tracer's poles array (checked against the counts below).
    counts = mock_counts()
    n_common = min(counts.values())
    for g, smin in enumerate(SMINS_SINGLE_MOCK):
        for i in range(g * MOCKS_PER_SMIN, min((g + 1) * MOCKS_PER_SMIN, n_common)):
            add(f'compact_mock{i:02d}_joint_smin{smin}', 'compact', 'mock_single', 'joint', FIT_KEYS, smin,
                mock_idx=i, priority=1)
    return specs


# ------------------------------------------------------------------ data
def dr1_paths(key):
    tracer, zmin, zmax = key
    stem = (f'allcounts_{tracer}_GCcomb_z{zmin}-{zmax}_default_FKP_lin_'
            f'nran{NRAN[tracer]}_njack0_split20_thetacut0.05')
    cov = DR1_COV_DIR / (f'xi024_{tracer}_GCcomb_{zmin}_{zmax}_default_FKP_lin4_s20-200_cov_RascalC_rescaled.txt')
    return EXTRACT_DIR / (stem + '.npz'), cov


def missing_files():
    """Data files the fits need but that do not exist (mock poles + covariance, DR1 extraction + covariance)."""
    paths = []
    for key in FIT_KEYS:
        tracer, zmin, zmax = key
        paths += [poles_path_for(tracer, TRACERS[tracer], zmin, zmax), cov_path_for(tracer, TRACERS[tracer], zmin, zmax),
                  *dr1_paths(key)]
    return [p for p in paths if not Path(p).exists()]


def window_at(W, s_fine, s_query):
    """Linearly interpolate each W[lo, lt, :] onto s_query (same simplification as the DR1 test scripts)."""
    n_lo, n_lt, _ = W.shape
    out = np.empty((n_lo, n_lt, len(s_query)))
    for i in range(n_lo):
        for j in range(n_lt):
            out[i, j] = np.interp(s_query, s_fine, W[i, j])
    return out


def load_mock_fit_data(key, smin, mock_idx=None, cov_by_nmock=False):
    """(s_data, data_vec, cov) for one tracer-bin; mock_idx=None -> mean over mocks. Covariance is the
    single-mock (unscaled) one, cut to s >= smin."""
    tracer, zmin, zmax = key
    s_data, poles, mask, cov = load_tracer_arrays(tracer, TRACERS[tracer], zmin, zmax)
    keep = s_data >= smin
    n = len(s_data)
    idx = np.concatenate([np.where(keep)[0] + k * n for k in range(len(ELLS_FIT))])
    mask2 = mask.copy()
    mask2[np.where(mask)[0][~keep]] = False
    cov = cov[np.ix_(idx, idx)]
    if cov_by_nmock:                                  # covariance of the mean of the available mocks
        cov = cov / poles.shape[0]
    return s_data[keep], data_vec_from(poles, mask2, mock_idx), cov


def load_dr1_fit_data(key, smin):
    """(s_data, data_vec, cov, W) for one DR1 tracer-bin: DR1 multipoles, DR1 covariance, theta-cut window."""
    npz_path, cov_path = dr1_paths(key)
    npz = np.load(npz_path)
    s_full, xi_full, W_fine, s_fine = npz['s_full'], npz['xi_full'], npz['W_fine'], npz['s_fine']
    assert tuple(npz['ells_obs']) == ELLS_FIT and tuple(npz['ells_theory']) == ELL_ALL, \
        'extracted DR1 file ells do not match common.ELLS_FIT / ELL_ALL'
    cov_full = np.loadtxt(cov_path)
    mask = (s_full <= SMAX_FIT) & (s_full >= smin)
    idx = np.concatenate([ELL_ALL.index(ell) * NBINS_COV + np.where(mask)[0] for ell in ELLS_FIT])
    s_data = s_full[mask]
    data_vec = np.concatenate([xi_full[ELL_ALL.index(ell)][mask] for ell in ELLS_FIT])
    return s_data, data_vec, cov_full[np.ix_(idx, idx)], window_at(W_fine, s_fine, s_data)


# ------------------------------------------------------------------ model
_emu_cache = {}


def get_emus(model, z_eff):
    """Load the three multipole emulators for (model, z_eff) once per process."""
    if (model, z_eff) not in _emu_cache:
        directory, suffix = MODELS[model]
        _emu_cache[(model, z_eff)] = tuple(
            Emulator(os.path.join(directory, f'compact_xi{ell}_{zdir(z_eff)}{suffix}')) for ell in (0, 2, 4))
    return _emu_cache[(model, z_eff)]


_DA_FID = {z: RESULTS_FID.angular_diameter_distance(z) for z in set(Z_EFF.values())}
_H_FID = {z: RESULTS_FID.hubble_parameter(z) for z in set(Z_EFF.values())}


def ap_multi(h, ombh2, omch2, ns, z_effs):
    """(q_perp, q_par) for every z in z_effs from ONE CAMB background call."""
    r = get_background(h, ombh2, omch2, ns)
    return [(r.angular_diameter_distance(z) / _DA_FID[z], _H_FID[z] / r.hubble_parameter(z)) for z in z_effs]


def model_one(emus, emu_params, s_data, W, q_perp, q_par):
    """Predicted multipole vector for one tracer-bin (windowed if W is given)."""
    _, xi0 = emus[0](emu_params)
    _, xi2 = emus[1](emu_params)
    _, xi4 = emus[2](emu_params)
    if W is None:
        return ap_multipoles(s_data, ELLS_FIT, xi0, xi2, xi4, q_perp, q_par).ravel()
    xi_ap = ap_multipoles(s_data, ELL_ALL, xi0, xi2, xi4, q_perp, q_par)
    return np.einsum('ots,ts->os', W, xi_ap).ravel()


class FitProblem:
    """Posterior for one fit spec. indiv: theta = [h, ombh2, omch2, lnAs, b1, b2, ns];
    joint: theta = [h, ombh2, omch2, lnAs, ns, b1_0, b2_0, b1_1, b2_1, ...] (one bias pair per tracer-bin)."""

    def __init__(self, spec):
        self.spec = spec
        self.joint = spec['mode'] == 'joint'
        self.data = []
        for key in spec['keys']:
            if spec['data'] == 'dr1':
                s_data, data_vec, cov, W = load_dr1_fit_data(key, spec['smin'])
            else:
                s_data, data_vec, cov = load_mock_fit_data(key, spec['smin'], spec['mock_idx'], spec.get('cov_by_nmock', False))
                W = None
            _, logdet = np.linalg.slogdet(2 * np.pi * cov)
            self.data.append(dict(z_eff=Z_EFF[key], s_data=s_data, data_vec=data_vec, W=W,
                                  cov_inv=np.linalg.inv(cov), logdet=logdet,
                                  emus=get_emus(spec['model'], Z_EFF[key])))
        self.z_effs = [d['z_eff'] for d in self.data]
        nb = len(self.data)
        if self.joint:
            self.ndim = 5 + 2 * nb
            self.theta0 = np.concatenate([[COSMO_FID[0], COSMO_FID[1], COSMO_FID[2], LNAS_FID, NS_FIXED], np.tile([1.0, 0.0], nb)])
            self.scale = np.concatenate([[0.01, 0.0001, 0.002, 0.02, 0.01], np.tile([0.02, 0.1], nb)])
            self.param_names = ['h', 'ombh2', 'omch2', 'logA', 'ns']
            self.param_labels = ['h', r'\omega_b', r'\omega_{cdm}', r'\ln(10^{10}A_s)', 'n_s']
            for key in spec['keys']:
                self.param_names += [f'b1_{kname(key)}', f'b2_{kname(key)}']
                self.param_labels += [f'b_1^{{{key[0]}}}', f'b_2^{{{key[0]}}}']
        else:
            self.ndim = 7
            self.theta0 = np.array([COSMO_FID[0], COSMO_FID[1], COSMO_FID[2], LNAS_FID, 1.0, 0.0, NS_FIXED])
            self.scale = np.array([0.01, 0.0001, 0.002, 0.02, 0.02, 0.1, 0.01])
            self.param_names, self.param_labels = PARAM_NAMES_7, PARAM_LABELS_7

    def unpack(self, theta):
        if self.joint:
            h, ombh2, omch2, lnAs, ns = theta[:5]
            return h, ombh2, omch2, lnAs, ns, [(theta[5 + 2 * i], theta[6 + 2 * i]) for i in range(len(self.data))]
        h, ombh2, omch2, lnAs, b1, b2, ns = theta
        return h, ombh2, omch2, lnAs, ns, [(b1, b2)]

    def log_prior(self, theta):
        h, ombh2, omch2, lnAs, ns, biases = self.unpack(theta)
        if not (0.55 < h < 0.91):     return -np.inf
        if not (0.08 < omch2 < 0.16): return -np.inf
        if not (2.0 < lnAs < 4.0):    return -np.inf
        for b1, b2 in biases:
            if not (0.0 < b1 < 2.0):   return -np.inf
            if not (-5.0 < b2 < 10.0): return -np.inf
        return -0.5 * ((ombh2 - OMBH2_MEAN) / OMBH2_SIG) ** 2 - 0.5 * ((ns - NS_MEAN) / NS_SIG) ** 2

    def log_likelihood(self, theta):
        h, ombh2, omch2, lnAs, ns, biases = self.unpack(theta)
        try:
            aps = ap_multi(h, ombh2, omch2, ns, self.z_effs)               # one CAMB call for all tracer-bins
            total = 0.0
            for d, (b1, b2), (q_perp, q_par) in zip(self.data, biases, aps):
                model = model_one(d['emus'], [b1, b2, ombh2, omch2, h, lnAs, ns], d['s_data'], d['W'], q_perp, q_par)
                if not np.all(np.isfinite(model)):
                    return -np.inf
                resid = model - d['data_vec']
                total += -0.5 * (resid @ d['cov_inv'] @ resid + d['logdet'])
        except Exception:
            return -np.inf
        return total

    def __call__(self, theta):
        lp = self.log_prior(theta)
        return -np.inf if not np.isfinite(lp) else lp + self.log_likelihood(theta)


_PROBLEM = None


def _log_prob(theta):      # module-level so the emcee pool pickles it by name; workers (fork) inherit _PROBLEM
    return _PROBLEM(theta)


# ------------------------------------------------------------------ sampling with the EFT-GSM convergence rule
def run_chain(problem, root, ncores, max_steps=MAX_STEPS, check_every=CHECK_EVERY,
              tau_factor=TAU_FACTOR, tau_tol=TAU_TOL, tag=''):
    global _PROBLEM
    _PROBLEM = problem
    root = Path(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    status_path = Path(f'{root}.status.json')
    if status_path.exists() and json.load(open(status_path)).get('converged'):
        print(f'{tag}already converged -> skipping', flush=True)
        return
    ndim = problem.ndim
    nwalkers = 2 * math.ceil(WALKERS_PER_PARAM * ndim / 2)

    backend = emcee.backends.HDFBackend(f'{root}.h5')
    resume = backend.initialized and backend.iteration > 0
    if not resume:
        backend.reset(nwalkers, ndim)
        rng = np.random.default_rng(zlib.crc32(str(root.name).encode()))
        pos = np.empty((nwalkers, ndim))
        i = 0
        while i < nwalkers:                                   # start inside the prior
            p = problem.theta0 + problem.scale * rng.standard_normal(ndim)
            if np.isfinite(problem.log_prior(p)):
                pos[i] = p
                i += 1
        state = pos
    else:
        state = backend.get_last_sample()                    # explicit: sample(None) is not reliable across emcee versions
    print(f'{tag}{"resuming at step " + str(backend.iteration) if resume else "starting"}: '
          f'ndim={ndim}, nwalkers={nwalkers}, ncores={ncores}', flush=True)

    pool = multiprocessing.get_context('fork').Pool(ncores) if ncores > 1 else None
    t0 = time.time()
    converged, old_tau = False, np.full(ndim, np.inf)
    try:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, _log_prob, pool=pool, backend=backend)
        for _ in sampler.sample(state, iterations=max_steps - sampler.iteration, progress=False):
            it = sampler.iteration
            if it % check_every:
                continue
            tau = sampler.get_autocorr_time(tol=0)
            long_enough = bool(np.all(tau * tau_factor < it))
            stable = bool(np.all(np.abs(old_tau - tau) / tau < tau_tol))
            if it % LOG_EVERY == 0:
                print(f'{tag}step {it}: tau_max={np.nanmax(tau):.0f}, N/tau_max={it / np.nanmax(tau):.0f}, '
                      f'long_enough={long_enough}, stable={stable}, {(time.time() - t0) / 60:.1f} min', flush=True)
            old_tau = tau
            if long_enough and stable:
                converged = True
                break
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    it = int(sampler.iteration)
    tau = sampler.get_autocorr_time(tol=0)
    tau_max = float(np.nanmax(tau))
    discard = min(int(BURN_TAUS * tau_max), it // 2)
    thin = max(1, int(THIN_TAUS * tau_max))
    save_getdist_chain(root, sampler, discard=discard, thin=thin,
                       param_names=problem.param_names, param_labels=problem.param_labels)
    with open(status_path, 'w') as f:
        json.dump(dict(converged=bool(converged), iterations=it, tau_max=tau_max, tau=[float(t) for t in tau],
                       n_over_tau_max=float(it / tau_max), nwalkers=int(nwalkers), ndim=int(ndim), discard=int(discard), thin=int(thin),
                       wall_minutes_this_run=(time.time() - t0) / 60, spec={k: v for k, v in problem.spec.items() if k != 'keys'},
                       keys=[kname(k) for k in problem.spec['keys']]), f, indent=1)
    print(f'{tag}{"CONVERGED" if converged else "NOT CONVERGED (hit max_steps)"} at step {it}: tau_max={tau_max:.0f}, '
          f'N/tau_max={it / tau_max:.0f} -> {root}.txt', flush=True)


def is_converged(spec):
    p = CHAIN_ROOT / spec['model'] / f"{spec['name']}.status.json"
    return p.exists() and bool(json.load(open(p)).get('converged'))


def run_spec(name, **overrides):
    lower_priority()
    spec = next(s for s in build_specs() if s['name'] == name)
    problem = FitProblem(spec)
    run_chain(problem, CHAIN_ROOT / spec['model'] / name, spec['ncores'], tag=f'[{name}] ', **overrides)