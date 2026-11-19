"""
Refit of the 11-parameter ("compact") and Gaussian model fits with the AP fixes built in. Standalone: it does NOT import fit_lib_11par; from the repo it
only uses nb/fits/common.py for data/emulator/chain I/O and the (s, mu) AP transform `ap_multipoles`. The AP parameters are computed here:

    q_perp = (h / h_fid) D_A / D_A^fid,   q_par = (h / h_fid) H^fid / H,   reference = AbacusSummit c000 (the DESI fiducial; h_fid = 0.6736)

because the data separations are in h_fid^-1 Mpc and the emulator output is in h^-1 Mpc at the sampled h. This does not depend on whether common.py has
been patched. For the w0wa fits the sampled background is a CAMB PPF w0wa one; the reference stays the c000 LCDM cosmology. Put this file next to
fit_lib_11par.py (nb/fits/11par_fits/); run it in the compact environment.

Priors: DESI DR1 forms (BBN Gaussian on omega_b, N(0.9649, 0.0422^2) on n_s), NOT clipped to the range the NN emulators were trained on by default: untruncated
Gaussians and the flat ranges WIDE (h 0.55-0.91, omega_cdm 0.08-0.16, lnAs 2-4, b1 0-2, b2 -5-10). The emulators extrapolate outside their training box, so check
the fraction of the posterior outside it (compare_apfix_chains.py, section 4c). Shift fits (spec flag shift='fixed' / 'marg', joint fits only): the model (emulator input and AP) is evaluated at theta + Delta, where theta are the sampled (corrected) parameters, the priors act on theta, and Delta
= mean over the single-mock baseline fits of the posterior mean - c000 truth for (h, omega_b, omega_cdm, ln A_s, n_s), i.e. the pipeline's mean offset is put into the likelihood. 'fixed': Delta is held fixed;
'marg': Delta is marginalized with a Gaussian prior of mean Delta and std SYS_FRACTION x |Delta| (the shift +- shift/2), as five extra sampled parameters shift_h ... shift_ns.
Spec flags (only the flatprior fits use flat_priors; none uses hard_box or fix_ns): flat_priors=True -> omega_b and n_s get FLAT priors over the NN training box (BOX) instead of the DESI
Gaussians, everything else unchanged; hard_box=True -> every prior cut at the training box (BOX); fix_ns=True -> n_s fixed at 0.9649 (as EFT-GSM) and dropped from the parameter vector; w0wa=True -> parameter vector gains
(w0, wa) after the LCDM block, and w0, wa are ALWAYS limited to the w0wa training box (the NN error outside it grows to percent level within ~0.1 in w0 / 0.2 in wa).

Chains go to CHAIN_ROOT/<model>/<name>.{h5,txt,status.json}, with the same <model> folders and <name>s as the old fits (plus compact_w0wa). Joint-fit parameter
order: [h, ombh2, omch2, logA, (ns), (w0, wa), b1_0, b2_0, ...]; the .paramnames file has the names. Every fit samples with emcee (HDF5 backend, resumable)
and stops on the EFT-GSM rule (chain length > TAU_FACTOR x tau for every parameter and tau stable to TAU_TOL).

Usage:
    python fits_apfix.py mocks                                 # how many mocks each tracer has, and which are missing (if the files store ids)
    python fits_apfix.py shift  [--force]                      # compute / show the likelihood-level shift from the single-mock fits
    python fits_apfix.py list   [--groups dr1,mock_mean,mock_single,w0wa,loo,flatprior,shift]
    python fits_apfix.py status [--groups ...]                 # progress of every fit
    python fits_apfix.py run NAME [NAME ...] [--max-steps N]   # run fits one after the other in this process (resumes if a chain exists)
    python fits_apfix.py launch [--groups ...] [--cores 48] [--only REGEX] [--dry-run]    # run many fits in parallel under a core budget
Default groups (all of them): dr1, mock_mean, mock_single, w0wa, loo (DR1 leave-one-out, one tracer dropped each), flatprior (baseline with flat omega_b, n_s priors), shift (baseline with the
single-mock shift put into the likelihood, see below): 70 fits. The shift fits need the single-mock fits to have converged: `launch` skips them with a message until then (rerun `launch` afterwards), or
run `python fits_apfix.py shift` to compute and print the shift from the single-mock chains (written to CHAIN_ROOT/shift.json, which the shift fits then read; delete it to recompute).
"""
import os
for _k, _v in (('OMP_NUM_THREADS', '1'), ('OPENBLAS_NUM_THREADS', '1'), ('MKL_NUM_THREADS', '1'),   # forced, before numpy loads: the fork-based emcee
               ('TF_CPP_MIN_LOG_LEVEL', '2'), ('TF_NUM_INTRAOP_THREADS', '1'), ('TF_NUM_INTEROP_THREADS', '1'),   # pool needs 1-thread BLAS/OpenMP/TF
               ('HDF5_USE_FILE_LOCKING', 'FALSE')):
    os.environ[_k] = _v

import argparse
import glob
import json
import math
import re
import subprocess
import sys
import time
import zlib
import multiprocessing
from pathlib import Path

import numpy as np
import camb
import emcee
import psutil

sys.path.insert(1, str(Path(__file__).resolve().parent.parent))   # finds nb/fits/common.py when run from a subfolder
from common import (NS_FIXED, TRACERS, ZBINS, Z_EFF, SMAX_FIT, NBINS_COV, ELL_ALL, ELLS_FIT,
                    get_background, ap_multipoles, load_tracer_arrays, data_vec_from,
                    poles_path_for, cov_path_for, save_getdist_chain, Emulator)

NICE = 10


def lower_priority():
    """Niceness NICE for this process (inherited by its emcee pool workers). Called at the start of each fit process, not at import: a process that
    inherited a higher niceness (e.g. 19 from a niced launcher) cannot lower it back to 10."""
    psutil.Process().nice(NICE)


# common.py excludes BGS (it predates the z = 0.295 emulator node). Add it here so all six DESI tracer-bins are fit.
BGS = 'BGS_BRIGHT-21.5'
TRACERS = {BGS: 'bright', **TRACERS}
ZBINS = {BGS: [(0.1, 0.4)], **ZBINS}
Z_EFF = {(BGS, 0.1, 0.4): 0.295, **Z_EFF}

# ------------------------------------------------------------------ settings
CHAIN_ROOT = Path('/spiffball/cosweeney/data/chains/fits_11par_apfix_v2')
MODELS = {   # model -> (emulator directory, file-name suffix); w0wa emulator input = [b1, b2, ombh2, omch2, h, lnAs, ns, w0, wa]
    'compact':       ('/spiffball/cosweeney/data/models_11par_avg_M/',      '_11par'),
    'gauss':         ('/spiffball/cosweeney/data/models_gauss_avg_M/',      '_gauss'),
    'compact_lowM':  ('/spiffball/cosweeney/data/models_11par_low_M/',      '_11par'),
    'compact_highM': ('/spiffball/cosweeney/data/models_11par_high_M/',     '_11par'),
    'compact_w0wa':  ('/spiffball/cosweeney/data/models_11par_w0wa_avg_M/', '_11par_w0wa'),
}
EXTRACT_DIR = Path('/spiffball/cosweeney/DESI/DR1/CorrFuncs/gccomb/gccomb_extracted')
DR1_COV_DIR = Path('/spiffball/cosweeney/DESI/DR1/covariances')
NRAN = {BGS: 1, 'LRG': 8, 'ELG_LOPnotqso': 10, 'QSO': 4}

AP_REF = (0.6736, 0.02237, 0.12)      # AP reference = DESI fiducial cosmology, AbacusSummit c000: (h, omega_b, omega_cdm)
LNAS_C000 = float(np.log(1e10 * 2.0830e-9))
START = dict(h=AP_REF[0], ombh2=AP_REF[1], omch2=AP_REF[2], logA=LNAS_C000, ns=NS_FIXED, w0=-1.0, wa=0.0)    # walkers start in a ball around c000
START_SCALE = dict(h=0.01, ombh2=0.0001, omch2=0.002, logA=0.02, ns=0.01, w0=0.05, wa=0.1)

# ---- likelihood-level shift (the 'shift' fits)
SHIFT_PARAMS = ['h', 'ombh2', 'omch2', 'logA', 'ns']          # parameters that carry the shift
SHIFT_NAMES = [f'shift_{p}' for p in SHIFT_PARAMS]
SYS_FRACTION = 0.5                                             # 'marg' fits: Gaussian prior on the shift with std = SYS_FRACTION x |shift|
SHIFT_PATH = CHAIN_ROOT / 'shift.json'
TRUTH_SAMPLED = dict(h=AP_REF[0], ombh2=AP_REF[1], omch2=AP_REF[2], logA=LNAS_C000, ns=NS_FIXED)      # c000

# ---- priors: DESI DR1 functional forms; BOX = NN training box (used only when hard_box, and for w0/wa), WIDE = looser flat ranges used otherwise ----
HARD_BOX = False                      # default for every fit: priors NOT clipped to the NN training box; per-spec override with hard_box=True (w0, wa always boxed)
OMBH2_PRIOR = (0.02198, 0.00053)      # DESI BBN prior on omega_b with Neff = 3.044 (DESI 2024 V Sec. 4.7, 2024 VII); Table 4 of 2024 V lists (0.02218, 0.00055)
NS_PRIOR = (0.9649, 0.0422)           # DESI "ns10"
BOX = {                               # range the NN emulators were trained on (generate_multiz_11par*.py BOUNDS); logA upper edge 3.91 = DESI's (box: 4)
    'h': (0.55, 0.90), 'ombh2': (0.0207, 0.0243), 'omch2': (0.104, 0.140), 'logA': (2.0, 3.91), 'ns': (0.901, 1.025),
    'b1': (0.0, 2.0), 'b2': (-5.0, 10.0), 'w0': (-1.271, -0.726), 'wa': (-0.628, 0.621)}
WIDE = {                              # flat ranges used by default (hard_box off)
    'h': (0.55, 0.91), 'omch2': (0.08, 0.16), 'logA': (2.0, 4.0), 'b1': (0.0, 2.0), 'b2': (-5.0, 10.0)}

SMINS = (20, 25, 30, 35, 40)
N_MOCKS_EXPECTED = 25         # fallback only: the number of single-mock fits is taken from the data (the mocks common to all tracers)
CORES_JOINT, CORES_INDIV = 4, 1       # cores per fit (emcee can use at most ~nwalkers/2); 4 packs the 112 cores of calvinball better than 3 (~14 h vs ~18 h for the full list)
WALKERS_PER_PARAM = 3
CHECK_EVERY, TAU_FACTOR, TAU_TOL = 100, 100, 0.01
MAX_STEPS = 1_000_000                 # safety cap; a fit that hits it is flagged converged = false (the first v2 baselines needed ~480k-590k steps)
BURN_TAUS, THIN_TAUS = 5, 0.5         # saved chain: discard 5 tau_max, thin by tau_max/2
LOG_EVERY = 1000
GROUPS = ('dr1', 'mock_mean', 'mock_single', 'w0wa', 'loo', 'flatprior', 'shift')
DEFAULT_GROUPS = GROUPS

FIT_KEYS = [(tracer, zmin, zmax) for tracer in TRACERS for zmin, zmax in ZBINS[tracer]]   # 6 tracer-bins: BGS, LRG1-3, ELG2, QSO


def kname(key):
    return f'{key[0]}_{key[1]}-{key[2]}'


SHORT_NAMES = dict(zip((kname(k) for k in FIT_KEYS), ['BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG2', 'QSO']))   # FIT_KEYS order


def zdir(z):
    return f"z{str(z).replace('.', 'p')}"


# ------------------------------------------------------------------ fit list
def mock_counts():
    return {key: np.load(poles_path_for(key[0], TRACERS[key[0]], key[1], key[2]))['poles'].shape[0] for key in FIT_KEYS}


def mock_report():
    """Print, for every tracer bin, the number of mocks in its poles file and any extra arrays in it (e.g. mock ids / phases); if such an id array exists in all files, also which ids are missing where."""
    ids, counts = {}, {}
    for key in FIT_KEYS:
        d = np.load(poles_path_for(key[0], TRACERS[key[0]], key[1], key[2]))
        n = d['poles'].shape[0]
        counts[kname(key)] = n
        extra = [k for k in d.files if k not in ('s', 'poles')]
        found = {k: np.asarray(d[k]).ravel() for k in extra if np.asarray(d[k]).ndim <= 1 and np.asarray(d[k]).size == n}
        if found:
            ids[kname(key)] = found
        print(f"{kname(key):28s} {n:3d} mocks | other arrays in the file: {extra if extra else 'none'}")
    nmin = min(counts.values())
    print(f"\nmocks common to all tracers (index 0-{nmin - 1}): {nmin}; the fits use these, assuming mock i is the same realization in every tracer's file"
          + ('' if len(set(counts.values())) == 1 else f'; counts differ: tracers with fewer mocks limit the list'))
    for name in sorted({k for f in ids.values() for k in f}):
        if all(name in f for f in ids.values()) and len(ids) == len(counts):
            union = sorted(set().union(*[set(f[name].tolist()) for f in ids.values()]))
            print(f"\nid array '{name}' (union of all files: {len(union)} values):")
            for tr, f in ids.items():
                missing = [v for v in union if v not in set(f[name].tolist())]
                print(f"  {tr:28s} missing {missing if missing else 'nothing'}")
            same = all(np.array_equal(f[name], next(iter(ids.values()))[name]) for f in ids.values())
            print(f"  same ids in the same order in every file: {same}  (the single-mock fits need this: mock i must be the same realization for every tracer)")
    if not ids:
        print('\nno per-mock id array is stored in the poles files, so which mocks are missing cannot be read from them: compare the list of mock files the poles were built from with the full list of realizations')


def build_specs():
    """All fits. Mock fits use the mean-of-mocks data vector ('mock_mean') or one mock ('mock_single') with the single-mock covariance, except the
    avgcov fit (covariance / number of mocks). The baseline is the compact (11-parameter, avg_M) joint fit of all tracers."""
    specs = []

    def add(name, model, data, mode, keys, smin, group, priority, mock_idx=None, **extra):
        specs.append(dict(name=name, model=model, data=data, mode=mode, keys=keys, smin=smin, group=group, priority=priority, mock_idx=mock_idx,
                          ncores=CORES_JOINT if mode == 'joint' else CORES_INDIV, **extra))

    for data, tag, group in (('dr1', 'dr1', 'dr1'), ('mock_mean', 'mean', 'mock_mean')):
        for smin in SMINS:                                                                 # baseline at each s_min
            add(f'compact_{tag}_joint_smin{smin}', 'compact', data, 'joint', FIT_KEYS, smin, group, 0 if smin == 20 else 1)
        for model in ('compact_lowM', 'compact_highM'):                                    # the other two mass schemes (avg_M is the baseline)
            add(f'{model}_{tag}_joint_smin20', model, data, 'joint', FIT_KEYS, 20, group, 1)
        add(f'gauss_{tag}_joint_smin20', 'gauss', data, 'joint', FIT_KEYS, 20, group, 1)   # Gaussian model
        for key in FIT_KEYS:                                                               # each tracer alone
            add(f'compact_{tag}_indiv_{kname(key)}_smin20', 'compact', data, 'indiv', [key], 20, group, 1)
    add('compact_mean_joint_avgcov_smin20', 'compact', 'mock_mean', 'joint', FIT_KEYS, 20, 'mock_mean', 1, cov_by_nmock=True)   # covariance / n_mocks

    counts = mock_counts()
    n_mocks = min(counts.values())                                                         # baseline fit to each mock available for ALL tracers (the first n_mocks)
    if len(set(counts.values())) > 1:
        print(f'WARNING: the tracers have different numbers of mocks {dict((kname(k), v) for k, v in counts.items())}; using the first {n_mocks} of each '
              f'(run `python fits_apfix.py mocks` to see which are missing)', flush=True)
    for i in range(n_mocks):
        add(f'compact_mock{i:02d}_joint_smin20', 'compact', 'mock_single', 'joint', FIT_KEYS, 20, 'mock_single', 2, mock_idx=i)

    for data, tag in (('dr1', 'dr1'), ('mock_mean', 'mean')):
        for smin in (20, 30):                                                              # w0waCDM baseline (w0, wa limited to the NN box)
            add(f'compact_w0wa_{tag}_joint_smin{smin}', 'compact_w0wa', data, 'joint', FIT_KEYS, smin, 'w0wa', 1, w0wa=True)

    for data, tag in (('dr1', 'dr1'), ('mock_mean', 'mean')):                               # baseline with flat omega_b, n_s priors over the NN box (instead of the DESI Gaussians)
        add(f'compact_flatprior_{tag}_joint_smin20', 'compact', data, 'joint', FIT_KEYS, 20, 'flatprior', 1, flat_priors=True)

    for data, tag in (('dr1', 'dr1'), ('mock_mean', 'mean')):                               # baseline with the single-mock shift in the likelihood (needs the mock_single fits)
        add(f'compact_shift_{tag}_joint_smin20', 'compact', data, 'joint', FIT_KEYS, 20, 'shift', 2, shift='fixed')
        add(f'compact_shiftsys_{tag}_joint_smin20', 'compact', data, 'joint', FIT_KEYS, 20, 'shift', 2, shift='marg')

    for left_out in FIT_KEYS:                                                              # DR1 leave-one-out (for the tracer-tension tests)
        add(f'compact_dr1_joint_no{SHORT_NAMES[kname(left_out)]}_smin20', 'compact', 'dr1', 'joint', [k for k in FIT_KEYS if k != left_out], 20,
            'loo', 3)
    return specs


def select(specs, groups):
    return [s for s in specs if s['group'] in groups]


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
        paths += [poles_path_for(tracer, TRACERS[tracer], zmin, zmax), cov_path_for(tracer, TRACERS[tracer], zmin, zmax), *dr1_paths(key)]
    return [p for p in paths if not Path(p).exists()]


def window_at(W, s_fine, s_query):
    """Linearly interpolate each W[lo, lt, :] onto s_query."""
    n_lo, n_lt, _ = W.shape
    out = np.empty((n_lo, n_lt, len(s_query)))
    for i in range(n_lo):
        for j in range(n_lt):
            out[i, j] = np.interp(s_query, s_fine, W[i, j])
    return out


def load_mock_fit_data(key, smin, mock_idx=None, cov_by_nmock=False):
    """(s_data, data_vec, cov) for one tracer-bin; mock_idx=None -> mean over mocks. Covariance is the single-mock one cut to s >= smin
    (divided by the number of mocks if cov_by_nmock)."""
    tracer, zmin, zmax = key
    s_data, poles, mask, cov = load_tracer_arrays(tracer, TRACERS[tracer], zmin, zmax)
    keep = s_data >= smin
    n = len(s_data)
    idx = np.concatenate([np.where(keep)[0] + k * n for k in range(len(ELLS_FIT))])
    mask2 = mask.copy()
    mask2[np.where(mask)[0][~keep]] = False
    cov = cov[np.ix_(idx, idx)]
    if cov_by_nmock:
        cov = cov / poles.shape[0]
    return s_data[keep], data_vec_from(poles, mask2, mock_idx), cov


def load_dr1_fit_data(key, smin):
    """(s_data, data_vec, cov, W) for one DR1 tracer-bin: DR1 multipoles, DR1 covariance, theta-cut window."""
    npz_path, cov_path = dr1_paths(key)
    npz = np.load(npz_path)
    s_full, xi_full, W_fine, s_fine = npz['s_full'], npz['xi_full'], npz['W_fine'], npz['s_fine']
    assert tuple(npz['ells_obs']) == ELLS_FIT and tuple(npz['ells_theory']) == ELL_ALL, 'extracted DR1 file ells do not match common.ELLS_FIT / ELL_ALL'
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
        _emu_cache[(model, z_eff)] = tuple(Emulator(os.path.join(directory, f'compact_xi{ell}_{zdir(z_eff)}{suffix}')) for ell in (0, 2, 4))
    return _emu_cache[(model, z_eff)]


_REF = get_background(*AP_REF, NS_FIXED)
_DA_REF = {z: _REF.angular_diameter_distance(z) for z in set(Z_EFF.values())}
_H_REF = {z: _REF.hubble_parameter(z) for z in set(Z_EFF.values())}


def get_background_w0wa(h, ombh2, omch2, ns, w0, wa):
    """Flat w0waCDM background (CAMB PPF); reduces exactly to LCDM at (w0, wa) = (-1, 0)."""
    pars = camb.set_params(H0=100.0 * h, ombh2=ombh2, omch2=omch2, ns=ns, w=w0, wa=wa, dark_energy_model='ppf')
    return camb.get_background(pars)


def ap_multi(h, ombh2, omch2, ns, z_effs, w0=None, wa=None):
    """(q_perp, q_par) for every z in z_effs from ONE CAMB background call, referenced to c000 (LCDM) and in h^-1 Mpc units (factor h / h_fid)."""
    r = get_background(h, ombh2, omch2, ns) if w0 is None else get_background_w0wa(h, ombh2, omch2, ns, w0, wa)
    f = h / AP_REF[0]
    return [(f * r.angular_diameter_distance(z) / _DA_REF[z], f * _H_REF[z] / r.hubble_parameter(z)) for z in z_effs]


def model_one(emus, emu_params, s_data, W, q_perp, q_par):
    """Predicted multipole vector for one tracer-bin (windowed if W is given)."""
    _, xi0 = emus[0](emu_params)
    _, xi2 = emus[1](emu_params)
    _, xi4 = emus[2](emu_params)
    if W is None:
        return ap_multipoles(s_data, ELLS_FIT, xi0, xi2, xi4, q_perp, q_par).ravel()
    xi_ap = ap_multipoles(s_data, ELL_ALL, xi0, xi2, xi4, q_perp, q_par)
    return np.einsum('ots,ts->os', W, xi_ap).ravel()


COSMO_LABELS = dict(h='h', ombh2=r'\omega_b', omch2=r'\omega_{cdm}', logA=r'\ln(10^{10}A_s)', ns='n_s', w0='w_0', wa='w_a', shift_h=r'\Delta h', shift_ombh2=r'\Delta\omega_b',
                    shift_omch2=r'\Delta\omega_{cdm}', shift_logA=r'\Delta\ln(10^{10}A_s)', shift_ns=r'\Delta n_s')


def compute_shift():
    """(dict, None) with the mean offset of the single-mock baseline fits from the c000 truth, or (None, reason) if those fits are not all converged: shift = mean over mocks of the posterior mean of
    (h, omega_b, omega_cdm, ln A_s, n_s) minus the truth; std_error = scatter of the means / sqrt(n_mocks)."""
    import h5py                                                    # imported late on purpose (see the thread-pinning note at the top)
    names = sorted(Path(q).name[:-len('.status.json')] for q in glob.glob(str(CHAIN_ROOT / 'compact' / 'compact_mock*_joint_smin20.status.json')))
    try:
        expected = min(mock_counts().values())                       # as many as the single-mock fit list has
    except Exception:
        expected = N_MOCKS_EXPECTED
    if len(names) < expected:
        return None, f'only {len(names)} single-mock fits have a status.json (expected {expected})'
    means = []
    for n in names:
        root = CHAIN_ROOT / 'compact' / n
        st = json.load(open(f'{root}.status.json'))
        if not st['converged']:
            return None, f'{n} has not converged'
        with h5py.File(f'{root}.h5', 'r') as f:
            it = int(f['mcmc'].attrs['iteration'])
            ch = f['mcmc/chain'][st['discard']:it:st['thin']]
        idx = [st['param_names'].index(p) for p in SHIFT_PARAMS]
        means.append(ch.reshape(-1, ch.shape[-1])[:, idx].mean(0))
    M = np.array(means)
    shift = M.mean(0) - np.array([TRUTH_SAMPLED[p] for p in SHIFT_PARAMS])
    return dict(names=SHIFT_PARAMS, shift=shift.tolist(), std_error=(M.std(0, ddof=1) / np.sqrt(len(M))).tolist(), n_mocks=len(M), mocks=names), None


def ensure_shift(force=False, verbose=True):
    """The shift dict from SHIFT_PATH, computing and writing it from the single-mock chains first if needed; None if that is not possible yet."""
    if SHIFT_PATH.exists() and not force:
        return json.load(open(SHIFT_PATH))
    d, why = compute_shift()
    if d is None:
        if verbose:
            print(f'shift not available yet: {why}', flush=True)
        return None
    SHIFT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(d, open(SHIFT_PATH, 'w'), indent=1)
    return d



class FitProblem:
    """Posterior for one fit spec. indiv: theta = [h, ombh2, omch2, logA, b1, b2, ns];
    joint: theta = [h, ombh2, omch2, logA, (ns), (w0, wa), b1_0, b2_0, b1_1, b2_1, ...] (one bias pair per tracer-bin; ns dropped if fix_ns, w0/wa added for w0wa)."""

    def __init__(self, spec):
        self.spec = spec
        self.joint = spec['mode'] == 'joint'
        self.w0wa = bool(spec.get('w0wa', False))
        self.fix_ns = bool(spec.get('fix_ns', False))
        self.hard_box = bool(spec.get('hard_box', HARD_BOX))
        self.flat_priors = bool(spec.get('flat_priors', False))
        self.shift_mode = spec.get('shift')                                          # None, 'fixed' or 'marg'
        assert self.joint or not (self.w0wa or self.fix_ns), 'w0wa / fix_ns are implemented for joint fits only'
        assert not self.shift_mode or (self.joint and not (self.w0wa or self.fix_ns)), 'shift fits are implemented for plain joint fits only'
        start, start_scale = dict(START), dict(START_SCALE)
        if self.shift_mode:
            sh = ensure_shift()
            if sh is None:
                raise RuntimeError('the shift is not available: finish the single-mock fits (or run `python fits_apfix.py shift`)')
            self.shift_vec = np.array(sh['shift'])
            self.shift_sigma = np.maximum(SYS_FRACTION * np.abs(self.shift_vec), 1e-12)
            self.shift_info = dict(mode=self.shift_mode, names=SHIFT_PARAMS, values=self.shift_vec.tolist(), std_error=sh['std_error'], n_mocks=sh['n_mocks'], source=str(SHIFT_PATH),
                                   sys_fraction=SYS_FRACTION if self.shift_mode == 'marg' else None)
            start.update(dict(zip(SHIFT_NAMES, self.shift_vec)))
            start_scale.update(dict(zip(SHIFT_NAMES, 0.25 * self.shift_sigma)))
        self.data = []
        for key in spec['keys']:
            if spec['data'] == 'dr1':
                s_data, data_vec, cov, W = load_dr1_fit_data(key, spec['smin'])
            else:
                s_data, data_vec, cov = load_mock_fit_data(key, spec['smin'], spec['mock_idx'], spec.get('cov_by_nmock', False))
                W = None
            _, logdet = np.linalg.slogdet(2 * np.pi * cov)
            self.data.append(dict(z_eff=Z_EFF[key], s_data=s_data, data_vec=data_vec, W=W, cov_inv=np.linalg.inv(cov), logdet=logdet,
                                  emus=get_emus(spec['model'], Z_EFF[key])))
        self.z_effs = [d['z_eff'] for d in self.data]
        nb = len(self.data)
        if self.joint:
            self.cosmo_names = ['h', 'ombh2', 'omch2', 'logA'] + ([] if self.fix_ns else ['ns']) + (['w0', 'wa'] if self.w0wa else []) + (SHIFT_NAMES if self.shift_mode == 'marg' else [])
            self.n_cosmo = len(self.cosmo_names)
            self.ndim = self.n_cosmo + 2 * nb
            self.theta0 = np.concatenate([[start[n] for n in self.cosmo_names], np.tile([1.0, 0.0], nb)])
            self.scale = np.concatenate([[start_scale[n] for n in self.cosmo_names], np.tile([0.02, 0.1], nb)])
            self.param_names = list(self.cosmo_names)
            self.param_labels = [COSMO_LABELS[n] for n in self.cosmo_names]
            for key in spec['keys']:
                self.param_names += [f'b1_{kname(key)}', f'b2_{kname(key)}']
                self.param_labels += [f'b_1^{{{key[0]}}}', f'b_2^{{{key[0]}}}']
        else:
            self.ndim = 7
            self.theta0 = np.array([START['h'], START['ombh2'], START['omch2'], START['logA'], 1.0, 0.0, START['ns']])
            self.scale = np.array([0.01, 0.0001, 0.002, 0.02, 0.02, 0.1, 0.01])
            self.param_names = ['h', 'ombh2', 'omch2', 'logA', 'b1', 'b2', 'ns']
            self.param_labels = ['h', r'\omega_b', r'\omega_{cdm}', r'\ln(10^{10}A_s)', 'b_1', 'b_2', 'n_s']

    def _split(self, theta):
        if self.joint:
            c = dict(zip(self.cosmo_names, theta[:self.n_cosmo]))
            biases = [(theta[self.n_cosmo + 2 * i], theta[self.n_cosmo + 2 * i + 1]) for i in range(len(self.data))]
        else:
            h, ombh2, omch2, lnAs, b1, b2, ns = theta
            c, biases = dict(h=h, ombh2=ombh2, omch2=omch2, logA=lnAs, ns=ns), [(b1, b2)]
        c.setdefault('ns', NS_FIXED)
        return c, biases

    def unpack(self, theta):
        c, biases = self._split(theta)
        return c['h'], c['ombh2'], c['omch2'], c['logA'], c['ns'], biases

    def shift_values(self, theta):
        """The offset (h, omega_b, omega_cdm, ln A_s, n_s) between the model and the sampled parameters: fixed, sampled ('marg'), or zero."""
        if self.shift_mode == 'marg':
            c, _ = self._split(theta)
            return np.array([c[n] for n in SHIFT_NAMES])
        return self.shift_vec if self.shift_mode else np.zeros(len(SHIFT_PARAMS))

    def model_params(self, theta):
        """(h, ombh2, omch2, lnAs, ns) at which the model is evaluated: the sampled parameters plus the shift."""
        h, ombh2, omch2, lnAs, ns, _ = self.unpack(theta)
        if not self.shift_mode:
            return h, ombh2, omch2, lnAs, ns
        d = self.shift_values(theta)
        return h + d[0], ombh2 + d[1], omch2 + d[2], lnAs + d[3], ns + d[4]

    def dark_energy(self, theta):
        """(w0, wa) for w0wa fits, (None, None) otherwise."""
        c, _ = self._split(theta)
        return (c['w0'], c['wa']) if self.w0wa else (None, None)

    def log_prior(self, theta):
        c, biases = self._split(theta)
        box = BOX if self.hard_box else WIDE
        for k in ('h', 'omch2', 'logA'):
            if not box[k][0] < c[k] < box[k][1]:
                return -np.inf
        for k in (['ombh2', 'ns'] if self.hard_box or self.flat_priors else []) + (['w0', 'wa'] if self.w0wa else []):      # w0, wa always; omega_b, n_s too when the box is their prior
            if not BOX[k][0] < c[k] < BOX[k][1]:
                return -np.inf
        for b1, b2 in biases:
            if not (box['b1'][0] < b1 < box['b1'][1] and box['b2'][0] < b2 < box['b2'][1]):
                return -np.inf
        if self.flat_priors:                                      # omega_b and n_s are flat over the NN box (checked above): no Gaussian terms
            return 0.0
        lp = -0.5 * ((c['ombh2'] - OMBH2_PRIOR[0]) / OMBH2_PRIOR[1]) ** 2
        if not self.fix_ns:
            lp += -0.5 * ((c['ns'] - NS_PRIOR[0]) / NS_PRIOR[1]) ** 2
        if self.shift_mode == 'marg':                                         # Gaussian prior on the shift: Delta +- SYS_FRACTION x |Delta|
            lp += -0.5 * float(np.sum(((np.array([c[n] for n in SHIFT_NAMES]) - self.shift_vec) / self.shift_sigma) ** 2))
        return lp

    def log_likelihood(self, theta):
        biases = self.unpack(theta)[5]
        h, ombh2, omch2, lnAs, ns = self.model_params(theta)               # the sampled parameters, plus the shift for the shift fits
        w0, wa = self.dark_energy(theta)
        try:
            aps = ap_multi(h, ombh2, omch2, ns, self.z_effs, w0, wa)           # one CAMB call for all tracer-bins
            total = 0.0
            for d, (b1, b2), (q_perp, q_par) in zip(self.data, biases, aps):
                emu_params = [b1, b2, ombh2, omch2, h, lnAs, ns] + ([w0, wa] if self.w0wa else [])
                model = model_one(d['emus'], emu_params, d['s_data'], d['W'], q_perp, q_par)
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
def chain_root(spec):
    return CHAIN_ROOT / spec['model'] / spec['name']


def run_chain(problem, root, ncores, max_steps=MAX_STEPS, check_every=CHECK_EVERY, tau_factor=TAU_FACTOR, tau_tol=TAU_TOL, tag=''):
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
        state = backend.get_last_sample()
    print(f'{tag}{"resuming at step " + str(backend.iteration) if resume else "starting"}: ndim={ndim}, nwalkers={nwalkers}, ncores={ncores}', flush=True)

    pool = multiprocessing.get_context('fork').Pool(ncores) if ncores > 1 else None
    t0 = time.time()
    converged, old_tau = False, np.full(ndim, np.inf)
    try:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, _log_prob, pool=pool, backend=backend)
        for _ in sampler.sample(state, iterations=max(0, max_steps - sampler.iteration), progress=False):
            it = sampler.iteration
            if it % check_every:
                continue
            tau = sampler.get_autocorr_time(tol=0)
            long_enough = bool(np.all(tau * tau_factor < it))
            stable = bool(np.all(np.abs(old_tau - tau) / tau < tau_tol))
            if it % LOG_EVERY == 0:
                print(f'{tag}step {it}: tau_max={np.nanmax(tau):.0f}, N/tau_max={it / np.nanmax(tau):.0f}, long_enough={long_enough}, stable={stable}, '
                      f'{(time.time() - t0) / 60:.1f} min', flush=True)
            old_tau = tau
            if long_enough and stable:
                converged = True
                break
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    it = int(sampler.iteration)
    if it == 0:
        print(f'{tag}no steps taken; nothing saved', flush=True)
        return
    tau = sampler.get_autocorr_time(tol=0)
    tau_max = float(np.nanmax(tau))
    discard = min(int(BURN_TAUS * tau_max), it // 2)
    thin = max(1, int(THIN_TAUS * tau_max))
    save_getdist_chain(root, sampler, discard=discard, thin=thin, param_names=problem.param_names, param_labels=problem.param_labels)
    spec = {k: v for k, v in problem.spec.items() if k != 'keys'}
    with open(status_path, 'w') as f:
        json.dump(dict(converged=bool(converged), iterations=it, tau_max=tau_max, tau=[float(t) for t in tau], n_over_tau_max=float(it / tau_max),
                       nwalkers=int(nwalkers), ndim=int(ndim), discard=int(discard), thin=int(thin), wall_minutes_this_run=(time.time() - t0) / 60,
                       spec=spec, keys=[kname(k) for k in problem.spec['keys']], param_names=list(problem.param_names),
                       shift=getattr(problem, 'shift_info', None),
                       priors=dict(hard_box=problem.hard_box, flat_priors=problem.flat_priors, fix_ns=problem.fix_ns, w0wa=problem.w0wa, ombh2=OMBH2_PRIOR, ns=NS_PRIOR,
                                   ranges=BOX if problem.hard_box else WIDE, de_box={k: BOX[k] for k in ('w0', 'wa')} if problem.w0wa else None),
                       ap=dict(reference=AP_REF, h_over_h_fid_factor=True, version=2)), f, indent=1)
    print(f'{tag}{"CONVERGED" if converged else "NOT CONVERGED (hit max_steps)"} at step {it}: tau_max={tau_max:.0f}, N/tau_max={it / tau_max:.0f} '
          f'-> {root}.txt', flush=True)


def is_converged(spec):
    p = Path(f'{chain_root(spec)}.status.json')
    return p.exists() and bool(json.load(open(p)).get('converged'))


def run_spec(spec, max_steps=MAX_STEPS):
    lower_priority()
    q_perp, q_par = ap_multi(*AP_REF, NS_FIXED, [next(iter(Z_EFF.values()))])[0]
    assert abs(q_perp - 1) < 1e-9 and abs(q_par - 1) < 1e-9, f'AP parameters are not 1 at the reference cosmology: {q_perp}, {q_par}'
    problem = FitProblem(spec)
    run_chain(problem, chain_root(spec), spec['ncores'], max_steps=max_steps, tag=f"[{spec['name']}] ")


# ------------------------------------------------------------------ command line
def status_line(spec):
    root = chain_root(spec)
    st_path, h5_path = Path(f'{root}.status.json'), Path(f'{root}.h5')
    if st_path.exists():
        st = json.load(open(st_path))
        return f"{'converged' if st['converged'] else 'NOT converged':14s} it={st['iterations']:>7d}  tau_max={st['tau_max']:>6.0f}  N/tau={st['n_over_tau_max']:>5.0f}"
    if h5_path.exists():
        import h5py
        with h5py.File(h5_path, 'r') as f:
            return f"{'running/stopped':14s} it={int(f['mcmc'].attrs['iteration']):>7d}"
    return 'not started'


def launch(specs, cores, dry_run=False):
    if any(s.get('shift') for s in specs):
        available = (SHIFT_PATH.exists() or compute_shift()[0] is not None) if dry_run else ensure_shift() is not None
        if not available:
            skipped = [s['name'] for s in specs if s.get('shift')]
            specs = [s for s in specs if not s.get('shift')]
            print(f'skipping {len(skipped)} shift fits until the single-mock fits have converged (rerun launch then): {skipped}', flush=True)
    pending = [s for s in sorted(specs, key=lambda s: s['priority']) if not is_converged(s)]
    print(f'{len(pending)} fits to run ({len(specs) - len(pending)} already converged), core budget {cores}', flush=True)
    if dry_run:
        for s in pending:
            print(f"  prio {s['priority']}  {s['ncores']} cores  {s['name']}")
        print(f"total cores if all ran at once: {sum(s['ncores'] for s in pending)}")
        return
    log_dir = CHAIN_ROOT / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    running, used = [], 0
    try:
        while pending or running:
            for item in running[:]:
                proc, sp, lf = item
                if proc.poll() is not None:
                    running.remove(item)
                    used -= sp['ncores']
                    lf.close()
                    print(f"{time.strftime('%H:%M:%S')} finished {sp['name']} (rc={proc.returncode}); {len(pending)} pending, {len(running)} running", flush=True)
            while True:
                nxt = next((s for s in pending if used + s['ncores'] <= cores), None)    # first pending fit that fits: small fits backfill
                if nxt is None:
                    break
                pending.remove(nxt)
                lf = open(log_dir / f"{nxt['name']}.log", 'a')
                proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'run', nxt['name']], stdout=lf, stderr=subprocess.STDOUT)
                running.append((proc, nxt, lf))
                used += nxt['ncores']
                print(f"{time.strftime('%H:%M:%S')} started {nxt['name']} ({nxt['ncores']} cores; {used}/{cores} used)", flush=True)
            time.sleep(10)
    except KeyboardInterrupt:
        print('interrupted: stopping running fits (rerun `launch` to resume them)', flush=True)
    finally:
        for proc, _sp, lf in running:
            proc.terminate()
            lf.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('Usage:')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('list', 'status', 'launch'):
        sp = sub.add_parser(name)
        sp.add_argument('--groups', default=','.join(DEFAULT_GROUPS), help='comma-separated: ' + ', '.join(GROUPS) + ' (or all)')
        sp.add_argument('--only', default=None, help='regex on the fit name')
        if name == 'launch':
            sp.add_argument('--cores', type=int, default=48)
            sp.add_argument('--dry-run', action='store_true')
    sub.add_parser('mocks')
    sp = sub.add_parser('shift')
    sp.add_argument('--force', action='store_true', help='recompute even if shift.json exists')
    sp = sub.add_parser('run')
    sp.add_argument('names', nargs='+')
    sp.add_argument('--max-steps', type=int, default=MAX_STEPS)
    args = ap.parse_args()

    if args.cmd == 'mocks':
        mock_report()
        return
    if args.cmd == 'shift':
        d = ensure_shift(force=args.force)
        if d is None:
            sys.exit(1)
        print(f"shift from {d['n_mocks']} single-mock fits (mean of the posterior means - c000 truth), written to {SHIFT_PATH}:")
        for n, v, e in zip(d['names'], d['shift'], d['std_error']):
            print(f'  {n:6s} {v:+.5g} +- {e:.2g}  ({v / e:+.1f} s.e.)   systematic used by the shiftsys fits: +- {SYS_FRACTION * abs(v):.2g}')
        return
    specs = build_specs()
    if args.cmd == 'run':
        by_name = {s['name']: s for s in specs}
        unknown = [n for n in args.names if n not in by_name]
        if unknown:
            sys.exit(f'unknown fit names: {unknown}')
        for n in args.names:
            run_spec(by_name[n], args.max_steps)
        return
    groups = set(GROUPS) if args.groups == 'all' else set(args.groups.split(','))
    chosen = [s for s in select(specs, groups) if args.only is None or re.search(args.only, s['name'])]
    if args.cmd == 'list':
        for s in chosen:
            print(f"{s['group']:12s} prio {s['priority']}  {s['ncores']} cores  {s['name']}")
        print(f'{len(chosen)} fits')
    elif args.cmd == 'status':
        for s in chosen:
            print(f"{s['name']:55s} {status_line(s)}")
    else:
        miss = missing_files()
        if miss:
            sys.exit('missing data files:\n  ' + '\n  '.join(map(str, miss[:10])))
        launch(chosen, args.cores, args.dry_run)


if __name__ == '__main__':
    main()