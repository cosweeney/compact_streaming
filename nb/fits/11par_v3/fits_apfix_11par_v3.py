"""
Fits with the 11par_v3 emulators (sigma_0^2 = A_sigma sigma_{v,lin}^2): tiers 1 and 2 of the plan, for both variants,
    compact_v3           cosmology slopes on   (models_11par_v3_avg_M,          file suffix _11par_v3)
    compact_v3_noslope   all slopes zero       (models_11par_v3_noslope_avg_M,  file suffix _11par_v3_noslope)

This is a thin layer over nb/fits/11par_fits/fits_apfix.py, which it imports (not copies): the data, covariances, priors, AP treatment,
likelihood, emcee sampler and EFT-GSM convergence rule are exactly those of the baseline script. Only three things change:
  * CHAIN_ROOT  -> /spiffball/cosweeney/data/chains/fits_11par_apfix_v2_v3 (one folder per model: compact_v3/, compact_v3_noslope/),
  * MODELS      += compact_v3, compact_v3_noslope,
  * the fit list: for each of the two models, copies of the baseline 'compact' (avg_M, 11-parameter) specs of
        - the DR1 and mean-of-mocks JOINT fits at s_min = 20, 25, 30, 35, 40   (tier 1 = s_min 20, tier 2a = the rest),
        - the six INDIVIDUAL-tracer fits at s_min = 20, DR1 and mean-of-mocks   (tier 2b),
    22 fits per model, 44 in all (20 joint at 4 cores + 24 individual at 1 core = 104 cores). Names are the baseline names with 'compact_'
    replaced by 'compact_v3_' / 'compact_v3_noslope_', e.g. compact_v3_dr1_joint_smin20, compact_v3_noslope_mean_indiv_LRG_0.4-0.6_smin20.
    Not included (no v3 emulators, or later): low_M / high_M, Gaussian, w0wa, single-mock, leave-one-out (except the de-biased no-QSO fit), flat-prior, marginalized-shift ('shiftsys') and avg-cov fits.

Because the specs are copied from fits_apfix.build_specs(), the priorities, core counts, data selections and covariances stay in sync with the baseline script.

DE-BIASED fits (group 'shift', 8 fits, all for the BASELINE MODEL compact_v3_noslope at s_min = 25): compact_v3_noslope_shift25_dr1_joint_smin25 (all six tracer-bins),
compact_v3_noslope_shift25_dr1_indiv_<tracer-bin>_smin25 (the six tracers alone) and compact_v3_noslope_shift25_dr1_joint_noQSO_smin25 (joint fit without the QSO). The baseline 'fixed' shift mode: the shift
is calibrated on the model's own mean-of-mocks joint fit at s_min = 25 (compact_v3_noslope_mean_joint_smin25; it must have converged first, `launch` holds the shift fits back until then):
Delta = posterior mean - c000 truth for (h, omega_b, omega_cdm, ln A_s, n_s); the model (emulator input and AP) is evaluated at theta + Delta, so L(p') = L(p - Delta_p) with Delta_p = -Delta, and the priors
act on theta. Nothing is marginalized: the shift is a fixed number (written to CHAIN_ROOT/shift_<model>_smin25.json, statistical error approximated by the posterior std / sqrt(n_mocks)). The same joint-calibrated shift
is applied to the individual-tracer fits (fits_apfix.py itself only allows shift fits for joint fits, so this layer builds those problems without the shift and attaches it). The systematic error (e.g. 1/2 of the shift) is
not in the chains: add it to the posterior afterwards (plot cells 13-14). SHIFT_EXCLUDE below leaves chosen parameters unshifted (e.g. ('ombh2',) when the omega_b offset is dominated by the BBN prior).
`python fits_apfix_11par_v3.py shift [--force]` prints / computes the shifts of both models at s_min = 20 and 25.

b1-ONLY fits (group 'b1only', 14 fits with the defaults below, baseline model compact_v3_noslope at s_min = 25): compact_v3_noslope_b1only_<dr1|mean>_joint_smin25 and
compact_v3_noslope_b1only_<dr1|mean>_indiv_<tracer-bin>_smin25. Same data, covariances, priors and emulators as the corresponding b1+b2 fits, but b2 is not sampled: it is set to B2_OF_B1(b1) (default 0) at every
likelihood call, so each tracer-bin has one bias parameter instead of two (joint fit: 11 sampled parameters instead of 17; individual fits: 6 instead of 7). The chains' param_names have no b2 entries, and
their status.json spec has b1_only = true. B1ONLY_SMIN, B1ONLY_DATASETS and B2_OF_B1 at the top of the script change the s_min, the datasets ('dr1', 'mean') and the b2 relation.

HEXADECAPOLE fits (group 'hex', 4 fits with the defaults, both v3 models at s_min = 25): compact_v3_hex_<dr1|mean>_joint_smin25 and compact_v3_noslope_hex_<dr1|mean>_joint_smin25. The data vector gets xi_4 as well as xi_0 and xi_2
(the covariance already contains it, 'xi024', and the emulators exist for ell = 4): the layer sets ELLS_FIT = (0, 2, 4) in common.py and fits_apfix.py before building each of these problems (and back to (0, 2) for every other fit).
The mock files have xi_4. The DR1 files' xi_full already holds xi_4 and the covariance 'xi024' too; what the hexadecapole fits need on top is the window row of the observed ell = 4. It is read from the keys W_fine_hex / ells_obs_hex that
extract_dr1_data_hex.py ADDS to the existing extracted files without changing any array in them (so the other fits and scripts read the same files as before), or, for a file extracted directly with ells_obs = (0, 2, 4), from W_fine / ells_obs.
fits_apfix.load_dr1_fit_data only knows W_fine / ells_obs, so for the hexadecapole fits this layer swaps in a copy of that function that picks the window by these rules. If the DR1 files have neither, `launch` holds the DR1 hexadecapole
fits back with a message and only the mock fits run. HEX_MODELS, HEX_SMINS, HEX_DATASETS and HEX_INDIVIDUAL (also the six individual tracers) at the top change the set. HEX_EXTRACT_DIR (default None) points the DR1 hexadecapole fits at a
different directory of extracted files. `python inspect_dr1_extracted.py [DIR]` shows what a directory contains.
"""
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FITS_APFIX_DIR = Path(os.environ.get('FITS_APFIX_DIR', HERE.parent / '11par_fits'))          # where the baseline fits_apfix.py lives
sys.path.insert(0, str(FITS_APFIX_DIR))
import fits_apfix as fa          # keep this import first: it pins the BLAS / OpenMP / TF threads to 1 BEFORE numpy loads (the fork-based emcee pool needs it)

V3_CHAIN_ROOT = Path('/spiffball/cosweeney/data/chains/fits_11par_apfix_v2_v3')
V3_MODELS = {   # model -> (emulator directory, file-name suffix), as in fits_apfix.MODELS
    'compact_v3':         ('/spiffball/cosweeney/data/models_11par_v3_avg_M/',         '_11par_v3'),
    'compact_v3_noslope': ('/spiffball/cosweeney/data/models_11par_v3_noslope_avg_M/', '_11par_v3_noslope'),
}
SHIFT_EXCLUDE = ()                                                               # parameters to leave unshifted, subset of ('h', 'ombh2', 'omch2', 'logA', 'ns')
SHIFT_MODEL = 'compact_v3_noslope'                                               # the baseline model of the de-biased fits
SHIFT_SMIN = 25                                                                  # s_min of the mean-of-mocks fit that calibrates the shift AND of the de-biased DR1 fits
KEEP_INDIV = re.compile(r'^compact_dr1_indiv_.+_smin20$')                        # baseline individual-tracer DR1 specs, copied with s_min = SHIFT_SMIN
B1ONLY_MODEL = 'compact_v3_noslope'                                              # model of the b1-only fits
B1ONLY_SMIN = 25                                                                 # s_min of the b1-only fits
B1ONLY_DATASETS = ('dr1', 'mean')                                                # 'dr1' and / or 'mean' (mean of the mocks)
B2_OF_B1 = lambda b1: 0.0                                                        # b2 of the b1-only fits as a function of b1 (default: b2 = 0)
HEX_MODELS = ('compact_v3', 'compact_v3_noslope')                                # hexadecapole fits: models,
HEX_SMINS = (25,)                                                                #   s_min values (any of 20, 25, 30, 35, 40),
HEX_DATASETS = ('dr1', 'mean')                                                   #   datasets,
HEX_INDIVIDUAL = False                                                           #   and whether the six individual-tracer fits are included too
HEX_EXTRACT_DIR = None                                                            # directory of DR1 files with the hexadecapole window for the DR1 hexadecapole fits (None: the same directory as the other fits)
KEEP = re.compile(r'^compact_(dr1|mean)_(joint_smin\d+|indiv_.+_smin20)$')      # baseline names to copy: joint at every s_min, individual tracers at s_min 20
N_EXPECTED = 22                                                                  # per model: 2 datasets x (5 joint + 6 individual)

_needed = ['CHAIN_ROOT', 'MODELS', 'GROUPS', 'DEFAULT_GROUPS', 'build_specs', 'launch', 'main', 'chain_root', 'FitProblem', 'ensure_shift', 'compute_shift', 'SHIFT_PATH',
           'SHIFT_PARAMS', 'TRUTH_SAMPLED', 'mock_counts', 'ELLS_FIT', 'dr1_paths', 'FIT_KEYS', 'kname', 'EXTRACT_DIR', 'load_dr1_fit_data', 'window_at', 'SMAX_FIT', 'ELL_ALL', 'NBINS_COV']
_missing = [n for n in _needed if not hasattr(fa, n)]
if _missing:
    sys.exit(f'{FITS_APFIX_DIR / "fits_apfix.py"} has no {_missing}: this layer was written against the fits_apfix.py of the 11par apfix v2 refit')

_baseline_build_specs = fa.build_specs


def build_specs_v3():
    baseline = _baseline_build_specs()
    base = [s for s in baseline if s['model'] == 'compact' and s['group'] in ('dr1', 'mock_mean') and KEEP.match(s['name'])]
    if len(base) != N_EXPECTED:
        print(f'WARNING: found {len(base)} baseline specs to copy, expected {N_EXPECTED}; check the fit list of fits_apfix.build_specs()', flush=True)
    by_name = {s['name']: s for s in baseline}
    templ = [by_name.get(f'compact_dr1_joint_smin{SHIFT_SMIN}')] + [s for n, s in by_name.items() if KEEP_INDIV.match(n)] + [by_name.get('compact_dr1_joint_noQSO_smin20')]
    if None in templ or len(templ) != 8:
        print(f'WARNING: found {len([t for t in templ if t is not None])} baseline specs for the de-biased fits, expected 8 (joint, six individual, joint without QSO)', flush=True)
    specs = []
    for model in V3_MODELS:
        for s in base:
            t = dict(s)
            t['model'] = model
            t['name'] = model + s['name'][len('compact'):]
            specs.append(t)
    for model in HEX_MODELS:                            # hexadecapole fits: the same specs with xi_4 in the data vector
        for tag in HEX_DATASETS:
            for smin in HEX_SMINS:
                tmpl = [by_name.get(f'compact_{tag}_joint_smin{smin}')]
                if HEX_INDIVIDUAL:
                    tmpl += [t for n, t in by_name.items() if re.match(rf'^compact_{tag}_indiv_.+_smin20$', n)]
                for t0 in tmpl:
                    if t0 is None:
                        print(f'WARNING: no baseline spec compact_{tag}_joint_smin{smin} for the hexadecapole fits', flush=True)
                        continue
                    t = dict(t0)
                    t.update(model=model, smin=smin, ells4=True, group='hex', priority=1)
                    t['name'] = f'{model}_hex' + t0['name'][len('compact'):].replace('_smin20', f'_smin{smin}')
                    specs.append(t)
    for tag in B1ONLY_DATASETS:                         # b1-only fits (b2 not sampled): joint and the six individual tracers, baseline model
        tmpl = [by_name.get(f'compact_{tag}_joint_smin{B1ONLY_SMIN}')] + [t for n, t in by_name.items() if re.match(rf'^compact_{tag}_indiv_.+_smin20$', n)]
        if None in tmpl or len(tmpl) != 7:
            print(f'WARNING: found {len([t for t in tmpl if t is not None])} baseline specs for the b1-only {tag} fits, expected 7 (joint and six individual)', flush=True)
        for t0 in tmpl:
            if t0 is None:
                continue
            t = dict(t0)
            t.update(model=B1ONLY_MODEL, smin=B1ONLY_SMIN, b1_only=True, group='b1only', priority=1)
            t['name'] = f'{B1ONLY_MODEL}_b1only' + t0['name'][len('compact'):].replace('_smin20', f'_smin{B1ONLY_SMIN}')
            specs.append(t)
    for s in templ:                                    # de-biased DR1 fits of the baseline model: shift from the s_min = SHIFT_SMIN mean-of-mocks fit
        if s is None:
            continue
        t = dict(s)
        t.update(model=SHIFT_MODEL, smin=SHIFT_SMIN, shift='fixed', shift_from_smin=SHIFT_SMIN, group='shift', priority=2)
        t['name'] = f'{SHIFT_MODEL}_shift{SHIFT_SMIN}' + s['name'][len('compact'):].replace('_smin20', f'_smin{SHIFT_SMIN}')
        specs.append(t)
    return specs


# ------------------------------------------------------------------ de-biased fits: shift from a model's own mean-of-mocks fit at a given s_min
import numpy as np

_ACTIVE = None        # (model, s_min of the calibrating fit) of the FitProblem being built (set by FitProblemV3), so that ensure_shift knows which shift to use


def shift_file(model, smin):
    return V3_CHAIN_ROOT / f'shift_{model}_smin{smin}.json'


def compute_shift_v3(model, smin):
    """(dict, None) with the offset of `model`'s mean-of-mocks joint fit at s_min = `smin` from the c000 truth, or (None, reason) if that fit has not converged.
    shift = posterior mean - truth for fa.SHIFT_PARAMS (parameters in SHIFT_EXCLUDE are set to 0); std_error ~ posterior std / sqrt(n_mocks): the mean of the mocks has the noise of one mock / sqrt(n_mocks)."""
    import h5py                                                    # imported late on purpose (the thread pinning of fits_apfix has to come first)
    name = f'{model}_mean_joint_smin{smin}'
    root = V3_CHAIN_ROOT / model / name
    if not Path(f'{root}.status.json').exists():
        return None, f'{name} has no status.json yet'
    st = json.load(open(f'{root}.status.json'))
    if not st['converged']:
        return None, f'{name} has not converged'
    with h5py.File(f'{root}.h5', 'r') as f:
        it = int(f['mcmc'].attrs['iteration'])
        ch = f['mcmc/chain'][st['discard']:it:st['thin']]
    flat = ch.reshape(-1, ch.shape[-1])[:, [st['param_names'].index(p) for p in fa.SHIFT_PARAMS]]
    mean, std = flat.mean(0), flat.std(0)
    truth = np.array([fa.TRUTH_SAMPLED[p] for p in fa.SHIFT_PARAMS])
    shift = np.where([p in SHIFT_EXCLUDE for p in fa.SHIFT_PARAMS], 0.0, mean - truth)
    try:
        n_mocks = min(fa.mock_counts().values())
    except Exception:
        n_mocks = getattr(fa, 'N_MOCKS_EXPECTED', 25)
    return dict(model=model, smin=smin, names=list(fa.SHIFT_PARAMS), shift=shift.tolist(), std_error=(std / np.sqrt(n_mocks)).tolist(), n_mocks=int(n_mocks), mocks=[name],
                posterior_mean=mean.tolist(), posterior_std=std.tolist(), truth=truth.tolist(), excluded=list(SHIFT_EXCLUDE)), None


def get_shift(model, smin, force=False, verbose=True):
    """The shift dict from shift_<model>_smin<smin>.json, computing and writing it from the mean-of-mocks chain first if needed; None if that is not possible yet."""
    path = shift_file(model, smin)
    if path.exists() and not force:
        return json.load(open(path))
    d, why = compute_shift_v3(model, smin)
    if d is None:
        if verbose:
            print(f'shift of {model} at s_min = {smin} not available yet: {why}', flush=True)
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(d, open(path, 'w'), indent=1)
    return d


def ensure_shift_v3(force=False, verbose=True):
    """Replaces fa.ensure_shift: the shift for the FitProblem being built. With none active (fa.launch's availability check, already filtered by launch_v3) any non-None value passes."""
    return {} if _ACTIVE is None else get_shift(*_ACTIVE, force=force, verbose=verbose)


_OrigFitProblem = fa.FitProblem


class FitProblemV3(_OrigFitProblem):
    def __init__(self, spec):
        global _ACTIVE
        _ACTIVE = (spec['model'], spec.get('shift_from_smin', SHIFT_SMIN))
        try:
            if spec.get('shift') and spec['mode'] != 'joint':
                # fits_apfix.FitProblem only allows shift fits for joint fits: build the individual-tracer problem without the shift, then attach the same (joint-calibrated) fixed shift
                super().__init__({**spec, 'shift': None})
                sh = get_shift(*_ACTIVE)
                if sh is None:
                    raise RuntimeError('the shift is not available: finish the mean-of-mocks fit that calibrates it')
                self.spec = spec
                self.shift_mode = spec['shift']
                self.shift_vec = np.array(sh['shift'])
                self.shift_sigma = np.maximum(getattr(fa, 'SYS_FRACTION', 0.5) * np.abs(self.shift_vec), 1e-12)
                self.shift_info = dict(mode=self.shift_mode, names=fa.SHIFT_PARAMS, values=self.shift_vec.tolist(), std_error=sh['std_error'], n_mocks=sh['n_mocks'],
                                       source=str(fa.SHIFT_PATH), sys_fraction=None)
            else:
                super().__init__(spec)
        finally:
            _ACTIVE = None


class FitProblemB1(FitProblemV3):
    """b1-only problem: the parameter vector of the parent without the b2 entries; b2 = B2_OF_B1(b1) is put back before the parent's prior and likelihood are evaluated."""

    def __init__(self, spec):
        super().__init__(spec)
        names = list(self.param_names)
        b2 = [i for i, n in enumerate(names) if n.startswith('b2')]
        self._full_dim = len(names)
        self._pairs = [(names.index('b1' + names[j][2:]), j) for j in b2]               # (index of b1, index of its b2) per tracer-bin
        self._keep = [i for i in range(len(names)) if i not in b2]
        self.param_names = [names[i] for i in self._keep]
        self.param_labels = [self.param_labels[i] for i in self._keep]
        self.theta0, self.scale = np.asarray(self.theta0)[self._keep], np.asarray(self.scale)[self._keep]
        self.ndim = len(self._keep)

    def _expand(self, theta):
        full = np.empty(self._full_dim)
        full[self._keep] = theta
        for i1, i2 in self._pairs:
            full[i2] = B2_OF_B1(full[i1])
        return full

    def log_prior(self, theta):
        return super().log_prior(self._expand(theta))

    def log_likelihood(self, theta):
        return super().log_likelihood(self._expand(theta))


_ELLS_DEFAULT = tuple(fa.ELLS_FIT)                                                 # (0, 2)


_EXTRACT_DEFAULT = fa.EXTRACT_DIR


def set_ells(four):
    """Multipoles of the data vector for the problem about to be built: common.py (data_vec_from, load_tracer_arrays) and fits_apfix.py (data loading, model_one) both read their ELLS_FIT at call time."""
    ells = (0, 2, 4) if four else _ELLS_DEFAULT
    sys.modules['common'].ELLS_FIT = ells
    fa.ELLS_FIT = ells
    fa.EXTRACT_DIR = Path(HEX_EXTRACT_DIR) if (four and HEX_EXTRACT_DIR) else _EXTRACT_DEFAULT          # DR1 files with ell = 4 for the hexadecapole fits only


def hex_window(npz):
    """(W_fine, ells_obs) of an extracted DR1 file for hexadecapole fits: the added keys W_fine_hex / ells_obs_hex if present, otherwise W_fine / ells_obs (a file extracted directly with (0, 2, 4))."""
    if 'W_fine_hex' in npz.files and 'ells_obs_hex' in npz.files:
        return npz['W_fine_hex'], tuple(int(e) for e in npz['ells_obs_hex'])
    return npz['W_fine'], tuple(int(e) for e in npz['ells_obs'])


_orig_load_dr1_fit_data = fa.load_dr1_fit_data


def load_dr1_fit_data_v3(key, smin):
    """fits_apfix.load_dr1_fit_data, unchanged for (0, 2); for the hexadecapole fits the same function with the window from hex_window()."""
    if tuple(fa.ELLS_FIT) != (0, 2, 4):
        return _orig_load_dr1_fit_data(key, smin)
    npz_path, cov_path = fa.dr1_paths(key)
    npz = np.load(npz_path)
    s_full, xi_full, s_fine = npz['s_full'], npz['xi_full'], npz['s_fine']
    W_fine, ells_obs = hex_window(npz)
    assert ells_obs == tuple(fa.ELLS_FIT) and tuple(int(e) for e in npz['ells_theory']) == tuple(fa.ELL_ALL), \
        f'{Path(npz_path).name}: observed multipoles {ells_obs} (window) do not match (0, 2, 4); run extract_dr1_data_hex.py'
    cov_full = np.loadtxt(cov_path)
    mask = (s_full <= fa.SMAX_FIT) & (s_full >= smin)
    idx = np.concatenate([fa.ELL_ALL.index(ell) * fa.NBINS_COV + np.where(mask)[0] for ell in fa.ELLS_FIT])
    s_data = s_full[mask]
    data_vec = np.concatenate([xi_full[fa.ELL_ALL.index(ell)][mask] for ell in fa.ELLS_FIT])
    return s_data, data_vec, cov_full[np.ix_(idx, idx)], fa.window_at(W_fine, s_fine, s_data)


def make_problem(spec):
    set_ells(bool(spec.get('ells4')))
    return (FitProblemB1 if spec.get('b1_only') else FitProblemV3)(spec)


_orig_launch = fa.launch


def dr1_files_without_ell4():
    """Tracer-bins whose extracted DR1 file does not list ell = 4 among the observed multipoles (or cannot be read)."""
    bad = []
    old_dir = fa.EXTRACT_DIR
    fa.EXTRACT_DIR = Path(HEX_EXTRACT_DIR) if HEX_EXTRACT_DIR else _EXTRACT_DEFAULT
    try:
        for key in fa.FIT_KEYS:
            try:
                ells = list(hex_window(np.load(fa.dr1_paths(key)[0]))[1])
            except Exception:
                ells = []
            if 4 not in ells:
                bad.append(fa.kname(key))
    finally:
        fa.EXTRACT_DIR = old_dir
    return bad


def launch_v3(specs, cores, dry_run=False):
    """fa.launch with the shift fits held back until the mean-of-mocks fit that calibrates their shift has converged."""
    ready = lambda s: shift_file(s['model'], s.get('shift_from_smin', SHIFT_SMIN)).exists() or compute_shift_v3(s['model'], s.get('shift_from_smin', SHIFT_SMIN))[0] is not None
    held = [s['name'] for s in specs if s.get('shift') and not ready(s)]
    if held:
        print(f'skipping {len(held)} shift fits until the mean-of-mocks fit that calibrates their shift has converged (rerun launch then): {held}', flush=True)
    hex_dr1 = [s['name'] for s in specs if s.get('ells4') and s['data'] == 'dr1']
    if hex_dr1:
        missing = dr1_files_without_ell4()
        if missing:
            held_hex = hex_dr1
            print(f'skipping {len(held_hex)} DR1 hexadecapole fits: the extracted DR1 files of {missing} (in {HEX_EXTRACT_DIR or _EXTRACT_DEFAULT}) have no hexadecapole window (keys W_fine_hex / ells_obs_hex, or ells_obs with 4); run extract_dr1_data_hex.py '
                  f'and rerun launch. The mock hexadecapole fits are not affected.', flush=True)
            held += held_hex
    _orig_launch([s for s in specs if s['name'] not in held], cores, dry_run)


def shift_cli(force=False):
    """Both models at s_min = 20 and 25 (only SHIFT_MODEL at SHIFT_SMIN is used by the fits; the others are for comparison)."""
    for model in V3_MODELS:
        for smin in (20, 25):
            d = get_shift(model, smin, force=force, verbose=False)
            if d is None:
                print(f'{model} at s_min = {smin}: not available (no converged mean-of-mocks fit)')
                continue
            used = ' [USED by the de-biased fits]' if (model, smin) == (SHIFT_MODEL, SHIFT_SMIN) else ''
            print(f"{model}, s_min = {smin}{used}: shift (posterior mean of {d['mocks'][0]} - c000 truth), stat. error ~ posterior std / sqrt({d['n_mocks']}), {shift_file(model, smin)}"
                  + (f"; left unshifted: {d['excluded']}" if d['excluded'] else ''))
            for n, v, e, sd in zip(d['names'], d['shift'], d['std_error'], d['posterior_std']):
                print(f'  {n:6s} {v:+.5g} +- {e:.2g}  ({v / e:+.1f} s.e.; {v / sd:+.2f} posterior std)   1/2 shift = {0.5 * abs(v):.2g}')


fa.CHAIN_ROOT = V3_CHAIN_ROOT
fa.MODELS.update(V3_MODELS)
fa.GROUPS = fa.DEFAULT_GROUPS = ('dr1', 'mock_mean', 'shift', 'b1only', 'hex')
fa.build_specs = build_specs_v3
fa.ensure_shift = ensure_shift_v3
fa.compute_shift = lambda: ({}, None)                  # only used by fa.launch's availability check, which launch_v3 has already done per model
fa.SHIFT_PATH = V3_CHAIN_ROOT / 'shift_<model>_smin<s>.json'   # shows up as the 'source' of the shift in each chain's status.json
fa.FitProblem = make_problem
fa.load_dr1_fit_data = load_dr1_fit_data_v3
fa.launch = launch_v3
fa.__file__ = str(Path(__file__).resolve())          # launch() starts "<this file> run NAME" in subprocesses, which must see the v3 fit list

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'shift':
        shift_cli(force='--force' in sys.argv)
    else:
        fa.main()