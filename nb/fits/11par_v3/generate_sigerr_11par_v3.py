"""
Training/test/validation sets of xi_0(s), xi_2(s), xi_4(s) from CompactStreamingModel with an EXTRA input sigma_err, for the QSO redshift-error
treatment by variance addition:   sigma_v^2 -> sigma_v^2 + sigma_err^2   (compact.model.Pv_compact, sigma_err in km/s, a PAIR dispersion).

    theta = [b1, b2, omega_b, omega_cdm, h, As, ns, sigma_err]

One redshift (default QSO, z = 1.491), one mass scheme (avg_M), Sobol sampled; sigma_err is uniform on SIGMA_ERR_BOUNDS (0 included, so any
log-uniform prior with a positive lower cut lies inside the training range). Same structure as generate_multiz_11par.py: resumable (each finished
set gets a '<xi file>.done' marker; a set with a checkpointed xi file but no marker is resumed, all-NaN rows are recomputed), thetas files are
regenerated from the seed and must match what is on disk, nothing is overwritten with different thetas. The first n points of the Sobol sequence do not
depend on n, so a set can be extended later by raising its size (the first rows stay identical; delete the '.done' marker to resume the extension).

Requires the sigma_err hook in compact/ (sigma_err.patch: Pv_compact(..., sigma_err=0.0) and CompactStreamingModel._pdf(..., sigma_err=0.0)).

Output:  <BASE_DIR>/<zdir>/{s,thetas_train,xi_train,thetas_test,xi_test,thetas_val,xi_val}.npy    (xi: (n, 3, len(cfg.r)))

    python generate_sigerr_11par_v3.py --selftest 20 --ref-dir <.../compact_multipole_training_set_11par_v3_noslope_avg_M/>   # no-op check at sigma_err = 0
    python generate_sigerr_11par_v3.py                                                                                                  # generate (resumes)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import importlib
import json
import sys
import time
import warnings

import numpy as np
import psutil
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                   # config_11par_v3.py lives next to the v3 generator (run this script from the same folder)

# ------------------------------------------------------------------ settings
Z = 1.491                                  # QSO
AP_PADDING = {1.491: (16.10, 171.44)}      # s range (h^-1 Mpc) the AP step needs from the model at this z, as in generate_multiz_11par.py
MASS_BIN = 'avg_M'
SETS = {'train': (100_000, 142), 'test': (10_000, 143), 'val': (10_000, 144)}     # (n samples, Sobol seed)
N_JOBS = 100
CHECKPOINT_EVERY = 500
ELLS = (0, 2, 4)
NICE = 19
CONFIG_DEFAULT = 'config_11par_v3:ConfigNoSlope'   # module:Class of the v3 no-slope config (override with --config)
BASE_DIR = '/spiffball/cosweeney/data/training/compact_multipole_training_set_11par_v3_noslope_sigerr_avg_M/'   # edit / override with --out

BOUNDS = {
    'b1':        (0.0,    2.0),
    'b2':        (-5.0,   10.0),
    'omega_b':   (0.0207, 0.0243),
    'omega_cdm': (0.104,  0.140),
    'h':         (0.55,   0.90),
    'lnAs_e10':  (2.0,    4.0),
    'ns':        (0.901,  1.025),
    'sigma_err': (0.0,    1000.0),         # km/s, PAIR dispersion = sqrt(2) x per-object RMS; 1000 covers a Lorentzian w ~ 500 km/s (L1 = 2000)
}
PARAM_NAMES = list(BOUNDS)
LOWER = np.array([BOUNDS[p][0] for p in PARAM_NAMES])
UPPER = np.array([BOUNDS[p][1] for p in PARAM_NAMES])

warnings.filterwarnings('ignore', message='.*balance properties.*')
warnings.filterwarnings('ignore', message='.*is outside the calibrated range.*')


def zdir(z):
    return f"z{str(z).replace('.', 'p')}"


def load_config(spec):
    mod, cls = spec.split(':')
    return getattr(importlib.import_module(mod), cls)()


def make_thetas(n_samples, seed):
    """(n, 8): [b1, b2, omega_b, omega_cdm, h, As, ns, sigma_err]."""
    s = qmc.scale(qmc.Sobol(d=len(PARAM_NAMES), scramble=True, seed=seed).random(n_samples), LOWER, UPPER)
    i = {n: k for k, n in enumerate(PARAM_NAMES)}
    return np.column_stack([s[:, i['b1']], s[:, i['b2']], s[:, i['omega_b']], s[:, i['omega_cdm']], s[:, i['h']],
                            np.exp(s[:, i['lnAs_e10']]) / 1e10, s[:, i['ns']], s[:, i['sigma_err']]])


def preflight(cfg, z=Z):
    lo, hi = AP_PADDING[z]
    if cfg.r[0] > lo or cfg.r[-1] < hi:
        raise ValueError(f"z={z}: cfg.r spans [{cfg.r[0]:.2f}, {cfg.r[-1]:.2f}] but AP needs [{lo:.2f}, {hi:.2f}] h^-1 Mpc. Widen Config.r.")


def predict_one(theta, z, cfg, model_cls):
    """(xi (len(ELLS), n_r), None) or (None, error string). LPT is computed for this theta only (no sharing: one scheme, one sigma_err per theta)."""
    b1, b2, ombh2, omch2, h, As, ns, sigma_err = theta
    cosmo = (h, ombh2, omch2, As, ns)
    model = model_cls(redshift=z, config=cfg)
    try:
        lpt = model._lpt_inputs(cosmo, np.array([b1, b2]))
        pdf = model._pdf(lpt, cosmo, mass_bin=MASS_BIN, sigma_err=sigma_err)
        return np.asarray(model._multipoles(pdf, ells=ELLS)), None
    except Exception as e:
        return None, str(e)


def run_one(i, theta, z, config_spec):
    psutil.Process().nice(NICE)
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from compact.predictions import CompactStreamingModel
    cfg = load_config(config_spec)
    xi, err = predict_one(theta, z, cfg, CompactStreamingModel)
    return i, xi, err


def _atomic_save(path, arr):
    tmp = path + '.tmp.npy'
    np.save(tmp, arr)
    os.replace(tmp, path)


def generate_set(name, n_samples, seed, z, config_spec, out_dir, n_jobs, run=run_one):
    from joblib import Parallel, delayed
    from tqdm import tqdm
    cfg = load_config(config_spec)
    n_r = len(cfg.r)
    os.makedirs(out_dir, exist_ok=True)
    s_path = os.path.join(out_dir, 's.npy')
    if not os.path.exists(s_path):
        np.save(s_path, np.asarray(cfg.r))
    thetas = make_thetas(n_samples, seed)
    thetas_path, xi_path = os.path.join(out_dir, f'thetas_{name}.npy'), os.path.join(out_dir, f'xi_{name}.npy')
    if os.path.exists(thetas_path):
        old = np.load(thetas_path)
        if old.shape == thetas.shape and not np.allclose(old, thetas, rtol=0, atol=0):
            raise RuntimeError(f"{thetas_path} exists but does not match make_thetas(seed={seed}) -- refusing to mix sample sets")
        if old.shape[0] > n_samples or old.shape[1] != thetas.shape[1]:
            raise RuntimeError(f"{thetas_path} has shape {old.shape}, requested {thetas.shape}")
        if old.shape[0] < n_samples:                           # extension: first rows must agree
            if not np.allclose(old, thetas[:old.shape[0]], rtol=0, atol=0):
                raise RuntimeError(f"{thetas_path}: the first {old.shape[0]} rows do not match the Sobol sequence for seed {seed}")
    np.save(thetas_path, thetas)
    if os.path.exists(xi_path + '.done'):
        done = json.load(open(xi_path + '.done'))
        if done['n_samples'] == n_samples:
            print(f"z={z} '{name}': already done, skipping")
            return {}
    xi = np.full((n_samples, len(ELLS), n_r), np.nan)
    if os.path.exists(xi_path):
        old_xi = np.load(xi_path)
        assert old_xi.shape[1:] == xi.shape[1:] and old_xi.shape[0] <= n_samples, f"{xi_path}: unexpected shape {old_xi.shape}"
        xi[:old_xi.shape[0]] = old_xi
    need = [i for i in range(n_samples) if np.isnan(xi[i]).all()]
    print(f"\n{'=' * 70}\nz={z}  '{name}': {len(need)}/{n_samples} samples to compute (seed={seed})\n{'=' * 70}")
    errors = {}
    t0 = time.time()
    with Parallel(n_jobs=n_jobs, batch_size=1, return_as="generator_unordered") as par:
        stream = par(delayed(run)(i, thetas[i], z, config_spec) for i in need)
        for done, (i, xi_i, err) in enumerate(tqdm(stream, total=len(need), desc=f"z={z} {name}"), 1):
            if err is not None:
                errors[i] = err
            else:
                xi[i] = xi_i
            if done % CHECKPOINT_EVERY == 0:
                _atomic_save(xi_path, xi)
    _atomic_save(xi_path, xi)
    n_bad = int(np.isnan(xi).all(axis=(1, 2)).sum())
    with open(xi_path + '.done', 'w') as f:
        json.dump({'n_samples': n_samples, 'n_failed': n_bad, 'seed': seed, 'param_names': PARAM_NAMES, 'bounds': BOUNDS}, f)
    if errors:
        with open(os.path.join(out_dir, f'errors_{name}.json'), 'w') as f:
            json.dump({str(k): v for k, v in errors.items()}, f)
        print(f"  {len(errors)} failed -- first error: {next(iter(errors.values()))}")
    el = time.time() - t0
    print(f"z={z} '{name}' done: {len(need)} thetas in {el / 60:.1f} min ({el / max(len(need), 1):.2f} s per theta wall, {n_jobs} workers)")
    return errors


def selftest(n, ref_dir, z, config_spec):
    """sigma_err = 0 must reproduce an existing 7-input avg_M training set row by row (the hook is a no-op), and sigma_err > 0 must change the multipoles."""
    from compact.predictions import CompactStreamingModel
    cfg = load_config(config_spec)
    ref = os.path.join(ref_dir, zdir(z))
    th = np.load(os.path.join(ref, 'thetas_test.npy'))[:n]
    xi_ref = np.load(os.path.join(ref, 'xi_test.npy'))[:n]
    worst = 0.0
    for t, x in zip(th, xi_ref):
        xi, err = predict_one(np.append(t, 0.0), z, cfg, CompactStreamingModel)
        if err is not None:
            print('failed:', err)
            continue
        worst = max(worst, float(np.max(np.abs(xi - x)) / np.max(np.abs(x))))
    print(f'sigma_err = 0 vs the existing set, {n} rows: max |dxi| / max|xi| = {worst:.2e}  (expect ~0 if the config and model are the same)')
    xi0, _ = predict_one(np.append(th[0], 0.0), z, cfg, CompactStreamingModel)
    for se in (150.0, 424.0):
        xi1, _ = predict_one(np.append(th[0], se), z, cfg, CompactStreamingModel)
        print(f'sigma_err = {se:5.0f}: max relative change of xi_0, xi_2 at s >= 25: '
              + ', '.join(f'{np.max(np.abs(xi1[l] - xi0[l])[cfg.r >= 25]) / np.max(np.abs(xi0[l][cfg.r >= 25])):.2e}' for l in (0, 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=CONFIG_DEFAULT, help='module:Class of the v3 no-slope compact config (module found next to this script)')
    ap.add_argument('--out', default=BASE_DIR)
    ap.add_argument('--redshift', type=float, default=Z)
    ap.add_argument('--n-jobs', type=int, default=N_JOBS)
    ap.add_argument('--sets', default='train,test,val')
    ap.add_argument('--selftest', type=int, default=0, metavar='N', help='check N rows of --ref-dir at sigma_err = 0, then exit')
    ap.add_argument('--ref-dir', default=None, help='existing 7-input avg_M training set directory (contains z1p491/thetas_test.npy, xi_test.npy)')
    args = ap.parse_args()
    preflight(load_config(args.config), args.redshift)
    if args.selftest:
        assert args.ref_dir, '--selftest needs --ref-dir'
        selftest(args.selftest, args.ref_dir, args.redshift, args.config)
        return
    print(f'config {args.config}, z = {args.redshift}, scheme {MASS_BIN}, parameters {PARAM_NAMES}\noutput {args.out}')
    for name in args.sets.split(','):
        n, seed = SETS[name]
        generate_set(name, n, seed, args.redshift, args.config, os.path.join(args.out, zdir(args.redshift)), args.n_jobs)


if __name__ == '__main__':
    main()