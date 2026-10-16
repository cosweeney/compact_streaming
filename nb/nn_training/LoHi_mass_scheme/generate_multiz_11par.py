"""
Generate train/test/val sets of xi_0(s), xi_2(s), xi_4(s) predictions from
CompactStreamingModel across six redshifts, with n_s varied as an input
parameter -- theta = [b1, b2, omega_b, omega_cdm, h, As, ns].

Updated for the 11-parameter compact velocity model: the skew-t tail shape is
nu(r) = 2 + (nu_0 - 2)/(1 + (r_c/r)^2) with omega and alpha set by moment matching
(compact.model), and the cosmology emulators in compact/config.py come from the
streamlined cosmo-grid fit (N_MIN = 50 everywhere; Delta-shape rows and r_c of the
high-mass bin extrapolated above z = 0.5 from bin 0 + the mean low-z shift).

All three mass-bin emulation schemes (low_M, high_M, avg_M) are generated in ONE pass:
the LPT inputs (CAMB + velocileptors, ~45% of a prediction) depend only on (theta, z),
so they are computed once per theta and reused for every scheme. Output is identical to
calling CompactStreamingModel.predict(theta, mass_bin=scheme) separately (checked in
smoke_test_11par.py). Each scheme still gets its own directory tree:
  .../compact_multipole_training_set_11par_low_M/z0p510/thetas_train.npy
  .../compact_multipole_training_set_11par_high_M/z0p510/xi_train.npy
  .../compact_multipole_training_set_11par_avg_M/z0p510/...
The same Sobol seeds are used for every scheme, so predictions are comparable point by point.

Resumable: each finished (scheme, z, set) gets a '<xi file>.done' marker; a set that has
a checkpointed xi file but no marker is resumed (rows that are all-NaN are recomputed --
this also retries rows that failed). thetas files are regenerated from the seed and must
match what is on disk. Nothing is ever overwritten with different thetas.

n_s prior is AbacusSummit's emulator-grid range, [0.901, 1.025].
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json
import numpy as np
from scipy.stats import qmc
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import warnings
import psutil

from compact.predictions import CompactStreamingModel
from compact.config import Config

# ------------------------------------------------------------------ settings
REDSHIFTS = [0.295, 0.510, 0.706, 0.919, 1.317, 1.491]

AP_PADDING = {
    0.295: (18.57, 141.71),
    0.510: (17.78, 148.91),
    0.706: (17.24, 154.95),
    0.919: (16.80, 160.65),
    1.317: (16.26, 168.79),
    1.491: (16.10, 171.44),
}

SETS = {
    'train':  (50_000,  42),
    'test':   (5_000,   43),
    'val':    (5_000,   44),
}
N_JOBS = 100
CHECKPOINT_EVERY = 500
ELLS = (0, 2, 4)
NICE = 19
N_COMPACT_PARAMS = 11

MASS_BIN_SCHEMES = ['low_M', 'high_M', 'avg_M']

BASE_DIR_TEMPLATE = '/spiffball/cosweeney/data/training/compact_multipole_training_set_11par_{scheme}/'

BOUNDS = {
    'b1':          (0.0,    2.0),
    'b2':          (-5.0,   10.0),
    'omega_b':     (0.0207, 0.0243),
    'omega_cdm':   (0.104,  0.140),
    'h':           (0.55,   0.90),
    'lnAs_e10':    (2.0,    4.0),
    'ns':          (0.901,  1.025),
}
PARAM_NAMES = list(BOUNDS.keys())
LOWER = np.array([BOUNDS[p][0] for p in PARAM_NAMES])
UPPER = np.array([BOUNDS[p][1] for p in PARAM_NAMES])

warnings.filterwarnings('ignore', message='.*balance properties.*')
warnings.filterwarnings('ignore', message='.*is outside the calibrated range.*')


def zdir(z):
    return f"z{str(z).replace('.', 'p')}"


def make_thetas(n_samples, seed):
    sampler = qmc.Sobol(d=len(PARAM_NAMES), scramble=True, seed=seed)
    unit_samples = sampler.random(n_samples)
    samples = qmc.scale(unit_samples, LOWER, UPPER)
    idx = {name: i for i, name in enumerate(PARAM_NAMES)}
    return np.column_stack([
        samples[:, idx['b1']],
        samples[:, idx['b2']],
        samples[:, idx['omega_b']],
        samples[:, idx['omega_cdm']],
        samples[:, idx['h']],
        np.exp(samples[:, idx['lnAs_e10']]) / 1e10,
        samples[:, idx['ns']],
    ])


def preflight(cfg, redshifts=REDSHIFTS, schemes=MASS_BIN_SCHEMES):
    """Fail fast if config.py is not the 11-parameter emulator, or the grid / nodes can't
    serve the requested redshifts. Returns a list of non-fatal warnings (rows with no fit
    that silently fall back to _MEAN_PARAMS in predictions.py)."""
    for z in redshifts:
        s_req_min, s_req_max = AP_PADDING[z]
        if cfg.r[0] > s_req_min or cfg.r[-1] < s_req_max:
            raise ValueError(
                f"z={z}: cfg.r spans [{cfg.r[0]:.2f}, {cfg.r[-1]:.2f}] but AP "
                f"needs [{s_req_min:.2f}, {s_req_max:.2f}] h^-1 Mpc. Widen Config.r.")

    nodes = sorted(z for z, v in cfg.compact_emulator.items() if v is not None)
    if not nodes:
        raise ValueError('config.compact_emulator has no populated redshift nodes')
    if min(redshifts) < nodes[0] or max(redshifts) > nodes[-1]:
        raise ValueError(f'requested z in [{min(redshifts)}, {max(redshifts)}] is outside the '
                         f'calibrated nodes [{nodes[0]}, {nodes[-1]}]')

    warns = []
    for zn in nodes:
        emu = cfg.compact_emulator[zn]
        if len(emu['mean_params']) != N_COMPACT_PARAMS:
            raise ValueError(f"z={zn}: mean_params has {len(emu['mean_params'])} entries, expected "
                             f"{N_COMPACT_PARAMS} -- is compact/config.py still the 9-parameter version?")
        for scheme in schemes:
            fits = emu['fits'] if scheme == 'avg_M' else emu['fits_indiv'][scheme]
            rows = {r for r, _, _ in fits}
            bad = [r for r in rows if not 0 <= r < N_COMPACT_PARAMS]
            if bad:
                raise ValueError(f'z={zn} {scheme}: fit rows {bad} out of range')
            miss = sorted(set(range(N_COMPACT_PARAMS)) - rows)
            if miss:
                warns.append(f'z={zn} {scheme}: no fit for parameter rows {miss} (falls back to _MEAN_PARAMS)')
    return warns


def run_one(i, theta, z, cfg, schemes):
    """One theta, every requested scheme, LPT computed once.
    Returns (i, {scheme: (xi (len(ELLS), n_r) or None, error string or None)})."""
    psutil.Process().nice(NICE)
    model = CompactStreamingModel(redshift=z, config=cfg)
    b1, b2, ombh2, omch2, h, As, ns = theta
    cosmo = (h, ombh2, omch2, As, ns)
    try:
        lpt = model._lpt_inputs(cosmo, np.array([b1, b2]))
    except Exception as e:
        return i, {s: (None, f'LPT: {e}') for s in schemes}
    out = {}
    for s in schemes:
        try:
            pdf = model._pdf(lpt, cosmo, mass_bin=s)
            out[s] = (np.asarray(model._multipoles(pdf, ells=ELLS)), None)
        except Exception as e:
            out[s] = (None, str(e))
    return i, out


def _atomic_save(path, arr):
    tmp = path + '.tmp.npy'
    np.save(tmp, arr)
    os.replace(tmp, path)


def generate_set(name, n_samples, seed, z, cfg, schemes, base_dir_template, n_jobs):
    """Generate (or resume) one (z, set) for every scheme in `schemes`."""
    out_dirs = {s: os.path.join(base_dir_template.format(scheme=s), zdir(z)) for s in schemes}
    thetas = make_thetas(n_samples, seed)
    n_r = len(cfg.r)

    all_xi, xi_paths, todo_schemes = {}, {}, []
    for s in schemes:
        os.makedirs(out_dirs[s], exist_ok=True)
        s_path = os.path.join(out_dirs[s], 's.npy')
        if not os.path.exists(s_path):
            np.save(s_path, np.asarray(cfg.r))
        thetas_path = os.path.join(out_dirs[s], f'thetas_{name}.npy')
        xi_path = os.path.join(out_dirs[s], f'xi_{name}.npy')
        xi_paths[s] = xi_path

        if os.path.exists(thetas_path):
            if not np.allclose(np.load(thetas_path), thetas, rtol=0, atol=0):
                raise RuntimeError(f"{thetas_path} exists but does not match make_thetas(seed={seed}) "
                                   f"-- refusing to mix sample sets")
        else:
            np.save(thetas_path, thetas)

        if os.path.exists(xi_path + '.done'):
            continue                                   # finished in an earlier run
        if os.path.exists(xi_path):
            all_xi[s] = np.load(xi_path)
            assert all_xi[s].shape == (n_samples, len(ELLS), n_r), f"{xi_path}: unexpected shape"
        else:
            all_xi[s] = np.full((n_samples, len(ELLS), n_r), np.nan)
        todo_schemes.append(s)

    if not todo_schemes:
        print(f"z={z} '{name}': all schemes already done, skipping")
        return {}

    # rows still needed, per scheme (all-NaN rows); one task per row computes only what's missing
    need = {i: [s for s in todo_schemes if np.isnan(all_xi[s][i]).all()] for i in range(n_samples)}
    need = {i: ss for i, ss in need.items() if ss}
    print(f"\n{'='*70}\nz={z}  '{name}': {len(need)}/{n_samples} samples to compute "
          f"for schemes {todo_schemes} (seed={seed})\n{'='*70}")

    errors = {s: {} for s in todo_schemes}
    t0 = time.time()
    with Parallel(n_jobs=n_jobs, batch_size=1, return_as="generator_unordered") as par:
        stream = par(delayed(run_one)(i, thetas[i], z, cfg, ss) for i, ss in need.items())
        for done, (i, res) in enumerate(tqdm(stream, total=len(need), desc=f"z={z} {name}"), 1):
            for s, (xi_i, err) in res.items():
                if err is not None:
                    errors[s][i] = err
                else:
                    all_xi[s][i] = xi_i
            if done % CHECKPOINT_EVERY == 0:
                for s in todo_schemes:
                    _atomic_save(xi_paths[s], all_xi[s])
    for s in todo_schemes:
        _atomic_save(xi_paths[s], all_xi[s])
        n_bad = int(np.isnan(all_xi[s]).all(axis=(1, 2)).sum())
        with open(xi_paths[s] + '.done', 'w') as f:
            json.dump({'n_samples': n_samples, 'n_failed': n_bad, 'seed': seed}, f)
        if errors[s]:
            with open(os.path.join(out_dirs[s], f'errors_{name}.json'), 'w') as f:
                json.dump({str(k): v for k, v in errors[s].items()}, f)

    elapsed = time.time() - t0
    print(f"z={z} '{name}' done: {len(need)} thetas x {len(todo_schemes)} schemes in {elapsed/60:.1f} min "
          f"({elapsed/max(len(need), 1):.2f} s per theta wall, {n_jobs} workers)")
    for s in todo_schemes:
        if errors[s]:
            print(f"  [{s}] {len(errors[s])} failed -- first error: {next(iter(errors[s].values()))}")
    return errors


def generate_all(schemes=MASS_BIN_SCHEMES, redshifts=REDSHIFTS, sets=SETS,
                 base_dir_template=BASE_DIR_TEMPLATE, n_jobs=N_JOBS, cfg=None):
    cfg = Config() if cfg is None else cfg
    for w in preflight(cfg, redshifts, schemes):
        print('PREFLIGHT WARNING:', w)
    for z in redshifts:
        for name, (n_samples, seed_base) in sets.items():
            seed = seed_base + 1000 * REDSHIFTS.index(z)
            generate_set(name, n_samples, seed, z, cfg, list(schemes), base_dir_template, n_jobs)


if __name__ == '__main__':
    print(f"schemes: {MASS_BIN_SCHEMES}\nredshifts: {REDSHIFTS}\noutput: {BASE_DIR_TEMPLATE}")
    generate_all()