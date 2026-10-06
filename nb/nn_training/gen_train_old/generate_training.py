"""
Generate three independent sets of xi_0(s), xi_2(s), xi_4(s) predictions from
CompactStreamingModel over Sobol sequences in (b1, b2, omega_b, omega_cdm, h, As):
  - train : 2**16 samples
  - test  : 2**12 samples
  - val   : 2**12 samples

Each set uses its own Sobol sampler (distinct seed) rather than slicing one
long sequence -- a prefix of a scrambled Sobol sequence is balanced at any
power-of-two length, but an arbitrary *slice* (e.g. samples 65536:69632) is
not guaranteed to be, so three independently-seeded samplers keep each set's
own balance property intact at the cost of a (usually negligible) risk of
mild correlation between sets.

Filenames are suffixed per set and distinct from the plain thetas.npy/xi.npy
of the existing 2**16 run in OUT_DIR -- this script refuses to run if any of
its own target files already exist, so a second accidental run doesn't
silently clobber a completed one either.
"""

import os
# must precede numpy/camb -- parallelism is at the task level, so each worker
# gets one thread and we avoid oversubscribing 110 processes
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from scipy.stats import qmc
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import psutil

from compact.predictions import CompactStreamingModel
from compact.config import Config

# ------------------------------------------------------------------ settings
SETS = {
    'train':  (50_000,  42), # not a power of two, but closer to the number we want. 
    'test':   (5_000,   43),
    'val':    (5_000,   44),
}
N_JOBS    = 110
CHECKPOINT_EVERY = 500   # completed predictions between saves, not a barrier
ELLS = (0, 2, 4)         # 4 is not fitted, but AP mixes it into the observed 0 and 2
OUT_DIR = '/spiffball/cosweeney/data/training/compact_multipole_training_set/z0p5/'
os.makedirs(OUT_DIR, exist_ok=True)

# s-range the AP interpolation will demand, from ap.s_prime_range over the prior box.
# Recompute if the priors change -- extrapolating the splines is silent and wrong.
S_REQUIRED = (16.0, 165.0)

BOUNDS = {
    'b1':          (0.0,    2.0),
    'b2':          (-5.0,   10.0),
    'omega_b':     (0.0207, 0.0243),
    'omega_cdm':   (0.104,  0.140),
    'h':           (0.55,   0.90),
    'lnAs_e10':    (2.0,    4.0),   # ln(1e10 * As); converted to As below
}
PARAM_NAMES = list(BOUNDS.keys())
LOWER = np.array([BOUNDS[p][0] for p in PARAM_NAMES])
UPPER = np.array([BOUNDS[p][1] for p in PARAM_NAMES])

NICE = 10   # calvinball is shared -- yield to interactive work

# ------------------------------------------------------------------ refuse to overwrite
# check every target path across all three sets up front, before any compute
target_paths = {}
for name in SETS:
    target_paths[name] = {
        'thetas': os.path.join(OUT_DIR, f'thetas_{name}.npy'),
        'xi':     os.path.join(OUT_DIR, f'xi_{name}.npy'),
    }
existing = [p for paths in target_paths.values() for p in paths.values() if os.path.exists(p)]
if existing:
    raise FileExistsError(
        "Refusing to run -- these target files already exist and would be "
        "overwritten:\n  " + "\n  ".join(existing) +
        "\nMove/rename them first if you intend to regenerate."
    )

# ------------------------------------------------------------------ shared setup
cfg = Config()

if cfg.r[0] > S_REQUIRED[0] or cfg.r[-1] < S_REQUIRED[1]:
    raise ValueError(
        f"cfg.r spans [{cfg.r[0]:.1f}, {cfg.r[-1]:.1f}] but AP needs "
        f"[{S_REQUIRED[0]:.1f}, {S_REQUIRED[1]:.1f}] h^-1 Mpc. Widen the grid."
    )
s_path = os.path.join(OUT_DIR, 's.npy')
if not os.path.exists(s_path):
    np.save(s_path, np.asarray(cfg.r))   # shared grid across all sets; written once


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
    ])


def run_one(i, theta):
    psutil.Process().nice(NICE)   # set in the worker; joblib children do not
                                  # inherit a nice value set after they spawn
    model = CompactStreamingModel(redshift=0.5, config=cfg)   # fresh instance per call -- safe for parallel workers
    try:
        return i, np.stack(model.predict(theta, ells=ELLS)), None
    except Exception as e:
        return i, None, str(e)


def generate_set(name, n_samples, seed):
    print(f"\n{'='*70}\ngenerating '{name}': {n_samples} samples (seed={seed})\n{'='*70}")

    thetas = make_thetas(n_samples, seed)
    np.save(target_paths[name]['thetas'], thetas)

    n_r = len(cfg.r)
    all_xi = np.full((n_samples, len(ELLS), n_r), np.nan)
    errors = {}
    xi_path = target_paths[name]['xi']

    t0 = time.time()
    with Parallel(n_jobs=N_JOBS, batch_size=1,
                  return_as="generator_unordered") as par:
        stream = par(delayed(run_one)(i, th) for i, th in enumerate(thetas))
        for done, (i, xi_i, err) in enumerate(tqdm(stream, total=n_samples, desc=name), 1):
            if err is not None:
                errors[i] = err
            else:
                all_xi[i] = xi_i
            if done % CHECKPOINT_EVERY == 0:
                np.save(xi_path, all_xi)
    np.save(xi_path, all_xi)

    elapsed = time.time() - t0
    print(f"'{name}' done: {n_samples} predictions in {elapsed/60:.1f} min "
          f"({elapsed/n_samples:.2f} s/prediction effective, {N_JOBS} workers)")
    if errors:
        print(f"{len(errors)} failed predictions in '{name}' -- indices: {list(errors.keys())}")
        for i, msg in errors.items():
            print(f"  [{i}] theta={thetas[i]}: {msg}")


# ------------------------------------------------------------------ run all three, largest first
for name, (n_samples, seed) in SETS.items():
    generate_set(name, n_samples, seed)