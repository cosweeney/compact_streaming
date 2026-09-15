"""
Generate train/test/val sets of xi_0(s), xi_2(s), xi_4(s) predictions from
CompactStreamingModel across five redshifts, with n_s varied as an input
parameter -- theta = [b1, b2, omega_b, omega_cdm, h, As, ns].

n_s prior is AbacusSummit's emulator-grid range, [0.901, 1.025]
(confirmed against Table 1 of arXiv:2501.01698 / arXiv:2502.19353,
fiducial 0.965).

Written to a separate "_free_ns" directory tree so this does NOT overwrite
the existing fixed-ns training data from generate_multi_z_training_sets.py.

z=1.317 and z=1.491 sit outside the calibrated compact-model grid (0.3-1.1)
and rely on _compact_params' cubic-spline-interp / linear-tail-extrapolation.
AP s-padding is unchanged by adding n_s (it only affects the background,
which n_s does not touch) -- reusing the same AP_PADDING as before.
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

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
REDSHIFTS = [0.510, 0.706, 0.919, 1.317, 1.491]

AP_PADDING = {
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
N_JOBS = 90 # ~80% of cores
CHECKPOINT_EVERY = 500
ELLS = (0, 2, 4)
BASE_DIR = '/spiffball/cosweeney/data/training/compact_multipole_training_set_free_ns/'
NICE = 10

BOUNDS = {
    'b1':          (0.0,    2.0),
    'b2':          (-5.0,   10.0),
    'omega_b':     (0.0207, 0.0243),
    'omega_cdm':   (0.104,  0.140),
    'h':           (0.55,   0.90),
    'lnAs_e10':    (2.0,    4.0),
    'ns':          (0.901,  1.025),   # AbacusSummit emulator-grid range
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


def run_one(i, theta, z, cfg):
    psutil.Process().nice(NICE)
    model = CompactStreamingModel(redshift=z, config=cfg)
    try:
        return i, np.stack(model.predict(theta, ells=ELLS)), None
    except Exception as e:
        return i, None, str(e)


def generate_set(name, n_samples, seed, z, cfg, out_dir):
    thetas_path = os.path.join(out_dir, f'thetas_{name}.npy')
    xi_path = os.path.join(out_dir, f'xi_{name}.npy')
    if os.path.exists(thetas_path) or os.path.exists(xi_path):
        raise FileExistsError(f"{thetas_path} or {xi_path} already exists -- refusing to overwrite")

    print(f"\n{'='*70}\nz={z}  '{name}' (free ns): {n_samples} samples (seed={seed})\n{'='*70}")
    thetas = make_thetas(n_samples, seed)
    np.save(thetas_path, thetas)

    n_r = len(cfg.r)
    all_xi = np.full((n_samples, len(ELLS), n_r), np.nan)
    errors = {}

    t0 = time.time()
    with Parallel(n_jobs=N_JOBS, batch_size=1,
                  return_as="generator_unordered") as par:
        stream = par(delayed(run_one)(i, th, z, cfg) for i, th in enumerate(thetas))
        for done, (i, xi_i, err) in enumerate(tqdm(stream, total=n_samples, desc=f"z={z} {name}"), 1):
            if err is not None:
                errors[i] = err
            else:
                all_xi[i] = xi_i
            if done % CHECKPOINT_EVERY == 0:
                np.save(xi_path, all_xi)
    np.save(xi_path, all_xi)

    elapsed = time.time() - t0
    print(f"z={z} '{name}' done: {n_samples} in {elapsed/60:.1f} min "
          f"({elapsed/n_samples:.2f} s/pred, {N_JOBS} workers)")
    if errors:
        print(f"{len(errors)} failed -- indices: {list(errors.keys())[:20]}"
              f"{'...' if len(errors) > 20 else ''}")


# ------------------------------------------------------------------ run
cfg = Config()

for z in REDSHIFTS:
    s_req_min, s_req_max = AP_PADDING[z]
    if cfg.r[0] > s_req_min or cfg.r[-1] < s_req_max:
        raise ValueError(
            f"z={z}: cfg.r spans [{cfg.r[0]:.2f}, {cfg.r[-1]:.2f}] but AP "
            f"needs [{s_req_min:.2f}, {s_req_max:.2f}] h^-1 Mpc. Widen Config.r."
        )

for z in REDSHIFTS:
    out_dir = os.path.join(BASE_DIR, zdir(z))
    os.makedirs(out_dir, exist_ok=True)

    s_path = os.path.join(out_dir, 's.npy')
    if not os.path.exists(s_path):
        np.save(s_path, np.asarray(cfg.r))

    for name, (n_samples, seed_base) in SETS.items():
        seed = seed_base + 1000 * REDSHIFTS.index(z)
        generate_set(name, n_samples, seed, z, cfg, out_dir)