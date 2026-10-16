"""
Train xi_0, xi_2, xi_4 emulators for each mass-bin scheme at each of
REDSHIFTS with n_s varied (theta = [b1, b2, omega_b, omega_cdm, h, As, ns]),
using the train/test/val sets from generate_multiz_11par.py (11-parameter compact
velocity model). Training data are read from, and models written to, 11par-specific
directories, so the earlier 9-parameter emulators are not touched.

Schemes are processed sequentially (outer loop); within each scheme all
(z, ell) jobs run as independent, fully separate processes
(ProcessPoolExecutor, spawn context).  Thread count per job is capped so
18 concurrent jobs don't oversubscribe the machine.

Reads from the per-scheme data directory, writes flat model files (no
per-z subdirectory) under a per-scheme MODEL_DIR.
"""

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'   # must precede any TF import, in every process
# thread-pin BEFORE numpy/TF import -- 18 concurrent jobs share ~110 cores,
# so each gets a bounded slice rather than each spawning its own full pool
N_THREADS_PER_JOB = 6   # 110 // 18, rounded down, leaves headroom
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = str(N_THREADS_PER_JOB)

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import psutil

import sys
sys.path.insert(0, '/home/cosweeney/code/EmulateLSS')

# ---- mass-bin emulation schemes to train (sequentially; all (z, ell) jobs of a scheme in parallel) ----
MASS_BIN_SCHEMES = [
    'low_M',
    'high_M',
    'avg_M',
]

REDSHIFTS = [0.295, 0.510, 0.706, 0.919, 1.317, 1.491]
ELLS = (0, 2, 4)

BASE_DIR_TEMPLATE = '/spiffball/cosweeney/data/training/compact_multipole_training_set_11par_{scheme}/'
MODEL_DIR_TEMPLATE = '/spiffball/cosweeney/data/models_11par_{scheme}/'

N_HIDDEN = [128, 128, 128, 128]
N_EPOCHS = 1000
LRS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
NBATCHS = [320, 640, 1280, 2560, 5120]
PATIENCE = 1000
NICE = 10


def zdir(z):
    return f"z{str(z).replace('.', 'p')}"


def to_params(thetas):
    p = thetas.copy()
    p[:, 5] = np.log(1e10 * thetas[:, 5])
    return p


def flag(xi, rough_ref=None):
    xi0 = xi[:, 0]
    finite = np.isfinite(xi).all(axis=(1, 2))
    collapsed = (np.median(xi0, axis=1) < -0.5) | (np.abs(xi[:, 1]).max(axis=1) < 1e-8)
    rough = np.abs(np.diff(xi0, n=2, axis=1)).mean(axis=1) / np.abs(xi0).mean(axis=1)
    if rough_ref is None:
        rough_ref = np.median(rough[finite & ~collapsed])
    noisy = rough > 5 * rough_ref
    amp = np.abs(xi).max(axis=(1, 2))
    bad = ~finite | (amp > 10) | collapsed | noisy
    return bad, rough_ref


def train_emu_with_val(Ptrain, Ftrain_asinh, Pval, Fval_asinh,
                        n_hidden, n_pcs, n_epochs, fstd, pmean, pstd,
                        outfile, lrs, nbatchs, restart_file=None):
    from train_nn_emu import Emulator
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    tf.config.threading.set_intra_op_parallelism_threads(N_THREADS_PER_JOB)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    mean = np.mean(Ftrain_asinh, axis=0).astype('float32')
    sigmas = np.std(Ftrain_asinh, axis=0).astype('float32')
    Ftrain_n = (Ftrain_asinh - mean) / sigmas
    Fval_n = (Fval_asinh - mean) / sigmas

    cov_matrix = np.cov(Ftrain_n.T)
    w, v = np.linalg.eigh(cov_matrix)
    w, v = np.flip(w), np.flip(v, axis=1).astype('float32')
    pc_train = Ftrain_n @ v
    pc_mean, pc_sigmas = pc_train.mean(0), pc_train.std(0)

    emulator = Emulator(n_params=Ptrain.shape[-1], nks=Ftrain_n.shape[-1],
                        pc_sigmas=pc_sigmas, pc_mean=pc_mean, v=v,
                        sigmas=sigmas, mean=mean, fstd=fstd,
                        param_mean=pmean, param_sigmas=pstd,
                        n_components=n_pcs, n_hidden=n_hidden)
    if restart_file is not None:
        emulator.load(restart_file)

    emulator.compile(optimizer='adam', loss='mse', metrics=['mse'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)

    for lr, nbatch in zip(lrs, nbatchs):
        emulator.optimizer.learning_rate.assign(lr)
        emulator.fit(Ptrain, Ftrain_n, epochs=n_epochs, batch_size=nbatch,
                     validation_data=(Pval, Fval_n), callbacks=[es], verbose=2)
        if outfile is not None:
            emulator.save(outfile)
    return emulator


def train_one(scheme, base_dir, model_dir, z, ell):
    """Everything needed for one (z, ell) job -- self-contained so it can run
    as an independent process. base_dir/model_dir/scheme are passed in
    explicitly (not read from module globals) because the spawn context
    re-imports this module fresh in each worker, so any state set in the
    parent after import time would not carry over."""
    psutil.Process().nice(NICE)

    z_dir = os.path.join(base_dir, zdir(z))
    for split in ('train', 'val'):
        marker = os.path.join(z_dir, f'xi_{split}.npy.done')
        if not os.path.exists(marker):
            raise FileNotFoundError(f"{marker} missing -- generation of this set did not finish")
    thetas_train = np.load(os.path.join(z_dir, 'thetas_train.npy'))
    xi_train = np.load(os.path.join(z_dir, 'xi_train.npy'))
    thetas_val = np.load(os.path.join(z_dir, 'thetas_val.npy'))
    xi_val = np.load(os.path.join(z_dir, 'xi_val.npy'))

    assert thetas_train.shape[1] == 7, \
        f"expected 7-parameter theta (with ns), got shape {thetas_train.shape}"

    bad_train, rough_ref = flag(xi_train)
    bad_val, _ = flag(xi_val, rough_ref=rough_ref)

    th_tr, xi_tr = thetas_train[~bad_train], xi_train[~bad_train]
    th_va, xi_va = thetas_val[~bad_val], xi_val[~bad_val]

    print(f"[{scheme}] z={z} ell={ell}: train {len(th_tr)}/{len(thetas_train)}, "
          f"val {len(th_va)}/{len(thetas_val)}")

    Ptrain = to_params(th_tr)
    Pval = to_params(th_va)
    Pmean, Pstd = Ptrain.mean(0), Ptrain.std(0)
    Ptrain_n = (Ptrain - Pmean) / Pstd
    Pval_n = (Pval - Pmean) / Pstd

    i = ELLS.index(ell)
    Ftrain = xi_tr[:, i]
    Fstd = np.std(Ftrain, axis=0)
    Ftrain_asinh = np.arcsinh(Ftrain / Fstd)

    Fval = xi_va[:, i]
    Fval_asinh = np.arcsinh(Fval / Fstd)

    os.makedirs(model_dir, exist_ok=True)
    outfile = os.path.join(model_dir, f'compact_xi{ell}_{zdir(z)}_11par')

    train_emu_with_val(Ptrain_n, Ftrain_asinh, Pval_n, Fval_asinh,
                       n_hidden=N_HIDDEN, n_pcs=Ftrain.shape[-1],
                       n_epochs=N_EPOCHS, fstd=Fstd, pmean=Pmean, pstd=Pstd,
                       outfile=outfile, lrs=LRS, nbatchs=NBATCHS)
    return f"[{scheme}] z={z} ell={ell} done -> {outfile}"


if __name__ == '__main__':
    for scheme in MASS_BIN_SCHEMES:
        base_dir = BASE_DIR_TEMPLATE.format(scheme=scheme)
        model_dir = MODEL_DIR_TEMPLATE.format(scheme=scheme)

        os.makedirs(model_dir, exist_ok=True)
        jobs = [(z, ell) for z in REDSHIFTS for ell in ELLS]
        print(f"\n{'#'*70}")
        print(f"# scheme: {scheme}  ({len(jobs)} jobs, {N_THREADS_PER_JOB} threads each)")
        print(f"# data:   {base_dir}")
        print(f"# models: {model_dir}")
        print(f"{'#'*70}")

        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=len(jobs), mp_context=ctx) as pool:
            futures = {
                pool.submit(train_one, scheme, base_dir, model_dir, z, ell): (z, ell)
                for z, ell in jobs
            }
            for fut in as_completed(futures):
                z, ell = futures[fut]
                try:
                    print(fut.result())
                except Exception as e:
                    print(f"[{scheme}] z={z} ell={ell} FAILED: {e}")