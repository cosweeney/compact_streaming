"""
Train xi_0, xi_2, xi_4 emulators at each of REDSHIFTS with n_s varied
(theta = [b1, b2, omega_b, omega_cdm, h, As, ns]), using the train/test/val
sets from generate_multi_z_training_sets_free_ns.py.

All 5 redshifts x 3 multipoles = 15 trainings run as independent, fully
separate processes (ProcessPoolExecutor, spawn context) rather than
sequentially -- each is a standalone tf.keras model/training loop, and
mixing several independent Keras models inside one process is more fragile
than just giving each its own interpreter. Thread count per job is capped
so 15 concurrent jobs don't oversubscribe the machine; calvinball is shared,
so each worker is niced same as the generation script.

Reads from the "_free_ns" data directory, writes flat model files (no
per-z subdirectory) under a "_free_ns" suffix into MODEL_DIR.
"""

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'   # must precede any TF import, in every process
# thread-pin BEFORE numpy/TF import -- 15 concurrent jobs share 110 cores,
# so each gets a bounded slice rather than each spawning its own full pool
N_THREADS_PER_JOB = 7   # 110 // 15, rounded down, leaves headroom
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = str(N_THREADS_PER_JOB)

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import psutil

import sys
sys.path.insert(0, '/home/cosweeney/code/EmulateLSS')

REDSHIFTS = [0.510, 0.706, 0.919, 1.317, 1.491]
ELLS = (0, 2, 4)
BASE_DIR = '/spiffball/cosweeney/data/training/compact_multipole_training_set_free_ns/'
MODEL_DIR = '/spiffball/cosweeney/data/models_free_ns/'

N_HIDDEN = [128, 128, 128, 128]
N_EPOCHS = 1000
LRS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
NBATCHS = [320, 640, 1280, 2560, 5120]
PATIENCE = 1000
NICE = 10


def zdir(z):
    # bare string only -- used both as a data-directory component (joined
    # with os.path.join) and as part of a model filename. Do NOT embed a
    # slash here: that silently turns "compact_xi2_z0p510_free_ns" into a
    # nested path "compact_xi2_z0p510/_free_ns" wherever it's used in an
    # f-string filename rather than a real path join.
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
    # imported here, not at module level -- must happen after this process's
    # own TF_USE_LEGACY_KERAS / thread-count env vars are set, and before any
    # TF op runs in this process
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


def train_one(z, ell):
    """Everything needed for one (z, ell) job -- self-contained so it can run
    as an independent process. Reloads/re-cleans this z's data once per job;
    a little redundant across the 3 ells sharing a z, but cheap relative to
    training time."""
    psutil.Process().nice(NICE)

    z_dir = os.path.join(BASE_DIR, zdir(z))
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

    # defensive: ensure MODEL_DIR exists even if this job runs standalone,
    # outside the __main__ dispatch loop that normally creates it once
    os.makedirs(MODEL_DIR, exist_ok=True)

    outfile = os.path.join(MODEL_DIR, f'compact_xi{ell}_{zdir(z)}_free_ns')
    train_emu_with_val(Ptrain_n, Ftrain_asinh, Pval_n, Fval_asinh,
                       n_hidden=N_HIDDEN, n_pcs=Ftrain.shape[-1],
                       n_epochs=N_EPOCHS, fstd=Fstd, pmean=Pmean, pstd=Pstd,
                       outfile=outfile, lrs=LRS, nbatchs=NBATCHS)
    return f"z={z} ell={ell} done -> {outfile}"


if __name__ == '__main__':
    os.makedirs(MODEL_DIR, exist_ok=True)
    jobs = [(z, ell) for z in REDSHIFTS for ell in ELLS]   # 15 independent jobs
    print(f"dispatching {len(jobs)} jobs, {N_THREADS_PER_JOB} threads each")

    ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=ctx) as pool:
        futures = {pool.submit(train_one, z, ell): (z, ell) for z, ell in jobs}
        for fut in as_completed(futures):
            z, ell = futures[fut]
            try:
                print(fut.result())
            except Exception as e:
                print(f"z={z} ell={ell} FAILED: {e}")