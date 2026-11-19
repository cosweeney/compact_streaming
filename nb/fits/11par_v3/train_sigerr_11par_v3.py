"""
Train xi_0, xi_2, xi_4 emulators with an EXTRA input sigma_err for the QSO redshift-error treatment by variance addition
(sigma_v^2 -> sigma_v^2 + sigma_err^2, sigma_err in km/s, a pair dispersion):

    theta = [b1, b2, omega_b, omega_cdm, h, As, ns, sigma_err]      (As -> ln(1e10 As) before the network, as in train_11par.py)

One redshift (QSO, z = 1.491), one scheme (avg_M), the v3 no-slope model. Reads the train/val sets written by generate_sigerr_11par_v3.py,
writes flat model files  <MODEL_DIR>/compact_xi<ell>_z1p491_11par_v3_noslope_sigerr.{json,...}  (the EmulateLSS naming used by the existing emulators:
<emulator dir>/compact_xi<ell>_<zdir><suffix>). The three multipole emulators train as three independent spawned processes.

Derived from train_11par.py: same network ([128]x4), learning-rate / batch-size schedule, outlier flagging, asinh + PCA output scaling. The only changes are the
8-parameter theta, the single (z, scheme), the paths, and the thread count per job (3 jobs instead of 18).

    python train_sigerr_11par_v3.py
"""

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'   # must precede any TF import, in every process
N_THREADS_PER_JOB = 12                    # 3 concurrent jobs; the original gave 6 threads to each of 18
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = str(N_THREADS_PER_JOB)

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import psutil

import sys
sys.path.insert(0, '/home/cosweeney/code/EmulateLSS')

Z = 1.491
ELLS = (0, 2, 4)
N_PARAMS = 8
PARAM_NAMES = ['b1', 'b2', 'omega_b', 'omega_cdm', 'h', 'As', 'ns', 'sigma_err']

BASE_DIR = '/spiffball/cosweeney/data/training/compact_multipole_training_set_11par_v3_noslope_sigerr_avg_M/'   # generate_sigerr_11par_v3.py
MODEL_DIR = '/spiffball/cosweeney/data/models_11par_v3_noslope_sigerr_avg_M/'
SUFFIX = '_11par_v3_noslope_sigerr'

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


def prepare(base_dir, z, ell):
    """Load, flag and normalise one (z, ell): returns everything train_emu_with_val needs (no TF here, so it can be checked on its own)."""
    z_dir = os.path.join(base_dir, zdir(z))
    for split in ('train', 'val'):
        marker = os.path.join(z_dir, f'xi_{split}.npy.done')
        if not os.path.exists(marker):
            raise FileNotFoundError(f"{marker} missing -- generation of this set did not finish")
    thetas_train = np.load(os.path.join(z_dir, 'thetas_train.npy'))
    xi_train = np.load(os.path.join(z_dir, 'xi_train.npy'))
    thetas_val = np.load(os.path.join(z_dir, 'thetas_val.npy'))
    xi_val = np.load(os.path.join(z_dir, 'xi_val.npy'))

    assert thetas_train.shape[1] == N_PARAMS, \
        f"expected {N_PARAMS}-parameter theta {PARAM_NAMES}, got shape {thetas_train.shape}"

    bad_train, rough_ref = flag(xi_train)
    bad_val, _ = flag(xi_val, rough_ref=rough_ref)

    th_tr, xi_tr = thetas_train[~bad_train], xi_train[~bad_train]
    th_va, xi_va = thetas_val[~bad_val], xi_val[~bad_val]
    print(f"z={z} ell={ell}: train {len(th_tr)}/{len(thetas_train)}, val {len(th_va)}/{len(thetas_val)}", flush=True)

    Ptrain, Pval = to_params(th_tr), to_params(th_va)
    Pmean, Pstd = Ptrain.mean(0), Ptrain.std(0)
    Ptrain_n, Pval_n = (Ptrain - Pmean) / Pstd, (Pval - Pmean) / Pstd

    i = ELLS.index(ell)
    Ftrain = xi_tr[:, i]
    Fstd = np.std(Ftrain, axis=0)
    return dict(Ptrain=Ptrain_n, Ftrain_asinh=np.arcsinh(Ftrain / Fstd), Pval=Pval_n,
                Fval_asinh=np.arcsinh(xi_va[:, i] / Fstd), n_pcs=Ftrain.shape[-1], fstd=Fstd, pmean=Pmean, pstd=Pstd)


def train_one(base_dir, model_dir, z, ell):
    """One ell, self-contained so it can run as an independent spawned process (paths are passed in, not read from module globals)."""
    psutil.Process().nice(NICE)
    d = prepare(base_dir, z, ell)
    os.makedirs(model_dir, exist_ok=True)
    outfile = os.path.join(model_dir, f'compact_xi{ell}_{zdir(z)}{SUFFIX}')
    train_emu_with_val(d['Ptrain'], d['Ftrain_asinh'], d['Pval'], d['Fval_asinh'],
                       n_hidden=N_HIDDEN, n_pcs=d['n_pcs'], n_epochs=N_EPOCHS, fstd=d['fstd'], pmean=d['pmean'], pstd=d['pstd'],
                       outfile=outfile, lrs=LRS, nbatchs=NBATCHS)
    return f"z={z} ell={ell} done -> {outfile}"


if __name__ == '__main__':
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"{'#' * 70}\n# sigma_err emulators: z = {Z}, ells {ELLS}, {N_THREADS_PER_JOB} threads each\n# data:   {BASE_DIR}\n# models: {MODEL_DIR}\n{'#' * 70}", flush=True)
    ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=len(ELLS), mp_context=ctx) as pool:
        futures = {pool.submit(train_one, BASE_DIR, MODEL_DIR, Z, ell): ell for ell in ELLS}
        for fut in as_completed(futures):
            ell = futures[fut]
            try:
                print(fut.result(), flush=True)
            except Exception as e:
                print(f"ell={ell} FAILED: {e}", flush=True)