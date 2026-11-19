"""
sigma8_design.py -- sigma_8 from CAMB for a design of (h, omega_b, omega_cdm, n_s) points, computed in parallel.

    python sigma8_design.py design_in.npz design_out.npz [--nproc N]

design_in.npz holds X (n, 4) and optionally Xt (m, 4), columns (h, ombh2, omch2, ns); design_out.npz receives y and yt (sigma_8 at A_s = A_S_REF, massive
neutrinos 0.06 eV, z = 0) and a_s_ref. sigma_8 scales exactly as sqrt(A_s), so any other A_s follows by rescaling.

This is a separate script (rather than a multiprocessing pool inside the notebook session) because forking a process that has already used CAMB's OpenMP
runtime can leave the workers deadlocked and idle. Here the workers are started with 'spawn' in a fresh interpreter, threading is limited to one thread per
process, and progress with an ETA is printed.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

A_S_REF = 2.1e-9


def sigma8_ref(x):
    import camb
    h, ombh2, omch2, ns = x
    pars = camb.set_params(H0=100 * h, ombh2=ombh2, omch2=omch2, As=A_S_REF, ns=ns, mnu=0.06, WantTransfer=True, redshifts=[0.0], kmax=2.0,
                           NonLinear=camb.model.NonLinear_none)
    pars.WantCls = False
    return float(camb.get_results(pars).get_sigma8_0())


def run(points, nproc):
    n = len(points)
    out = np.empty(n)
    t0 = time.time()
    step = max(1, n // 20)
    with ProcessPoolExecutor(max_workers=nproc, mp_context=mp.get_context('spawn')) as ex:
        for i, v in enumerate(ex.map(sigma8_ref, [tuple(p) for p in points], chunksize=1)):
            out[i] = v
            if (i + 1) % step == 0 or i + 1 == n:
                el = time.time() - t0
                print(f'  {i + 1}/{n} CAMB runs done, {el:.0f} s elapsed, about {el / (i + 1) * (n - i - 1):.0f} s left', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('infile')
    ap.add_argument('outfile')
    ap.add_argument('--nproc', type=int, default=max(1, min(os.cpu_count() or 1, 32)))
    a = ap.parse_args()
    d = np.load(a.infile)
    X = d['X']
    Xt = d['Xt'] if 'Xt' in d.files else np.empty((0, 4))
    print(f'sigma8_design: {len(X) + len(Xt)} CAMB runs on {a.nproc} worker processes', flush=True)
    y = run(np.vstack([X, Xt]), a.nproc)
    np.savez(a.outfile, y=y[:len(X)], yt=y[len(X):], a_s_ref=A_S_REF)
    print('sigma8_design: done', flush=True)


if __name__ == '__main__':
    main()