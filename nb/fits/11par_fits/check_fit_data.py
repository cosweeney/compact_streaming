"""
Check that every input the fits need exists where fit_lib_11par.py expects it, and loads cleanly.
Uses the fits' own path functions and data loaders, so a pass here means the fits will find and read their data.

    python check_fit_data.py            (run from the same folder / environment as the fits)

Checks, for all six tracer-bins (BGS, LRG1-3, ELG2, QSO):
  mocks    poles npz (keys, number of mocks, NaNs), mock covariance, and the actual loader for every s_min used
  DR1      extracted multipoles+window npz, DR1 covariance, and the actual loader for s_min = 20 and 40
  models   emulator files for both models (compact, gauss) at every effective redshift, ell = 0, 2, 4
  system   emcee/h5py importable, output directory writable
When something is missing it lists similarly named files in that folder, to show how the real names differ.
Exit code 1 if anything is missing or fails.
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np

import fit_lib_11par as fl
import common

problems = []


def ok(msg):
    print(f'  ok       {msg}')


def bad(msg, folder=None, tracer=None):
    print(f'  MISSING  {msg}')
    problems.append(msg)
    if folder is not None and Path(folder).is_dir():
        token = (tracer or '').split('_')[0]
        near = sorted(p.name for p in Path(folder).glob(f'*{token}*'))[:6]
        print(f'           similar files in {folder}: {near if near else "none"}')
    elif folder is not None:
        print(f'           (folder {folder} does not exist)')


def pd(cov):
    try:
        np.linalg.cholesky(cov)
        return True
    except np.linalg.LinAlgError:
        return False


def kn(key):
    return fl.kname(key)


# ------------------------------------------------------------------ mocks
print('MOCKS (poles, covariance, loaders)')
counts = {}
for key in fl.FIT_KEYS:
    tracer, zmin, zmax = key
    prog = fl.TRACERS[tracer]
    pp, cp = common.poles_path_for(tracer, prog, zmin, zmax), common.cov_path_for(tracer, prog, zmin, zmax)
    if not pp.exists():
        bad(f'{kn(key)}: mock poles {pp}', pp.parent, tracer)
        continue
    d = np.load(pp)
    poles = d['poles']
    counts[key] = poles.shape[0]
    n_nan = int(np.isnan(poles).any(axis=(1, 2)).sum())
    shape_ok = poles.ndim == 3 and poles.shape[1] == 3 and poles.shape[2] // common.REBIN == common.NBINS_COV
    (ok if shape_ok and not n_nan else bad)(
        f'{kn(key)}: poles {poles.shape}, {poles.shape[0]} mocks, {n_nan} with NaNs, npz keys {list(d.keys())}'
        + ('' if shape_ok else f' -- expected (n_mock, 3, {common.NBINS_COV * common.REBIN})'))
    if not cp.exists():
        bad(f'{kn(key)}: mock covariance {cp}', cp.parent, tracer)
        continue
    for smin in sorted(set(fl.SMINS_SINGLE_MOCK) | {20, 40}):
        try:
            s_data, data_vec, cov = fl.load_mock_fit_data(key, smin)
            good = cov.shape == (len(data_vec), len(data_vec)) and np.allclose(cov, cov.T) and pd(cov) and np.isfinite(data_vec).all()
            if not good:
                bad(f'{kn(key)}: loader at s_min={smin} gave a bad covariance/data vector (shape {cov.shape}, symmetric/PD/finite required)')
        except Exception as e:
            bad(f'{kn(key)}: loader failed at s_min={smin}: {type(e).__name__}: {e}')
    ok(f'{kn(key)}: covariance + loader fine for s_min in {sorted(set(fl.SMINS_SINGLE_MOCK) | {20, 40})}')
if counts:
    n_common = min(counts.values())
    print(f'  mock counts per tracer-bin: { {kn(k): v for k, v in counts.items()} }')
    print(f'  -> single-mock joint fits can use mocks 0..{n_common - 1} (row i is assumed to be the same mock in every tracer)')

# ------------------------------------------------------------------ DR1
print('\nDR1 (extracted multipoles + window, covariance, loaders)')
for key in fl.FIT_KEYS:
    npz_path, cov_path = fl.dr1_paths(key)
    if not npz_path.exists():
        bad(f'{kn(key)}: DR1 extracted file {npz_path}', npz_path.parent, key[0])
        raw = Path('/spiffball/cosweeney/DESI/DR1/CorrFuncs/gccomb/gccomb') / (npz_path.stem + '.npy')
        print(f'           raw allcounts for extraction {"exists" if raw.exists() else "also missing"}: {raw}')
        continue
    if not cov_path.exists():
        bad(f'{kn(key)}: DR1 covariance {cov_path}', cov_path.parent, key[0])
        continue
    for smin in (20, 40):
        try:
            s_data, data_vec, cov, W = fl.load_dr1_fit_data(key, smin)
            good = (cov.shape == (len(data_vec), len(data_vec)) and np.allclose(cov, cov.T) and pd(cov)
                    and W.shape[-1] == len(s_data) and np.isfinite(data_vec).all() and np.isfinite(W).all())
            if not good:
                bad(f'{kn(key)}: DR1 loader at s_min={smin} gave inconsistent shapes or a non-PD covariance (cov {cov.shape}, W {W.shape})')
        except Exception as e:
            bad(f'{kn(key)}: DR1 loader failed at s_min={smin}: {type(e).__name__}: {e}')
    ok(f'{kn(key)}: DR1 npz + covariance + loader fine for s_min = 20, 40')

# ------------------------------------------------------------------ emulators
print('\nEMULATOR FILES')
for model, (directory, suffix) in fl.MODELS.items():
    if not os.path.isdir(directory):
        bad(f'{model}: model directory {directory}')
        continue
    for z in sorted(set(fl.Z_EFF.values())):
        for ell in (0, 2, 4):
            prefix = os.path.join(directory, f'compact_xi{ell}_{fl.zdir(z)}{suffix}')
            if glob.glob(prefix + '*') or os.path.isdir(prefix):
                continue
            avail = sorted({Path(p).name.split('_')[2] for p in glob.glob(os.path.join(directory, f'compact_xi{ell}_z*'))})
            bad(f'{model}: z={z} ell={ell}: no file starting with {prefix}')
            print(f'           redshift tags present for ell={ell}: {avail if avail else "none"}')
    ok(f'{model}: checked {directory}')

# ------------------------------------------------------------------ system
print('\nSYSTEM')
try:
    import emcee
    import h5py
    ok(f'emcee {emcee.__version__}, h5py {h5py.__version__}')
except Exception as e:
    bad(f'emcee/h5py import failed: {e}')
root = fl.CHAIN_ROOT
while not root.exists() and root != root.parent:
    root = root.parent
(ok if os.access(root, os.W_OK) else bad)(f'chain output location writable: {fl.CHAIN_ROOT} (nearest existing folder: {root})')

print(f'\n{"ALL CHECKS PASSED" if not problems else str(len(problems)) + " PROBLEM(S) FOUND"}')
sys.exit(1 if problems else 0)