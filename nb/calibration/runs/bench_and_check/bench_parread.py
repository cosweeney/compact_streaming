"""
The catalogue read is CPU-bound on gzip decompression, not I/O or layout.

Established by benchmark_catread.py on base_c130 z0p500:
  * layout is a COMPOUND table, so a mass-first two-stage read saves
    nothing (MASS_ONLY 119.3 s vs FULL ~121 s) -- that idea is dead.
  * the file is 15.1 GB gzip-compressed, 395,393,780 rows x 60 B = ~23.7 GB
    uncompressed, read in ~120 s = ~200 MB/s, which is about what
    single-threaded gzip decompression does.
  * chunks=(6034,) means ~65,500 INDEPENDENTLY compressed chunks.

Independent chunks can be decompressed by independent processes.  This
times a parallel read where each worker opens the file itself, reads its
own row range, applies the mass selection, and returns ONLY the selected
rows -- which also avoids the ~19 GB of full-length pos/vel arrays the
current code builds and then discards 99.4% of.

Slab boundaries are aligned to the 6034-row chunk size: a read that starts
mid-chunk still forces that whole chunk to be decompressed, so unaligned
slabs would duplicate work at every boundary.

TWO CONFOUNDS, stated plainly:
  * PAGE CACHE -- this file is 15.1 GB and the machine has ~1.5 TB of RAM,
    so after the earlier benchmarks it may be entirely cached, making the
    I/O side of every number here look better than a cold read.  The
    DECOMPRESSION cost is CPU work that the cache cannot remove, and that
    is the part being parallelized, so the speedup should still be real --
    but the absolute times may be optimistic.  For a cold number, run this
    on a sim/z nothing has touched yet.
  * FILESYSTEM CONTENTION -- many processes reading one file on network
    storage may not scale the way local decompression does, which is why
    the worker count is swept rather than assumed.

Usage:
    python benchmark_parread.py            # default sim/z below
    python benchmark_parread.py 131 0.5    # ideally an UNREAD sim/z
"""

import os
import sys
import time

import numpy as np
import h5py as h5
from joblib import Parallel, delayed

import nb.calibration.runs.auto_runs.abacus_cosmo_core_auto as core

SIM_NUM = 130
Z = 0.5
COLS = ['mass', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'gid']
N_JOBS_SWEEP = (8, 16, 32, 56)


def _read_slab(hpath, key, i0, i1, m_edges):
    """One worker: decompress its own row range, select, return only the
    selected rows.  Returns small arrays, not the slab."""
    with h5.File(hpath, 'r') as hdf:
        sub = hdf[key][i0:i1]
    mass = sub['mass']
    keep = np.zeros(len(mass), dtype=bool)
    for lo, hi in m_edges:
        keep |= (mass >= lo) & (mass <= hi)
    if not keep.any():
        return None
    sel = sub[keep]
    return dict((nm, np.ascontiguousarray(sel[nm])) for nm in COLS)


def parallel_read(hpath, key, n_rows, chunk, m_edges, n_jobs):
    # align slabs to chunk boundaries -- a slab starting mid-chunk forces
    # that chunk to be decompressed twice
    chunks_total = int(np.ceil(n_rows / chunk))
    bounds = np.linspace(0, chunks_total, n_jobs + 1).astype(int) * chunk
    bounds[-1] = n_rows
    slabs = [(int(bounds[i]), int(bounds[i + 1])) for i in range(n_jobs)
             if bounds[i + 1] > bounds[i]]

    t0 = time.time()
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_read_slab)(hpath, key, i0, i1, m_edges) for i0, i1 in slabs)
    parts = [p for p in parts if p is not None]
    out = dict((nm, np.concatenate([p[nm] for p in parts])) for nm in COLS)
    return time.time() - t0, len(out['mass'])


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else SIM_NUM
    z = float(sys.argv[2]) if len(sys.argv) > 2 else Z

    hpath = core.halo_path(n, z)
    if not os.path.exists(hpath):
        raise FileNotFoundError(hpath)

    par = core.read_params(n, z)
    m_p = core.particle_mass(par['Omega_M'])
    m_edges = [(lo * m_p, hi * m_p) for lo, hi in core.BIN_NPART]

    with h5.File(hpath, 'r') as hdf:
        key = next(k for k in hdf.keys()
                   if getattr(hdf[k], 'dtype', None) is not None
                   and hdf[k].dtype.names)
        d = hdf[key]
        n_rows = d.shape[0]
        chunk = d.chunks[0] if d.chunks else 1
        itemsize = d.dtype.itemsize

    print(f'{hpath}')
    print(f'  {n_rows:,} rows x {itemsize} B = {n_rows*itemsize/1e9:.1f} GB uncompressed, '
          f'{os.path.getsize(hpath)/1e9:.1f} GB on disk, chunks={chunk}')
    print(f'  mass bins: ' + ', '.join(f'[{lo:.3e}, {hi:.3e}]' for lo, hi in m_edges))
    print()

    print('--- parallel read (selection done in the workers) ---')
    results = {}
    for nj in N_JOBS_SWEEP:
        t, n_sel = parallel_read(hpath, key, n_rows, chunk, m_edges, nj)
        results[nj] = t
        print(f'  n_jobs={nj:3d} -> {t:6.1f} s  ({n_sel:,} halos selected)')

    best_nj = min(results, key=results.get)
    t_best = results[best_nj]
    print(f'  best: n_jobs={best_nj} ({t_best:.1f} s)')

    t_serial = 121.0   # measured for this file by benchmark_catread.py
    print(f'\nserial baseline (measured previously): {t_serial:.1f} s')
    print(f'speedup at n_jobs={best_nj}: {t_serial/t_best:.1f}x')

    n_reads = len(core.SIM_NUMS) * 8
    print(f'\nover the grid ({n_reads} catalogue reads):')
    print(f'  serial   : {t_serial * n_reads / 3600:5.1f} h')
    print(f'  parallel : {t_best * n_reads / 3600:5.1f} h  '
          f'(saves {(t_serial - t_best) * n_reads / 3600:.1f} h)')
    print('\n(reminder: this file may be page-cached from earlier runs -- see the '
          'module docstring. Re-run on an untouched sim/z for a cold number.)')


if __name__ == '__main__':
    main()