"""
Quick benchmark: how much faster would the cosmology-grid auto-statistics
run under each of the three parallelization changes?

Rather than running anything at full scale, this measures each method's
TWO cost components on small seed samples:
    T(n_seeds) = fixed_overhead + slope * n_seeds
by timing two different seed counts and taking the slope.  Extrapolating
that line to the bin's real seed count gives a full-run estimate for each
method, for a few minutes of actual compute.

Methods compared:
  A  file-based, TASKS_PER_SUB_BOX=1  -- what the code does now
  A2 file-based, TASKS_PER_SUB_BOX=2  -- suggestion (a), 128 tasks so all
     112 cores have work instead of only 64
  B  in-memory, one cKDTree over the bin, parallel over seed chunks --
     suggestion (b), no sub-box files and no 27-neighbour load per task

Suggestion (c), running several (sim, z) jobs concurrently, is NOT directly
benchmarked -- that needs multiple full jobs running together to mean
anything.  Instead this reports how method B scales from half to full
n_jobs: if it is already saturating (doubling cores well short of halving
the time), job-level concurrency has headroom and is worth trying; if it
scales near-linearly, the cores are already busy and (c) would mostly just
create contention.

Usage:
    python benchmark_parallel.py            # default sim/z/bin below
    python benchmark_parallel.py 130 0.5 0  # sim, z, mass bin

I have not run this -- no cluster access -- so the structure is checkable
but the numbers are yours to produce.
"""

import os
import shutil
import sys
import time

import numpy as np
import h5py as h5
from joblib import Parallel, delayed
from scipy.spatial import cKDTree

import nb.calibration.runs.auto_runs.abacus_cosmo_core_auto as core
from compact.velocities.pairhist import PairHist
from compact.velocities.pairhist_direct import get_pairwise_hist
from compact.catalog.accumulate import (generate_sub_box_ids, split_simulation_into_sub_boxes,
                                        relative_coordinates, get_zw13_vr_vt)

import psutil
psutil.Process().nice(19)

SIM_NUM = 130
Z = 0.5
K_BIN = 0            # 0 = [12.5,12.6] (the more populous, slower bin)
# Must be large enough that real work DOMINATES the ~20-45 s of fixed
# overhead -- (400, 1600) was far too small and produced negative slopes
# (pure noise).  With ~8000 neighbours per seed in this bin, these give
# several seconds of measurable parallel work.
SEED_SAMPLES = (50_000, 200_000)
N_JOBS_FULL = core.N_JOBS
# the benchmark itself showed 112 workers was SLOWER than 56 on this
# workload, so sweep rather than assume
N_JOBS_SWEEP = (28, 56, 112)
FLUSH = 5_000_000


# ----------------------------------------------------------- method B ------
def _inmem_chunk(pos_all, vel_all, seed_idx, r_max, rlos_max, boxsize,
                 n_M, R_edges, rlos_edges, v_lo, v_hi, n_v, conv):
    """One worker: build the tree over the WHOLE bin (cheap for a narrow
    mass bin), then process this chunk of seeds. No file I/O at all."""
    tree = cKDTree(pos_all, boxsize=boxsize)
    h = PairHist(n_M, R_edges, rlos_edges, v_lo, v_hi, n_v)
    r_query = float(np.hypot(r_max, rlos_max))

    mb, mR, ml, mv, mvr, nbuf = [], [], [], [], [], 0
    for i in seed_idx:
        nb = tree.query_ball_point(pos_all[i], r_query, return_sorted=False)
        if not nb:
            continue
        nb = np.asarray(nb)
        rel_pos = relative_coordinates(pos_all[i], pos_all[nb], boxsize)
        R = np.sqrt(rel_pos[:, 0] ** 2 + rel_pos[:, 1] ** 2)
        keep = (R <= r_max) & (R > 0) & (np.abs(rel_pos[:, 2]) <= rlos_max)
        if not keep.any():
            continue
        rel_pos = rel_pos[keep]
        rel_vel = vel_all[nb[keep]] - vel_all[i]
        vr, _ = get_zw13_vr_vt(rel_pos, rel_vel)

        mb.append(np.zeros(keep.sum(), dtype=np.int32))
        mR.append(R[keep].astype(np.float32))
        ml.append(rel_pos[:, 2].astype(np.float32))
        mv.append(rel_vel[:, 2].astype(np.float32))
        mvr.append(vr.astype(np.float32))
        nbuf += int(keep.sum())
        if nbuf >= FLUSH:
            h.add(np.concatenate(mb), np.concatenate(mR), np.concatenate(ml),
                  conv * np.concatenate(mv), conv * np.concatenate(mvr))
            mb, mR, ml, mv, mvr, nbuf = [], [], [], [], [], 0

    if nbuf:
        h.add(np.concatenate(mb), np.concatenate(mR), np.concatenate(ml),
              conv * np.concatenate(mv), conv * np.concatenate(mvr))
    return h


def time_inmem(pos_k, vel_k, n_sample, n_jobs, rng):
    seed_idx = rng.choice(len(pos_k), size=n_sample, replace=False)
    chunks = np.array_split(seed_idx, n_jobs)
    chunks = [c for c in chunks if c.size]
    t0 = time.time()
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_inmem_chunk)(pos_k, vel_k, c, core.r_max, core.RLOS_MAX,
                              core.box_length, 1, core.R_edges, core.rlos_edges,
                              core.V_LO, core.V_HI, core.N_V, core.CONV)
        for c in chunks)
    total = parts[0]
    for p in parts[1:]:
        total += p
    return time.time() - t0


# ----------------------------------------------------------- method A ------
def time_filebased(pos_k, vel_k, hid_k, sb_path, n_sample, tasks_per_sb, n_jobs, rng):
    seed_mbin = np.full(len(pos_k), -1, dtype=np.int32)
    seed_mbin[rng.choice(len(pos_k), size=n_sample, replace=False)] = 0
    t0 = time.time()
    get_pairwise_hist(
        pos_k, vel_k, seed_mbin,
        r_max=core.r_max, boxsize=core.box_length, subsize=core.subsize,
        path=sb_path, n_M=1, R_edges=core.R_edges, rlos_edges=core.rlos_edges,
        v_lo=core.V_LO, v_hi=core.V_HI, n_v=core.N_V, conv=core.CONV,
        n_jobs=n_jobs, tasks_per_sub_box=tasks_per_sb,
        rlos_max=core.RLOS_MAX, progress=False)
    return time.time() - t0


def fit_line(ns, ts, label=''):
    """T(n) = overhead + slope*n, from two points."""
    slope = (ts[1] - ts[0]) / (ns[1] - ns[0])
    overhead = ts[0] - slope * ns[0]
    if slope <= 0:
        print(f'  WARNING [{label}]: non-positive slope ({slope*1e3:.3f} ms/seed) -- '
              f'the seed samples are still too small for the real work to rise above '
              f'fixed overhead and timing noise. Any extrapolation below is meaningless; '
              f'raise SEED_SAMPLES and re-run.')
    return overhead, slope


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else SIM_NUM
    z = float(sys.argv[2]) if len(sys.argv) > 2 else Z
    k = int(sys.argv[3]) if len(sys.argv) > 3 else K_BIN
    rng = np.random.default_rng(0)

    hpath = core.halo_path(n, z)
    if not os.path.exists(hpath):
        raise FileNotFoundError(hpath)

    par = core.read_params(n, z)
    m_p = core.particle_mass(par['Omega_M'])
    lo_np, hi_np = core.BIN_NPART[k]

    print(f'reading {hpath}')
    t0 = time.time()
    with h5.File(hpath, 'r') as hdf:
        col = core._read_cols(hdf, ['mass', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'gid'])
    t_read = time.time() - t0
    mass = col['mass']
    pos = np.array([col['x'], col['y'], col['z']]).T % core.box_length
    vel = np.array([col['vx'], col['vy'], col['vz']]).T
    hid = col['gid']
    del col
    print(f'catalogue read in {t_read:.1f} s ({len(mass):,} halos)')

    sel = (mass >= lo_np * m_p) & (mass <= hi_np * m_p)
    pos_k = np.ascontiguousarray(pos[sel])
    vel_k = np.ascontiguousarray(vel[sel])
    hid_k = hid[sel]
    n_bin = len(pos_k)
    print(f'bin {k}: {n_bin:,} halos '
          f'({pos_k.nbytes/1e6:.0f} MB positions, {vel_k.nbytes/1e6:.0f} MB velocities)\n')
    del pos, vel, hid, mass

    # --- sub-box build: method A's fixed setup cost, which B skips entirely
    sb_path = core.SB_TMPL.format(sim=core.sim_name(n), zdir=core.zdir(z), k=k)
    shutil.rmtree(sb_path, ignore_errors=True)
    os.makedirs(sb_path, exist_ok=True)
    t0 = time.time()
    chunksize = max(n_bin // 1000, 1)
    generate_sub_box_ids(pos_k, core.box_length, core.subsize, chunksize, sb_path, name='halo')
    split_simulation_into_sub_boxes(
        positions=pos_k, velocities=vel_k, ids=hid_k, upids=hid_k,
        boxsize=core.box_length, subsize=core.subsize, chunksize=chunksize,
        dtypes=[pos_k.dtype, vel_k.dtype], path=sb_path, name='halo')
    t_subbox = time.time() - t0
    print(f'sub-box build (method A/A2 only): {t_subbox:.1f} s\n')

    results = {}

    for label, tps in (('A  (tasks_per_sb=1)', 1), ('A2 (tasks_per_sb=2)', 2)):
        ts = []
        for ns in SEED_SAMPLES:
            t = time_filebased(pos_k, vel_k, hid_k, sb_path, ns, tps, N_JOBS_FULL, rng)
            ts.append(t)
            print(f'{label}: {ns:6d} seeds -> {t:7.1f} s')
        oh, sl = fit_line(SEED_SAMPLES, ts, label)
        results[label] = (oh + t_subbox, sl)
        print(f'{label}: overhead {oh:.1f} s (+{t_subbox:.1f} s sub-box build), '
              f'{sl*1e3:.3f} ms/seed\n')

    ts = []
    for ns in SEED_SAMPLES:
        t = time_inmem(pos_k, vel_k, ns, N_JOBS_FULL, rng)
        ts.append(t)
        print(f'B  (in-memory)     : {ns:6d} seeds -> {t:7.1f} s')
    oh_b, sl_b = fit_line(SEED_SAMPLES, ts, 'B in-memory')
    results['B  (in-memory)'] = (oh_b, sl_b)
    print(f'B  (in-memory)     : overhead {oh_b:.1f} s (no sub-box build), '
          f'{sl_b*1e3:.3f} ms/seed\n')

    # --- n_jobs sweep for method B: which worker count is actually best,
    # and is there headroom for running several (sim,z) jobs at once?
    print('\n--- method B, n_jobs sweep at %s seeds ---' % format(SEED_SAMPLES[1], ','))
    sweep = {}
    for nj in N_JOBS_SWEEP:
        t = time_inmem(pos_k, vel_k, SEED_SAMPLES[1], nj, rng)
        sweep[nj] = t
        print(f'  n_jobs={nj:4d} -> {t:7.1f} s')
    best_nj = min(sweep, key=sweep.get)
    print(f'  best: n_jobs={best_nj} ({sweep[best_nj]:.1f} s)')
    if best_nj < max(N_JOBS_SWEEP):
        print(f'  -> more workers than {best_nj} makes it SLOWER, so the cores are not '
              f'the binding constraint: running several (sim,z) jobs concurrently at '
              f'n_jobs={best_nj} each (suggestion c) should raise throughput.')
    else:
        print('  -> still improving at the largest worker count: cores ARE the binding '
              'constraint, so (c) would mostly add contention.')

    print(f'\n--- extrapolated to the full bin ({n_bin:,} seeds), per (sim,z,bin) ---')
    base = None
    for label, (oh, sl) in results.items():
        tot = oh + sl * n_bin
        if base is None:
            base = tot
        print(f'{label}: {tot/60:8.1f} min   ({base/tot:.2f}x vs current)')

    n_jobs_total = len(core.SIM_NUMS) * 8 * len(core.BIN_NPART)
    print(f'\n--- projected over the whole grid ({len(core.SIM_NUMS)} cosmologies '
          f'x 8 redshifts x {len(core.BIN_NPART)} bins = {n_jobs_total} measurements) ---')
    print('(this bin is only one of the two, and counts fall at higher z, so treat '
          'these as an UPPER bound scaled from the slowest case)')
    for label, (oh, sl) in results.items():
        tot = (oh + sl * n_bin) * n_jobs_total / len(core.BIN_NPART)
        t_reads = t_read * len(core.SIM_NUMS) * 8
        print(f'{label}: {(tot + t_reads)/3600:7.1f} h  (incl. {t_reads/3600:.1f} h of catalogue reads)')

    if not core.KEEP_SUBBOXES:
        shutil.rmtree(sb_path, ignore_errors=True)


if __name__ == '__main__':
    main()