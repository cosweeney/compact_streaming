"""
Two independent ways to estimate how long run_abacus_batch_jk.py would take
for all available phases, BEFORE actually committing to that full run.

  1. ANALYTICAL (instant): pure arithmetic from your already-measured
     ~95 min/phase baseline (at effectively 64-way task parallelism, since
     the original N_JOBS=84 exceeds the 64 real tasks) plus the planned
     concurrency settings. Assumes every one of the 64 sub-box tasks costs
     about the same -- if the halo density is lumpy across the box (which
     is likely, at least somewhat), the slowest task sets the real wall
     time, not the average, so this is probably an OPTIMISTIC estimate,
     not a guarantee.

  2. EMPIRICAL (takes a few minutes, real work): times a SMALL number of
     actual sub-box tasks -- not the full 64 -- using the exact same
     get_pairwise_hist_jk code path, by restricting the seed catalog to
     just those sub-boxes' halos. This grounds the per-task cost in your
     real data instead of assuming uniformity, but sampling only a few of
     64 sub-boxes still might not catch the single slowest one -- treat it
     as a sanity check on #1, not a full replacement for it.

Neither of these has been run against your actual data -- I have no
cluster access. Run this yourself before scheduling the full job.
"""

import os
import sys
import time

import numpy as np
import h5py as h5
from scipy.spatial import cKDTree
from colossus.cosmology import cosmology

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compact.catalog.accumulate import (generate_sub_box_ids, split_simulation_into_sub_boxes,
                                        get_sub_box_id, relative_coordinates, get_zw13_vr_vt)
from pairhist_direct_jk import get_pairwise_hist_jk, PairHist, _Buffer

# ---------------------------------------------------------------- config ----
# must match run_abacus_batch_jk.py
HALO_TMPL = ('/spiff/hengweichang/AbacusSummit_base/snapshots/'
             'AbacusSummit_base_{sim}/halos/{z}/halo_combined.h5')
# RAM-backed (tmpfs), not /spiffball -- see run_abacus_batch_jk.py's SB_TMPL
# comment for why. Running benchmark_n_tasks() with this change vs. the
# earlier /spiffball-based version is exactly the test that tells us
# whether local staging closes the ~6.5x gap you measured.
SB_TMPL = '/dev/shm/{sim}_SBp_estimate/'
REDSHIFT = 'z0.200'
BENCHMARK_SIM = 'c000_ph000'   # any phase you already have on disk

box_length = 2000.0
subsize = box_length / 4
r_max = 140.0
M_NEIGHBOUR = 1e12

log_edges = np.array([[12, 12.01], [12.5, 12.6], [13, 13.1],
                      [13.5, 13.6], [14.0, 14.2], [14.4, 14.7]])
massbins = 10 ** log_edges

r_perp = np.linspace(5, 140, 135 + 1)
r_los = np.linspace(5, 140, 135 + 1)

cosmo = cosmology.setCosmology('planck18')
redshift = 0.2
a_scale = 1 / (1 + redshift)
conv = a_scale * cosmo.Hz(redshift) / cosmo.h * box_length
m_p = 2e9

R_edges, rlos_edges = r_perp, r_los
V_LO, V_HI, N_V = -4000.0, 4000.0, 400

N_TASKS = 64   # 4x4x4 sub-box grid, tasks_per_sub_box=1
N_PHASES_PLANNED = 16   # or 25 -- change to match what you're actually planning
TOTAL_CORES = 112


# ==================================================================== part 1
def analytical_estimate(baseline_min_per_phase, n_phases=N_PHASES_PLANNED,
                         total_cores=TOTAL_CORES, n_tasks=N_TASKS,
                         concurrency_options=(1, 2, 3, 4)):
    print(f'\n--- analytical estimate (baseline: {baseline_min_per_phase:.0f} min/phase) ---')
    for n_conc in concurrency_options:
        n_jobs = max(1, total_cores // n_conc)
        if n_jobs >= n_tasks:
            phase_min = baseline_min_per_phase   # full parallelism already, matches baseline
        else:
            waves = np.ceil(n_tasks / n_jobs)
            phase_min = baseline_min_per_phase * waves
        n_waves_of_phases = np.ceil(n_phases / n_conc)
        total_h = phase_min * n_waves_of_phases / 60
        print(f'{n_conc} concurrent phase(s), {n_jobs} jobs each: '
              f'~{phase_min:.0f} min/phase, ~{total_h:.1f} h total for {n_phases} phases')


# ==================================================================== part 2
def _periodic_bbox_mask(pos, lo, hi, boxsize, padding):
    """Boolean mask for positions within [lo-padding, hi+padding] in every
    dimension, using the minimum-image convention so it wraps correctly for
    a target sub-box sitting at the edge of the periodic box. Approximate
    (assumes hi-lo+2*padding doesn't exceed boxsize, true here: subsize=500
    vs boxsize=2000)."""
    mask = np.ones(len(pos), dtype=bool)
    for d in range(3):
        center = 0.5 * (lo[d] + hi[d])
        half_width = 0.5 * (hi[d] - lo[d]) + padding
        diff = (pos[:, d] - center + boxsize / 2) % boxsize - boxsize / 2
        mask &= np.abs(diff) <= half_width
    return mask


def benchmark_fast(sim=BENCHMARK_SIM, n_tasks_to_sample=1, padding=5.0):
    """Fast, IN-MEMORY-ONLY per-task timing check -- skips the sub-box FILE
    machinery entirely. That's what cost ~5 min in a naive benchmark
    (generate_sub_box_ids + split_simulation_into_sub_boxes), and that cost
    is FIXED regardless of how many tasks you're timing afterward, since it
    processes the whole box's tracer population every time. The catalog is
    already sitting in memory right after being read, so this just filters
    it directly with a boolean mask instead of writing it back out to disk
    and reading it back in per-task -- the same core computation
    (cKDTree build, neighbor query, velocity moments, PairHist
    accumulation), without that round-trip.

    CAVEAT, stated plainly: this skips the real per-task file I/O the
    production pipeline actually does (each of the 64 tasks in a real run
    reads its own sub-box file fresh from disk). If that I/O is a
    meaningful fraction of the true per-task cost, this UNDERESTIMATES it.
    Treat this as a fast check on the computational cost specifically, not
    a fully faithful reproduction of real per-task wall time -- a genuine
    trade-off for the <10 min turnaround, not a free lunch.
    """
    halo_path = HALO_TMPL.format(sim=sim, z=REDSHIFT)
    if not os.path.exists(halo_path):
        raise FileNotFoundError(f'{halo_path} -- change BENCHMARK_SIM to a phase you have')

    print(f'\n--- fast in-memory benchmark: {n_tasks_to_sample} task(s) from {sim} ---')
    t_read = time.time()
    with h5.File(halo_path, 'r') as hdf:
        hcat = hdf['halos'][:]
    Mvir = m_p * hcat['N_total']
    pos = np.array([hcat['x'], hcat['y'], hcat['z']]).T % box_length
    vel = np.array([hcat['vx'], hcat['vy'], hcat['vz']]).T
    del hcat
    print(f'catalogue read in {time.time() - t_read:.1f} s ({len(pos):,} halos)')

    nb = Mvir >= M_NEIGHBOUR
    seed_mbin = np.full(len(Mvir), -1, dtype=np.int32)
    for k, (lo, hi) in enumerate(massbins):
        seed_mbin[(Mvir >= lo) & (Mvir <= hi)] = k

    sb_id_all = get_sub_box_id(pos.copy(), box_length, subsize)
    real_ids = sorted(set(sb_id_all[seed_mbin >= 0].tolist()))
    sample_ids = real_ids[:n_tasks_to_sample]

    r_query = float(r_max)   # rlos_max defaults to r_max in the real pipeline too

    task_times = []
    for target_id in sample_ids:
        seed_sel = (sb_id_all == target_id) & (seed_mbin >= 0)
        n_seeds = int(seed_sel.sum())
        if n_seeds == 0:
            continue

        lo = pos[sb_id_all == target_id].min(axis=0)
        hi = pos[sb_id_all == target_id].max(axis=0)
        nbr_mask = _periodic_bbox_mask(pos[nb], lo, hi, box_length, padding)
        pos_local = np.ascontiguousarray(pos[nb][nbr_mask] % box_length)
        vel_local = vel[nb][nbr_mask]

        sp, sv, sm = pos[seed_sel], vel[seed_sel], seed_mbin[seed_sel]

        t0 = time.time()
        tree = cKDTree(pos_local, boxsize=box_length)
        h = PairHist(len(massbins), R_edges, rlos_edges, V_LO, V_HI, N_V)
        buf = _Buffer(h, conv)
        for i in range(n_seeds):
            nbi = tree.query_ball_point(sp[i], r_query, return_sorted=False)
            if not nbi:
                continue
            nbi = np.asarray(nbi)
            rel_pos = relative_coordinates(sp[i], pos_local[nbi], box_length)
            R = np.sqrt(rel_pos[:, 0] ** 2 + rel_pos[:, 1] ** 2)
            keep_mask = (R <= r_max) & (R > 0) & (np.abs(rel_pos[:, 2]) <= r_max)
            if not keep_mask.any():
                continue
            rel_pos = rel_pos[keep_mask]
            rel_vel = vel_local[nbi[keep_mask]] - sv[i]
            vr, _ = get_zw13_vr_vt(rel_pos, rel_vel)
            buf.add(sm[i], R[keep_mask], rel_pos[:, 2], rel_vel[:, 2], vr)
        buf.flush()
        elapsed = time.time() - t0
        task_times.append(elapsed)
        print(f'sub-box {target_id}: {n_seeds:,} seeds, {len(pos_local):,} neighbor pool, '
              f'{elapsed:.1f} s')

    if not task_times:
        print('no valid sub-boxes sampled -- try a different BENCHMARK_SIM or more samples')
        return None

    mean_t = np.mean(task_times)
    print(f'\nmean: {mean_t:.1f} s/task over {len(task_times)} sample(s)'
          + (f' (range: {min(task_times):.1f}-{max(task_times):.1f} s)' if len(task_times) > 1 else ''))
    print(f'if representative of all 64 tasks: full-phase estimate at full parallelism '
          f'~ {mean_t / 60:.1f} min -- remember this skips real per-task file I/O, so treat '
          f'it as a lower bound. Compare against your measured ~95 min baseline: a big gap '
          f'points at file I/O and/or load imbalance across sub-boxes mattering a lot, which '
          f'this quick check can\'t rule out with only {len(task_times)} sample(s).')
    return task_times


# ==================================================== part 2b: slower, thorough alternative
def benchmark_n_tasks(sim=BENCHMARK_SIM, n_tasks_to_time=3, n_jobs=None):
    """The SLOWER, more faithful alternative to benchmark_fast() above --
    goes through the real sub-box FILE machinery (generate_sub_box_ids +
    split_simulation_into_sub_boxes), so it captures real per-task file I/O
    that benchmark_fast() skips, at the cost of a ~5 min fixed setup cost
    regardless of how many tasks you time. Worth running once you have a
    quick read from benchmark_fast() and want to check whether file I/O
    explains a gap between that and your measured ~95 min baseline.
    """
    halo_path = HALO_TMPL.format(sim=sim, z=REDSHIFT)
    sb_path = SB_TMPL.format(sim=sim, z=REDSHIFT)
    if not os.path.exists(halo_path):
        raise FileNotFoundError(f'{halo_path} -- change BENCHMARK_SIM to a phase you have')

    print(f'\n--- empirical benchmark: {n_tasks_to_time} real tasks from {sim} ---')
    print('reading catalogue...')
    with h5.File(halo_path, 'r') as hdf:
        hcat = hdf['halos'][:]
    Mvir = m_p * hcat['N_total']
    pos = np.array([hcat['x'], hcat['y'], hcat['z']]).T % box_length
    vel = np.array([hcat['vx'], hcat['vy'], hcat['vz']]).T
    hid = hcat['hid_cleaned']
    del hcat

    nb = Mvir >= M_NEIGHBOUR

    print('building sub-box files (setup cost, not part of the per-task timing below)...')
    import shutil
    shutil.rmtree(sb_path, ignore_errors=True)
    os.makedirs(sb_path, exist_ok=True)
    chunksize = max(nb.sum() // 1000, 1)
    generate_sub_box_ids(pos[nb], box_length, subsize, chunksize, sb_path, name='halo')
    split_simulation_into_sub_boxes(
        positions=pos[nb], velocities=vel[nb], ids=hid[nb], upids=hid[nb],
        boxsize=box_length, subsize=subsize, chunksize=chunksize,
        dtypes=[pos.dtype, vel.dtype], path=sb_path, name='halo')

    seed_mbin = np.full(len(Mvir), -1, dtype=np.int32)
    for k, (lo, hi) in enumerate(massbins):
        seed_mbin[(Mvir >= lo) & (Mvir <= hi)] = k

    # restrict the SEEDS to just n_tasks_to_time real sub-box IDs -- the
    # sub-box FILES still cover the whole box (load_halos needs adjacent
    # sub-boxes for edge padding regardless of which are being timed), only
    # the seed list (hence the task list get_pairwise_hist_jk builds) shrinks.
    sb_id_all = get_sub_box_id(pos.copy(), box_length, subsize)
    real_ids = sorted(set(sb_id_all[seed_mbin >= 0].tolist()))
    if len(real_ids) < n_tasks_to_time:
        raise RuntimeError(f'only {len(real_ids)} sub-boxes have any valid seeds')
    keep_ids = set(real_ids[:n_tasks_to_time])
    seed_mbin_subset = np.where(np.isin(sb_id_all, list(keep_ids)), seed_mbin, -1)
    n_seeds = int((seed_mbin_subset >= 0).sum())
    print(f'timing sub-boxes {sorted(keep_ids)} ({n_seeds:,} seeds total)...')

    t0 = time.time()
    total, regions = get_pairwise_hist_jk(
        pos, vel, seed_mbin_subset,
        r_max=r_max, boxsize=box_length, subsize=subsize, path=sb_path,
        n_M=len(massbins), R_edges=R_edges, rlos_edges=rlos_edges,
        v_lo=V_LO, v_hi=V_HI, n_v=N_V,
        conv=conv, n_jobs=n_tasks_to_time, tasks_per_sub_box=1, progress=False,
    )
    elapsed = time.time() - t0
    per_task = elapsed / n_tasks_to_time
    print(f'{n_tasks_to_time} tasks: {elapsed:.1f} s total, {per_task:.1f} s/task '
          f'({n_seeds/n_tasks_to_time:.0f} seeds/task on average)')

    shutil.rmtree(sb_path, ignore_errors=True)

    # extrapolate: at full (>=64-way) parallelism, wall time ~ the SLOWEST
    # task, not the average -- this only reports the average, which is a
    # lower bound if the sampled tasks aren't representative.
    print(f'if every one of the 64 real tasks costs about the same as these '
          f'{n_tasks_to_time}: full-phase estimate at full parallelism '
          f'~ {per_task/60:.1f} min (compare against your measured ~95 min --'
          f' a big gap either way means load imbalance across sub-boxes, or '
          f'meaningful fixed overhead per phase beyond the per-task search cost).')
    return per_task


if __name__ == '__main__':
    analytical_estimate(baseline_min_per_phase=95.0)
    benchmark_fast(n_tasks_to_sample=1)   # quick, <10 min; increase n_tasks_to_sample
                                            # for a feel of load imbalance, or call
                                            # benchmark_n_tasks() separately for the
                                            # slower, file-I/O-inclusive check