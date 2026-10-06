"""
Runs AbacusSummit base-suite phases using
pairhist_direct_jk.get_pairwise_hist_jk instead of the original
get_pairwise_hist, to also get a delete-one-region jackknife covariance for
<v_r|r> per mass bin.

Modeled closely on nb/calibration/runs/run_abacus_batch.py (same halo
selection, same mass bins, same grids, same M_PART=2e9 convention -- kept
exactly as that script uses it, NOT the 2.1e9 correction used elsewhere in
this project, so this stays comparable to the ALREADY-existing
pairhist_allbins.h5 files) -- but does NOT edit that script; this is a
separate file.

Concurrency (N_CONCURRENT_PHASES, N_JOBS_PER_PHASE): originally set up to
run several phases at once, on the assumption that the ~95 min/phase cost
was I/O-latency-bound and therefore cheap to share. That assumption turned
out to be wrong: a direct benchmark showed a single real task takes ~2609s
even with local, RAM-backed sub-box staging (no network latency at all),
and 64 such tasks running with zero contention should finish a phase in
about that same time (~43.5 min) -- but the measured real phase time is
~95 min, a ~2.2x gap showing real contention exists even within ONE phase
at n_jobs=84. Given that, running multiple phases concurrently (each
wanting the same heavy, large-neighborhood tree builds) would likely make
things worse, not better. This now defaults to N_CONCURRENT_PHASES=1
(sequential) and N_JOBS_PER_PHASE=84 -- exactly the configuration your own
~95 min/phase measurement came from, i.e. the one setup we actually know
works, kept as the default rather than a hoped-for optimization. See the
comments by those two constants below if you want to carefully explore
concurrency further later, with real monitoring.

Sub-box files are staged on /dev/shm (RAM-backed tmpfs), not /spiffball --
this is a separate, low-risk change from the concurrency question above:
it changes WHERE files are staged, not WHAT gets computed, so it can only
help or be neutral. It wasn't enough on its own to close the gap above
(the dominant cost turned out to be the full 27-sub-box neighbor loading
itself, not network latency), but there's no reason to revert it.

STORAGE: the 64 region-level PairHist pieces per phase are NOT saved to
disk -- at ~51 MB each (matching the existing single-box PairHist file
size), keeping all 64 would be ~3.2 GB per phase, ~50-80 GB across
16-25 phases, which would revisit the exact spiffball space concern from
earlier in this project. Instead, the jackknife covariance is computed
from them immediately, in this same run, and only that (tiny) covariance
matrix plus the usual `total` PairHist are written out.

I have not run this myself -- no access to your cluster or data -- so
treat this as a structurally-complete first draft to check over, the same
way as everything else in this pipeline so far.
"""

import gc
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import h5py as h5
from colossus.cosmology import cosmology

from compact.catalog.accumulate import generate_sub_box_ids, split_simulation_into_sub_boxes

from pairhist_direct_jk import get_pairwise_hist_jk, vr_jackknife_covariance

def vr_profile(h, k, r_edges=None, r_max_trust=None):
    """Pair-weighted <v_r>(r) and sigma_vr(r) for mass bin k. From
    ASbase_compact_calibration.ipynb -- unchanged."""
    Rc = 0.5 * (h.R_edges[1:] + h.R_edges[:-1])
    lc = 0.5 * (h.rlos_edges[1:] + h.rlos_edges[:-1])
    r3 = np.sqrt(Rc[:, None] ** 2 + lc[None, :] ** 2).ravel()
    n = h.n[k].ravel().astype(float)
    s1 = h.sum_vr[k].ravel()
    s2 = h.sum_vr2[k].ravel()
    if r_edges is None:
        top = r_max_trust if r_max_trust is not None else h.R_edges[-1]
        r_edges = np.arange(0.0, top + 2.0, 2.0)
    nb_bins = len(r_edges) - 1
    idx = np.searchsorted(r_edges, r3, side='right') - 1
    ok = (idx >= 0) & (idx < nb_bins) & (n > 0)

    def acc(w):
        return np.bincount(idx[ok], weights=w[ok], minlength=nb_bins)

    N = acc(n)
    r_eff = np.where(N > 0, acc(n * r3) / np.where(N > 0, N, 1), np.nan)
    mu = np.where(N > 0, acc(s1) / np.where(N > 0, N, 1), np.nan)
    m2 = np.where(N > 0, acc(s2) / np.where(N > 0, N, 1), np.nan)
    sd = np.sqrt(np.maximum(m2 - mu ** 2, 0.0))
    return r_eff, mu, sd, N.astype(np.int64)

import psutil
psutil.Process().nice(19)

# ---------------------------------------------------------------- config ----
PHASES = [f'c000_ph{i:03d}' for i in range(0, 10)]   # first 10 phases, per plan  # match what's actually available; adjust if not
REDSHIFT = 'z0.200'

HALO_TMPL = ('/spiff/hengweichang/AbacusSummit_base/snapshots/'
             'AbacusSummit_base_{sim}/halos/{z}/halo_combined.h5')
# RAM-backed (tmpfs), not /spiffball -- the halo catalog read itself still
# has to hit the network (that's the ~159s you measured), but the sub-box
# files get read back MANY times over during the search (each task also
# reads its neighbors' sub-boxes for padding, so the same sub-box is
# re-fetched by every task that borders it). Staging them locally turns
# that redundant re-reading into RAM-speed access instead of repeated
# network round-trips, which is very likely the dominant cost given the
# ~6.5x gap between the in-memory and real-pipeline benchmarks. At ~2 GB
# per phase (per run_abacus_batch.py's own docstring) this is trivial
# against the 1.5 TB this project's notes say the machine has -- but check
# `df -h /dev/shm` yourself before relying on it, since tmpfs size limits
# and quotas vary by system.
SB_TMPL = '/dev/shm/{sim}_SBp_jk/'
OUT_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base/halos/'
            'AbacusSummit_base_{sim}/halos/{z}/data/pairhist_allbins_jk.h5')   # NEW path --
                                                            # does not touch/overwrite the
                                                            # existing pairhist_allbins.h5
LOG_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base/logs/'
            'pairhist_jk_{sim}.log')

box_length = 2000.0
subsize = box_length / 4
r_max = 140.0
M_NEIGHBOUR = 1e12

log_edges = np.array([[12, 12.01], [12.5, 12.6], [13, 13.1],
                      [13.5, 13.6], [14.0, 14.2], [14.4, 14.7]])
massbins = 10**log_edges

r_perp = np.linspace(5, 140, 135 + 1)
r_los = np.linspace(5, 140, 135 + 1)

cosmo = cosmology.setCosmology('planck18')
redshift = 0.2
a = 1 / (1 + redshift)
conv = a * cosmo.Hz(redshift) / cosmo.h * box_length

m_p = 2e9   # h^-1 M_sun -- matches run_abacus_batch.py's OWN convention exactly,
            # not the 2.1e9 correction used elsewhere, so this stays comparable
            # to the already-existing pairhist_allbins.h5 files

R_edges = r_perp
rlos_edges = r_los
V_LO, V_HI, N_V = -4000.0, 4000.0, 400
CONV = conv
M_PART = m_p

# jackknife r-bins for <v_r|r> -- must match gather_paper_data.py's
# r_edges_shared, since vr_jackknife_covariance reuses gather_paper_data's
# own vr_profile() to reduce each region.
VR_R_EDGES = np.arange(2.0, 142.0, 2.0)

RESUME = True
KEEP_SUBBOXES = False

TOTAL_CORES = 112                # this machine's core count -- adjust if wrong

# Reverted from an earlier attempt at N_CONCURRENT_PHASES=2: that was based
# on the assumption that the ~95 min/phase cost was I/O-latency-bound and
# therefore cheap to share across concurrent phases. A direct benchmark
# showed otherwise -- even a single real task (with local RAM-backed
# staging, no network latency at all) took ~2609s, and 64 of those running
# together should, in a contention-free world, finish in about that same
# ~2609s (43.5 min) -- but the actual measured full-phase time is ~95 min,
# a ~2.2x gap that means there's real contention (memory bandwidth, CPU
# cache, or similar) even within ONE phase at n_jobs=84. Adding a second
# concurrent phase's worth of the same heavy, large-neighborhood tree
# builds would very plausibly make that worse, not better. Defaulting back
# to sequential, proven-safe execution until there's actual evidence
# (e.g., a real 2-concurrent-phase trial, watched closely) that some
# concurrency is safe here.
N_CONCURRENT_PHASES = 1

# Hardcoded to match run_abacus_batch.py's own deliberate choice, not
# derived from TOTAL_CORES // N_CONCURRENT_PHASES (which would give 112) --
# that script explicitly used 84 instead of the full 112, which reads as an
# intentional courtesy margin on a shared machine, consistent with this
# project's other conventions (niceing jobs, etc.). This is also the exact
# setting your own ~95 min/phase measurement came from, so it's the one
# configuration we actually know works.
N_JOBS_PER_PHASE = 84


def log(sim, msg):
    log_path = LOG_TMPL.format(sim=sim)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(log_path, 'a') as f:
        f.write(line + '\n')


def run_phase_jk(sim, n_jobs):
    """One phase, region-preserving. Returns (sim, elapsed_seconds, error_or_None)."""
    t0 = time.time()
    try:
        halo_path = HALO_TMPL.format(sim=sim, z=REDSHIFT)
        sb_path = SB_TMPL.format(sim=sim, z=REDSHIFT)
        out_path = OUT_TMPL.format(sim=sim, z=REDSHIFT)

        if not os.path.exists(halo_path):
            raise FileNotFoundError(halo_path)

        log(sim, f'{sim}: reading catalogue')
        with h5.File(halo_path, 'r') as hdf:
            hcat = hdf['halos'][:]
        Mvir = M_PART * hcat['N_total']
        pos = np.array([hcat['x'], hcat['y'], hcat['z']]).T % box_length
        vel = np.array([hcat['vx'], hcat['vy'], hcat['vz']]).T
        hid = hcat['hid_cleaned']
        del hcat
        gc.collect()

        nb = Mvir >= M_NEIGHBOUR
        log(sim, f'{sim}: {nb.sum():,} tracers, n_jobs={n_jobs}')

        shutil.rmtree(sb_path, ignore_errors=True)
        os.makedirs(sb_path, exist_ok=True)
        chunksize = max(nb.sum() // 1000, 1)
        generate_sub_box_ids(pos[nb], box_length, subsize, chunksize,
                             sb_path, name='halo')
        split_simulation_into_sub_boxes(
            positions=pos[nb], velocities=vel[nb], ids=hid[nb], upids=hid[nb],
            boxsize=box_length, subsize=subsize, chunksize=chunksize,
            dtypes=[pos.dtype, vel.dtype], path=sb_path, name='halo')

        seed_mbin = np.full(len(Mvir), -1, dtype=np.int32)
        for k, (lo, hi) in enumerate(massbins):
            seed_mbin[(Mvir >= lo) & (Mvir <= hi)] = k
        counts = [int((seed_mbin == k).sum()) for k in range(len(massbins))]
        log(sim, f'{sim}: seeds per bin {counts}, total {sum(counts):,}')

        total, regions = get_pairwise_hist_jk(
            pos, vel, seed_mbin,
            r_max=r_max, boxsize=box_length, subsize=subsize, path=sb_path,
            n_M=len(massbins), R_edges=R_edges, rlos_edges=rlos_edges,
            v_lo=V_LO, v_hi=V_HI, n_v=N_V,
            conv=CONV, n_jobs=n_jobs, tasks_per_sub_box=1,
            progress=False,   # tqdm bars from several concurrent phases would
                               # interleave into garbage; rely on the per-sim
                               # log lines instead
        )
        log(sim, f'{sim}: search done, {len(regions)} regions, computing jackknife covariances')

        vr_jk_covs = []
        for k in range(len(massbins)):
            cov_k = vr_jackknife_covariance(regions, k, VR_R_EDGES, vr_profile)
            vr_jk_covs.append(cov_k)
        del regions   # NOT saved to disk -- see module docstring on storage
        gc.collect()

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        total.save(out_path)
        with h5.File(out_path, 'a') as hdf:
            hdf.attrs.update(sim=sim, redshift=REDSHIFT, boxsize=box_length,
                             subsize=subsize, r_max=r_max,
                             M_neighbour=M_NEIGHBOUR, conv=CONV, m_part=M_PART,
                             n_tracers=int(nb.sum()))
            hdf.create_dataset('massbins', data=massbins)
            hdf.create_dataset('seeds_per_bin', data=np.array(counts))
            hdf.create_dataset('vr_r_edges', data=VR_R_EDGES)
            for k, cov_k in enumerate(vr_jk_covs):
                hdf.create_dataset(f'vr_jk_cov_mbin{k}', data=cov_k)
        log(sim, f'{sim}: saved {out_path}')

        if not KEEP_SUBBOXES:
            shutil.rmtree(sb_path, ignore_errors=True)

        del pos, vel, hid, Mvir, nb, seed_mbin, total
        gc.collect()
        return sim, time.time() - t0, None

    except Exception:
        return sim, time.time() - t0, traceback.format_exc()


def main():
    if len(sys.argv) > 1:
        phases = [sys.argv[1]]
    else:
        phases = PHASES

    phases = [s for s in phases
              if not (RESUME and os.path.exists(OUT_TMPL.format(sim=s, z=REDSHIFT)))]
    if not phases:
        print('nothing to do -- all requested phases already have output')
        return 0

    print(f'running {len(phases)} phases, {N_CONCURRENT_PHASES} at a time, '
          f'{N_JOBS_PER_PHASE} jobs each ({N_CONCURRENT_PHASES * N_JOBS_PER_PHASE} '
          f'of {TOTAL_CORES} cores)')

    failed = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_CONCURRENT_PHASES) as pool:
        futures = {pool.submit(run_phase_jk, sim, N_JOBS_PER_PHASE): sim for sim in phases}
        for fut in as_completed(futures):
            sim, elapsed, error = fut.result()
            if error:
                failed.append(sim)
                log(sim, f'{sim}: FAILED after {elapsed/60:.1f} min\n{error}')
            else:
                log(sim, f'{sim}: done in {elapsed/60:.1f} min')

    log('batch', f'batch done in {(time.time() - t0)/3600:.1f} h; '
        f'{len(failed)} failed: {failed}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())