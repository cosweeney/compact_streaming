"""
Adapted from run_abacus_batch_jk.py for the Salcedo base_c000 z=0.3 halo
catalog (path: /spiffball/salcedo/AbacusSummit/Snapshots/base_c000/
base_{sim}/z0p300/halo_base_{sim}_z0p300.h5), which has a different schema
('mass', 'x','y','z','vx','vy','vz','rvir','cvir','vrms','gid' -- no
N_total/hid_cleaned) and different conventions:

  - 'mass' is ALREADY in M_sun/h (confirmed by user), so no M_PART *
    N_total conversion is needed or done here -- Mvir = hcat['mass']
    directly. This also means there is no equivalent of the
    2.1e9-vs-2e9 M_PART mismatch documented in the original
    compute_xihh_jk.py / run_abacus_batch_jk.py pair: both this script
    and its compute_xihh_salcedo_jk.py counterpart use the same 'mass'
    field, so mass-bin membership is guaranteed identical on both sides
    here.

  - 'vx','vy','vz' are ALREADY proper km/s (confirmed by user against
    hcat2['vx'] sample values), so CONV=1.0 -- do NOT apply the
    a*H(a)/h*box_length rescale the original script computes via
    colossus, which was only needed because that catalog's velocities
    were in a Mpc/h-type convention. Applying it here would inflate
    these values by ~1.8e5x.

  - 'gid' substitutes for 'hid_cleaned' as the sub-box id/upid field.

READ-ONLY on the salcedo path: only ever opens HALO_TMPL for reading.
All outputs (pairhist_allbins_jk.h5, logs) go to a NEW subtree under
cosweeney's own directories (.../AbacusSummit_base_salcedo/...) that
does not overlap with the original hengweichang-catalog output tree, so
neither this script nor its outputs ever touch salcedo's or the
original z=0.2 run's files. Every output directory is created with
os.makedirs(..., exist_ok=True) before any write, same as the original.

Only c000_ph000 is confirmed to exist at this path so far -- extend
PHASES once more phases are confirmed present under the same
base_c000/base_{sim}/z0p300/ pattern.

Everything else (concurrency defaults, /dev/shm staging, jackknife
region-preservation, not saving the 64 per-region PairHist pieces) is
unchanged from run_abacus_batch_jk.py -- see that file's docstring for
the reasoning, which still applies as-is.

I have not run this myself -- no access to your cluster or data -- so
treat this as a structurally-complete first draft to check over, same
as everything else in this pipeline.
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
PHASES = ['c000_ph000']   # only phase confirmed to exist at this path so far;
                          # extend once more are confirmed present

REDSHIFT_ON_DISK = 'z0p300'   # Salcedo's own on-disk naming convention
REDSHIFT_OUT = 'z0.300'       # matches this project's own "z0.200"-style
                              # convention for cosweeney output paths

# READ-ONLY -- never written to.
HALO_TMPL = ('/spiffball/salcedo/AbacusSummit/Snapshots/'
             'base_c000/base_{sim}/{zd}/halo_base_{sim}_{zd}.h5')

SB_TMPL = '/dev/shm/{sim}_SBp_jk_salcedo/'

# NEW subtree under cosweeney -- distinct from the original hengweichang-
# catalog output path, so this never collides with or overwrites it.
OUT_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base_salcedo/halos/'
            'AbacusSummit_base_{sim}/halos/{z}/data/pairhist_allbins_jk.h5')
LOG_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base_salcedo/logs/'
            'pairhist_jk_salcedo_{sim}.log')

box_length = 2000.0
subsize = box_length / 4
r_max = 140.0
M_NEIGHBOUR = 1e12

log_edges = np.array([[12, 12.01], [12.5, 12.6], [13, 13.1],
                      [13.5, 13.6], [14.0, 14.2], [14.4, 14.7]])
massbins = 10**log_edges

r_perp = np.linspace(5, 140, 135 + 1)
r_los = np.linspace(5, 140, 135 + 1)

CONV = 1.0   # velocities in this catalog are ALREADY proper km/s -- do NOT
             # apply the a*H(a)/h*box_length rescale the original script
             # uses; see module docstring.

R_edges = r_perp
rlos_edges = r_los
V_LO, V_HI, N_V = -4000.0, 4000.0, 400

# jackknife r-bins for <v_r|r> -- must match gather_paper_data.py's
# r_edges_shared, since vr_jackknife_covariance reuses gather_paper_data's
# own vr_profile() to reduce each region.
VR_R_EDGES = np.arange(2.0, 142.0, 2.0)

RESUME = True
KEEP_SUBBOXES = False

TOTAL_CORES = 112                # this machine's core count -- adjust if wrong

# Same reasoning/measurement as run_abacus_batch_jk.py -- see that file's
# docstring. Kept sequential and at 84 jobs/phase for the same reasons.
N_CONCURRENT_PHASES = 1
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
        halo_path = HALO_TMPL.format(sim=sim, zd=REDSHIFT_ON_DISK)
        sb_path = SB_TMPL.format(sim=sim)
        out_path = OUT_TMPL.format(sim=sim, z=REDSHIFT_OUT)

        if not os.path.exists(halo_path):
            raise FileNotFoundError(halo_path)

        log(sim, f'{sim}: reading catalogue')
        with h5.File(halo_path, 'r') as hdf:
            hcat = hdf['halos'][:]
        Mvir = hcat['mass']   # already M_sun/h -- no M_PART conversion needed
        pos = np.array([hcat['x'], hcat['y'], hcat['z']]).T % box_length
        vel = np.array([hcat['vx'], hcat['vy'], hcat['vz']]).T
        hid = hcat['gid']
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
            progress=False,
        )
        log(sim, f'{sim}: search done, {len(regions)} regions, computing jackknife covariances')

        vr_jk_covs = []
        for k in range(len(massbins)):
            cov_k = vr_jackknife_covariance(regions, k, VR_R_EDGES, vr_profile)
            vr_jk_covs.append(cov_k)
        del regions
        gc.collect()

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        total.save(out_path)
        with h5.File(out_path, 'a') as hdf:
            hdf.attrs.update(sim=sim, redshift=REDSHIFT_OUT, boxsize=box_length,
                             subsize=subsize, r_max=r_max,
                             M_neighbour=M_NEIGHBOUR, conv=CONV,
                             mass_convention='direct_mass_field',
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
              if not (RESUME and os.path.exists(OUT_TMPL.format(sim=s, z=REDSHIFT_OUT)))]
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