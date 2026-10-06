"""
Adapted from run_salcedo_batch_jk.py: measures v_r + its jackknife
covariance for ONE mass bin covering ALL halos above M = 1e12 M_sun/h
(no upper cut), rather than the narrow, DISJOINT LOG_MASS_EDGES slices
the original script uses. This is the principled "no mass split"
measurement -- every halo above the mass threshold contributes, not
just those falling in one of the original slices' narrow ranges.

Everything else (schema handling, CONV=1.0 for already-km/s velocities,
'mass' field used directly, 'gid' as the id field, read-only on the
salcedo path, output only under cosweeney's own directory, concurrency
defaults, /dev/shm staging, jackknife region-preservation) is unchanged
from run_salcedo_batch_jk.py -- see that file's docstring for the
reasoning, which still applies as-is.

Output goes to a NEW file (pairhist_allbins_jk_allmass.h5, distinct
suffix) so it never collides with or overwrites the per-mass-bin
pairhist_allbins_jk.h5 already produced by run_salcedo_batch_jk.py.

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
PHASES = ['c000_ph000']   # only phase needed for this run

REDSHIFT_ON_DISK = 'z0p300'
REDSHIFT_OUT = 'z0.300'

# READ-ONLY -- never written to.
HALO_TMPL = ('/spiffball/salcedo/AbacusSummit/Snapshots/'
             'base_c000/base_{sim}/{zd}/halo_base_{sim}_{zd}.h5')

SB_TMPL = '/dev/shm/{sim}_SBp_jk_salcedo_allmass/'

# NEW filename suffix -- does not collide with the per-mass-bin output.
OUT_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base_salcedo/halos/'
            'AbacusSummit_base_{sim}/halos/{z}/data/pairhist_allbins_jk_allmass.h5')
LOG_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base_salcedo/logs/'
            'pairhist_jk_salcedo_allmass_{sim}.log')

box_length = 2000.0
subsize = box_length / 4
r_max = 140.0

# Single mass bin: ALL halos with M >= MASS_LO, no upper cut -- NOT the
# narrow disjoint LOG_MASS_EDGES slices from run_salcedo_batch_jk.py.
MASS_LO = 1e12
massbins = np.array([[MASS_LO, np.inf]])   # shape (1,2) -- N_MASS=1 throughout

r_perp = np.linspace(5, 140, 135 + 1)
r_los = np.linspace(5, 140, 135 + 1)

CONV = 1.0   # velocities already proper km/s -- see run_salcedo_batch_jk.py

R_edges = r_perp
rlos_edges = r_los
V_LO, V_HI, N_V = -4000.0, 4000.0, 400

VR_R_EDGES = np.arange(2.0, 142.0, 2.0)

RESUME = True
KEEP_SUBBOXES = False

TOTAL_CORES = 112

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
    """One phase, region-preserving, single mass bin (M >= MASS_LO).
    Returns (sim, elapsed_seconds, error_or_None)."""
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
        Mvir = hcat['mass']   # already M_sun/h
        pos = np.array([hcat['x'], hcat['y'], hcat['z']]).T % box_length
        vel = np.array([hcat['vx'], hcat['vy'], hcat['vz']]).T
        hid = hcat['gid']
        del hcat
        gc.collect()

        nb = Mvir >= MASS_LO
        log(sim, f'{sim}: {nb.sum():,} tracers with M >= {MASS_LO:.2e}, n_jobs={n_jobs}')

        shutil.rmtree(sb_path, ignore_errors=True)
        os.makedirs(sb_path, exist_ok=True)
        chunksize = max(nb.sum() // 1000, 1)
        generate_sub_box_ids(pos[nb], box_length, subsize, chunksize,
                             sb_path, name='halo')
        split_simulation_into_sub_boxes(
            positions=pos[nb], velocities=vel[nb], ids=hid[nb], upids=hid[nb],
            boxsize=box_length, subsize=subsize, chunksize=chunksize,
            dtypes=[pos.dtype, vel.dtype], path=sb_path, name='halo')

        # single bin -- every tracer above MASS_LO gets seed_mbin=0
        seed_mbin = np.full(len(Mvir), -1, dtype=np.int32)
        seed_mbin[nb] = 0
        counts = [int(nb.sum())]
        log(sim, f'{sim}: seeds in bin 0: {counts[0]:,}')

        total, regions = get_pairwise_hist_jk(
            pos, vel, seed_mbin,
            r_max=r_max, boxsize=box_length, subsize=subsize, path=sb_path,
            n_M=1, R_edges=R_edges, rlos_edges=rlos_edges,
            v_lo=V_LO, v_hi=V_HI, n_v=N_V,
            conv=CONV, n_jobs=n_jobs, tasks_per_sub_box=1,
            progress=False,
        )
        log(sim, f'{sim}: search done, {len(regions)} regions, computing jackknife covariance')

        vr_jk_cov = vr_jackknife_covariance(regions, 0, VR_R_EDGES, vr_profile)
        del regions
        gc.collect()

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        total.save(out_path)
        with h5.File(out_path, 'a') as hdf:
            hdf.attrs.update(sim=sim, redshift=REDSHIFT_OUT, boxsize=box_length,
                             subsize=subsize, r_max=r_max,
                             mass_lo=MASS_LO, conv=CONV,
                             mass_convention='direct_mass_field_single_bin_no_upper_cut',
                             n_tracers=int(nb.sum()))
            hdf.create_dataset('massbins', data=massbins)
            hdf.create_dataset('seeds_per_bin', data=np.array(counts))
            hdf.create_dataset('vr_r_edges', data=VR_R_EDGES)
            hdf.create_dataset('vr_jk_cov_mbin0', data=vr_jk_cov)
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