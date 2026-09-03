"""PairHist for AbacusSummit_base c000_ph001 .. ph024 at z=0.200.

Two ways to run:
    python run_abacus_batch.py                 # all phases, serially
    python run_abacus_batch.py c000_ph007      # one phase (job-array element)

Under Slurm the phase is taken from SLURM_ARRAY_TASK_ID (1-based) if no
argument is given, and N_JOBS from SLURM_CPUS_PER_TASK.  Exits non-zero on
failure so the scheduler records it.

One output file per phase, all on an identical grid so they can be stacked or
differenced directly.

Design notes:
  * RESUMABLE -- a phase whose output already exists is skipped, so you can
    kill this and restart without losing work.
  * FAULT-TOLERANT -- a phase that raises is logged and the loop continues;
    the failures are listed again at the end.
  * Sub-box files are rebuilt per phase and deleted afterwards (~2 GB each).
    Set KEEP_SUBBOXES = True to keep them.
  * Run parameters (M_NEIGHBOUR, r_max, conv, grids) are written into each
    output file, because the accumulator alone does not record them and they
    define what the histogram means.
"""

import gc
import os
import shutil
import sys
import time
import traceback

import numpy as np
import h5py as h5
from colossus.cosmology import cosmology


from compact.velocities.pairhist import PairHist
from compact.velocities.pairhist_direct import get_pairwise_hist
from compact.catalog.accumulate import generate_sub_box_ids, split_simulation_into_sub_boxes

import psutil
psutil.Process().nice(19)

# ---------------------------------------------------------------- config ----
PHASES = [f'c000_ph{i:03d}' for i in range(2, 25)]
REDSHIFT = 'z0.200'

HALO_TMPL = ('/spiff/hengweichang/AbacusSummit_base/snapshots/'
             'AbacusSummit_base_{sim}/halos/{z}/halo_combined.h5')
SB_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base/halos/'
           'AbacusSummit_base_{sim}/halos/{z}/SBp/')
OUT_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base/halos/'
            'AbacusSummit_base_{sim}/halos/{z}/data/pairhist_allbins.h5')
LOG_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base/logs/'
            'pairhist_{sim}.log')

box_length = 2000.0
subsize = box_length / 4
r_max = 140.0
M_NEIGHBOUR = 1e12

log_edges = np.array([[12, 12.01], [12.5, 12.6], [13, 13.1],
                      [13.5, 13.6], [14.0, 14.2], [14.4, 14.7]])
massbins = 10**log_edges


r_perp = np.linspace(5, 140, 135+1)
r_los = np.linspace(5, 140, 135+1)

r_p = 0.5*(r_perp[1:]+r_perp[:-1])
r_l = 0.5*(r_los[1:]+r_los[:-1])

cosmo = cosmology.setCosmology('planck18')

redshift = 0.2
a = 1/(1+redshift)

conv = a*cosmo.Hz(redshift)/cosmo.h * box_length

box_length = 2000 #Mpc/h, AS box side length

m_p = 2e9 # h^-1 M_sun, particle mass

R_edges = r_perp              # noqa: F821  identical for every phase
rlos_edges = r_los            # noqa: F821
V_LO, V_HI, N_V = -4000.0, 4000.0, 400
CONV = conv                   # noqa: F821
M_PART = m_p                  # noqa: F821  c000 -> same for all phases

RESUME = True
KEEP_SUBBOXES = False
# One task must not claim more cores than the scheduler gave it.
N_JOBS = 84 #int(os.environ.get('SLURM_CPUS_PER_TASK', 112))
# tasks = 64 sub-boxes * this; want a multiple of N_JOBS
TASKS_PER_SUB_BOX = 1

_log_path = None          # set per phase, so array tasks never share a file


def log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    if _log_path:
        with open(_log_path, 'a') as f:
            f.write(line + '\n')


def run_phase(sim):
    global _log_path
    _log_path = LOG_TMPL.format(sim=sim)
    os.makedirs(os.path.dirname(_log_path), exist_ok=True)

    halo_path = HALO_TMPL.format(sim=sim, z=REDSHIFT)
    sb_path = SB_TMPL.format(sim=sim, z=REDSHIFT)
    out_path = OUT_TMPL.format(sim=sim, z=REDSHIFT)

    if not os.path.exists(halo_path):
        raise FileNotFoundError(halo_path)

    log(f'{sim}: reading catalogue')
    with h5.File(halo_path, 'r') as hdf:
        hcat = hdf['halos'][:]
    Mvir = M_PART * hcat['N_total']
    pos = np.array([hcat['x'], hcat['y'], hcat['z']]).T % box_length
    vel = np.array([hcat['vx'], hcat['vy'], hcat['vz']]).T
    hid = hcat['hid_cleaned']
    del hcat
    gc.collect()

    nb = Mvir >= M_NEIGHBOUR
    n_bar = nb.sum() / box_length**3
    log(f'{sim}: {nb.sum():,} tracers, n_bar {n_bar:.3e}, '
        f'{n_bar * np.pi * r_max**2 * 2 * r_max:,.0f} pairs/seed')

    shutil.rmtree(sb_path, ignore_errors=True)   # append mode -> start clean
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
    log(f'{sim}: seeds per bin {counts}, total {sum(counts):,}')

    h = get_pairwise_hist(
        pos, vel, seed_mbin,
        r_max=r_max, boxsize=box_length, subsize=subsize, path=sb_path,
        n_M=len(massbins), R_edges=R_edges, rlos_edges=rlos_edges,
        v_lo=V_LO, v_hi=V_HI, n_v=N_V,
        conv=CONV, n_jobs=N_JOBS, tasks_per_sub_box=TASKS_PER_SUB_BOX,
    )

    off = int((h.n - h.counts.sum(-1)).sum())
    log(f'{sim}: pairs per bin {h.n.sum(axis=(1, 2)).tolist()}, off-grid {off}')
    if off:
        log(f'{sim}: WARNING {off} pairs fell off the velocity grid')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    h.save(out_path)
    with h5.File(out_path, 'a') as hdf:       # what the accumulator forgets
        hdf.attrs.update(sim=sim, redshift=REDSHIFT, boxsize=box_length,
                         subsize=subsize, r_max=r_max,
                         M_neighbour=M_NEIGHBOUR, conv=CONV, m_part=M_PART,
                         n_tracers=int(nb.sum()))
        hdf.create_dataset('massbins', data=massbins)
        hdf.create_dataset('seeds_per_bin', data=np.array(counts))
    log(f'{sim}: saved {out_path}')

    if not KEEP_SUBBOXES:
        shutil.rmtree(sb_path, ignore_errors=True)

    del pos, vel, hid, Mvir, nb, seed_mbin, h
    gc.collect()


def main():
    if len(sys.argv) > 1:
        phases = [sys.argv[1]]
    elif 'SLURM_ARRAY_TASK_ID' in os.environ:
        phases = [PHASES[int(os.environ['SLURM_ARRAY_TASK_ID']) - 1]]
    else:
        phases = PHASES

    failed, t0 = [], time.time()
    for n, sim in enumerate(phases, 1):
        out_path = OUT_TMPL.format(sim=sim, z=REDSHIFT)
        if RESUME and os.path.exists(out_path):
            print(f'{sim}: exists, skipping ({n}/{len(phases)})', flush=True)
            continue

        t = time.time()
        try:
            run_phase(sim)
            log(f'{sim}: done in {(time.time() - t) / 60:.1f} min')
        except Exception:
            failed.append(sim)
            log(f'{sim}: FAILED\n{traceback.format_exc()}')

    if len(phases) > 1:
        log(f'batch done in {(time.time() - t0) / 3600:.1f} h; '
            f'{len(failed)} failed: {failed}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())