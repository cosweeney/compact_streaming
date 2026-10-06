"""
Adapted from compute_xihh_jk.py for the Salcedo base_c000 z=0.3 halo
catalog. Needs pycorr -- run in that environment, never the fitting one,
same as the original.

Unlike the original, there is no M_PART-mismatch caveat here: this
catalog provides 'mass' directly (confirmed M_sun/h), and
run_salcedo_batch_jk.py's v_r side uses that exact same field with no
separate particle-mass constant -- so both observables' mass-bin
membership is guaranteed identical, not just approximately so.

Unlike the original compute_xihh_jk.py, this does NOT wait for a
phase's v_r jackknife output to exist first -- it computes xi_hh for
every phase in ALL_PLANNED_PHASES unconditionally (only skipping a
phase if its OWN xi_hh output already exists, to stay resumable). xi_hh
measurement has no dependency on the v_r pipeline, so there's no reason
to gate it on that here.

FID_HALO_TMPL/REDSHIFT_ON_DISK read from salcedo's path (read-only).
XI_JK_OUT_TMPL points at the same new cosweeney subtree
run_salcedo_batch_jk.py writes its own (separate) v_r output to
(.../AbacusSummit_base_salcedo/...) -- kept as the same subtree purely
for consistency of layout, not because this script depends on v_r's
output.

ALL_PLANNED_PHASES covers all 5 available phases (ph000-ph004).

I have not run this myself -- no cluster access -- so treat this as a
structurally-complete first draft to check over, same as everything else
in this pipeline.
"""

import os
import numpy as np
import h5py as h5
from pycorr import TwoPointCorrelationFunction

REDSHIFT_ON_DISK = 'z0p300'   # Salcedo's own on-disk naming convention
REDSHIFT_OUT = 'z0.300'       # matches run_salcedo_batch_jk.py's REDSHIFT_OUT

# READ-ONLY -- never written to.
FID_HALO_TMPL = ('/spiffball/salcedo/AbacusSummit/Snapshots/'
                 'base_c000/base_{sim}/{zd}/halo_base_{sim}_{zd}.h5')

XI_JK_OUT_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base_salcedo/halos/'
                  'AbacusSummit_base_{sim}/halos/{z}/data/xi_hh_jk.npz')

BOX_LENGTH = 2000.0

LOG_MASS_EDGES = np.array([[12, 12.01], [12.5, 12.6], [13, 13.1],
                            [13.5, 13.6], [14.0, 14.2], [14.4, 14.7]])
MASSBINS = 10 ** LOG_MASS_EDGES
N_MASS = len(MASSBINS)
N_JK_1D = 4   # matches the 4x4x4=64 sub-box grid used for v_r's own jackknife

ALL_PLANNED_PHASES = [f'c000_ph{i:03d}' for i in range(5)]   # ph000-ph004, all
                                                               # 5 confirmed
                                                               # available

R_EDGES = np.arange(2.0, 142.0, 2.0)
NTHREADS = os.cpu_count()


def compute_one_phase(sim, r_edges=R_EDGES, n_jk_1d=N_JK_1D, nthreads=NTHREADS):
    halo_path = FID_HALO_TMPL.format(sim=sim, zd=REDSHIFT_ON_DISK)
    with h5.File(halo_path, 'r') as hdf:
        hcat = hdf['halos'][:]
        Mvir = hcat['mass']   # already M_sun/h -- no conversion needed
        pos = np.array([hcat['x'], hcat['y'], hcat['z']]).T % BOX_LENGTH

    n_r = len(r_edges) - 1
    xi_per_bin = np.zeros((N_MASS, n_r))
    cov_jk_per_bin = np.zeros((N_MASS, n_r, n_r))
    n_halos_per_bin = np.zeros(N_MASS, dtype=int)

    for k in range(N_MASS):
        lo, hi = MASSBINS[k]
        sel = (Mvir >= lo) & (Mvir <= hi)
        pos_bin = pos[sel]

        grid_idx = np.clip((pos_bin / BOX_LENGTH * n_jk_1d).astype(int), 0, n_jk_1d - 1)
        samples = grid_idx[:, 0] + n_jk_1d * grid_idx[:, 1] + (n_jk_1d ** 2) * grid_idx[:, 2]

        result = TwoPointCorrelationFunction(
            mode='s', edges=r_edges, data_positions1=pos_bin, data_samples1=samples,
            engine='corrfunc', boxsize=BOX_LENGTH, position_type='pos',
            nthreads=nthreads, compute_sepsavg=False,
        )
        xi_per_bin[k] = result.corr
        cov_jk_per_bin[k] = result.cov()
        n_halos_per_bin[k] = int(sel.sum())

    return xi_per_bin, cov_jk_per_bin, n_halos_per_bin


def main():
    r_cens = 0.5 * (R_EDGES[1:] + R_EDGES[:-1])
    n_done = 0
    for sim in ALL_PLANNED_PHASES:
        xi_path = XI_JK_OUT_TMPL.format(sim=sim, z=REDSHIFT_OUT)

        if os.path.exists(xi_path):
            print(f'{sim}: xi_hh jk already computed, skipping')
            n_done += 1
            continue

        print(f'{sim}: computing xi_hh + jackknife covariance...')
        xi_per_bin, cov_jk_per_bin, n_halos_per_bin = compute_one_phase(sim)

        os.makedirs(os.path.dirname(xi_path), exist_ok=True)
        np.savez(xi_path, r_cens=r_cens, xi_per_bin=xi_per_bin,
                 cov_jk_per_bin=cov_jk_per_bin, n_halos_per_bin=n_halos_per_bin,
                 n_jk_1d=N_JK_1D)
        print(f'{sim}: saved {xi_path}')
        n_done += 1

    print(f'done: {n_done}/{len(ALL_PLANNED_PHASES)} phase(s) have xi_hh jk data')


if __name__ == '__main__':
    main()