"""
Adapted from compute_xihh_salcedo_jk.py: computes xi_hh + jackknife
covariance for ALL halos with M >= 1e12 M_sun/h (no upper cut),
matching run_salcedo_batch_allmass_jk.py's selection, rather than the
narrow disjoint LOG_MASS_EDGES slices. Needs pycorr -- run in that
environment, never the fitting one.

Decoupled from v_r's availability, same as compute_xihh_salcedo_jk.py's
current form -- xi_hh measurement has no dependency on the v_r pipeline.

Output goes to a NEW file (xi_hh_jk_allmass.npz) so it never collides
with the per-mass-bin xi_hh_jk.npz.

I have not run this myself -- no cluster access -- so treat this as a
structurally-complete first draft to check over, same as everything else
in this pipeline.
"""

import os
import numpy as np
import h5py as h5
from pycorr import TwoPointCorrelationFunction

REDSHIFT_ON_DISK = 'z0p300'
REDSHIFT_OUT = 'z0.300'

# READ-ONLY -- never written to.
FID_HALO_TMPL = ('/spiffball/salcedo/AbacusSummit/Snapshots/'
                 'base_c000/base_{sim}/{zd}/halo_base_{sim}_{zd}.h5')

XI_JK_OUT_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base_salcedo/halos/'
                  'AbacusSummit_base_{sim}/halos/{z}/data/xi_hh_jk_allmass.npz')

BOX_LENGTH = 2000.0

# Same mass cut as run_salcedo_batch_allmass_jk.py -- must match, since
# these two scripts' outputs get paired up in a fitting script.
MASS_LO = 1e12

N_JK_1D = 4   # matches the 4x4x4=64 sub-box grid used for v_r's own jackknife

ALL_PLANNED_PHASES = [f'c000_ph{i:03d}' for i in range(5)]   # ph000-ph004

R_EDGES = np.arange(2.0, 142.0, 2.0)
NTHREADS = os.cpu_count()


def compute_one_phase(sim, r_edges=R_EDGES, n_jk_1d=N_JK_1D, nthreads=NTHREADS):
    halo_path = FID_HALO_TMPL.format(sim=sim, zd=REDSHIFT_ON_DISK)
    with h5.File(halo_path, 'r') as hdf:
        hcat = hdf['halos'][:]
        Mvir = hcat['mass']   # already M_sun/h
        pos = np.array([hcat['x'], hcat['y'], hcat['z']]).T % BOX_LENGTH

    sel = Mvir >= MASS_LO
    pos_bin = pos[sel]

    grid_idx = np.clip((pos_bin / BOX_LENGTH * n_jk_1d).astype(int), 0, n_jk_1d - 1)
    samples = grid_idx[:, 0] + n_jk_1d * grid_idx[:, 1] + (n_jk_1d ** 2) * grid_idx[:, 2]

    result = TwoPointCorrelationFunction(
        mode='s', edges=r_edges, data_positions1=pos_bin, data_samples1=samples,
        engine='corrfunc', boxsize=BOX_LENGTH, position_type='pos',
        nthreads=nthreads, compute_sepsavg=False,
    )
    xi = result.corr
    cov_jk = result.cov()
    n_halos = int(sel.sum())

    return xi, cov_jk, n_halos


def main():
    r_cens = 0.5 * (R_EDGES[1:] + R_EDGES[:-1])
    n_done = 0
    for sim in ALL_PLANNED_PHASES:
        xi_path = XI_JK_OUT_TMPL.format(sim=sim, z=REDSHIFT_OUT)

        if os.path.exists(xi_path):
            print(f'{sim}: xi_hh (all-mass) jk already computed, skipping')
            n_done += 1
            continue

        print(f'{sim}: computing xi_hh (all-mass) + jackknife covariance...')
        xi, cov_jk, n_halos = compute_one_phase(sim)

        os.makedirs(os.path.dirname(xi_path), exist_ok=True)
        np.savez(xi_path, r_cens=r_cens, xi=xi, cov_jk=cov_jk,
                 n_halos=n_halos, n_jk_1d=N_JK_1D, mass_lo=MASS_LO)
        print(f'{sim}: saved {xi_path}')
        n_done += 1

    print(f'done: {n_done}/{len(ALL_PLANNED_PHASES)} phase(s) have xi_hh (all-mass) jk data')


if __name__ == '__main__':
    main()