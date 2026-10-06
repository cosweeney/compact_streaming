"""
Computes xi_hh(r) + its within-box jackknife covariance, per mass bin, for
whichever phases ALREADY have v_r auto-statistics data available (i.e. all
6 pairhist_auto_mbin{k}.h5 files exist for that phase, from
gather_auto_stats.py) -- so the two datasets can be paired up phase-by-phase
for a joint fit.

UPDATED from the earlier version: that gated on pairhist_allbins_jk.h5,
the OLD cross-statistic v_r pipeline's output (one file per phase). The
current v_r pipeline (gather_auto_stats.py) produces auto-statistics
instead, one file PER MASS BIN (pairhist_auto_mbin{k}.h5) -- this now
checks for all 6 of those per phase instead.

Also updated: M_PART is now 2e9, matching gather_auto_stats.py's own
convention exactly, rather than the earlier 2.1e9 "corrected" value used
elsewhere in this project. Previously this was a deliberate, flagged
mismatch (xi_hh and the OLD v_r pipeline used different M_PART values,
~5% apart, so they weren't selecting quite the same halos for "the same"
bin) -- now that both sides are genuinely auto-statistics meant to be
fit jointly, they should select identical halos for a given bin, so this
mismatch is removed rather than carried forward.

Needs pycorr -- run this in that environment, never the fitting one. This
is the ONLY script in this pipeline that needs pycorr at all; the
fitting/plotting scripts never import it.

Entirely self-contained: defines its own copies of every constant/
convention needed (mass bins, particle mass, box size, halo catalog path)
rather than importing gather_paper_data.py, so this has no dependency on
that module.

I have not run this myself -- no cluster access -- so treat this as a
structurally-complete first draft to check over, same as everything else
in this pipeline.
"""

import os
import numpy as np
import h5py as h5
from pycorr import TwoPointCorrelationFunction

FID_HALO_TMPL = ('/spiff/hengweichang/AbacusSummit_base/snapshots/'
                 'AbacusSummit_base_{sim}/halos/{z}/halo_combined.h5')
# matches gather_auto_stats.py's OUT_TMPL exactly -- one file per (phase, bin)
VR_AUTO_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base/halos/'
                'AbacusSummit_base_{sim}/halos/{z}/data/pairhist_auto_mbin{k}.h5')
XI_JK_OUT_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base/halos/'
                  'AbacusSummit_base_{sim}/halos/{z}/data/xi_hh_jk.npz')

FID_REDSHIFT = 'z0.200'
BOX_LENGTH = 2000.0
M_PART = 2e9   # matches gather_auto_stats.py's convention exactly -- see
               # module docstring; NOT the 2.1e9 value used elsewhere

LOG_MASS_EDGES = np.array([[12, 12.01], [12.5, 12.6], [13, 13.1],
                            [13.5, 13.6], [13.9, 14.3]])
MASSBINS = 10 ** LOG_MASS_EDGES
N_MASS = len(MASSBINS)
N_JK_1D = 4   # matches the 4x4x4=64 sub-box grid already used for v_r's own
              # jackknife, for direct comparability between the two

ALL_PLANNED_PHASES = [f'c000_ph{i:03d}' for i in range(0, 25)]
R_EDGES = np.arange(2.0, 142.0, 2.0)
NTHREADS = os.cpu_count()


def v_r_auto_ready(sim):
    """True only if ALL 6 mass bins' auto-statistics are present for this
    phase -- the strict interpretation of "the same phases as the velocity
    data", so a joint fit later never silently mixes a fully-measured xi_hh
    against a partially-measured v_r for the same nominal phase."""
    return all(os.path.exists(VR_AUTO_TMPL.format(sim=sim, z=FID_REDSHIFT, k=k))
               for k in range(N_MASS))


def compute_one_phase(sim, r_edges=R_EDGES, n_jk_1d=N_JK_1D, nthreads=NTHREADS):
    halo_path = FID_HALO_TMPL.format(sim=sim, z=FID_REDSHIFT)
    with h5.File(halo_path, 'r') as hdf:
        hcat = hdf['halos'][:]
        Mvir = M_PART * hcat['N_total']
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
    n_done, n_skipped_not_ready = 0, 0
    for sim in ALL_PLANNED_PHASES:
        xi_path = XI_JK_OUT_TMPL.format(sim=sim, z=FID_REDSHIFT)

        if not v_r_auto_ready(sim):
            n_skipped_not_ready += 1
            continue   # v_r auto-stats aren't fully ready for this phase yet
        if os.path.exists(xi_path):
            print(f'{sim}: xi_hh jk already computed, skipping')
            n_done += 1
            continue

        print(f'{sim}: computing xi_hh + jackknife covariance...')
        xi_per_bin, cov_jk_per_bin, n_halos_per_bin = compute_one_phase(sim)

        os.makedirs(os.path.dirname(xi_path), exist_ok=True)
        np.savez(xi_path, r_cens=r_cens, xi_per_bin=xi_per_bin,
                 cov_jk_per_bin=cov_jk_per_bin, n_halos_per_bin=n_halos_per_bin,
                 n_jk_1d=N_JK_1D, m_part=M_PART)
        print(f'{sim}: saved {xi_path}')
        n_done += 1

    print(f'done: {n_done} phase(s) have xi_hh jk data; '
          f'{n_skipped_not_ready} skipped (v_r auto-stats not fully ready yet)')


if __name__ == '__main__':
    main()