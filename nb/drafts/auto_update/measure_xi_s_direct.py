"""
Direct redshift-space correlation function of one mass bin, per phase -- RUN IN THE pycorr ENVIRONMENT (needs pycorr, h5py, numpy
only; no compact). Same pattern as compute_xihh_jk.py. Not run by me (no cluster access): treat as a first draft to check.

For every phase with a halo catalogue, for mass bin K_XIS (same selection as gather_auto_halo_stats.py: M = M_PART * N_total):
  xi_rppi  xi^s(s_perp, s_par) on 1 h^-1 Mpc bins, 0-130 in both, LOS = z, halos moved to redshift space with z + vz * BOX
           (v_kms/(aH/h) = vz * BOX, the pairhist convention); pi < 0 and pi > 0 are folded (equal bin volumes -> average of xi)
  xi_r     real-space xi_hh(r) on 1 h^-1 Mpc bins, r = 0.5-210 (bins with no pairs give xi = -1)
Periodic box, no randoms: pycorr uses the analytic RR (natural estimator), as in compute_xihh_jk.py.
One npz per phase, resumable; build_paper_cache.py (section xi_s, in the compact environment) reads these files.

    python measure_xi_s_direct.py
"""
import os
import time

import numpy as np
import h5py as h5
from pycorr import TwoPointCorrelationFunction

HALO_TMPL = ('/spiff/hengweichang/AbacusSummit_base/snapshots/'
             'AbacusSummit_base_{sim}/halos/z0.200/halo_combined.h5')
OUT_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base/halos/'
            'AbacusSummit_base_{sim}/halos/z0.200/data/xi_s_direct_mbin{k}.npz')    # read by build_paper_cache.py

PHASES = [f'c000_ph{i:03d}' for i in range(0, 25)]
BOX, M_PART = 2000.0, 2e9
LOG_MASS_EDGES = np.array([[12, 12.01], [12.5, 12.6], [13, 13.1], [13.5, 13.6], [13.9, 14.3]])
K_XIS = 2                                             # keep equal to K_XIS in build_paper_cache.py
S_MAX = 130
RP_EDGES = np.arange(0.0, S_MAX + 1.0)                # s_perp
PI_EDGES = np.arange(-S_MAX, S_MAX + 1.0)             # signed s_par, folded below
R_EDGES = np.arange(0.5, 211.0, 1.0)
NTHREADS = os.cpu_count()


def measure(sim):
    with h5.File(HALO_TMPL.format(sim=sim), 'r') as hdf:
        hc = hdf['halos'].fields(['N_total', 'x', 'y', 'z', 'vz'])[:]
    lo, hi = 10 ** LOG_MASS_EDGES[K_XIS]
    sel = (M_PART * hc['N_total'] >= lo) & (M_PART * hc['N_total'] <= hi)
    pos = np.array([hc['x'][sel], hc['y'][sel], hc['z'][sel]]).T % BOX
    pos_s = pos.copy()
    pos_s[:, 2] = (pos[:, 2] + hc['vz'][sel] * BOX) % BOX

    res_s = TwoPointCorrelationFunction('rppi', edges=(RP_EDGES, PI_EDGES), data_positions1=pos_s, engine='corrfunc',
                                        boxsize=BOX, position_type='pos', los='z', nthreads=NTHREADS, compute_sepsavg=False)
    corr = np.asarray(res_s.corr)                     # (n_rp, 2 * S_MAX)
    xi_rppi = 0.5 * (corr[:, S_MAX:] + corr[:, :S_MAX][:, ::-1])

    res_r = TwoPointCorrelationFunction('s', edges=R_EDGES, data_positions1=pos, engine='corrfunc', boxsize=BOX,
                                        position_type='pos', nthreads=NTHREADS, compute_sepsavg=False)
    return dict(xi_rppi=xi_rppi, xi_r=np.asarray(res_r.corr), r_cens=0.5 * (R_EDGES[1:] + R_EDGES[:-1]),
                s_edges=RP_EDGES, n_halos=len(pos), mass_bin=K_XIS)


def main():
    for sim in PHASES:
        out = OUT_TMPL.format(sim=sim, k=K_XIS)
        if os.path.exists(out):
            print(f'{sim}: already done, skipping')
            continue
        if not os.path.exists(HALO_TMPL.format(sim=sim)):
            print(f'{sim}: no halo catalogue, skipping')
            continue
        t0 = time.time()
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez(out, **measure(sim))
        print(f'{sim}: saved {out} ({(time.time() - t0) / 60:.1f} min)')


if __name__ == '__main__':
    main()