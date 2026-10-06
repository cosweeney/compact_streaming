"""
Compares <v_r|r> for logM in [14.0, 14.2] (mass bin 4) from c000_ph000,
computed two ways:
  1. The CURRENT pipeline: reads the already-computed PairHist output
     (pairhist_allbins_jk.h5 if available, else the original
     pairhist_allbins.h5) and reduces it via vr_profile -- no recomputation,
     since that raw accumulation is expensive (~95-190 min).
  2. compact.velocities.pw_moments.pairwise_velocity_moments (the compiled
     pairvel/Cython implementation) -- computed directly from the halo
     catalog, TWICE: once on raw catalog velocities, once with the same
     `conv` factor the current pipeline applies internally. This directly
     tests whether a missing velocity-unit conversion explains the
     magnitude discrepancy you're seeing -- pairwise_velocity_moments has
     no conv parameter at all, unlike the current pipeline's _Buffer.add().

Checked directly against the actual source (not assumed) before writing
this: the two implementations' sign/radial-vector conventions match (both
project the velocity difference onto the pair-separation vector, with
consistent i-j ordering on both sides) -- so that's NOT a suspected source
of any discrepancy. A separate, smaller possible issue (not confirmed) is
noted in the printed diagnostics below: pairvel.pyx's core loop doesn't
appear to apply a minimum-image correction to the raw position difference,
only to the cell-list neighbor search itself -- if right, this would drop
a small fraction of true near-boundary pairs rather than mis-bin them, a
minor completeness effect rather than a large systematic bias.

I have not run this myself -- no cluster access -- so treat this as a
structurally-complete first draft to check over, same as everything else
in this pipeline.
"""

import os
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology

from compact.velocities.pairhist import PairHist
from compact.velocities.pw_moments import pairwise_velocity_moments

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

# plotting config
import matplotlib
matplotlib.rcParams.update({'text.usetex': True,
                            'font.family': 'Computer Modern Roman'})
plt.rcParams['axes.labelsize'] = 20

import os
import warnings
warnings.filterwarnings('ignore')

import psutil
psutil.Process().nice(10)


SIM = 'c000_ph000'
FID_REDSHIFT = 'z0.200'
BOX_LENGTH = 2000.0
M_PART = 2e9   # matches run_abacus_batch.py / run_abacus_batch_jk.py's OWN
               # convention for the v_r side specifically -- NOT the 2.1e9
               # correction used on the xi_hh side elsewhere in this
               # project. Matching this deliberately, so both methods here
               # select the SAME halos for the SAME nominal mass bin.
K_BIN = 4      # logM in [14.0, 14.2]
LOG_MASS_EDGES = np.array([[12, 12.01], [12.5, 12.6], [13, 13.1],
                            [13.5, 13.6], [14.0, 14.2], [14.4, 14.7]])
MASSBINS = 10 ** LOG_MASS_EDGES

HALO_TMPL = ('/spiff/hengweichang/AbacusSummit_base/snapshots/'
             'AbacusSummit_base_{sim}/halos/{z}/halo_combined.h5')
VR_JK_PATH = ('/spiffball/cosweeney/simulations/AbacusSummit_base/halos/'
              f'AbacusSummit_base_{SIM}/halos/{FID_REDSHIFT}/data/pairhist_allbins_jk.h5')
VR_ORIG_PATH = ('/spiffball/cosweeney/simulations/AbacusSummit_base/halos/'
                f'AbacusSummit_base_{SIM}/halos/{FID_REDSHIFT}/data/pairhist_allbins.h5')

R_EDGES = np.arange(2.0, 142.0, 2.0)

# same conv formula the current pipeline uses (run_abacus_batch.py /
# run_abacus_batch_jk.py), reproduced here rather than imported, since
# this script is meant to be self-contained for this specific check
cosmo = cosmology.setCosmology('planck18')
redshift = 0.2
a_scale = 1 / (1 + redshift)
CONV = a_scale * cosmo.Hz(redshift) / cosmo.h * BOX_LENGTH


def main():
    # ---------------- 1. current pipeline: read existing output, no recompute ----------------
    if os.path.exists(VR_JK_PATH):
        vr_path = VR_JK_PATH
    elif os.path.exists(VR_ORIG_PATH):
        vr_path = VR_ORIG_PATH
    else:
        raise FileNotFoundError(
            f'no existing PairHist output found at either\n  {VR_JK_PATH}\nor\n  {VR_ORIG_PATH}\n'
            f'-- fix the path(s) above to wherever c000_ph000 actually landed.')
    print(f'current pipeline: reading {vr_path}')
    total = PairHist.load(vr_path)
    r_current, vr_current, sd_current, npairs_current = vr_profile(total, K_BIN, r_edges=R_EDGES)
    good_current = np.isfinite(vr_current) & (npairs_current > 1000)
    r_current, vr_current = r_current[good_current], vr_current[good_current]

    # ---------------- 2. pw_moments: recompute directly from the halo catalog ----------------
    halo_path = HALO_TMPL.format(sim=SIM, z=FID_REDSHIFT)
    print(f'pw_moments: reading catalogue from {halo_path}')
    with h5.File(halo_path, 'r') as hdf:
        hcat = hdf['halos'][:]
    Mvir = M_PART * hcat['N_total']
    pos = np.array([hcat['x'], hcat['y'], hcat['z']]).T % BOX_LENGTH
    vel_raw = np.array([hcat['vx'], hcat['vy'], hcat['vz']]).T

    lo, hi = MASSBINS[K_BIN]
    sel = (Mvir >= lo) & (Mvir <= hi)
    pos_bin = np.ascontiguousarray(pos[sel], dtype=np.float64)
    vel_bin_raw = np.ascontiguousarray(vel_raw[sel], dtype=np.float64)
    print(f'pw_moments: {sel.sum():,} halos in bin {K_BIN} '
          f'(logM in [{np.log10(lo):.2f},{np.log10(hi):.2f}])')

    print('pw_moments: computing on RAW (unconverted) velocities...')
    result_raw = pairwise_velocity_moments(pos_bin, vel_bin_raw, R_EDGES, BOX_LENGTH)
    r_pw = 0.5 * (R_EDGES[1:] + R_EDGES[:-1])
    vr_pw_raw = result_raw['m10']
    n_pw = result_raw['counts']

    print('pw_moments: computing WITH the same conv factor the current pipeline applies...')
    vel_bin_converted = vel_bin_raw * CONV
    result_conv = pairwise_velocity_moments(pos_bin, vel_bin_converted, R_EDGES, BOX_LENGTH)
    vr_pw_conv = result_conv['m10']

    good_pw = (n_pw > 1000) & np.isfinite(vr_pw_raw)

    # ---------------- diagnostics ----------------
    print(f'\nCONV factor used: {CONV:.6g}')
    print('(if the raw-velocity pw_moments curve is off from the current pipeline by '
          'roughly this factor, that directly confirms the missing-conversion hypothesis)')

    # compare on the OVERLAPPING, good region of both r-grids
    common_r = np.intersect1d(np.round(r_current, 6), np.round(r_pw[good_pw], 6))
    if len(common_r) > 3:
        idx_cur = np.isin(np.round(r_current, 6), common_r)
        idx_pw = np.isin(np.round(r_pw, 6), common_r) & good_pw
        ratio_raw = np.nanmedian(vr_pw_raw[idx_pw] / vr_current[idx_cur])
        ratio_conv = np.nanmedian(vr_pw_conv[idx_pw] / vr_current[idx_cur])
        print(f'median (pw_moments RAW / current pipeline)       = {ratio_raw:.4g}  '
              f'(compare against 1/CONV = {1/CONV:.4g})')
        print(f'median (pw_moments CONVERTED / current pipeline) = {ratio_conv:.4g}  '
              f'(expect close to 1 if everything else matches)')
        if abs(ratio_conv - 1.0) > 0.1:
            print('FLAG: even after applying the same conv factor, the converted pw_moments '
                  'curve differs from the current pipeline by >10% -- something beyond the '
                  'unit conversion is different (binning, mass selection, PBC handling, or a '
                  'real bug in one implementation). Worth a closer look before trusting either.')
    else:
        print('not enough overlapping r-bins between the two grids to compute a direct ratio '
              '-- check R_EDGES / vr_profile\'s own r_eff convention if this happens.')

    print('\nPossible minimum-image/PBC handling gap in pairvel.pyx (see module docstring) -- '
          'if you see the pw_moments curves behave oddly specifically near large r (approaching '
          'R_EDGES[-1]), that would be consistent with this, though it should be a small, '
          'edge-localized effect, not a bulk magnitude shift.')

    # ---------------- plot ----------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    ax[0].plot(r_current, vr_current, 'o-', ms=4, c='C0', label='current pipeline (PairHist)')
    ax[0].plot(r_pw[good_pw], vr_pw_raw[good_pw], 's-', ms=4, c='C1', label='pw_moments, RAW velocities')
    ax[0].plot(r_pw[good_pw], vr_pw_conv[good_pw], '^-', ms=4, c='C2', label='pw_moments, conv-corrected')
    ax[0].axhline(0, c='gray', lw=0.7)
    ax[0].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
    ax[0].set_ylabel(r'$\langle v_r|r\rangle$ [km/s or raw units, see legend]')
    ax[0].legend(fontsize=8)
    ax[0].set_title(f'{SIM}, bin {K_BIN}: raw comparison')

    ax[1].plot(r_current, vr_current, 'o-', ms=4, c='C0', label='current pipeline (PairHist)')
    ax[1].plot(r_pw[good_pw], vr_pw_conv[good_pw], '^-', ms=4, c='C2', label='pw_moments, conv-corrected')
    ax[1].axhline(0, c='gray', lw=0.7)
    ax[1].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
    ax[1].set_ylabel(r'$\langle v_r|r\rangle$ [km/s]')
    ax[1].legend(fontsize=8)
    ax[1].set_title('same units (both converted) -- should overlay if consistent')

    fig.suptitle(f'v_r(r) cross-check: current pipeline vs. pw_moments, {SIM}, logM in [14.0,14.2]')
    fig.tight_layout()
    out_path = f'vr_crosscheck_{SIM}_bin{K_BIN}.png'
    fig.savefig(out_path, dpi=150)
    print(f'\nsaved {out_path}')


if __name__ == '__main__':
    main()