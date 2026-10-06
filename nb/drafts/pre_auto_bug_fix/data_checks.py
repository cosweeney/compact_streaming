"""
data_sanity_checks.py -- five data-side diagnostics for the mass-bin-3
xi_hh/v_r tension, aimed at distinguishing a DATA/measurement issue from
a model issue (the covariance treatment itself was ruled out as the
driver -- switching between diagonal and full covariance didn't change
the picture).

Five independently-runnable cells after Setup:
  1. Mass-bin index consistency: does bin index k mean the same mass
     range in the xi_hh npz files and the pairhist h5 file?
  2. Phase-to-phase scatter vs. jackknife-predicted scatter for xi_hh:
     does the jackknife covariance underestimate real sample variance
     for bin 3 specifically?
  3. good-mask attrition for v_r: does bin 3 lose a different fraction of
     radial bins to the npairs>1000/finite/r<110 cut than its neighbors?
  4. Smooth-trend check across all 6 mass bins (not just the 4 plotted):
     does xi_hh's amplitude at fixed r vary smoothly with mass bin, or is
     there a kink/discontinuity at bin 3?
  5. Independent bias sanity check: does bin 3's fitted linear bias
     deviate from a standard Tinker et al. (2010) bias(M) prediction more
     than its neighbors do?

Entirely self-contained, never imports pycorr, per this project's
convention. Run Setup first; the five checks can then run independently.
"""
#%% Setup -- run first

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

from compact.lpt import GetLPTInputs
from compact.velocities.pairhist import PairHist

FID_COSMO = [0.6736, 0.02237, 0.1200, 2.083e-9, 0.9649]   # AbacusSummit c000
FID_REDSHIFT_VAL = 0.3
R_MIN_FIT = 15.0
N_MASS = 6
LOG_MASS_EDGES = np.array([[12, 12.01], [12.5, 12.6], [13, 13.1],
                            [13.5, 13.6], [14.0, 14.2], [14.4, 14.7]])

SIM = 'c000_ph000'
XI_PHASES = [f'c000_ph{i:03d}' for i in range(5)]
REDSHIFT_OUT = 'z0.300'
VR_JK_PATH = ('/spiffball/cosweeney/simulations/AbacusSummit_base_salcedo/halos/'
              f'AbacusSummit_base_{SIM}/halos/{REDSHIFT_OUT}/data/pairhist_allbins_jk.h5')
XI_JK_PATH_TEMPLATE = ('/spiffball/cosweeney/simulations/AbacusSummit_base_salcedo/halos/'
                        'AbacusSummit_base_{sim}/halos/' + REDSHIFT_OUT +
                        '/data/xi_hh_jk.npz')
R_EDGES_SHARED = np.arange(2.0, 142.0, 2.0)


def vr_profile(h, k, r_edges=None):
    Rc = 0.5 * (h.R_edges[1:] + h.R_edges[:-1])
    lc = 0.5 * (h.rlos_edges[1:] + h.rlos_edges[:-1])
    r3 = np.sqrt(Rc[:, None] ** 2 + lc[None, :] ** 2).ravel()
    n = h.n[k].ravel().astype(float)
    s1 = h.sum_vr[k].ravel()
    s2 = h.sum_vr2[k].ravel()
    nb_bins = len(r_edges) - 1
    idx = np.searchsorted(r_edges, r3, side='right') - 1
    ok = (idx >= 0) & (idx < nb_bins) & (n > 0)

    def acc(w):
        return np.bincount(idx[ok], weights=w[ok], minlength=nb_bins)

    N = acc(n)
    r_eff = np.where(N > 0, acc(n * r3) / np.where(N > 0, N, 1), np.nan)
    mu = np.where(N > 0, acc(s1) / np.where(N > 0, N, 1), np.nan)
    return r_eff, mu, N.astype(np.int64)


#%% Check 1 -- mass-bin index consistency across data sources
# n_halos_per_bin should be strictly decreasing with mass-bin index, and
# massbins (from the pairhist h5) should exactly equal LOG_MASS_EDGES. A
# silent index mismatch between the xi_hh and v_r pipelines would produce
# exactly "one bin looks anomalously bad" without any real physics or
# covariance cause.

xi_data0 = np.load(XI_JK_PATH_TEMPLATE.format(sim=XI_PHASES[0]))
n_halos = xi_data0['n_halos_per_bin']
print('n_halos_per_bin (xi npz):', n_halos)
print('  strictly decreasing?   ', bool(np.all(np.diff(n_halos) < 0)))
print('n_jk_1d (xi npz):        ', xi_data0['n_jk_1d'])

with h5.File(VR_JK_PATH, 'r') as hdf:
    massbins = hdf['massbins'][()]
print('massbins (pairhist h5):\n', massbins)
print('LOG_MASS_EDGES:\n', LOG_MASS_EDGES)
print('massbins == LOG_MASS_EDGES?', bool(np.allclose(massbins, LOG_MASS_EDGES)))


#%% Check 2 -- phase-to-phase scatter vs. jackknife-predicted scatter for
# xi_hh, all 6 bins. If bin 3's TRUE phase-to-phase scatter sits well
# above what its own jackknife covariance predicts (while its neighbors
# don't), that's evidence the jackknife variance is underestimating real
# sample variance specifically for this bin -- unrelated to diagonal vs.
# full covariance, and unrelated to the model.

fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
for k, ax in zip(range(N_MASS), axes.flat):
    xi_list, cov_list, r_xi = [], [], None
    for sim in XI_PHASES:
        xi_data = np.load(XI_JK_PATH_TEMPLATE.format(sim=sim))
        if r_xi is None:
            r_xi = xi_data['r_cens']
        xi_list.append(xi_data['xi_per_bin'][k])
        cov_list.append(xi_data['cov_jk_per_bin'][k])
    xi_arr = np.array(xi_list)             # (5 phases, n_r)
    phase_std = xi_arr.std(axis=0, ddof=1)  # TRUE across-phase scatter (of a single phase)
    jk_std = np.sqrt(np.diag(np.mean(cov_list, axis=0)))  # jackknife-predicted scatter
    # jk_std above is for the 5-PHASE MEAN (already divided by n_phases in
    # the project's normal convention) -- multiply by sqrt(5) for a
    # single-phase-equivalent comparison against phase_std
    jk_std_single_phase = jk_std * np.sqrt(len(XI_PHASES))

    ratio = np.nanmedian(jk_std_single_phase[10:]) / np.nanmedian(phase_std[10:])
    lo, hi = LOG_MASS_EDGES[k]
    ax.plot(r_xi, phase_std, 'k-', label='true phase-to-phase std (single phase)')
    ax.plot(r_xi, jk_std_single_phase, 'r--', label='jackknife-predicted std (single phase)')
    ax.set_title(rf'bin {k}: $\log M\in[{lo:.2f},{hi:.2f}]$, ratio={ratio:.2f}', fontsize=9)
    ax.set_yscale('log')
    if k == 0:
        ax.legend(fontsize=7)
fig.suptitle('xi_hh: true phase-to-phase scatter vs. jackknife-predicted scatter, all bins\n'
             '(ratio >> 1 or << 1 relative to OTHER bins flags that bin\'s jackknife '
             'covariance specifically)')
fig.tight_layout()


#%% Check 3 -- good-mask attrition for v_r, all 6 bins: how many radial
# bins survive the (finite & npairs>1000 & var>0 & r<110) cut, and at
# which r do the survivors start/stop? A bin losing a very different
# fraction (or range) of points than its neighbors is fitting on a
# different effective r-range, which can look like "worse tension" for
# reasons that have nothing to do with the model.

total = PairHist.load(VR_JK_PATH)
with h5.File(VR_JK_PATH, 'r') as hdf:
    vr_jk_cov_per_bin = {k: hdf[f'vr_jk_cov_mbin{k}'][()] for k in range(N_MASS)}

print(f'{"bin":>4}  {"logM range":>14}  {"n_total":>8}  {"n_good":>7}  '
      f'{"frac_good":>10}  {"r_good_min":>11}  {"r_good_max":>11}')
for k in range(N_MASS):
    r_vr, vr, npairs = vr_profile(total, k, r_edges=R_EDGES_SHARED)
    cov_vr_full = vr_jk_cov_per_bin[k]
    var_vr_full = np.diag(cov_vr_full)
    good = np.isfinite(vr) & (npairs > 1000) & (var_vr_full > 0) & (r_vr < 110)
    lo, hi = LOG_MASS_EDGES[k]
    r_good = r_vr[good]
    rng = f'[{lo:.2f},{hi:.2f}]'
    print(f'{k:>4}  {rng:>14}  {len(r_vr):>8}  {good.sum():>7}  '
          f'{good.sum()/len(r_vr):>10.2%}  {r_good.min() if good.any() else np.nan:>11.2f}  '
          f'{r_good.max() if good.any() else np.nan:>11.2f}')


#%% Check 4 -- smooth-trend check: xi_hh amplitude at fixed r, across ALL
# 6 mass bins (not just the 4 usually plotted). A kink or discontinuity
# specifically at bin 3 (rather than a smooth increase with mass) would
# point at a bin-3-specific data issue; a smooth curve through bin 3
# argues against one.

R_CHECK = [50.0, 70.0, 90.0]
xi_per_bin_list, r_xi = [], None
for sim in XI_PHASES:
    xi_data = np.load(XI_JK_PATH_TEMPLATE.format(sim=sim))
    if r_xi is None:
        r_xi = xi_data['r_cens']
    xi_per_bin_list.append(xi_data['xi_per_bin'])
xi_per_bin_avg = np.mean(xi_per_bin_list, axis=0)   # (6 bins, n_r)

fig, ax = plt.subplots(figsize=(7, 5))
bin_centers = LOG_MASS_EDGES.mean(axis=1)
for r_target in R_CHECK:
    ir = np.argmin(np.abs(r_xi - r_target))
    vals = xi_per_bin_avg[:, ir]
    ax.plot(bin_centers, vals, 'o-', label=rf'$r\approx{r_xi[ir]:.0f}\,h^{{-1}}$Mpc')
ax.axvline(bin_centers[3], color='gray', ls=':', lw=1, label='bin 3')
ax.set_yscale('log')
ax.set_xlabel(r'mass-bin center, $\log_{10} M$')
ax.set_ylabel(r'$\xi_{hh}(r)$')
ax.legend(fontsize=8)
ax.set_title('xi_hh amplitude at fixed r vs. mass-bin index, all 6 bins')
fig.tight_layout()


#%% Check 5 -- independent bias sanity check: fitted linear bias (from
# xi_hh alone, this project's own diagonal/no-Hartlap method) vs. a
# standard Tinker et al. (2010) bias(M) prediction, for all 6 bins.
#
# CAVEATS, stated explicitly rather than assumed away:
#  - mean halo mass per bin is APPROXIMATED as the geometric mean of the
#    log-mass bin edges (10**mean(log10 edges)) -- the true pair/number-
#    weighted mean mass isn't saved anywhere queried so far in this
#    project's files, so this is a proxy, not the exact value.
#  - the halo mass DEFINITION Tinker's fit expects (SO, some overdensity
#    convention) is assumed to match CompaSO's own definition ('200m',
#    i.e. Delta=200 times the mean matter density, per CompaSO's
#    published spherical-overdensity criterion) -- this hasn't been
#    independently confirmed against this project's specific catalog
#    processing and is the single biggest assumption in this check.
#  - sigma8 is taken from CAMB's own z=0 output for this project's
#    FID_COSMO (via compact.lpt's CAMB setup, evaluated at z=0 specifically
#    -- NOT at FID_REDSHIFT_VAL, since colossus's setCosmology expects the
#    present-day normalization and applies its own growth factor when
#    queried at z=FID_REDSHIFT_VAL later; confirmed these differ by ~15%
#    for this cosmology, so this distinction matters).

from colossus.cosmology import cosmology as ccosmology
from colossus.lss import bias as cbias

lpt_cosmo_z0 = GetLPTInputs(redshift=0.0, cosmo=FID_COSMO)   # z=0 for sigma8 -- see caveat below
sigma8 = float(lpt_cosmo_z0.results.get_sigma8()[-1])
h, ombh2, omch2 = FID_COSMO[0], FID_COSMO[1], FID_COSMO[2]
Om0 = (ombh2 + omch2) / h**2
Ob0 = ombh2 / h**2
ccosmology.setCosmology('abacus_c000', params=dict(
    flat=True, H0=100*h, Om0=Om0, Ob0=Ob0, sigma8=sigma8, ns=FID_COSMO[4]))
print(f'colossus cosmology set: H0={100*h:.4f}, Om0={Om0:.5f}, Ob0={Ob0:.5f}, '
      f'sigma8(z=0)={sigma8:.5f}, ns={FID_COSMO[4]}')
# NOTE: sigma8 is deliberately taken at z=0 here, NOT FID_REDSHIFT_VAL --
# colossus's setCosmology expects the present-day normalization and
# applies its OWN growth factor when bias.haloBias(..., z=FID_REDSHIFT_VAL)
# is called below. Feeding it sigma8 already evaluated at z=0.3 would
# double-apply growth suppression (confirmed numerically while building
# this script: CAMB gives sigma8(z=0)=0.808 vs sigma8(z=0.3)=0.689 for
# this cosmology -- a ~15% difference, not negligible).

MDEF_ASSUMED = '200m'   # ASSUMED to match CompaSO -- see caveats above

lpt = GetLPTInputs(redshift=FID_REDSHIFT_VAL, cosmo=FID_COSMO)
lpt.get_pklin_gsm()


def model_xihh(r, b1, b2, bs=0.0):
    pars = np.array([b1, b2, bs, 0, 0, 0, 0, 0, 0])
    return lpt.xi_real(r, pars)


def fit_bias_xi_only(r_xi, xihh, var_xi, r_min=R_MIN_FIT):
    m = r_xi > r_min
    r_fit, xi_fit, var_fit = r_xi[m], xihh[m], var_xi[m]

    def loss(params):
        b1, b2 = params
        try:
            res = model_xihh(r_fit, b1, b2) - xi_fit
        except Exception:
            return np.inf
        c = np.sum(res**2 / var_fit)
        return c if np.isfinite(c) else np.inf

    result = differential_evolution(loss, [(-0.3, 5.0), (-10.0, 10.0)],
                                     tol=1e-6, seed=0, maxiter=150, popsize=15, polish=True)
    return result.x


print(f'\n{"bin":>4}  {"logM range":>14}  {"M_proxy [Msun/h]":>18}  '
      f'{"b1_Lagr":>9}  {"b_Eulerian":>11}  {"b_Tinker10":>11}  {"ratio":>7}')
for k in range(N_MASS):
    xi_list, cov_list, r_xi = [], [], None
    for sim in XI_PHASES:
        xi_data = np.load(XI_JK_PATH_TEMPLATE.format(sim=sim))
        if r_xi is None:
            r_xi = xi_data['r_cens']
        xi_list.append(xi_data['xi_per_bin'][k])
        cov_list.append(xi_data['cov_jk_per_bin'][k])
    xihh = np.mean(xi_list, axis=0)
    var_xi = np.diag(np.mean(cov_list, axis=0) / len(XI_PHASES))

    b1_lagr, b2 = fit_bias_xi_only(r_xi, xihh, var_xi)
    b_eulerian = 1.0 + b1_lagr

    lo, hi = LOG_MASS_EDGES[k]
    M_proxy = 10**np.mean([lo, hi])   # geometric mean of bin edges, in Msun/h
    b_tinker = cbias.haloBias(M_proxy, model='tinker10', z=FID_REDSHIFT_VAL, mdef=MDEF_ASSUMED)

    rng = f'[{lo:.2f},{hi:.2f}]'
    print(f'{k:>4}  {rng:>14}  {M_proxy:>18.3e}  {b1_lagr:>9.4f}  '
          f'{b_eulerian:>11.4f}  {b_tinker:>11.4f}  {b_eulerian/b_tinker:>7.3f}')

print('\nA ratio (b_eulerian/b_Tinker10) close to 1 across MOST bins, with bin 3 a clear '
      'outlier, is evidence for a bin-3-specific data issue. A ratio drifting smoothly '
      'and similarly across all bins is more consistent with this project\'s own bs=0 '
      '(no tidal bias) simplification, or the M_proxy/mdef approximations above, than '
      'with anything specific to bin 3.')