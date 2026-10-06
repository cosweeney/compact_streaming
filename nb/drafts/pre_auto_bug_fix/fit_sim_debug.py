"""
fit_sim_debug.py -- multi-mass-bin debugging companion to fit_sim_jk.py:
joint vs. individual xi_hh/v_r fits, DIAGONAL covariance only
correction (chi2 = sum((model-data)^2/var), var taken at face value,
unscaled -- isolates fit behavior from the small-n_jk Hartlap deflation
seen in this project's other diagnostics).

Four independently-runnable cells, each looping over ALL mass bins
(range(N_MASS-2), matching main()'s existing subset) and plotting every
bin together in one 4-panel figure, colored/offset per bin -- same style
as main() -- rather than one bin at a time:
  1. Joint (xi_hh + v_r) fit, all bins
  2. Individual fits (xi_hh and v_r each fit on their own), all bins
  3. xi_hh-only fit per bin, with the v_r PREDICTION from that fit
     overlaid dashed, all bins
  4. v_r-only fit per bin, with the xi_hh PREDICTION from that fit
     overlaid dashed, all bins

Entirely self-contained -- never imports pycorr, matches this project's
convention that pycorr is only ever used for MEASURING, never FITTING.
xi_hh averaged over all 5 phases, no "/2" applied; v_r stays single-phase
(SIM).
"""
#%% Setup -- run once before any plot cell below

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

from compact.lpt import GetLPTInputs
from compact.velocities.pairhist import PairHist

FID_COSMO = [0.6736, 0.02237, 0.1200, 2.083e-9, 0.9649]   # official AbacusSummit c000
FID_REDSHIFT_VAL = 0.3   # matches the Salcedo catalog's z0p300
R_MIN_FIT = 15.0
N_MASS = 6
LOG_MASS_EDGES = np.array([[12, 12.01], [12.5, 12.6], [13, 13.1],
                            [13.5, 13.6], [14.0, 14.2], [14.4, 14.7]])

MASS_BINS_TO_PLOT = range(N_MASS - 2)   # matches main()'s existing subset

SIM = 'c000_ph000'
XI_PHASES = [f'c000_ph{i:03d}' for i in range(5)]
REDSHIFT_OUT = 'z0.300'

VR_JK_PATH = ('/spiffball/cosweeney/simulations/AbacusSummit_base_salcedo/halos/'
              f'AbacusSummit_base_{SIM}/halos/{REDSHIFT_OUT}/data/pairhist_allbins_jk.h5')
XI_JK_PATH_TEMPLATE = ('/spiffball/cosweeney/simulations/AbacusSummit_base_salcedo/halos/'
                        'AbacusSummit_base_{sim}/halos/' + REDSHIFT_OUT +
                        '/data/xi_hh_jk.npz')
R_EDGES_SHARED = np.arange(2.0, 142.0, 2.0)

lpt = GetLPTInputs(redshift=FID_REDSHIFT_VAL, cosmo=FID_COSMO)
lpt.get_pklin_gsm()


def vr_profile(h, k, r_edges=None, r_max_trust=None):
    """Pair-weighted <v_r>(r) and sigma_vr(r) for mass bin k."""
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


def model_xihh(r, b1, b2, bs=0.0):
    pars = np.array([b1, b2, bs, 0, 0, 0, 0, 0, 0])
    return lpt.xi_real(r, pars)


def model_vr(r, b1, b2, bs=0.0):
    pars = np.array([b1, b2, bs, 0, 0, 0, 0, 0, 0])
    # unwindowed xi_real for normalization consistency -- see this
    # project's earlier debugging of velocileptors' pwv_mean bug
    lpt._ensure_cumulants(pars)
    v_int = np.interp(r, lpt.gsm.rint, lpt.gsm.veft)
    xi_consistent = lpt.gsm.compute_xi_real(r, *pars)
    return lpt.conv * lpt.f * v_int / (1 + xi_consistent)


def load_bin_data(k):
    """Phase-averaged xi_hh (+ diagonal variance, no /2) and single-phase
    v_r (+ diagonal variance) for mass bin k."""
    xi_list, cov_list, r_xi = [], [], None
    for sim in XI_PHASES:
        xi_data = np.load(XI_JK_PATH_TEMPLATE.format(sim=sim))
        if r_xi is None:
            r_xi = xi_data['r_cens']
        xi_list.append(xi_data['xi_per_bin'])
        cov_list.append(xi_data['cov_jk_per_bin'])
    n_phases = len(XI_PHASES)
    xihh = np.mean(xi_list, axis=0)[k]
    var_xi = np.diag(np.mean(cov_list, axis=0)[k] / n_phases)

    total = PairHist.load(VR_JK_PATH)
    with h5.File(VR_JK_PATH, 'r') as hdf:
        cov_vr_full = hdf[f'vr_jk_cov_mbin{k}'][()]
    r_vr, vr, sd_vr, npairs = vr_profile(total, k, r_edges=R_EDGES_SHARED)
    var_vr_full = np.diag(cov_vr_full)
    good = np.isfinite(vr) & (npairs > 1000) & (var_vr_full > 0) & (r_vr < 110)
    r_vr_k, vr_k, var_vr_k = r_vr[good], vr[good], var_vr_full[good]

    return r_xi, xihh, var_xi, r_vr_k, vr_k, var_vr_k, n_phases


def fit_bias_diag(mode, r_min=R_MIN_FIT, fit_bs=False,
                   r_xi=None, xihh=None, var_xi=None,
                   r_vr=None, vr=None, var_vr=None,
                   popsize=15, maxiter=150, tol=1e-6, seed=0):
    """Diagonal-covariance bias fit correction.
    chi2 = sum((model-data)^2/var), var unscaled. mode in ('joint','xi','vr')."""
    if mode in ('joint', 'xi'):
        m_xi = r_xi > r_min
        r_xi_fit, xihh_fit, var_xi_fit = r_xi[m_xi], xihh[m_xi], var_xi[m_xi]
    if mode in ('joint', 'vr'):
        m_vr = r_vr > r_min
        r_vr_fit, vr_fit, var_vr_fit = r_vr[m_vr], vr[m_vr], var_vr[m_vr]

    def loss(params):
        b1, b2 = params[0], params[1]
        bs = params[2] if fit_bs else 0.0
        c = 0.0
        if mode in ('joint', 'xi'):
            try:
                res = model_xihh(r_xi_fit, b1, b2, bs) - xihh_fit
            except Exception:
                return np.inf
            c += np.sum(res**2 / var_xi_fit)
        if mode in ('joint', 'vr'):
            try:
                res = model_vr(r_vr_fit, b1, b2, bs) - vr_fit
            except Exception:
                return np.inf
            c += np.sum(res**2 / var_vr_fit)
        return c if np.isfinite(c) else np.inf

    bounds = [(-0.3, 5.0), (-10.0, 10.0)] + ([(-5.0, 5.0)] if fit_bs else [])
    result = differential_evolution(loss, bounds, tol=tol, seed=seed,
                                     maxiter=maxiter, popsize=popsize, polish=True)

    n_data = 0
    if mode in ('joint', 'xi'):
        n_data += len(xihh_fit)
    if mode in ('joint', 'vr'):
        n_data += len(vr_fit)
    result.dof = max(n_data - len(bounds), 1)
    result.chi2 = result.fun
    result.reduced_chi2 = result.fun / result.dof
    return result


def plot_debug_4panel_multibin(title, per_bin, r_min=R_MIN_FIT, offset_step=0.5):
    """per_bin: list of dicts, one per mass bin, each with keys
      k, r_vr, vr, var_vr, vr_curves, r_xi, xihh, var_xi, xi_curves
    where *_curves is a list of (y, style, label) tuples (color is fixed
    to C{k} for that bin's data AND all its curves, matching main()'s
    per-bin-color convention). Ratio panels offset by k*offset_step, as
    in main()."""
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharex='col',
                            gridspec_kw={'height_ratios': [1.2, 1]})

    for bd in per_bin:
        k = bd['k']
        color = f'C{k}'
        r_vr, vr, var_vr = bd['r_vr'], bd['vr'], bd['var_vr']
        r_xi, xihh, var_xi = bd['r_xi'], bd['xihh'], bd['var_xi']
        vr_err, xi_err = np.sqrt(var_vr), np.sqrt(var_xi)
        lo, hi = LOG_MASS_EDGES[k]
        baseline = 1 + k * offset_step

        ax[0, 0].plot(r_vr, vr, 'o', ms=3, c=color, zorder=2,
                       label=rf'$\log M \in [{lo:.2f}, {hi:.2f}]$')
        ax[0, 0].fill_between(r_vr, vr - vr_err, vr + vr_err,
                               color=color, alpha=0.25, lw=0, zorder=1)
        for y, style, _ in bd['vr_curves']:
            ax[0, 0].plot(r_vr, y, style, c=color)

        xi_r2 = r_xi**2 * xihh
        xi_err_r2 = r_xi**2 * xi_err
        ax[0, 1].plot(r_xi, xi_r2, 'o', ms=3, c=color, zorder=2)
        ax[0, 1].fill_between(r_xi, xi_r2 - xi_err_r2, xi_r2 + xi_err_r2,
                               color=color, alpha=0.25, lw=0, zorder=1)
        for y, style, _ in bd['xi_curves']:
            ax[0, 1].plot(r_xi, r_xi**2 * y, style, c=color)

        ax[1, 0].axhline(baseline, c=color, lw=0.7, alpha=0.5, zorder=0)
        for y, style, _ in bd['vr_curves']:
            ratio = y / vr
            ratio_err = np.abs(ratio) * (vr_err / np.abs(vr))
            ax[1, 0].fill_between(r_vr, baseline - 2*ratio_err, baseline + 2*ratio_err,
                                   color=color, alpha=0.12, lw=0, zorder=0)
            ax[1, 0].fill_between(r_vr, baseline - ratio_err, baseline + ratio_err,
                                   color=color, alpha=0.22, lw=0, zorder=0)
            ax[1, 0].plot(r_vr, ratio + k * offset_step, style, c=color, zorder=2)

        xi_good = np.abs(xihh) > xi_err
        ax[1, 1].axhline(baseline, c=color, lw=0.7, alpha=0.5, zorder=0)
        for y, style, _ in bd['xi_curves']:
            ratio = y / xihh
            ratio_err = np.abs(ratio) * (xi_err / np.abs(xihh))
            ax[1, 1].fill_between(r_xi[xi_good], baseline - 2*ratio_err[xi_good],
                                   baseline + 2*ratio_err[xi_good],
                                   color=color, alpha=0.12, lw=0, zorder=0)
            ax[1, 1].fill_between(r_xi[xi_good], baseline - ratio_err[xi_good],
                                   baseline + ratio_err[xi_good],
                                   color=color, alpha=0.22, lw=0, zorder=0)
            ax[1, 1].plot(r_xi[xi_good], ratio[xi_good] + k * offset_step,
                           style, c=color, zorder=2)

    for ax_ in ax.flat:
        ax_.axvline(r_min, color='gray', ls=':', lw=1, zorder=0)
        ax_.margins(x=0)

    ax[0, 0].set_ylabel(r'$\langle v_r|r\rangle$ [km/s]')
    ax[0, 1].set_ylabel(r'$r^2\,\xi_{hh}(r|M)$ [$(h^{-1}\,{\rm Mpc})^2$]')
    ax[1, 0].set_ylabel('model / data (+ offset)')
    ax[1, 1].set_ylabel('model / data (+ offset)')
    ax[1, 0].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
    ax[1, 1].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
    ax[0, 0].legend(fontsize=8, ncol=2)
    ax[0, 1].set_yscale('log')
    ax[0, 1].set_ylim((1, 100))
    ax[1, 0].set_ylim((0.5, 3.0))
    ax[1, 1].set_ylim((0.5, 3.0))
    fig.suptitle(title)
    fig.tight_layout()
    return fig, ax


#%% Cell 1 -- joint (xi_hh + v_r) fit, diagonal covariance,
# looped over all mass bins

per_bin = []
for k in MASS_BINS_TO_PLOT:
    r_xi, xihh, var_xi, r_vr_k, vr_k, var_vr_k, n_phases = load_bin_data(k)
    res = fit_bias_diag(mode='joint', r_min=R_MIN_FIT,
                         r_xi=r_xi, xihh=xihh, var_xi=var_xi,
                         r_vr=r_vr_k, vr=vr_k, var_vr=var_vr_k)
    b1, b2 = res.x
    print(f'bin {k}: joint  b1={b1:.3f}, b2={b2:.3f}, '
          f'reduced_chi2={res.reduced_chi2:.3f} (dof={res.dof})')

    per_bin.append(dict(
        k=k, r_vr=r_vr_k, vr=vr_k, var_vr=var_vr_k,
        vr_curves=[(model_vr(r_vr_k, b1, b2), '-', 'joint fit')],
        r_xi=r_xi, xihh=xihh, var_xi=var_xi,
        xi_curves=[(model_xihh(r_xi, b1, b2), '-', 'joint fit')]))

plot_debug_4panel_multibin(
    f'Joint fit, all bins (diagonal cov, no intr scatter): {SIM}, '
    , per_bin)


#%% Cell 2 -- individual fits (xi_hh and v_r each fit on their own),
# looped over all mass bins

per_bin = []
for k in MASS_BINS_TO_PLOT:
    r_xi, xihh, var_xi, r_vr_k, vr_k, var_vr_k, n_phases = load_bin_data(k)
    res_xi = fit_bias_diag(mode='xi', r_min=R_MIN_FIT, r_xi=r_xi, xihh=xihh, var_xi=var_xi)
    res_vr = fit_bias_diag(mode='vr', r_min=R_MIN_FIT, r_vr=r_vr_k, vr=vr_k, var_vr=var_vr_k)
    b1_xi, b2_xi = res_xi.x
    b1_vr, b2_vr = res_vr.x
    print(f'bin {k}: xi-only  b1={b1_xi:.3f}, b2={b2_xi:.3f}, '
          f'reduced_chi2={res_xi.reduced_chi2:.3f} (dof={res_xi.dof})')
    print(f'bin {k}: vr-only  b1={b1_vr:.3f}, b2={b2_vr:.3f}, '
          f'reduced_chi2={res_vr.reduced_chi2:.3f} (dof={res_vr.dof})')

    per_bin.append(dict(
        k=k, r_vr=r_vr_k, vr=vr_k, var_vr=var_vr_k,
        vr_curves=[(model_vr(r_vr_k, b1_vr, b2_vr), '-', 'vr-only fit')],
        r_xi=r_xi, xihh=xihh, var_xi=var_xi,
        xi_curves=[(model_xihh(r_xi, b1_xi, b2_xi), '-', 'xi-only fit')]))

plot_debug_4panel_multibin(
    f'Individual fits, all bins (diagonal cov, no intr scatter): {SIM}, '
    , per_bin)


#%% Cell 3 -- xi_hh-only fit per bin, with the v_r PREDICTION from that fit
# overlaid dashed, looped over all mass bins

per_bin = []
for k in MASS_BINS_TO_PLOT:
    r_xi, xihh, var_xi, r_vr_k, vr_k, var_vr_k, n_phases = load_bin_data(k)
    res_xi = fit_bias_diag(mode='xi', r_min=R_MIN_FIT, r_xi=r_xi, xihh=xihh, var_xi=var_xi)
    b1_xi, b2_xi = res_xi.x
    print(f'bin {k}: xi-only  b1={b1_xi:.3f}, b2={b2_xi:.3f}, '
          f'reduced_chi2={res_xi.reduced_chi2:.3f} (dof={res_xi.dof})')

    per_bin.append(dict(
        k=k, r_vr=r_vr_k, vr=vr_k, var_vr=var_vr_k,
        vr_curves=[(model_vr(r_vr_k, b1_xi, b2_xi), '--', 'predicted from xi-only fit')],
        r_xi=r_xi, xihh=xihh, var_xi=var_xi,
        xi_curves=[(model_xihh(r_xi, b1_xi, b2_xi), '-', 'xi-only fit')]))

plot_debug_4panel_multibin(
    f'xi_hh-only fit, v_r predicted, all bins (diagonal cov, no intr scatter): {SIM}, '
    , per_bin)


#%% Cell 4 -- v_r-only fit per bin, with the xi_hh PREDICTION from that fit
# overlaid dashed, looped over all mass bins

per_bin = []
for k in MASS_BINS_TO_PLOT:
    r_xi, xihh, var_xi, r_vr_k, vr_k, var_vr_k, n_phases = load_bin_data(k)
    res_vr = fit_bias_diag(mode='vr', r_min=R_MIN_FIT, r_vr=r_vr_k, vr=vr_k, var_vr=var_vr_k)
    b1_vr, b2_vr = res_vr.x
    print(f'bin {k}: vr-only  b1={b1_vr:.3f}, b2={b2_vr:.3f}, '
          f'reduced_chi2={res_vr.reduced_chi2:.3f} (dof={res_vr.dof})')

    per_bin.append(dict(
        k=k, r_vr=r_vr_k, vr=vr_k, var_vr=var_vr_k,
        vr_curves=[(model_vr(r_vr_k, b1_vr, b2_vr), '-', 'vr-only fit')],
        r_xi=r_xi, xihh=xihh, var_xi=var_xi,
        xi_curves=[(model_xihh(r_xi, b1_vr, b2_vr), '--', 'predicted from vr-only fit')]))

plot_debug_4panel_multibin(
    f'v_r-only fit, xi_hh predicted, all bins (diagonal cov, no intrinsic scatter): {SIM}, '
    , per_bin)

 
#%% Cell 5 -- b1/b2 comparison plot across joint / xi-only / vr-only fits,
# using the values already printed by Cells 1-2 (hardcoded here, not
# refit) -- one point per (bin, fit type) in the (b1,b2) plane, colored by
# bin (matching Cells 1-4's C{k} convention), shaped by fit type, with a
# thin line connecting each bin's three fits so the spread/tension is
# visible directly as a triangle's size.
 
FIT_RESULTS = {
    # k: dict(joint=(b1,b2), xi=(b1,b2), vr=(b1,b2))
    0: dict(joint=(-0.028, 0.590), xi=(-0.082, -0.468), vr=(0.034, -0.106)),
    1: dict(joint=(0.140, -0.223), xi=(0.114, -0.836), vr=(0.133, -0.146)),
    2: dict(joint=(0.394, -1.030), xi=(0.476, 1.093), vr=(0.283, -0.126)),
    3: dict(joint=(0.753, -1.399), xi=(1.009, 1.084), vr=(0.547, 0.036)),
}
MARKERS = dict(joint='o', xi='s', vr='^')
 
fig, ax = plt.subplots(figsize=(7, 6))
for k, fits in FIT_RESULTS.items():
    color = f'C{k}'
    lo, hi = LOG_MASS_EDGES[k]
    pts = [fits['xi'], fits['joint'], fits['vr']]   # a sensible drawing order
    ax.plot([p[0] for p in pts], [p[1] for p in pts], '-', c=color, lw=0.8, alpha=0.6)
    for fit_type, (b1, b2) in fits.items():
        ax.plot(b1, b2, MARKERS[fit_type], c=color, ms=9, mec='k', mew=0.5)
    ax.annotate(rf'$\log M \in [{lo:.2f},{hi:.2f}]$', fits['joint'],
                textcoords='offset points', xytext=(6, 6), fontsize=8, color=color)
 
# legend for marker shape (fit type) -- separate from the bin-color legend,
# since color already carries bin identity above
from matplotlib.lines import Line2D
shape_legend = [Line2D([0], [0], marker=MARKERS[t], color='gray', linestyle='',
                        markeredgecolor='k', markersize=9, label=label)
                for t, label in [('joint', 'joint fit'), ('xi', 'xi-only fit'),
                                  ('vr', 'vr-only fit')]]
ax.legend(handles=shape_legend, fontsize=9, loc='best')
 
ax.axhline(0, c='gray', lw=0.5)
ax.axvline(0, c='gray', lw=0.5)
ax.set_xlabel(r'$b_1$')
ax.set_ylabel(r'$b_2$')
ax.set_title('Joint / xi-only / vr-only best-fit (b1,b2) per mass bin\n'
             '(larger triangle = more tension between xi_hh and v_r)')
fig.tight_layout()
 
 
#%% Cell 6 -- v_r variance sanity check: for each mass bin, compare the
# JACKKNIFE-predicted uncertainty (sqrt(var_vr), what the fit actually
# uses) against an EMPIRICAL local-scatter estimate derived directly from
# the v_r(r) profile itself (residuals from a smooth Savitzky-Golay fit,
# rolling std) -- if sqrt(var_vr) sits well ABOVE the empirical scatter
# across a bin, that's consistent with an overestimated jackknife
# variance for that bin (and with vr-only's suspiciously low reduced_chi2
# values from the joint-fit tension discussion).
 
from scipy.signal import savgol_filter
import pandas as pd
 
def local_scatter_estimate(r, y, window=9, polyorder=3):
    """Empirical local noise estimate: residual from a smooth Savitzky-Golay
    fit, then a rolling std of that residual -- NOT derived from var_vr at
    all, so this is a genuinely independent estimate of the noise level."""
    w = min(window, len(y) - (1 - len(y) % 2))
    if w % 2 == 0:
        w -= 1
    w = max(w, polyorder + 2 + (polyorder % 2 == 0))
    smooth = savgol_filter(y, w, polyorder)
    resid = y - smooth
    rolling_std = pd.Series(resid).rolling(w, center=True, min_periods=3).std().to_numpy()
    return rolling_std, resid, smooth
 
 
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
for k, ax in zip(MASS_BINS_TO_PLOT, axes.flat):
    r_xi, xihh, var_xi, r_vr, vr, var_vr, n_phases = load_bin_data(k)
    color = f'C{k}'
    lo, hi = LOG_MASS_EDGES[k]
 
    empirical_scatter, resid, smooth = local_scatter_estimate(r_vr, vr, window=31)
    jk_scatter = np.sqrt(var_vr)
 
    ax.plot(r_vr, jk_scatter, '-', c=color, lw=2,
             label=r'jackknife $\sqrt{\mathrm{var\_vr}}$')
    ax.plot(r_vr, empirical_scatter, '--', c='k', lw=1.5,
             label='empirical (Savitzky-Golay residual, rolling std)')
    ax.set_title(rf'bin {k}: $\log M \in [{lo:.2f},{hi:.2f}]$', fontsize=10)
    ax.set_ylabel(r'$\sigma_{v_r}$ [km/s]')
    if ax in axes[-1, :]:
        ax.set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
    ax.legend(fontsize=7)
 
    ratio = np.nanmedian(jk_scatter) / np.nanmedian(empirical_scatter)
    print(f'bin {k}: median jackknife sigma / median empirical sigma = {ratio:.2f} '
          f'{"(jackknife LOOKS INFLATED)" if ratio > 1.5 else ""}')
 
fig.suptitle('v_r sanity check: jackknife-predicted vs. empirical (profile-derived) scatter')
fig.tight_layout()
#%% Cell 6b -- Cell 6's diagnostic, repeated with a MUCH wider
# Savitzky-Golay window, to check whether the ~35-40x "inflation" found
# with window=9 survives or was an overfitting artifact (a narrow window's
# local cubic fit tracks v_r(r) almost exactly once the profile flattens
# at large r, driving the residual toward zero for reasons unrelated to
# real noise). If the ratio here comes down close to O(1), Cell 6's
# result was the artifact, not the jackknife covariance.
 
for k in MASS_BINS_TO_PLOT:
    r_xi, xihh, var_xi, r_vr, vr, var_vr, n_phases = load_bin_data(k)
    empirical_scatter, resid, smooth = local_scatter_estimate(r_vr, vr, window=31, polyorder=3)
    jk_scatter = np.sqrt(var_vr)
    ratio = np.nanmedian(jk_scatter) / np.nanmedian(empirical_scatter)
    print(f'bin {k}: window=31  median jk/empirical sigma ratio = {ratio:.2f}')
 
 
#%% Cell 7 -- LITERAL fit using Cell 6's empirical variance in place of
# var_vr, exactly as asked -- but treat this as a stress-test, NOT a
# trustworthy alternative fit. Per the concerns above (Cell 6's empirical
# sigma likely reflects overfitting, not real noise), this replaces v_r's
# real uncertainty with something ~35-40x too small in every bin, which
# will force the joint fit to match v_r almost exactly regardless of what
# xi_hh wants -- expect b1/b2 here to land very close to the vr-only
# values from Cell 2, not because v_r is more informative, but because
# its artificially tiny errors make xi_hh's contribution to chi2
# negligible by comparison.
 
per_bin = []
for k in MASS_BINS_TO_PLOT:
    r_xi, xihh, var_xi, r_vr, vr, var_vr, n_phases = load_bin_data(k)
    empirical_scatter, resid, smooth = local_scatter_estimate(r_vr, vr)
    # rolling std has NaN at the edges (min_periods=3) -- fill with the
    # nearest valid value so every point has a usable (if untrustworthy)
    # variance
    empirical_scatter = pd.Series(empirical_scatter).bfill().ffill().to_numpy()
    var_vr_empirical = empirical_scatter**2
 
    res = fit_bias_diag(mode='joint', r_min=R_MIN_FIT,
                         r_xi=r_xi, xihh=xihh, var_xi=0*var_xi,
                         r_vr=r_vr, vr=vr, var_vr=0*var_vr_empirical)
    b1, b2 = res.x
    print(f'bin {k}: joint w/ EMPIRICAL v_r var  b1={b1:.3f}, b2={b2:.3f}, '
          f'reduced_chi2={res.reduced_chi2:.3f} (dof={res.dof})  '
          f'[STRESS TEST -- see caveats above]')
 
    per_bin.append(dict(
        k=k, r_vr=r_vr, vr=vr, var_vr=var_vr_empirical,
        vr_curves=[(model_vr(r_vr, b1, b2), '-', 'joint fit (empirical var_vr)')],
        r_xi=r_xi, xihh=xihh, var_xi=var_xi,
        xi_curves=[(model_xihh(r_xi, b1, b2), '-', 'joint fit (empirical var_vr)')]))
 
plot_debug_4panel_multibin(
    'Joint fit using EMPIRICAL (likely overfit-artifact) v_r variance -- '
    'STRESS TEST, not a validated result', per_bin)
#%% Cell 8 -- fit with NO covariance: unweighted least squares,
# chi2 = sum((model-data)^2), var=1 for every point in both xi_hh and
# v_r. NOTE: xi_hh and v_r are on very different scales (xi_hh ~ O(1-100),
# v_r ~ O(50-250) km/s), so with no normalization at all, whichever
# statistic has the larger raw squared residuals will dominate the joint
# fit almost entirely -- this is a sensitivity check on the earlier
# weighted results, not a fit result to trust on its own.
 
per_bin = []
for k in MASS_BINS_TO_PLOT:
    r_xi, xihh, var_xi, r_vr, vr, var_vr, n_phases = load_bin_data(k)
    ones_xi = np.ones_like(xihh)
    ones_vr = np.ones_like(vr)
 
    res = fit_bias_diag(mode='joint', r_min=R_MIN_FIT,
                         r_xi=r_xi, xihh=xihh, var_xi=ones_xi,
                         r_vr=r_vr, vr=vr, var_vr=ones_vr)
    b1, b2 = res.x
    print(f'bin {k}: joint, NO covariance  b1={b1:.3f}, b2={b2:.3f}, '
          f'reduced_chi2={res.reduced_chi2:.3f} (dof={res.dof})')
 
    per_bin.append(dict(
        k=k, r_vr=r_vr, vr=vr, var_vr=var_vr,   # plot with the REAL jk var_vr
        vr_curves=[(model_vr(r_vr, b1, b2), '-', 'joint fit (no covariance)')],
        r_xi=r_xi, xihh=xihh, var_xi=var_xi,    # (band shown is real, fit used var=1)
        xi_curves=[(model_xihh(r_xi, b1, b2), '-', 'joint fit (no covariance)')]))
 
plot_debug_4panel_multibin(
    'Joint fit with NO covariance weighting (unweighted least squares) -- '
    'error bands shown are the real jackknife values, NOT what the fit used',
    per_bin)
#%% Cell 9 -- same no-covariance idea as Cell 8, but with v_r converted
# from km/s into h^-1 Mpc units before fitting (dividing by conv*f, the
# SAME prefactor model_vr applies to go from the raw LPT displacement
# integral to km/s), so the two statistics' raw squared residuals are on
# a more comparable scale than Cell 8's direct km/s vs (h^-1 Mpc)^2
# comparison. Still unweighted (var=1 for both, no jackknife covariance
# at all) -- this addresses the UNIT mismatch flagged in Cell 8, not the
# weighting question itself.
 
def model_vr_hinvmpc(r, b1, b2, bs=0.0):
    """model_vr, converted from km/s to h^-1 Mpc by dividing out the same
    conv*f prefactor model_vr applies -- i.e. the raw LPT
    veft/(1+xi) quantity, in comoving h^-1 Mpc."""
    return model_vr(r, b1, b2, bs) / (lpt.conv * lpt.f)
 
 
def fit_bias_novar_vrmpc(r_min=R_MIN_FIT, fit_bs=False,
                          r_xi=None, xihh=None,
                          r_vr=None, vr_mpc=None,
                          popsize=15, maxiter=150, tol=1e-6, seed=0):
    """Unweighted joint fit (var=1 for both statistics), with v_r already
    converted to h^-1 Mpc units by the caller."""
    m_xi = r_xi > r_min
    r_xi_fit, xihh_fit = r_xi[m_xi], xihh[m_xi]
    m_vr = r_vr > r_min
    r_vr_fit, vr_fit = r_vr[m_vr], vr_mpc[m_vr]
 
    def loss(params):
        b1, b2 = params[0], params[1]
        bs = params[2] if fit_bs else 0.0
        try:
            res_xi = model_xihh(r_xi_fit, b1, b2, bs) - xihh_fit
            res_vr = model_vr_hinvmpc(r_vr_fit, b1, b2, bs) - vr_fit
        except Exception:
            return np.inf
        c = np.sum(res_xi**2) + np.sum(res_vr**2)
        return c if np.isfinite(c) else np.inf
 
    bounds = [(-0.3, 5.0), (-10.0, 10.0)] + ([(-5.0, 5.0)] if fit_bs else [])
    result = differential_evolution(loss, bounds, tol=tol, seed=seed,
                                     maxiter=maxiter, popsize=popsize, polish=True)
    n_data = len(xihh_fit) + len(vr_fit)
    result.dof = max(n_data - len(bounds), 1)
    result.chi2 = result.fun
    result.reduced_chi2 = result.fun / result.dof
    return result
 
 
per_bin = []
for k in MASS_BINS_TO_PLOT:
    r_xi, xihh, var_xi, r_vr, vr, var_vr, n_phases = load_bin_data(k)
    vr_mpc = vr / (lpt.conv * lpt.f)
 
    res = fit_bias_novar_vrmpc(r_min=R_MIN_FIT, r_xi=r_xi, xihh=xihh,
                                r_vr=r_vr, vr_mpc=vr_mpc)
    b1, b2 = res.x
    print(f'bin {k}: joint, no covariance, v_r in h^-1 Mpc  b1={b1:.3f}, b2={b2:.3f}, '
          f'reduced_chi2={res.reduced_chi2:.3f} (dof={res.dof})')
 
    per_bin.append(dict(
        k=k, r_vr=r_vr, vr=vr, var_vr=var_vr,   # plotted back in km/s, real jk var shown
        vr_curves=[(model_vr(r_vr, b1, b2), '-', 'joint fit (no cov, v_r in Mpc/h)')],
        r_xi=r_xi, xihh=xihh, var_xi=var_xi,
        xi_curves=[(model_xihh(r_xi, b1, b2), '-', 'joint fit (no cov, v_r in Mpc/h)')]))
 
plot_debug_4panel_multibin(
    'Joint fit, no covariance, v_r converted to h^-1 Mpc before fitting -- '
    'plotted here back in km/s for comparison to Cell 8', per_bin)

#%% fit_bias_diag_scatter -- fit_bias_diag's diagonal-covariance, no-Hartlap
# approach, extended with one intrinsic-scatter parameter per included
# statistic (eps_xi and/or eps_vr): C_ii = var_i + eps^2*data_i^2, with the
# +sum(log(C)) term the loss needs since eps is now a free parameter
# (omitting it lets eps collapse to 0, per this project's established
# requirement -- same reasoning as fit_bias_scatter in fit_sim_jk.py, just
# rebuilt here to loop over mass bins in fit_sim_debug.py's style).
 
def fit_bias_diag_scatter(mode, r_min=R_MIN_FIT, fit_bs=False,
                           r_xi=None, xihh=None, var_xi=None,
                           r_vr=None, vr=None, var_vr=None,
                           eps_bounds=(1e-6, 0.5),
                           popsize=15, maxiter=150, tol=1e-6, seed=0):
    if mode in ('joint', 'xi'):
        m_xi = r_xi > r_min
        r_xi_fit, xihh_fit, var_xi_fit = r_xi[m_xi], xihh[m_xi], var_xi[m_xi]
    if mode in ('joint', 'vr'):
        m_vr = r_vr > r_min
        r_vr_fit, vr_fit, var_vr_fit = r_vr[m_vr], vr[m_vr], var_vr[m_vr]
 
    n_bs = 1 if fit_bs else 0
    n_eps_xi = 1 if mode in ('joint', 'xi') else 0
    n_eps_vr = 1 if mode in ('joint', 'vr') else 0
 
    def unpack(params):
        b1, b2 = params[0], params[1]
        i = 2
        bs = params[i] if fit_bs else 0.0
        i += n_bs
        eps_xi = params[i] if n_eps_xi else None
        i += n_eps_xi
        eps_vr = params[i] if n_eps_vr else None
        return b1, b2, bs, eps_xi, eps_vr
 
    def loss(params):
        b1, b2, bs, eps_xi, eps_vr = unpack(params)
        c = 0.0
        if mode in ('joint', 'xi'):
            try:
                res = model_xihh(r_xi_fit, b1, b2, bs) - xihh_fit
            except Exception:
                return np.inf
            C = var_xi_fit + eps_xi**2 * xihh_fit**2
            if np.any(C <= 0):
                return np.inf
            c += np.sum(res**2 / C) + np.sum(np.log(C))
        if mode in ('joint', 'vr'):
            try:
                res = model_vr(r_vr_fit, b1, b2, bs) - vr_fit
            except Exception:
                return np.inf
            C = var_vr_fit + eps_vr**2 * vr_fit**2
            if np.any(C <= 0):
                return np.inf
            c += np.sum(res**2 / C) + np.sum(np.log(C))
        return c if np.isfinite(c) else np.inf
 
    bounds = [(-0.3, 5.0), (-10.0, 10.0)]
    if fit_bs:
        bounds += [(-5.0, 5.0)]
    if n_eps_xi:
        bounds += [eps_bounds]
    if n_eps_vr:
        bounds += [eps_bounds]
 
    result = differential_evolution(loss, bounds, tol=tol, seed=seed,
                                     maxiter=maxiter, popsize=popsize, polish=True)
    b1, b2, bs, eps_xi, eps_vr = unpack(result.x)
 
    # chi2/dof reported WITHOUT the log-det term (pure quadratic piece),
    # matching this project's usual reduced_chi2 convention
    chi2 = 0.0
    n_data = 0
    if mode in ('joint', 'xi'):
        res = model_xihh(r_xi_fit, b1, b2, bs) - xihh_fit
        C = var_xi_fit + eps_xi**2 * xihh_fit**2
        chi2 += np.sum(res**2 / C)
        n_data += len(xihh_fit)
    if mode in ('joint', 'vr'):
        res = model_vr(r_vr_fit, b1, b2, bs) - vr_fit
        C = var_vr_fit + eps_vr**2 * vr_fit**2
        chi2 += np.sum(res**2 / C)
        n_data += len(vr_fit)
 
    result.dof = max(n_data - len(bounds), 1)
    result.chi2 = chi2
    result.reduced_chi2 = chi2 / result.dof
    result.eps = {}
    if n_eps_xi:
        result.eps['xi'] = eps_xi
    if n_eps_vr:
        result.eps['vr'] = eps_vr
    return result

#%% Cell 10 -- joint (xi_hh + v_r) fit WITH intrinsic scatter (eps_xi,
# eps_vr), diagonal cov, looped over all mass bins -- same
# structure as Cell 1, fit_bias_diag_scatter in place of fit_bias_diag.
 
per_bin = []
for k in MASS_BINS_TO_PLOT:
    r_xi, xihh, var_xi, r_vr_k, vr_k, var_vr_k, n_phases = load_bin_data(k)
    res = fit_bias_diag_scatter(mode='joint', r_min=R_MIN_FIT,
                                 r_xi=r_xi, xihh=xihh, var_xi=var_xi,
                                 r_vr=r_vr_k, vr=vr_k, var_vr=var_vr_k)
    b1, b2 = res.x[0], res.x[1]
    print(f'bin {k}: joint (scatter)  b1={b1:.3f}, b2={b2:.3f}, '
          f'eps_xi={res.eps["xi"]:.2e}, eps_vr={res.eps["vr"]:.2e}, '
          f'reduced_chi2={res.reduced_chi2:.3f} (dof={res.dof})')
 
    per_bin.append(dict(
        k=k, r_vr=r_vr_k, vr=vr_k, var_vr=var_vr_k,
        vr_curves=[(model_vr(r_vr_k, b1, b2), '-', 'joint fit (scatter)')],
        r_xi=r_xi, xihh=xihh, var_xi=var_xi,
        xi_curves=[(model_xihh(r_xi, b1, b2), '-', 'joint fit (scatter)')]))
 
plot_debug_4panel_multibin(
    f'Joint fit WITH intrinsic scatter, all bins (diagonal cov): {SIM} '
    , per_bin)
 
 
#%% Cell 11 -- individual fits (xi_hh and v_r each fit on their own) WITH
# intrinsic scatter (eps_xi for the xi-only fit, eps_vr for the vr-only
# fit), diagonal cov, looped over all mass bins -- same
# structure as Cell 2, fit_bias_diag_scatter in place of fit_bias_diag.
 
per_bin = []
for k in MASS_BINS_TO_PLOT:
    r_xi, xihh, var_xi, r_vr_k, vr_k, var_vr_k, n_phases = load_bin_data(k)
    res_xi = fit_bias_diag_scatter(mode='xi', r_min=R_MIN_FIT,
                                    r_xi=r_xi, xihh=xihh, var_xi=var_xi)
    res_vr = fit_bias_diag_scatter(mode='vr', r_min=R_MIN_FIT,
                                    r_vr=r_vr_k, vr=vr_k, var_vr=var_vr_k)
    b1_xi, b2_xi = res_xi.x[0], res_xi.x[1]
    b1_vr, b2_vr = res_vr.x[0], res_vr.x[1]
    print(f'bin {k}: xi-only (scatter)  b1={b1_xi:.3f}, b2={b2_xi:.3f}, '
          f'eps_xi={res_xi.eps["xi"]:.2e}, reduced_chi2={res_xi.reduced_chi2:.3f} '
          f'(dof={res_xi.dof})')
    print(f'bin {k}: vr-only (scatter)  b1={b1_vr:.3f}, b2={b2_vr:.3f}, '
          f'eps_vr={res_vr.eps["vr"]:.2e}, reduced_chi2={res_vr.reduced_chi2:.3f} '
          f'(dof={res_vr.dof})')
 
    per_bin.append(dict(
        k=k, r_vr=r_vr_k, vr=vr_k, var_vr=var_vr_k,
        vr_curves=[(model_vr(r_vr_k, b1_vr, b2_vr), '-', 'vr-only fit (scatter)')],
        r_xi=r_xi, xihh=xihh, var_xi=var_xi,
        xi_curves=[(model_xihh(r_xi, b1_xi, b2_xi), '-', 'xi-only fit (scatter)')]))
 
plot_debug_4panel_multibin(
    f'Individual fits WITH intrinsic scatter, all bins (diagonal cov): {SIM} '
    , per_bin)
#%%

#%% Fit xi_hh with ONLY b1 (b2 fixed to 0), diagonal cov, no Hartlap, for
# each mass bin -- model_xihh is compact.lpt's xi_real (same model used
# throughout this project's fit_sim_jk.py/fit_sim_debug.py). Uses all
# N_MASS bins, not just MASS_BINS_TO_PLOT's usual subset, since this only
# needs xi_hh (no v_r availability restriction applies here).

from scipy.optimize import differential_evolution

def fit_b1_only(r_xi, xihh, var_xi, r_min=R_MIN_FIT):
    m = r_xi > r_min
    r_fit, xi_fit, var_fit = r_xi[m], xihh[m], var_xi[m]

    def loss(params):
        b1 = params[0]
        try:
            res = model_xihh(r_fit, b1, 0.0) - xi_fit
        except Exception:
            return np.inf
        c = np.sum(res**2 / var_fit)
        return c if np.isfinite(c) else np.inf

    result = differential_evolution(loss, [(-0.3, 5.0)], tol=1e-6, seed=0,
                                     maxiter=150, popsize=15, polish=True)
    dof = max(m.sum() - 1, 1)
    return result.x[0], result.fun, dof


b1_per_bin, redchi2_per_bin = [], []
for k in range(N_MASS):
    r_xi, xihh, var_xi, r_vr_k, vr_k, var_vr_k, n_phases = load_bin_data(k)
    b1, chi2, dof = fit_b1_only(r_xi, xihh, var_xi)
    redchi2 = chi2 / dof
    b1_per_bin.append(b1)
    redchi2_per_bin.append(redchi2)
    lo, hi = LOG_MASS_EDGES[k]
    print(f'bin {k}: logM in [{lo:.2f},{hi:.2f}]  b1={b1:.4f}  '
          f'reduced_chi2={redchi2:.3f} (dof={dof})')

bin_centers = LOG_MASS_EDGES.mean(axis=1)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

axs[0].plot(bin_centers, b1_per_bin, 'o-', color='C0')
axs[0].set_xlabel(r'mass-bin center, $\log_{10} M$')
axs[0].set_ylabel(r'$b_1$ (Lagrangian, $b_2=0$ fixed)')
axs[0].set_title(r'$b_1$ vs. mass')

axs[1].plot(bin_centers, redchi2_per_bin, 'o-', color='C3')
axs[1].axhline(1, color='gray', lw=0.7, ls=':')
axs[1].set_xlabel(r'mass-bin center, $\log_{10} M$')
axs[1].set_ylabel(r'reduced $\chi^2$')
axs[1].set_title('fit quality vs. mass')

fig.suptitle('xi_hh fit with b1 only (b2=0), diagonal cov, no scatter')
fig.tight_layout()

#%%
#%% Validation plot: xi_hh data vs. the b1-only fit, all mass bins,
# same 2-panel (amplitude + offset ratio) style as this project's other
# multi-bin plots. Reuses b1_per_bin from the previous cell -- run that
# first.

fig, ax = plt.subplots(2, 1, figsize=(8, 9), sharex=True,
                        gridspec_kw={'height_ratios': [1.4, 1]})
offset_step = 0.5

for k in range(N_MASS):
    r_xi, xihh, var_xi, r_vr_k, vr_k, var_vr_k, n_phases = load_bin_data(k)
    b1 = b1_per_bin[k]
    xi_model = model_xihh(r_xi, b1, 0.0)
    xi_err = np.sqrt(var_xi)

    color = f'C{k}'
    lo, hi = LOG_MASS_EDGES[k]

    xi_r2 = r_xi**2 * xihh
    xi_model_r2 = r_xi**2 * xi_model
    ax[0].plot(r_xi, xi_r2, 'o', ms=3, c=color, zorder=2,
               label=rf'$\log M \in [{lo:.2f}, {hi:.2f}]$, $b_1={b1:.2f}$')
    ax[0].fill_between(r_xi, xi_r2 - r_xi**2*xi_err, xi_r2 + r_xi**2*xi_err,
                        color=color, alpha=0.25, lw=0, zorder=1)
    ax[0].plot(r_xi, xi_model_r2, '-', c=color)

    xi_ratio = xi_model / xihh
    xi_ratio_err = np.abs(xi_ratio) * (xi_err / np.abs(xihh))
    xi_good = np.abs(xihh) > xi_err
    baseline = 1 + k * offset_step
    ax[1].axhline(baseline, c=color, lw=0.7, alpha=0.5, zorder=0)
    ax[1].fill_between(r_xi[xi_good], baseline - 2*xi_ratio_err[xi_good],
                        baseline + 2*xi_ratio_err[xi_good],
                        color=color, alpha=0.12, lw=0, zorder=0)
    ax[1].fill_between(r_xi[xi_good], baseline - xi_ratio_err[xi_good],
                        baseline + xi_ratio_err[xi_good],
                        color=color, alpha=0.22, lw=0, zorder=0)
    ax[1].plot(r_xi[xi_good], xi_ratio[xi_good] + k * offset_step,
               'o-', ms=3, c=color, zorder=2)

for ax_ in ax:
    ax_.axvline(R_MIN_FIT, color='gray', ls=':', lw=1, zorder=0)
    ax_.margins(x=0)

ax[0].set_ylabel(r'$r^2\,\xi_{hh}(r|M)$ [$(h^{-1}\,{\rm Mpc})^2$]')
ax[0].set_yscale('log')
ax[0].legend(fontsize=8, ncol=2)
ax[1].set_ylabel('model / data (+ offset)')
ax[1].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
ax[1].set_ylim(0.5, 1 + N_MASS*offset_step + 0.5)

fig.suptitle('xi_hh: b1-only fit (b2=0 fixed) vs. data, all mass bins')
fig.tight_layout()

#%%

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

axs[0].plot(bin_centers, np.array(b1_per_bin), 'o-', color='C0')
axs[0].set_xlabel(r'mass-bin center, $\log_{10} M$')
axs[0].set_ylabel(r'$b_1$ (Lagrangian, $b_2=0$ fixed)')
axs[0].set_title(r'$b_1$ vs. mass')
# axs[0].set_yscale('log')
# axs[0].plot(bin_centers, bin_centers**0.33)

axs[1].plot(bin_centers, redchi2_per_bin, 'o-', color='C3')
axs[1].axhline(1, color='gray', lw=0.7, ls=':')
axs[1].set_xlabel(r'mass-bin center, $\log_{10} M$')
axs[1].set_ylabel(r'reduced $\chi^2$')
axs[1].set_title('fit quality vs. mass')

fig.suptitle('xi_hh fit with b1 only (b2=0), diagonal cov, no scatter')
fig.tight_layout()