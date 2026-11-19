"""
Talk figures -- PLOT ONLY. Reads paper_figs_cache.h5 (group `auto/fig7`), no fits are re-run.

P(v_los | r_perp, r_los) from AbacusSummit (mass bin 2, stacked phases) at r_perp = r_los = 15, 30, 60 h^-1 Mpc,
three log-y panels, built up in three stages so the slides can be flipped through:
  1. histograms only
  2. + best-fit Gaussian   (maximum likelihood = the histogram's own mean and standard deviation)
  3. + skew-t model        (11-parameter defaults Config._MEAN_PARAMS, cm.Pv_compact)

Axis limits depend only on the histogram, so they are identical in all three stages.
Run from the directory containing paper_figs_cache.h5.
"""

#%% setup
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import h5py as h5
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import InterpolatedUnivariateSpline

import compact.model as cm
from compact.config import Config

try:
    from matplotlib_inline.backend_inline import set_matplotlib_formats
    set_matplotlib_formats('retina')          # sharp inline figures when pasted into slides
except ImportError:
    pass

matplotlib.rcParams.update({'text.usetex': True,
                            'font.family': 'serif',
                            'font.serif': ['Computer Modern Roman'],
                            'mathtext.fontset': 'cm',
                            'axes.labelsize': 22,
                            'xtick.labelsize': 16,
                            'ytick.labelsize': 16,
                            'axes.xmargin': 0})

CACHE = 'paper_figs_cache.h5'
FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

PARS = np.asarray(Config._MEAN_PARAMS, dtype=float)      # 11-parameter model defaults

# same grid as the accumulator (needed to map r_perp, r_los -> cell and to evaluate the model)
r_perp_edges = np.linspace(5, 140, 135 + 1)
r_los_edges = np.linspace(5, 140, 135 + 1)
r_p = 0.5 * (r_perp_edges[1:] + r_perp_edges[:-1])
r_l = 0.5 * (r_los_edges[1:] + r_los_edges[:-1])

V_LIM, V_PLOT = 2500.0, 2000.0     # model-curve range / displayed x-range [km/s]
Y_FLOOR = 1e-5                     # log-y lower limit as a fraction of the panel's histogram peak
REBIN = 5                          # 400 fine v-bins -> 80 plotted bins

with h5.File(CACHE, 'r') as f:
    g = f['auto/fig7']
    RP_SEL, RL_SEL = g['rp_sel'][()], g['rl_sel'][()]
    v_fine = g['v_centers'][()]
    gb = g['bin']
    vr_spline = InterpolatedUnivariateSpline(gb['r_tab'][()], gb['vr_model'][()], k=3, ext=3)
    COUNTS = {k: gb[k]['counts'][()] for k in gb if k.startswith('cell_')}

cent = v_fine.reshape(-1, REBIN).mean(axis=1)
dv = (v_fine[1] - v_fine[0]) * REBIN
edges = np.r_[cent - dv / 2, cent[-1] + dv / 2]
vv = np.linspace(-V_LIM, V_LIM, 600)


#%% figure function
def talk_fig(stage, fname):
    """stage 1 = histograms, 2 = + best-fit Gaussian, 3 = + skew-t."""
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), layout='constrained', sharex=True)
    for ax, rp_t, rl_t in zip(axs, RP_SEL, RL_SEL):
        j, i = int(np.argmin(np.abs(r_p - rp_t))), int(np.argmin(np.abs(r_l - rl_t)))
        counts_fine = COUNTS[f'cell_{j}_{i}']
        counts = counts_fine.reshape(-1, REBIN).sum(axis=1)
        hist = counts / (counts.sum() * dv)

        ax.stairs(np.where(counts >= 1, hist, 1e-30), edges, color='k', lw=1.5)
        if stage >= 2:
            mu_g = np.sum(counts_fine * v_fine) / counts_fine.sum()
            sd_g = np.sqrt(np.sum(counts_fine * (v_fine - mu_g)**2) / counts_fine.sum())
            ax.plot(vv, norm.pdf(vv, mu_g, sd_g), c='C1', lw=2.5)
        if stage >= 3:
            ax.plot(vv, cm.Pv_compact(vv, r_p[j], r_l[i], vr_spline, PARS), c='C0', lw=2.5)

        ax.set_yscale('log')
        ax.set_xlim(-V_PLOT, V_PLOT)
        ax.set_ylim(hist.max() * Y_FLOOR, hist.max() * 2)
        ax.set_title(rf'$r_\perp = r_{{\rm los}} = {int(round(rp_t - 0.5))}\ h^{{-1}}$Mpc', fontsize=20)
        ax.set_xlabel(r'$v_{\rm los}$ [km/s]')
    axs[0].set_ylabel(r'$P(v_{\rm los}\,|\,r_\perp, r_{\rm los})$')

    handles = [plt.Line2D([], [], c='k', lw=2, label='AbacusSummit')]
    if stage >= 2:
        handles.append(plt.Line2D([], [], c='C1', lw=2.5, label='Best-fit Gaussian'))
    if stage >= 3:
        handles.append(plt.Line2D([], [], c='C0', lw=2.5, label='Skew-$t$ model'))
    axs[0].legend(handles=handles, fontsize=15, loc='lower center')

    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=300)
    print(f'saved {os.path.join(FIGURES_DIR, fname)}')
    return fig


#%% 1 -- histograms only
talk_fig(1, 'talk_pvd_1_hist.png')

#%% 2 -- + best-fit Gaussian
talk_fig(2, 'talk_pvd_2_gauss.png')

#%% 3 -- + skew-t model
talk_fig(3, 'talk_pvd_3_skewt.png')


#%% toy setup -- the model's skew-t as a Gaussian mixture over "environments"
# Azzalini & Capitanio: a skew-t is exactly a Gaussian mixture over two latent variables,
#   v | w, u ~ N( xi + omega d u / sqrt(w),  omega^2 (1 - d^2) / w ),   d = alpha / sqrt(1 + alpha^2)
#   w ~ Gamma(nu/2, rate nu/2)   local precision: small w = hot (dense) environment -> wide Gaussian -> heavy tails
#   u ~ |N(0,1)|                 one-sided shift of the local mean -> skewness
# xi, omega, alpha, nu are the model's own values in each cell -- nothing is tuned. The latent w and u are not
# measured densities; reading them as "local density" is the interpretation, not a measurement.
from scipy.stats import gamma, halfnorm, t as student_t, skewnorm

W_EDGES = np.array([0, .05, .4, 1.0])   # probability-weight bins in w: hot tail, middle, bulk
U_EDGES = np.array([0, .5, 1.0])
CMAP = plt.cm.plasma


def cell_params(rp, rl):
    """(xi, omega, alpha, nu) of the skew-t in a cell, built from model.py exactly as in cm.Pv_compact."""
    r = np.hypot(rp, rl)
    x = r - rp
    mu = vr_spline(r) * rl / r
    sig_v = cm.sigma_mod(x, PARS[0], cm.A_rho_mod(rp, PARS[1], PARS[2]), PARS[3])
    delta = cm.delta_mod_alpha(x, cm.A_delta_mod(rp, PARS[4], PARS[5]), PARS[6], PARS[7], PARS[8])
    nu = cm.nu_of_r(r, PARS[9], PARS[10])
    omega, alpha = cm.skewt_params_from_delta(delta, sig_v, nu)
    return mu + delta, omega, alpha, nu


def reps(edges, ppf):
    """Representative values (bin-median quantile) and probability weights of equal-quantile bins."""
    return ppf(0.5 * (edges[1:] + edges[:-1])), np.diff(edges)


def draw_components(ax, weights, m, s):
    """One thin Gaussian per environment, height weighted by its probability, coloured by its width."""
    order = np.argsort(s)
    for rank, k in enumerate(order):
        ax.plot(vv, weights[k] * norm.pdf(vv, m[k], s[k]), c=CMAP(0.1 + 0.8 * rank / (len(s) - 1)), lw=1.2, alpha=0.8)


def mixture_components(xi, om, al, nu, use_w=True, use_u=True):
    """(weights, means, widths) of the environment Gaussians; a latent that is switched off is fixed at 1 (w) or 0 (u, d)."""
    d = al / np.sqrt(1 + al**2)
    w, pw = reps(W_EDGES, lambda q: gamma.ppf(q, nu / 2, scale=2 / nu)) if use_w else (np.ones(1), np.ones(1))
    u, pu = reps(U_EDGES, halfnorm.ppf) if use_u else (np.zeros(1), np.ones(1))
    d = d if use_u else 0.0
    W, U = np.meshgrid(w, u, indexing='ij')
    P = np.outer(pw, pu)
    return P.ravel(), (xi + om * d * U / np.sqrt(W)).ravel(), (om * np.sqrt(1 - d**2) / np.sqrt(W)).ravel()


#%% toy 1 -- histogram, skew-t, and the individual environment Gaussians that sum to it
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), layout='constrained', sharex=True)
for ax, rp_t, rl_t in zip(axs, RP_SEL, RL_SEL):
    j, i = int(np.argmin(np.abs(r_p - rp_t))), int(np.argmin(np.abs(r_l - rl_t)))
    counts = COUNTS[f'cell_{j}_{i}'].reshape(-1, REBIN).sum(axis=1)
    hist = counts / (counts.sum() * dv)
    xi, om, al, nu = cell_params(r_p[j], r_l[i])

    draw_components(ax, *mixture_components(xi, om, al, nu))
    ax.stairs(np.where(counts >= 1, hist, 1e-30), edges, color='k', lw=1.5)
    ax.plot(vv, cm.Pv_compact(vv, r_p[j], r_l[i], vr_spline, PARS), c='C0', lw=2.5)
    ax.set_yscale('log')
    ax.set_xlim(-V_PLOT, V_PLOT)
    ax.set_ylim(hist.max() * Y_FLOOR, hist.max() * 2)
    ax.set_title(rf'$r_\perp = r_{{\rm los}} = {int(round(rp_t - 0.5))}\ h^{{-1}}$Mpc', fontsize=20)
    ax.set_xlabel(r'$v_{\rm los}$ [km/s]')
axs[0].set_ylabel(r'$P(v_{\rm los}\,|\,r_\perp, r_{\rm los})$')
axs[0].legend(handles=[plt.Line2D([], [], c='k', lw=2, label='AbacusSummit'),
                       plt.Line2D([], [], c='C0', lw=2.5, label='Skew-$t$ model'),
                       plt.Line2D([], [], c=CMAP(0.5), lw=1.5, label='Single environments')],
              fontsize=15, loc='lower center')
fig.savefig(os.path.join(FIGURES_DIR, 'talk_toy_1_mixture.png'), dpi=300)

#%% toy 3 -- which mixing produces which moment (cell index TOY_CELL: 0, 1, 2 = r = 15, 30, 60; both together = toy 1)
TOY_CELL = 0
rp_t, rl_t = RP_SEL[TOY_CELL], RL_SEL[TOY_CELL]
j, i = int(np.argmin(np.abs(r_p - rp_t))), int(np.argmin(np.abs(r_l - rl_t)))
xi, om, al, nu = cell_params(r_p[j], r_l[i])
d = al / np.sqrt(1 + al**2)

panels = [  # title, components, exact mixture, same-mean-and-variance Gaussian
    (r'Spread in local $\sigma$ $\Rightarrow$ heavy tails',
     mixture_components(xi, om, al, nu, use_u=False),
     student_t.pdf((vv - xi) / om, nu) / om,
     norm.pdf(vv, xi, student_t(nu, xi, om).std())),
    (r'Spread in local mean $\Rightarrow$ skewness',
     mixture_components(xi, om, al, nu, use_w=False),
     skewnorm.pdf(vv, al, loc=xi, scale=om),
     norm.pdf(vv, skewnorm(al, xi, om).mean(), skewnorm(al, xi, om).std())),
]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained', sharex=True)
for ax, (title, comps, pdf_mix, pdf_g) in zip(axs, panels):
    draw_components(ax, *comps)
    ax.plot(vv, pdf_g, c='C1', lw=2.5)
    ax.plot(vv, pdf_mix, c='C0', lw=2.5)
    ax.set_yscale('log')
    ax.set_xlim(-V_PLOT, V_PLOT)
    ax.set_ylim(pdf_mix.max() * Y_FLOOR, pdf_mix.max() * 2)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(r'$v_{\rm los}$ [km/s]')
axs[0].set_ylabel(r'$P(v_{\rm los})$')
axs[0].legend(handles=[plt.Line2D([], [], c='C0', lw=2.5, label='Sum over environments'),
                       plt.Line2D([], [], c='C1', lw=2.5, label='Gaussian, same mean and variance'),
                       plt.Line2D([], [], c=CMAP(0.5), lw=1.5, label='Single environments')],
              fontsize=13, loc='lower center')
fig.savefig(os.path.join(FIGURES_DIR, 'talk_toy_3_moments.png'), dpi=300)