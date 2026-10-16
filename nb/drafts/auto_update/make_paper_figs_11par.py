"""
Paper Figs 0-5 and 7 -- PLOT ONLY. Every fit lives in build_paper_cache.py; this script reads
paper_figs_cache.h5 (group `auto/`) and uses compact.model (cm) for every model evaluation.

Model defaults: the 11 parameters are Config._MEAN_PARAMS, the pivot is cm.R_PIV_FIXED, and
nu(r) is cm.nu_of_r. Nothing here is refit except Fig 0's one-histogram skew-t fit (400 points).

Each '#%%' cell is one figure and only touches the loaded cache, so re-running a cell to adjust
a plot never re-runs a fit. A figure whose cache section is missing prints a message and skips.
"""

#%% setup
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import h5py as h5
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge, Rectangle

import compact.model as cm
from compact.config import Config

USETEX = True
matplotlib.rcParams.update({'text.usetex': USETEX,
                            'font.family': 'serif',
                            'font.serif': ['Computer Modern Roman'],
                            'mathtext.fontset': 'cm',
                            'axes.labelsize': 20,
                            'axes.xmargin': 0})          # x-margins zero everywhere

CACHE = 'paper_figs_cache.h5'
FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

PARS = np.asarray(Config._MEAN_PARAMS, dtype=float)      # 11-parameter model defaults

LOG_MASS_EDGES = np.array([[12, 12.01], [12.5, 12.6], [13, 13.1],
                           [13.5, 13.6], [13.9, 14.3]])
N_MASS = len(LOG_MASS_EDGES)
M_CENS = (10**LOG_MASS_EDGES[:, 0] * 10**LOG_MASS_EDGES[:, 1])**0.5
NU_FIXED = 6
R_MIN_FIT = 15.0
SIG_FLOOR = 0.1

# grid -- same as the accumulator's R / r_los edges
r_perp_edges = np.linspace(5, 140, 135 + 1)
r_los_edges = np.linspace(5, 140, 135 + 1)
r_p = 0.5 * (r_perp_edges[1:] + r_perp_edges[:-1])
r_l = 0.5 * (r_los_edges[1:] + r_los_edges[:-1])
r_p_2d, r_l_2d = np.meshgrid(r_p, r_l, indexing='ij')
r3_2d = np.sqrt(r_p_2d**2 + r_l_2d**2)
x_grid = r3_2d - r_p_2d

# Figs 2-4 display restriction
RP_MAX_PLOT, R3_MAX_PLOT = 30.0, 70.0
plot_rows = np.where(r_p <= RP_MAX_PLOT)[0]


def read_group(g):
    out = {k: (read_group(v) if isinstance(v, h5.Group) else v[()]) for k, v in g.items()}
    out['_attrs'] = dict(g.attrs)
    return out


with h5.File(CACHE, 'r') as f:
    C = read_group(f['auto']) if 'auto' in f else {}
print('cache sections:', [k for k in C if k != '_attrs'])


def bins_in(group):
    return sorted(int(k[4:]) for k in group if k.startswith('bin_'))


def decode(arr):
    return [s.decode() if isinstance(s, bytes) else str(s) for s in arr]


def jackknife_mean_err(arr):
    n = arr.shape[0]
    total = arr.sum(axis=0)
    mean_vals = total / n
    loo_means = (total - arr) / (n - 1)
    return mean_vals, np.sqrt((n - 1) / n * np.sum((loo_means - mean_vals)**2, axis=0))


def save(fig, name):
    fig.savefig(os.path.join(FIGURES_DIR, name), dpi=150)
    print(f'saved {os.path.join(FIGURES_DIR, name)}')


def row_colors(rows):
    cmap = plt.cm.viridis(np.linspace(0, 1, len(rows)))
    norm = plt.Normalize(vmin=r_p[rows[0]], vmax=r_p[rows[-1]])
    return cmap, plt.cm.ScalarMappable(cmap='viridis', norm=norm)


#%% Fig 0 -- stacked P(v_los) in one mass bin, motivating nu
if 'fig0' not in C:
    print('Fig 0: no cache section -- skipping')
else:
    d = C['fig0']
    v0, counts0 = d['v_centers'], d['counts']
    hist0 = counts0 / (counts0.sum() * (v0[1] - v0[0]))
    mu0 = np.average(v0, weights=counts0)
    width0 = np.sqrt(np.average((v0 - mu0)**2, weights=counts0))
    p_free, _ = curve_fit(cm.skewed_t_pdf, v0, hist0, p0=[mu0, width0, -0.1, 6.0], method='lm', maxfev=10_000)
    p_fixed, _ = curve_fit(lambda x, xi_, om_, al_: cm.skewed_t_pdf(x, xi_, om_, al_, NU_FIXED),
                           v0, hist0, p0=[mu0, width0, -0.1], method='lm', maxfev=10_000)
    print(f'Fig 0: free nu={p_free[3]:.2f} | fixed nu={NU_FIXED}')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(v0, hist0, 'o', ms=3, c='C0', alpha=0.5, label='data')
    vv = np.linspace(v0.min(), v0.max(), 500)
    ax.plot(vv, cm.skewed_t_pdf(vv, *p_free), '-', c='C1', label=rf'free $\nu={p_free[3]:.1f}$')
    ax.plot(vv, cm.skewed_t_pdf(vv, *p_fixed, NU_FIXED), '--', c='C2', label=rf'fixed $\nu={NU_FIXED}$')
    ax.set_xlabel(r'$v_{\rm los}$ [km/s]')
    ax.set_ylabel(r'$P(v_{\rm los})$')
    ax.legend()
    fig.tight_layout()
    save(fig, 'fig0_stacked_pvlos.png')


#%% Fig 1 -- joint (b1, b2) fit with intrinsic scatter: v_r and xi_hh, model / data
fig1_ks = bins_in(C['fig1']) if 'fig1' in C else []

if not fig1_ks:
    print('Fig 1: no cache section -- skipping')
else:
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharex='col', gridspec_kw={'height_ratios': [1.2, 1]})
    offset_step = 0.5

    def plot_ratio_bands(ax_, r, baseline, err, color):
        ax_.axhline(baseline, c=color, lw=0.7, alpha=0.5, zorder=0)
        ax_.fill_between(r, baseline - 2*err, baseline + 2*err, color=color, alpha=0.12, lw=0, zorder=0)
        ax_.fill_between(r, baseline - err, baseline + err, color=color, alpha=0.22, lw=0, zorder=0)

    for n, k in enumerate(fig1_ks):
        d = C['fig1'][f'bin_{k}']
        lo, hi = LOG_MASS_EDGES[k]
        ax[0, 0].plot(d['r_vr'], d['vr'], 'o', ms=3, c=f'C{n}', zorder=2, label=rf'$\log M \in [{lo:.2f}, {hi:.2f}]$')
        ax[0, 0].fill_between(d['r_vr'], d['vr'] - d['vr_err'], d['vr'] + d['vr_err'], color=f'C{n}', alpha=0.25, lw=0)
        ax[0, 0].plot(d['r_vr'], d['vr_model'], '-', c=f'C{n}')

        xi_r2 = d['r_xi']**2 * d['xihh']
        ax[0, 1].plot(d['r_xi'], xi_r2, 'o', ms=3, c=f'C{n}', zorder=2)
        ax[0, 1].fill_between(d['r_xi'], xi_r2 - d['r_xi']**2 * d['xi_err'], xi_r2 + d['r_xi']**2 * d['xi_err'],
                              color=f'C{n}', alpha=0.25, lw=0)
        ax[0, 1].plot(d['r_xi'], d['r_xi']**2 * d['xi_model'], '-', c=f'C{n}')

        baseline = 1 + n * offset_step
        vr_ratio = d['vr_model'] / d['vr']
        plot_ratio_bands(ax[1, 0], d['r_vr'], baseline, np.abs(vr_ratio) * d['vr_err'] / np.abs(d['vr']), f'C{n}')
        ax[1, 0].plot(d['r_vr'], vr_ratio + n*offset_step, 'o-', ms=3, c=f'C{n}', zorder=2)

        xi_ratio = d['xi_model'] / d['xihh']
        good = np.abs(d['xihh']) > d['xi_err']
        plot_ratio_bands(ax[1, 1], d['r_xi'][good], baseline, (np.abs(xi_ratio) * d['xi_err'] / np.abs(d['xihh']))[good], f'C{n}')
        ax[1, 1].plot(d['r_xi'][good], xi_ratio[good] + n*offset_step, 'o-', ms=3, c=f'C{n}', zorder=2)

    for ax_ in ax.flat:
        ax_.axvline(R_MIN_FIT, color='gray', ls=':', lw=1)
    ax[0, 0].set_ylabel(r'$\langle v_r|r\rangle$ [km/s]')
    ax[0, 1].set_ylabel(r'$r^2\,\xi_{\rm hh}(r|M)$ [$(h^{-1}\,{\rm Mpc})^2$]')
    ax[1, 0].set_ylabel('model / data (+ offset)')
    ax[1, 0].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
    ax[1, 1].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
    ax[0, 0].legend(fontsize=8, ncol=2)
    ax[1, 1].sharey(ax[1, 0])
    ax[1, 1].tick_params(labelleft=False)
    fig.tight_layout()
    save(fig, 'fig1_joint_fit.png')


#%% Fig 1 -- best-fit b1, b2 vs mass
if fig1_ks:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a_, key, c in zip(ax, ('b1', 'b2'), ('C0', 'C1')):
        a_.plot(M_CENS[fig1_ks], [C['fig1'][f'bin_{k}'][key] for k in fig1_ks], 'o-', c=c)
        a_.set_xlabel(r'$M\ [h^{-1} M_\odot]$')
        a_.set_ylabel(rf'$b_{key[1]}$')
        a_.set_xscale('log')
    fig.tight_layout()
    save(fig, 'fig1_b1_b2_vs_mass.png')


#%% Figs 2 & 3 -- sigma_v^2(r|r_perp), sigma_v^2(x), A_rho(r_perp)
if 'fig23' not in C:
    print('Figs 2 & 3: no cache section -- skipping')
else:
    d = C['fig23']
    sig23, sp = d['sig'], d['sig_params']          # sp = sigma0, A_rho0, A_rho_p, sigma_rho
    cmap, sm = row_colors(plot_rows)

    fig, ax = plt.subplots(1, 2, figsize=(11, 5), layout='constrained')
    for c, j in zip(cmap, plot_rows):
        m = (sig23[j] > SIG_FLOOR) & (r3_2d[j] <= R3_MAX_PLOT)
        ax[0].plot(r3_2d[j][m], sig23[j][m]**2, 'o', ms=3, c=c)
        ax[1].plot(x_grid[j][m], sig23[j][m]**2, 'o', ms=3, c=c)
        if m.sum() < 2:
            continue
        xs = np.linspace(x_grid[j][m].min(), x_grid[j][m].max(), 500)
        ax[1].plot(xs, cm.sigma_mod(xs, sp[0], cm.A_rho_mod(r_p[j], sp[1], sp[2]), sp[3])**2, '--', c=c, lw=1)
    ax[0].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
    ax[0].set_ylabel(r'$\sigma_v^2(r|r_\perp)$ [(km/s)$^2$]')
    ax[1].set_xlabel(r'$x = r - r_\perp$ [$h^{-1}$ Mpc]')
    ax[1].set_ylabel(r'$\sigma_v^2(x|r_\perp)$ [(km/s)$^2$]')
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    fig.colorbar(sm, ax=ax, location='right', pad=0.02, label=r'$r_\perp$ [$h^{-1}$ Mpc]')
    save(fig, 'fig2_sigmav.png')

    skip = 2
    valid = np.isfinite(d['A_rho_R'])
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    ax3.errorbar(r_p[valid][::skip], d['A_rho_R'][valid][::skip], d['A_rho_R_err'][valid][::skip],
                 fmt='o', c='k', ms=4, label=r'per-$r_\perp$ fit')
    ax3.plot(r_p, cm.A_rho_mod(r_p, sp[1], sp[2]), c='C1',
             label=rf'$A_{{\rho,0}}e^{{-(r_\perp-{cm.R_PIV_FIXED:g})/A_{{\rho,p}}}}$')
    ax3.set_xlabel(r'$r_\perp$ [$h^{-1}$ Mpc]')
    ax3.set_ylabel(r'$A_\rho(r_\perp)$')
    ax3.legend(fontsize=8)
    fig3.tight_layout()
    save(fig3, 'fig3_A_rho.png')


#%% Fig 4 -- Delta(r|r_perp), Delta(x), A_Delta(r_perp)
if 'fig23' not in C:
    print('Fig 4: no cache section -- skipping')
else:
    D4, dp = C['fig23']['Delta'], C['fig23']['del_params']   # dp = A_delta0, r_delta, sigma_delta, beta_delta, alpha
    cmap, sm = row_colors(plot_rows)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for c, j in zip(cmap, plot_rows):
        m = np.isfinite(D4[j]) & (r3_2d[j] <= R3_MAX_PLOT)
        ax[0].plot(r3_2d[j][m], D4[j][m], 'o', ms=3, c=c)
        ax[1].plot(x_grid[j][m], D4[j][m], 'o', ms=3, c=c)
        if m.sum() < 2:
            continue
        xs = np.linspace(x_grid[j][m].min(), x_grid[j][m].max(), 500)
        ax[1].plot(xs, cm.delta_mod_alpha(xs, cm.A_delta_mod(r_p[j], dp[0], dp[1]), *dp[2:]), '--', c=c, lw=1)
    ax[0].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
    ax[0].set_ylabel(r'$\Delta(r|r_\perp)$ [km/s]')
    ax[1].set_xlabel(r'$x = r - r_\perp$ [$h^{-1}$ Mpc]')
    ax[1].set_ylabel(r'$\Delta(x)$ [km/s]')
    fig.colorbar(sm, ax=ax, location='right', pad=0.02, label=r'$r_\perp$ [$h^{-1}$ Mpc]')
    save(fig, 'fig4_delta.png')

    skip = 2
    d = C['fig23']
    valid = np.isfinite(d['A_delta_R'])
    fig5, ax5 = plt.subplots(figsize=(6, 5))
    ax5.errorbar(r_p[valid][::skip], d['A_delta_R'][valid][::skip], d['A_delta_R_err'][valid][::skip],
                 fmt='o', c='k', ms=4, label=r'per-$r_\perp$ fit')
    ax5.plot(r_p, cm.A_delta_mod(r_p, dp[0], dp[1]), c='C1', label=r'$A_{\Delta,0}\,e^{-r_\perp/r_\Delta}$')
    ax5.set_xlabel(r'$r_\perp$ [$h^{-1}$ Mpc]')
    ax5.set_ylabel(r'$A_\Delta(r_\perp)$')
    ax5.legend(fontsize=8)
    fig5.tight_layout()
    save(fig5, 'fig4b_A_delta.png')


#%% Fig 5 -- the 11 velocity-model parameters vs mass, b1, b2 (per-phase fits + jackknife mean)
LABELS = [r'$\sigma_0$ [km/s]', r'$A_{\rho,0}$', r'$A_{\rho,p}$ [$h^{-1}$ Mpc]', r'$\sigma_\rho$ [$h^{-1}$ Mpc]',
          r'$A_{\Delta,0}$', r'$r_\Delta$ [$h^{-1}$ Mpc]', r'$\sigma_\Delta$ [$h^{-1}$ Mpc]',
          r'$\beta_\Delta$', r'$\alpha_\Delta$', r'$\nu_0$', r'$r_c$ [$h^{-1}$ Mpc]']

P5, S5 = {}, {}
for k in (bins_in(C['fig5']) if 'fig5' in C else []):
    P5[k] = C['fig5'][f'bin_{k}']['params']
    S5[k] = decode(C['fig5'][f'bin_{k}']['sims'])


def param_panels(x_by_bin, xlabel, fname, logx=False, bands=False):
    """3x4 grid, one panel per parameter: thin lines = individual phases, black points = jackknife mean."""
    ks = sorted(set(P5) & set(x_by_bin))
    if not ks:
        print(f'{fname}: no bins available -- skipping')
        return
    xs = np.array([x_by_bin[k] for k in ks])
    fig, axs = plt.subplots(3, 4, figsize=(20, 13), layout='constrained')
    axs = axs.ravel()
    all_sims = sorted(set(s for k in ks for s in S5[k]))
    for i, label in enumerate(LABELS):
        a_ = axs[i]
        for sim in all_sims:
            ks_s = [k for k in ks if sim in S5[k]]
            a_.plot([x_by_bin[k] for k in ks_s], [P5[k][S5[k].index(sim), i] for k in ks_s], marker='o', alpha=0.3)
        mean_, jack_ = np.array([jackknife_mean_err(P5[k][:, i]) for k in ks]).T
        a_.errorbar(xs, mean_, yerr=jack_, marker='o', color='k', ls='none', capsize=3, zorder=5,
                    label='jackknife mean' if i == 0 else None)
        if bands:
            sd = np.array([P5[k][:, i].std() for k in ks])
            a_.plot(xs, mean_ + sd, 'k--', lw=2, label=r'$\pm$ standard dev over phases' if i == 0 else None)
            a_.plot(xs, mean_ - sd, 'k--', lw=2)
            a_.axhline(np.concatenate([P5[k][:, i] for k in ks]).mean(), c='k', ls=':', lw=3,
                       label='mean over all phases, bins' if i == 0 else None)
        if logx:
            a_.set_xscale('log')
            a_.set_xlim(xs.min() / 1.5, xs.max() * 1.5)
        a_.set_ylabel(label)
        if i >= 7:
            a_.set_xlabel(xlabel)
        else:
            a_.tick_params(labelbottom=False)
    h_, l_ = axs[0].get_legend_handles_labels()
    axs[11].axis('off')
    axs[11].legend(h_, l_, fontsize=20, loc='center')
    save(fig, fname)


if not P5:
    print('Fig 5: no cache section -- skipping')
else:
    param_panels({k: M_CENS[k] for k in P5}, r'$M\ [h^{-1} M_\odot]$', 'fig5_mass_dependence.png', logx=True, bands=True)
    grand = np.concatenate([P5[k] for k in P5], axis=0).mean(axis=0)
    print('grand-mean parameters vs Config._MEAN_PARAMS (fractional difference):')
    print(np.round(grand / PARS - 1, 4))

    for key in ('b1', 'b2'):
        xb = {k: C['fig1'][f'bin_{k}'][key] for k in fig1_ks}
        param_panels(xb, rf'$b_{key[1]}$', f'fig5_params_vs_{key}.png')

    # A_Delta,0 vs alpha_Delta
    ks = sorted(set(P5) & set(fig1_ks))
    fig, ax = plt.subplots(figsize=(6, 5))
    for sim in sorted(set(s for k in ks for s in S5[k])):
        ks_s = [k for k in ks if sim in S5[k]]
        ax.plot([P5[k][S5[k].index(sim), 8] for k in ks_s], [P5[k][S5[k].index(sim), 4] for k in ks_s], marker='o', alpha=0.3)
    mj = np.array([jackknife_mean_err(P5[k]) for k in ks])        # (bins, 2, 11)
    ax.errorbar(mj[:, 0, 8], mj[:, 0, 4], xerr=mj[:, 1, 8], yerr=mj[:, 1, 4], marker='o', color='k', ls='none',
                zorder=5, label='jackknife mean')
    ax.set_xlabel(LABELS[8])
    ax.set_ylabel(LABELS[4])
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, 'fig5_Adelta0_vs_alpha.png')


#%% Fig 6 -- cosmology dependence at z=0.2: grid-sim fits (points) vs emulator lines, three figures (sigma_v, Delta, nu)
# Cache contract (auto/fig6): cosmo (N, 4) in emulator order [omega_b, omega_cdm, ln(1e10 A_s), n_s];
# params (N, 2, 11) = the 11 model parameters per grid sim for [low-M, high-M] at z = 0.2 (NaN where a fit failed).
Z6 = 0.2
COSMO_SHOW = [2, 1]                       # columns plotted: ln(1e10 A_s), omega_cdm (add 3 for n_s, 0 for omega_b)
COSMO_LABELS = [r'$\omega_{\rm b}$', r'$\omega_{\rm cdm}$', r'$\log(10^{10}A_s)$', r'$n_s$']
COL_LO6, COL_HI6, COL_AVG6 = 'C0', 'C3', 'C4'      # blue, red, purple
LEG6 = [r'low-M ($12.5<\log_{10}M<12.6$)', r'high-M ($13.5<\log_{10}M<13.6$)', 'average']
H_DUMMY = 0.6736                          # h is not used by the emulator, only needed to build the cosmo tuple

if 'fig6' not in C:
    print('Fig 6: no cache section -- skipping')
else:
    from compact.predictions import CompactStreamingModel
    emu = CompactStreamingModel(Z6, Config)
    cos6, par6 = C['fig6']['cosmo'], C['fig6']['params']
    fid6 = np.asarray(Config.fiducial_cosmo, dtype=float)

    def emu_pars(theta, scheme):
        """Emulator parameter vector at z=Z6 for theta = [omega_b, omega_cdm, ln(1e10 A_s), n_s]."""
        cosmo = (H_DUMMY, theta[0], theta[1], np.exp(theta[2]) / 1e10, theta[3])
        return emu._compact_params_at_node(Z6, cosmo, scheme)

    schemes = ('low_M', 'high_M', 'avg_M')
    colors = (COL_LO6, COL_HI6, COL_AVG6)
    # average-emulator prediction at every grid sim's full cosmology, and the mean of the two bins' fits
    pred_avg = np.array([emu_pars(t, 'avg_M') for t in cos6])
    data_avg = np.nanmean(par6, axis=1)

    def cosmo_figure(p_idx, labels, fname):
        n_p, n_c = len(p_idx), len(COSMO_SHOW)
        fig = plt.figure(figsize=(3.5 * n_p + 0.5, 4.8 * n_c))
        outer = fig.add_gridspec(n_c, 1, hspace=0.28)
        for r_, c_col in enumerate(COSMO_SHOW):
            gs = outer[r_].subgridspec(2, n_p, height_ratios=[3, 1], hspace=0.08, wspace=0.45)
            xs = np.linspace(cos6[:, c_col].min(), cos6[:, c_col].max(), 100)
            lines = {}
            for s in schemes:
                th = np.tile(fid6, (len(xs), 1))
                th[:, c_col] = xs
                lines[s] = np.array([emu_pars(t, s) for t in th])
            for c, p in enumerate(p_idx):
                ax = fig.add_subplot(gs[0, c])
                axr = fig.add_subplot(gs[1, c], sharex=ax)
                for k, col in enumerate((COL_LO6, COL_HI6)):
                    ax.plot(cos6[:, c_col], par6[:, k, p], 'o', ms=4, c=col, alpha=0.6, mec='none')
                for s, col, lab in zip(schemes, colors, LEG6):
                    ax.plot(xs, lines[s][:, p], c=col, lw=1.8, label=lab)
                ax.set_ylabel(labels[c])
                ax.tick_params(labelbottom=False)

                resid = (pred_avg[:, p] / data_avg[:, p] - 1)
                axr.axhspan(-0.10, 0.10, color='gray', alpha=0.2, lw=0)
                axr.axhspan(-0.05, 0.05, color='gray', alpha=0.3, lw=0)
                axr.axhline(0, c='k', ls='--', lw=1)
                axr.plot(cos6[:, c_col], resid, 'o', ms=3, c=COL_AVG6)
                axr.set_ylim(-0.15, 0.15)
                axr.set_xlabel(COSMO_LABELS[c_col])
                if c == 0:
                    axr.set_ylabel(r'perc. diff.')
                if r_ == 0 and c == 0:
                    ax.legend(fontsize=7, loc='upper left')
        save(fig, fname)

    cosmo_figure([0, 1, 2, 3], LABELS[0:4], 'fig6_sigma.png')
    cosmo_figure([4, 5, 6, 7, 8], LABELS[4:9], 'fig6_delta.png')
    cosmo_figure([9, 10], LABELS[9:11], 'fig6_nu.png')




#%% Fig 7 setup -- shared by the diagonal (six-panel) and 3x3 figures
# P(v_los | r_perp, r_los) in mass bin 2: histogram vs skew-t model (Config._MEAN_PARAMS) vs Gaussian with the model's mean and variance
V_LIM, V_PLOT = 2500.0, 2000.0     # model-curve range / displayed x-range [km/s]
MIN_COUNT = 1                      # rebinned v-bins with fewer pairs are masked
Y_FLOOR = 1e-5                     # log-y lower limit, as a fraction of the panel's peak
R_INSET = 85.0                    # quarter-disk radius in the inset
INSET_PAD = 0.0 * R_INSET
RING_W = 5.0                       # schematic width of the quarter ring at the bin's r [h^-1 Mpc]
REBIN = 5
 
def gauss_pdf(v, mu, sd):
    return np.exp(-0.5 * ((v - mu) / sd)**2) / (np.sqrt(2 * np.pi) * sd)
 
 
if 'fig7' not in C:
    print('Fig 7: no cache section -- skipping')
else:
    d7 = C['fig7']
    RP_SEL, RL_SEL = d7['rp_sel'], d7['rl_sel']
    g7 = d7['bin']
    vr_spline = InterpolatedUnivariateSpline(g7['r_tab'], g7['vr_model'], k=3, ext=3)
    v7 = d7['v_centers']
    cent = v7.reshape(-1, REBIN).mean(axis=1)
    dv = (v7[1] - v7[0]) * REBIN
    edges = np.r_[cent - dv / 2, cent[-1] + dv / 2]
    vv = np.linspace(-V_LIM, V_LIM, 600)
    theta = np.linspace(0, np.pi / 2, 100)
    legend_handles = [Line2D([], [], c='k', lw=1.5, label='AbacusSummit (stacked phases)'),
                      Line2D([], [], c='C0', lw=2, label='Skew-t model'),
                      Line2D([], [], c='C1', lw=2, label='Gaussian')]
 
    def vlos_panel(ax_, rp_t, rl_t, logy, fancy_inset=False, inset=True):
        j, i = int(np.argmin(np.abs(r_p - rp_t))), int(np.argmin(np.abs(r_l - rl_t)))
        counts = g7[f'cell_{j}_{i}']['counts'].reshape(-1, REBIN).sum(axis=1)
        hist = counts / (counts.sum() * dv)
        r3 = r3_2d[j, i]
        mu = vr_spline(r3) * r_l[i] / r3                                   # skew-t mean (by construction)
        sig = cm.sigma_mod(r3 - r_p[j], PARS[0], cm.A_rho_mod(r_p[j], PARS[1], PARS[2]), PARS[3])   # skew-t std
        pdf_c = cm.Pv_compact(vv, r_p[j], r_l[i], vr_spline, PARS)
        pdf_g = gauss_pdf(vv, mu, sig)                                     # same two lowest moments
        ax_.stairs(np.where(counts >= MIN_COUNT, hist, 1e-30), edges, color='k', lw=1.2)
        ax_.plot(vv, pdf_g, c='C1', lw=2)
        ax_.plot(vv, pdf_c, c='C0', lw=2)
        peak = hist.max() if logy else max(hist.max(), pdf_c.max(), pdf_g.max())
        ax_.axvline(0, c='gray', ls=':', lw=1)
        ax_.set_xlim(-V_PLOT, V_PLOT)
        if logy:
            ax_.set_yscale('log')
            ax_.set_ylim(peak * Y_FLOOR, peak * 2)
        else:
            ax_.set_ylim(0, 1.1 * peak)
        ax_.set_title(rf'$r_\perp\in[{r_perp_edges[j]:.2f}, {r_perp_edges[j+1]:.2f}]$, '
                      rf'$r_{{\rm los}}\in[{r_los_edges[i]:.2f}, {r_los_edges[i+1]:.2f}]$ '
                      r'$h^{-1}$Mpc', fontsize=12)

        if not inset:
            return
 
        # inset: where this bin sits on the (R, r_los) quarter-disk
        if fancy_inset:
            ins = ax_.inset_axes([0.78, 0.60, 0.20, 0.36])
            ins.set_box_aspect(1)                                  # square
            ins.set_anchor('NE')
            lim = 1.1 * R_INSET                                    # disk stays inside the axes, so the arc is visible
            ins.add_patch(Wedge((0, 0), R_INSET, 0, 90, fc='gray', ec='none', alpha=0.3))
            ins.add_patch(Wedge((0, 0), np.hypot(r_p[j], r_l[i]) + RING_W / 2, 0, 90, width=RING_W,
                                fc='gray', ec='none', alpha=0.8))  # quarter ring at this bin's r
            ins.set_xlim(-INSET_PAD, lim)
            ins.set_ylim(-INSET_PAD, lim)
            ins.spines[['top', 'right']].set_visible(False)
            lo = -INSET_PAD                                        # lines run from the axes to the intersection point only
            ins.add_patch(Rectangle((r_perp_edges[j], lo), r_perp_edges[j + 1] - r_perp_edges[j], r_l[i] - lo,
                                    fc='k', alpha=0.4, lw=0))
            ins.add_patch(Rectangle((lo, r_los_edges[i]), r_p[j] - lo, r_los_edges[i + 1] - r_los_edges[i],
                                    fc='k', alpha=0.4, lw=0))
            ins.vlines(r_p[j], lo, r_l[i], colors='k', lw=1)
            ins.hlines(r_l[i], lo, r_p[j], colors='k', lw=1)
        else:
            ins = ax_.inset_axes([0.82, 0.72, 0.16, 0.24])
            ins.fill(np.r_[-INSET_PAD, R_INSET, R_INSET * np.cos(theta), -INSET_PAD],
                     np.r_[-INSET_PAD, -INSET_PAD, R_INSET * np.sin(theta), R_INSET],
                     c='gray', alpha=0.4, lw=0)
            ins.set_xlim(-INSET_PAD, R_INSET)
            ins.set_ylim(-INSET_PAD, R_INSET)
            ins.axvspan(r_perp_edges[j], r_perp_edges[j + 1], color='k', alpha=0.4, lw=0)
            ins.axhspan(r_los_edges[i], r_los_edges[i + 1], color='k', alpha=0.4, lw=0)
            ins.axvline(r_p[j], c='k', lw=1)
            ins.axhline(r_l[i], c='k', lw=1)
        ins.set_xticks([])
        ins.set_yticks([])
        ins.set_xlabel(r'$r_\perp$', fontsize=15, labelpad=1)
        ins.set_ylabel(r'$r_{\rm los}$', fontsize=15, labelpad=1)

#%% Fig 7a -- r_los = r_perp = r_ii bins, one column per r_ii: linear y (top row), log y (bottom row); inset in the top row only
if 'fig7' in C:
    n_ii = len(RP_SEL)
    fig, axs = plt.subplots(2, n_ii, figsize=(5 * n_ii, 8), layout='constrained', sharex=True)
    for n in range(n_ii):
        for r_, logy in enumerate((False, True)):
            vlos_panel(axs[r_, n], RP_SEL[n], RL_SEL[n], logy, fancy_inset=True, inset=(r_ == 0))
    for ax_ in axs[-1]:
        ax_.set_xlabel(r'$v_{\rm los}$ [km/s]')
    axs[1, 0].legend(handles=legend_handles, fontsize=12, loc='lower center')
    fig.supylabel(r'$P(v_{\rm los}\,|\,r_\perp, r_{\rm los})$', fontsize=35)
    save(fig, 'fig7a_diagonal_lin_log.png')

#%% Fig 7b -- 3x3 grid over the same r_perp, r_los values, linear y-axis
if 'fig7' in C:
    nr, nc = len(RL_SEL), len(RP_SEL)
    fig, axs = plt.subplots(nr, nc, figsize=(5 * nc, 4 * nr), layout='constrained', sharex=True)
    axs = np.atleast_2d(axs)
    for a, rl_t in enumerate(RL_SEL):
        for b_, rp_t in enumerate(RP_SEL):
            vlos_panel(axs[a, b_], rp_t, rl_t, logy=False, fancy_inset=True)
            if a == nr - 1:
                axs[a, b_].set_xlabel(r'$v_{\rm los}$ [km/s]')
    axs[0, 0].legend(handles=legend_handles, fontsize=12, loc='lower right')
    fig.supylabel(r'$P(v_{\rm los}\,|\,r_\perp, r_{\rm los})$', fontsize=35)
    save(fig, 'fig7b_grid_linear.png')

#%% Diagnostic -- <v_los^2> vs r: simulation vs compact sigma_v model vs velocileptors (mass bin 2, same phase as Figs 2-4)
# Everything is a second moment about zero: sim = sum v^2 / n; compact = sigma_v^2 + mu^2 (mu = LPT mean v_r projected on
# the LOS); velocileptors = conv^2 f^2 (s0 + P2 s2)/(1+xi), pwv_variance without its mean subtraction.
R_LO_D, R_HI_D = 15.0, 130.0
RP_DIAG = [5.5, 15.5, 30.5, 60.5]
 
if 'lpt_sigma' not in C or 'fig23' not in C:
    print('sigma_v vs velocileptors: needs cache sections lpt_sigma and fig23 -- skipping')
else:
    dl, sp = C['lpt_sigma'], C['fig23']['sig_params']
    vr_d = InterpolatedUnivariateSpline(dl['r_tab'], dl['vr_model'], k=3, ext=3)
    n_c = len(RP_DIAG)
    fig, axs = plt.subplots(2, n_c, figsize=(4.5 * n_c, 7), sharex='col', layout='constrained',
                            gridspec_kw={'height_ratios': [3, 1]})
    for c, rp_t in enumerate(RP_DIAG):
        j = int(np.argmin(np.abs(r_p - rp_t)))
        r3 = r3_2d[j]
        mu = vr_d(r3) * r_l / r3
        x = r3 - r_p[j]
        m2_fit = cm.sigma_mod(x, sp[0], cm.A_rho_mod(r_p[j], sp[1], sp[2]), sp[3])**2 + mu**2
        m2_def = cm.sigma_mod(x, PARS[0], cm.A_rho_mod(r_p[j], PARS[1], PARS[2]), PARS[3])**2 + mu**2
        m2_sim, m2_lpt, m2_lfit = dl['m2_meas'][j], dl['m2_lpt'][j], dl['m2_lpt_fit'][j]
        ok = (r3 >= R_LO_D) & (r3 <= R_HI_D) & np.isfinite(m2_sim)
 
        ax, axr = axs[0, c], axs[1, c]
        ax.plot(r3[ok], m2_sim[ok], 'o', ms=3, c='k', label='simulation')
        ax.plot(r3[ok], m2_fit[ok], c='C0', lw=2, label=r'compact, bin-2 $\sigma_v$ fit')
        ax.plot(r3[ok], m2_def[ok], c='C2', lw=2, ls='--', label=r'compact, default parameters')
        ax.plot(r3[ok], m2_lpt[ok], c='C3', lw=2, ls=':', label='velocileptors (no counterterms)')
        ax.plot(r3[ok], m2_lfit[ok], c='C1', lw=2, ls='-.', label=r'velocileptors + fitted $s_{2,\rm fog}$')
        axr.axhline(0, c='k', ls='--', lw=1)
        for m2_, col, ls in ((m2_fit, 'C0', '-'), (m2_def, 'C2', '--'), (m2_lpt, 'C3', ':'), (m2_lfit, 'C1', '-.')):
            axr.plot(r3[ok], (m2_ - m2_sim)[ok], c=col, lw=2, ls=ls)
        ax.set_title(rf'$r_\perp\in[{r_perp_edges[j]:.0f}, {r_perp_edges[j+1]:.0f}]$ $h^{{-1}}$Mpc', fontsize=12)
        axr.set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
        axr.set_xlim(R_LO_D, R_HI_D)
        if c == 0:
            ax.set_ylabel(r'$\langle v_{\rm los}^2\rangle$ [(km/s)$^2$]')
            axr.set_ylabel('model $-$ sim [(km/s)$^2$]')
            ax.legend(fontsize=8, loc='upper right')
    print(f"velocileptors fitted s2fog = {float(dl['s2fog']):.4g} (Mpc/h)^2")
    save(fig, 'diag_sigma_v_vs_velocileptors.png')

 

#%% Fig 8 -- 2D redshift-space xi^s(s_perp, s_los), mass bin 2: simulation vs skew-t model vs Gaussian (contours), skew-t residual map
# Simulation: xi^s measured directly from the halo catalogs with Corrfunc (every phase, mean + phase scatter). Models: streaming
# integral over the full real-space grid with the measured xi_hh(r) as the (1+xi) weight (build_paper_cache.py, section xi_s).
LEVELS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0]
S_MIN, S_MAX = 0.0, 40.0    # both axes span S_MIN -> S_MAX
SMOOTH_SIGMA = 1.0      # Gaussian smoothing of the SIMULATION contour map only, in pixels (1 h^-1 Mpc); 0 = off. The residual map is unsmoothed.
SIGMA_MODE = 'mean'     # residual normalisation: 'mean' = phase scatter / sqrt(N_phases) (error on the sim mean); 'single' = single-phase scatter
RESID_LIM = 3.0
 
if 'xi_s' not in C:
    print('Fig 8: no cache section xi_s -- skipping')
else:
    from scipy.ndimage import gaussian_filter
    dx = C['xi_s']
    print(f"Fig 8: simulation map is the mean of {int(dx['n_phases'])} phases")
    e_perp, e_par = dx['s_perp_edges'], dx['s_par_edges']
    c_perp, c_par = 0.5 * (e_perp[1:] + e_perp[:-1]), 0.5 * (e_par[1:] + e_par[:-1])
    tr = lambda Z: Z.T                                   # (s_perp, s_par) -> plotted as (s_par rows, s_perp columns)
 
    def diagonal_label_points(cs):
        """For each contour level, the vertex closest to the s_perp = s_los diagonal (levels that miss the plot are skipped)."""
        pts = []
        for segs in cs.allsegs:
            v = np.vstack(segs) if len(segs) else np.empty((0, 2))
            v = v[(v[:, 0] >= S_MIN) & (v[:, 0] <= S_MAX) & (v[:, 1] >= S_MIN) & (v[:, 1] <= S_MAX)]
            if len(v):
                pts.append(tuple(v[np.argmin(np.abs(v[:, 0] - v[:, 1]))]))
        return pts
 
    def frame(ax):
        ax.set_xlim(S_MIN, S_MAX)
        ax.set_ylim(S_MIN, S_MAX)
        ax.set_xlabel(r'$s_\perp$ [$h^{-1}$ Mpc]')
        ax.set_ylabel(r'$s_{\rm los}$ [$h^{-1}$ Mpc]')
 
    sim_map = gaussian_filter(dx['xi_sim'], SMOOTH_SIGMA, mode='nearest') if SMOOTH_SIGMA > 0 else dx['xi_sim']
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for Z, col, ls, lw in ((sim_map, 'k', '-', 3.0), (dx['xi_skewt'], 'C0', '-', 1.8), (dx['xi_gauss'], 'C1', '--', 1.8)):
        cs = ax.contour(c_perp, c_par, tr(Z), levels=LEVELS, colors=col, linestyles=ls, linewidths=lw)
        if col == 'k':
            ax.clabel(cs, manual=diagonal_label_points(cs), fmt='%g', fontsize=9, inline=True)
    frame(ax)
    ax.legend(handles=[Line2D([], [], c='k', lw=3, label=r'\textsc{AbacusSummit} (stacked phases)'),
                       Line2D([], [], c='C0', lw=1.8, label='Skew-t model'),
                       Line2D([], [], c='C1', lw=1.8, ls='--', label='Gaussian')],
              fontsize=9, loc='upper right')
    ax.set_title(r'$\xi(s_\perp, s_{\rm los})$')
    fig.tight_layout()
    save(fig, 'fig8a_xi_s_contours.png')
 
    sigma = dx['xi_sim_std'] / (np.sqrt(float(dx['n_phases'])) if SIGMA_MODE == 'mean' else 1.0)
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 6.5), sharey=True, layout='constrained')
    for ax, key, title in zip(axs, ('xi_skewt', 'xi_gauss'), ('Skew-t model', 'Gaussian')):
        resid = np.where(sigma > 0, (dx[key] - dx['xi_sim']) / np.where(sigma > 0, sigma, 1), np.nan)
        pm = ax.pcolormesh(e_perp, e_par, tr(resid), cmap='RdBu_r', vmin=-RESID_LIM, vmax=RESID_LIM, rasterized=True)
        frame(ax)
        ax.set_title(title)
    axs[1].set_ylabel('')
    fig.colorbar(pm, ax=axs, location='top', label=r'$(\xi^s_{\rm model}-\langle\xi^s_{\rm sim}\rangle)/\sigma_\xi$')
    save(fig, 'fig8b_xi_s_residual.png')
 
 
 
#%% Fig 9 -- redshift-space multipoles xi_ell(s), ell = 0, 2, 4, out to s = 130: simulation vs skew-t (fit to ell = 0, 2) vs Gaussian
# Simulation: direct pycorr measurement (mean over phases); the grey band is the SINGLE-phase scatter (std over phases).
# Models (production streaming integral, LPT xi_real and v_r at the Fig 1 b1, b2): skew-t with all 13 parameters fit to the simulation
# xi_0 and xi_2 over s >= fit_s_min (xi_4 is a prediction), the same skew-t at its default parameters (Config._MEAN_PARAMS), and the
# Gaussian with the default skew-t's mean and variance (never fit). Top: s^2 xi_ell. Bottom: s^2 (model - sim).
if 'multipoles' not in C or 'multipoles_fit' not in C:
    print('Fig 9: needs cache sections multipoles and multipoles_fit -- skipping')
else:
    dm, df = C['multipoles'], C['multipoles_fit']
    s_m, ells = dm['s'], [int(l) for l in dm['ells']]
    sim = dm['xi_sim_phases'].mean(axis=0)
    sig = dm['xi_sim_phases'].std(axis=0, ddof=1)                   # single-phase scatter
    print(f"Fig 9: {int(dm['n_phases'])} phases; fit chi2 {float(df['chi2_default']):.0f} -> {float(df['chi2_fit']):.0f} "
          f"for {int(df['dof'])} dof (s >= {float(df['fit_s_min']):g})")
    fig, axs = plt.subplots(2, len(ells), figsize=(5.5 * len(ells), 8), sharex=True, layout='constrained',
                            gridspec_kw={'height_ratios': [3, 1.3]})
    for c, l in enumerate(ells):
        s2 = s_m**2
        ax, axr = axs[0, c], axs[1, c]
        ax.fill_between(s_m, s2 * (sim[c] - sig[c]), s2 * (sim[c] + sig[c]), color='k', alpha=0.25, lw=0)
        ax.plot(s_m, s2 * sim[c], c='k', lw=1.5)
        ax.plot(s_m, s2 * df['xi_default'][c], c='C0', lw=1.8, ls='--')
        ax.plot(s_m, s2 * df['xi_fit'][c], c='C0', lw=2)
        ax.plot(s_m, s2 * df['xi_gauss'][c], c='C1', lw=2)
        ax.set_title(rf'$\ell={l}$' + (' (fit)' if l in (0, 2) else ' (prediction)'))
        ax.set_ylabel(r'$s^2\,\xi_\ell(s)$ [$(h^{-1}\,{\rm Mpc})^2$]')
        axr.axhline(0, c='k', lw=1)
        axr.fill_between(s_m, -s2 * sig[c], s2 * sig[c], color='k', alpha=0.25, lw=0)
        axr.plot(s_m, s2 * (df['xi_default'][c] - sim[c]), c='C0', lw=1.8, ls='--')
        axr.plot(s_m, s2 * (df['xi_fit'][c] - sim[c]), c='C0', lw=2)
        axr.plot(s_m, s2 * (df['xi_gauss'][c] - sim[c]), c='C1', lw=2)
        for a_ in (ax, axr):
            a_.axvline(float(df['fit_s_min']), c='gray', ls=':', lw=1)
        axr.set_xlabel(r'$s$ [$h^{-1}$ Mpc]')
        axr.set_ylabel(r'$s^2\,\Delta\xi_\ell$ (model $-$ sim)')
        axr.set_xlim(s_m[0], s_m[-1])
    axs[0, 0].legend(handles=[Line2D([], [], c='k', lw=1.5, label='AbacusSummit (mean; band = single-phase scatter)'),
                              Line2D([], [], c='C0', lw=2, label=r'Skew-t model (fit to $\ell=0,2$)'),
                              Line2D([], [], c='C0', lw=1.8, ls='--', label='Skew-t model (default parameters)'),
                              Line2D([], [], c='C1', lw=2, label='Gaussian (default parameters)')], fontsize=9, loc='upper right')
    save(fig, 'fig9_multipoles.png')
