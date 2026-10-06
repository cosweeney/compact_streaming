"""
Checks the mass dependence of the sigma_v / Delta compact-model parameters
as auto-statistics data comes in from gather_auto_stats.py.

Model definitions matched EXACTLY to vel_mod_mass_dep_plot.py (the current,
updated reference) rather than the earlier 2-parameter Delta guess:
  - sigma_v: 4 params (sigma0, A_piv, b, sigma_rho) -- unchanged from before.
  - Delta: 5 params (A_delta0, r_delta, sigma_delta, beta_delta, alpha) --
    an r_perp-dependent amplitude A_delta_mod(R) = A_delta0/(1+(R/r_delta)^2)
    and a free exponent alpha on x, via delta_mod_alpha_R. NOT the old
    2-parameter A_delta*x/(sigma_delta+x) form.

Difference from vel_mod_mass_dep_plot.py's own data flow: that script reads
already-fit ST_pars/meas_sigv out of an existing pdf_fits.h5 (built by a
separate, already-run pipeline for the cross-statistic data). There's no
equivalent cache yet for the new auto-statistics, so this script still has
to fit the skew-t PDF per (R, rlos) cell itself, from the raw PairHist
files gather_auto_stats.py produces -- that per-cell fit is the expensive
step, so it's what gets parallelized below, at the (phase, mass bin) level
(each task does its own bin's ~135x135 cell loop sequentially) rather than
nesting parallelism inside each bin as well.

b1/b2 (the xi_hh bias fit) is intentionally omitted per your last message.

I have not run this myself -- no cluster access -- so treat this as a
structurally-complete first draft to check over, same as everything else
in this pipeline.
"""

#%% imports and setup
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import t as student_t
from scipy.interpolate import InterpolatedUnivariateSpline
from joblib import Parallel, delayed
import psutil
psutil.Process().nice(10)

from compact.velocities.pairhist import PairHist

OUT_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base/halos/'
            'AbacusSummit_base_{sim}/halos/z0.200/data/pairhist_auto_mbin{k}.h5')
ALL_PLANNED_PHASES = [f'c000_ph{i:03d}' for i in range(0, 25)]
N_MASS = 6
LOG_MASS_EDGES = np.array([[12, 12.01], [12.5, 12.6], [13, 13.1],
                           [13.5, 13.6], [14.0, 14.2], [14.4, 14.7]])
M_CENS = (10**LOG_MASS_EDGES[:, 0] * 10**LOG_MASS_EDGES[:, 1])**0.5

N_JOBS = -1   # this is a standalone checking script, not sharing a cluster
              # job's resources the way the raw accumulation scripts do --
              # adjust down if you're running this alongside anything else

# grid -- matches gather_auto_stats.py's R_edges/rlos_edges exactly
r_perp_edges = np.linspace(5, 140, 135 + 1)
r_los_edges = np.linspace(5, 140, 135 + 1)
r_p = 0.5 * (r_perp_edges[1:] + r_perp_edges[:-1])
r_l = 0.5 * (r_los_edges[1:] + r_los_edges[:-1])
r_p_2d, r_l_2d = np.meshgrid(r_p, r_l, indexing='ij')
r3_2d = np.sqrt(r_p_2d**2 + r_l_2d**2)
x_grid = r3_2d - r_p_2d

VR_R_EDGES = np.arange(2.0, 142.0, 2.0)


# ---------------------------------------------------------------- sigma_v model
# unchanged from vel_mod_mass_dep_plot.py / gather_paper_data.py
SIG_FLOOR = 0.1
DEL_FLOOR = -1000.0
DEL_P0 = [125.0, 50.0, 10.0, 2.0, 1.0]   # A_delta0, r_delta, sigma_delta, beta_delta, alpha


def sigma_mod(x, sigma0, A_rho, sigma_rho):
    arg = 2.0 * sigma0**2 * (1.0 - A_rho * np.exp(-x / sigma_rho))
    return np.sqrt(np.clip(arg, 1e-8, None))


def A_rho_mod(R, A_piv, b, R_piv):
    return A_piv * np.exp(-(R - R_piv) / b)


def sigma_mod_2d(coords, sigma0, A_piv, b, sigma_rho, R_piv):
    x, R = coords
    return sigma_mod(x, sigma0, A_rho_mod(R, A_piv, b, R_piv), sigma_rho)


def _seed_sigma(x, R, sig, R_piv):
    sigma0 = sig.max() / np.sqrt(2.0) * 1.02
    y = 2.0 * sigma0**2 - sig**2
    m = y > 0
    G = np.column_stack([np.ones(m.sum()), -(R[m] - R_piv), -x[m]])
    c = np.linalg.lstsq(G, np.log(y[m]), rcond=None)[0]
    b = 1.0 / c[1] if c[1] > 0 else np.ptp(R) + 1.0
    sigma_rho = 1.0 / c[2] if c[2] > 0 else 10.0
    A_piv = np.clip(np.exp(c[0]) / (2.0 * sigma0**2), 1e-3, 0.99)
    return [sigma0, A_piv, b, sigma_rho]


def fit_sigma_joint(x, r_p_arr, sig):
    R = np.broadcast_to(r_p_arr[:, None], x.shape)
    m = np.isfinite(x) & np.isfinite(sig) & (sig > SIG_FLOOR)
    X, RR, S = x[m], R[m], sig[m]
    R_piv = np.median(RR)
    popt, _ = curve_fit(
        lambda c, s0, Ap, b, sr: sigma_mod_2d(c, s0, Ap, b, sr, R_piv),
        (X, RR), S, p0=_seed_sigma(X, RR, S, R_piv),
        bounds=([0.0, 1e-4, 1e-3, 1e-3], [np.inf, 1.0, np.inf, np.inf]), maxfev=20_000)
    return popt


# ------------------------------------------------------- Delta model, 5 params
def A_delta_mod(R, A_delta0, r_delta):
    return A_delta0 / (1 + (R / r_delta)**2)


def delta_mod_alpha(x, A, sigma_delta, beta_delta, alpha):
    return A * x**alpha / (1 + (x / sigma_delta)**beta_delta)


def delta_mod_alpha_R(coords, A_delta0, r_delta, sigma_delta, beta_delta, alpha):
    x, R = coords
    return delta_mod_alpha(x, A_delta_mod(R, A_delta0, r_delta), sigma_delta, beta_delta, alpha)


def fit_delta_joint(x, r_p_arr, D, p0=DEL_P0):
    R = np.broadcast_to(r_p_arr[:, None], x.shape)
    m = np.isfinite(x) & np.isfinite(D) & (D > DEL_FLOOR)
    X, RR, DD = x[m], R[m], D[m]
    popt, _ = curve_fit(delta_mod_alpha_R, (X, RR), DD, p0=p0, maxfev=20_000)
    return popt


# ------------------------------------------------- skew-t per-cell fit + meas_sigv
def skewed_t_pdf(x, xi, omega, alpha, nu):
    z = (x - xi) / omega
    t_pdf = student_t.pdf(z, df=nu)
    t_cdf = student_t.cdf(alpha * z * np.sqrt((nu + 1) / (nu + z**2)), df=nu + 1)
    return (2 / omega) * t_pdf * t_cdf


NU_FIXED = 6
REBIN = 5
N_MIN = 500


def meas_sigv_grid(h, k=0):
    n = h.n[k].astype(float)
    mu = np.where(n >= N_MIN, h.sum_v[k] / np.where(n > 0, n, 1), np.nan)
    m2 = np.where(n >= N_MIN, h.sum_v2[k] / np.where(n > 0, n, 1), np.nan)
    return np.sqrt(np.maximum(m2 - mu**2, 0.0))


def fit_skewt_grid(h, k=0):
    """ST_pars[j,i,:] = [xi, omega, alpha] per (R, rlos) cell. This is the
    expensive part (up to 135x135 independent curve_fit calls) -- kept
    sequential within one (phase, bin) task, since parallelism happens one
    level up, across (phase, bin) tasks instead."""
    n_k, sum_v_k, sum_v2_k, counts_k = h.n[k], h.sum_v[k], h.sum_v2[k], h.counts[k]
    cent_fine = h.v_centers
    dv = (cent_fine[1] - cent_fine[0]) * REBIN
    cent = cent_fine.reshape(-1, REBIN).mean(axis=1)

    n_R, n_l = n_k.shape
    ST_pars = np.full((n_R, n_l, 3), np.nan)
    for j in range(n_R):
        for i in range(n_l):
            n = n_k[j, i]
            if n < N_MIN:
                continue
            mu = sum_v_k[j, i] / n
            width = np.sqrt(max(sum_v2_k[j, i] / n - mu**2, 0.0))

            counts = counts_k[j, i].reshape(-1, REBIN).sum(axis=1)
            sel = np.abs(cent - mu) <= 2500
            c_w, x_w = counts[sel], cent[sel]
            if c_w.sum() == 0:
                continue
            hist = c_w / (c_w.sum() * dv)

            try:
                p, _ = curve_fit(
                    lambda xx, xi_, omega_, alpha_: skewed_t_pdf(xx, xi_, omega_, alpha_, NU_FIXED),
                    x_w, hist, p0=[mu, width, -0.1], method='lm', maxfev=5_000)
                ST_pars[j, i] = p
            except RuntimeError:
                continue
    return ST_pars


def vr_profile(h, k, r_edges):
    Rc = 0.5 * (h.R_edges[1:] + h.R_edges[:-1])
    lc = 0.5 * (h.rlos_edges[1:] + h.rlos_edges[:-1])
    r3 = np.sqrt(Rc[:, None]**2 + lc[None, :]**2).ravel()
    n = h.n[k].ravel().astype(float)
    s1 = h.sum_vr[k].ravel()
    nb_bins = len(r_edges) - 1
    idx = np.searchsorted(r_edges, r3, side='right') - 1
    ok = (idx >= 0) & (idx < nb_bins) & (n > 0)

    def acc(w):
        return np.bincount(idx[ok], weights=w[ok], minlength=nb_bins)

    N = acc(n)
    r_eff = np.where(N > 0, acc(n * r3) / np.where(N > 0, N, 1), np.nan)
    mu = np.where(N > 0, acc(s1) / np.where(N > 0, N, 1), np.nan)
    return r_eff, mu, N.astype(np.int64)


def fit_one_phase_bin(sim, k):
    """One full (phase, mass bin) task: load, compute meas_sigv + skew-t
    grid + Delta, fit both compact models. Returns None on any failure
    (missing file, too little data, a fit not converging) rather than
    raising, so one bad (phase, bin) doesn't kill the whole batch."""
    path = OUT_TMPL.format(sim=sim, k=k)
    if not os.path.exists(path):
        return sim, k, None, None

    try:
        h = PairHist.load(path)
        sig = meas_sigv_grid(h, 0)
        ST_pars = fit_skewt_grid(h, 0)

        r_vr, vr_mean, npairs_vr = vr_profile(h, 0, VR_R_EDGES)
        good_vr = np.isfinite(vr_mean) & (npairs_vr > 1000)
        if good_vr.sum() < 4:
            return sim, k, None, None
        vr_interp = InterpolatedUnivariateSpline(r_vr[good_vr], vr_mean[good_vr], k=3, ext=3)
        Delta = ST_pars[:, :, 0] - vr_interp(r3_2d) * r_l_2d / r3_2d

        sig_params = fit_sigma_joint(x_grid, r_p, sig)
        try:
            del_params = fit_delta_joint(x_grid, r_p, Delta, p0=DEL_P0)
        except RuntimeError:
            del_params = None

        return sim, k, sig_params, del_params
    except Exception:
        return sim, k, None, None


def jackknife_mean_err(arr):
    """Delete-one-PHASE jackknife, matching vel_mod_mass_dep_plot.py exactly.
    arr shape: (n_phases_available_for_this_bin, n_params)."""
    n = arr.shape[0]
    total = arr.sum(axis=0)
    mean_vals = total / n
    loo_means = (total - arr) / (n - 1)
    err = np.sqrt((n - 1) / n * np.sum((loo_means - mean_vals)**2, axis=0))
    return mean_vals, err


#%% scan for available (phase, bin) combinations and run the parallelized fit
tasks = [(sim, k) for sim in ALL_PLANNED_PHASES for k in range(N_MASS)
         if os.path.exists(OUT_TMPL.format(sim=sim, k=k))]
print(f'{len(tasks)} (phase, bin) combinations available -- fitting with n_jobs={N_JOBS}')

results = Parallel(n_jobs=N_JOBS, verbose=10)(
    delayed(fit_one_phase_bin)(sim, k) for sim, k in tasks)

# collect per-bin, ragged across phases (not every bin necessarily has the
# same set of completed phases) -- keyed by sim, not just appended, so
# sigma_v and Delta fits can be matched by ACTUAL phase identity below,
# and so each phase can get a consistent color across every bin/panel
sig_by_bin = {k: {} for k in range(N_MASS)}
del_by_bin = {k: {} for k in range(N_MASS)}
for sim, k, sig_params, del_params in results:
    if sig_params is not None:
        sig_by_bin[k][sim] = sig_params
    if del_params is not None:
        del_by_bin[k][sim] = del_params

for k in range(N_MASS):
    print(f'bin {k}: {len(sig_by_bin[k])} sigma_v fits, {len(del_by_bin[k])} Delta fits')

#%% jackknife across phases, per mass bin
labels = [r"$\sigma_0$ [km/s]", r"$A_{\rho,0}$", r"$A_{\rho, p}$ [$h^{-1}$ Mpc]", r"$\sigma_\rho$ [$h^{-1}$ Mpc]",
          r"$A_{\Delta,0}$", r"$r_\Delta$ [$h^{-1}$ Mpc]", r"$\sigma_\Delta$ [$h^{-1}$ Mpc]",
          r"$\beta_\Delta$", r"$\alpha_\Delta$"]

mean_vals = np.full((N_MASS, 9), np.nan)
jack_err = np.full((N_MASS, 9), np.nan)
per_bin_params = {}   # for the scatter-of-phases overlay below
per_bin_sims = {}     # which phase each row of per_bin_params[k] actually came from

for k in range(N_MASS):
    common_sims = sorted(set(sig_by_bin[k]) & set(del_by_bin[k]))   # only phases where
                                                                       # BOTH fits succeeded
    if len(common_sims) >= 2:
        sig_arr = np.array([sig_by_bin[k][s] for s in common_sims])
        del_arr = np.array([del_by_bin[k][s] for s in common_sims])
        params = np.concatenate([sig_arr, del_arr], axis=-1)
        mean_vals[k], jack_err[k] = jackknife_mean_err(params)
        per_bin_params[k] = params
        per_bin_sims[k] = common_sims
    else:
        print(f'bin {k}: fewer than 2 phases with both fits succeeding for the SAME phase -- skipping jackknife')
#%% main mass-dependence plot (9 panels) -- full set, and excluding the highest mass bin
import matplotlib
matplotlib.rcParams.update({'text.usetex': True,
                            'font.family': 'Computer Modern Roman'})
plt.rcParams['axes.labelsize'] = 20

import os
import warnings
warnings.filterwarnings('ignore')

available_ks = sorted(per_bin_params.keys())

def make_mass_dep_plot(ks_to_plot, fname, title_suffix):
    if not ks_to_plot:
        print(f'{fname}: no bins to plot, skipping')
        return
    all_sims_used = sorted(set(s for k in ks_to_plot for s in per_bin_sims[k]))

    fig, ax = plt.subplots(1, 9, figsize=(29, 5), layout='constrained')
    for i, label in enumerate(labels):
        for sim in all_sims_used:
            ks_for_sim = [k for k in ks_to_plot if sim in per_bin_sims[k]]
            if not ks_for_sim:
                continue
            vals = [per_bin_params[k][per_bin_sims[k].index(sim), i] for k in ks_for_sim]
            ax[i].plot(M_CENS[ks_for_sim], vals, marker='o', alpha=0.3)

        ax[i].errorbar(M_CENS[ks_to_plot], mean_vals[ks_to_plot, i],
                        yerr=jack_err[ks_to_plot, i], marker='o', color='k',
                        label='jackknife mean', zorder=5)

        median_val = np.median(mean_vals[ks_to_plot, i])
        mean_val = np.mean(mean_vals[ks_to_plot, i])
        ax[i].axhline(mean_val, c='k', ls=':', lw=1.2, label='mean of jackknife means')
        ax[i].axhline(median_val, c='k', ls='--', lw=1.2, label='median of jackknife means')

        ax[i].set_ylabel(label)
        ax[i].set_xlabel(r'$M\ [h^{-1} M_\odot]$')
        ax[i].set_xscale('log')
    ax[0].legend(fontsize=7)

    fig.suptitle(f'sigma_v / Delta parameters vs mass -- {len(ks_to_plot)}/{N_MASS} bins{title_suffix}')
    fig.savefig(fname, dpi=150)
    print(f'saved {fname}')


make_mass_dep_plot(available_ks, 'vel_mod_mass_dep_plot.png', '')

ks_excl_highest = available_ks[:-1]   # available_ks is sorted ascending by
                                        # mass bin index, so [:-1] drops the
                                        # highest-mass bin among those available
make_mass_dep_plot(ks_excl_highest, 'vel_mod_mass_dep_plot_excl_highest.png',
                    ' (highest mass bin excluded)')

#%% raw sigma_v^2(x) and Delta(x) diagnostics, low vs high mass bin, one phase
K_LOW, K_HIGH = 0, N_MASS - 3
example_sim = None
for sim in ALL_PLANNED_PHASES:
    if os.path.exists(OUT_TMPL.format(sim=sim, k=K_LOW)) and os.path.exists(OUT_TMPL.format(sim=sim, k=K_HIGH)):
        example_sim = sim
        break

if example_sim is None:
    print(f'no single phase has both bin {K_LOW} and bin {K_HIGH} available yet -- skipping raw diagnostics')
else:
    print(f'using {example_sim} for the raw sigma_v^2(x)/Delta(x) diagnostics')
    skip = 15   # show every 15th r_p curve, else this is unreadably dense

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    for col, k in enumerate([K_LOW, K_HIGH]):
        h = PairHist.load(OUT_TMPL.format(sim=example_sim, k=k))
        sig = meas_sigv_grid(h, 0)
        ST_pars = fit_skewt_grid(h, 0)

        r_vr, vr_mean, npairs_vr = vr_profile(h, 0, VR_R_EDGES)
        good_vr = np.isfinite(vr_mean) & (npairs_vr > 1000)
        vr_interp = InterpolatedUnivariateSpline(r_vr[good_vr], vr_mean[good_vr], k=3, ext=3)
        Delta = ST_pars[:, :, 0] - vr_interp(r3_2d) * r_l_2d / r3_2d

        cmap = plt.cm.viridis(np.linspace(0, 1, len(range(0, len(r_p), skip))))
        for c, j in zip(cmap, range(0, len(r_p), skip)):
            ax[0, col].scatter(x_grid[j], sig[j]**2, c=c, lw=1)
            ax[1, col].scatter(x_grid[j], Delta[j], c=c, lw=1)

        ax[0, col].set_title(f'bin {k} (logM~{LOG_MASS_EDGES[k].mean():.2f}), {example_sim}')
        ax[0, col].set_ylabel(r'$\sigma_v^2(x)$ [(km/s)$^2$]')
        ax[1, col].set_ylabel(r'$\Delta(x)$ [km/s]')
        ax[1, col].set_xlabel(r'$x = r - r_\perp$ [$h^{-1}$ Mpc]')

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=r_p[0], vmax=r_p[-1]))
    fig.colorbar(sm, ax=ax, label=r'$r_\perp$ [$h^{-1}$ Mpc]', shrink=0.6)
    fig.suptitle(f'raw sigma_v^2(x), Delta(x) across r_perp -- low (left) vs high (right) mass bin')
    fig.savefig('vel_mod_raw_diagnostics.png', dpi=150)
    print('saved vel_mod_raw_diagnostics.png')
     
#%% overlay best-fit sigma_v/Delta models on the raw diagnostic panels
K_LOW, K_HIGH = 0, N_MASS - 3
if example_sim is None:
    print('skipping model overlay -- no example_sim from the previous cell')
else:
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    for col, k in enumerate([K_LOW, K_HIGH]):
        h = PairHist.load(OUT_TMPL.format(sim=example_sim, k=k))
        sig = meas_sigv_grid(h, 0)
        ST_pars = fit_skewt_grid(h, 0)
 
        r_vr, vr_mean, npairs_vr = vr_profile(h, 0, VR_R_EDGES)
        good_vr = np.isfinite(vr_mean) & (npairs_vr > 1000)
        vr_interp = InterpolatedUnivariateSpline(r_vr[good_vr], vr_mean[good_vr], k=3, ext=3)
        Delta = ST_pars[:, :, 0] - vr_interp(r3_2d) * r_l_2d / r3_2d
 
        sig_params = fit_sigma_joint(x_grid, r_p, sig)
        # replicate fit_sigma_joint's own masking exactly, so this R_piv is
        # bit-identical to the one actually used inside that fit -- not an
        # approximation, unlike the next cell below
        R_bcast = np.broadcast_to(r_p[:, None], x_grid.shape)
        mask = np.isfinite(x_grid) & np.isfinite(sig) & (sig > SIG_FLOOR)
        R_piv = np.median(R_bcast[mask])
        sig_model = sigma_mod_2d((x_grid, r_p_2d), *sig_params, R_piv)
 
        try:
            del_params = fit_delta_joint(x_grid, r_p, Delta, p0=DEL_P0)
            del_model = delta_mod_alpha_R((x_grid, r_p_2d), *del_params)
        except RuntimeError:
            del_params, del_model = None, None
 
        cmap = plt.cm.viridis(np.linspace(0, 1, len(range(0, len(r_p), skip))))
        for c, j in zip(cmap, range(0, len(r_p), skip)):
            ax[0, col].plot(x_grid[j], sig[j]**2, 'o', ms=3, c=c, alpha=0.6)
            ax[0, col].plot(x_grid[j], sig_model[j]**2, '-', lw=1.5, c=c)
            ax[1, col].plot(x_grid[j], Delta[j], 'o', ms=3, c=c, alpha=0.6)
            if del_model is not None:
                ax[1, col].plot(x_grid[j], del_model[j], '-', lw=1.5, c=c)
 
        ax[0, col].set_title(f'bin {k} (logM~{LOG_MASS_EDGES[k].mean():.2f}), {example_sim}')
        ax[0, col].set_ylabel(r'$\sigma_v^2(x)$ [(km/s)$^2$]')
        ax[1, col].set_ylabel(r'$\Delta(x)$ [km/s]')
        ax[1, col].set_xlabel(r'$x = r - r_\perp$ [$h^{-1}$ Mpc]')
 
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=r_p[0], vmax=r_p[-1]))
    fig.colorbar(sm, ax=ax, label=r'$r_\perp$ [$h^{-1}$ Mpc]', shrink=0.6)
    fig.suptitle('markers: data, lines: best-fit model -- low (left) vs high (right) mass bin')
    fig.savefig('vel_mod_model_overlay.png', dpi=150)
    print('saved vel_mod_model_overlay.png')
 
#%% r_perp-dependent shape of A_rho(R) and A_delta(R), across mass bins
# R_piv here is an APPROXIMATION -- the exact R_piv used by each individual
# (phase, bin) fit isn't stored in mean_vals (fit_sigma_joint computes it
# internally and never returns it), so this uses the plain grid median
# instead. Should be very close in practice, since the fit's own mask
# rarely excludes a large, asymmetric fraction of the r_p range -- but
# it's not bit-identical the way the previous cell's R_piv is.
R_piv_plot = np.median(r_p)
 
if available_ks:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    cmap = plt.cm.plasma(np.linspace(0, 1, len(available_ks)))
    for c, k in zip(cmap, available_ks):
        sigma0, A_piv, b_, sigma_rho = mean_vals[k, :4]
        A_delta0, r_delta, sigma_delta, beta_delta, alpha = mean_vals[k, 4:]
 
        ax[0].plot(r_p, A_rho_mod(r_p, A_piv, b_, R_piv_plot), c=c,
                   label=rf'$\log M \approx {LOG_MASS_EDGES[k].mean():.2f}$')
        ax[1].plot(r_p, A_delta_mod(r_p, A_delta0, r_delta), c=c)
 
    ax[0].set_xlabel(r'$r_\perp$ [$h^{-1}$ Mpc]')
    ax[0].set_ylabel(r'$A_\rho(r_\perp)$')
    ax[1].set_xlabel(r'$r_\perp$ [$h^{-1}$ Mpc]')
    ax[1].set_ylabel(r'$A_\Delta(r_\perp)$')
    ax[0].legend(fontsize=7)
    fig.suptitle('r_perp-dependent amplitude shapes, by mass bin (jackknife-mean parameters)')
    fig.tight_layout()
    fig.savefig('vel_mod_r_perp_dependence.png', dpi=150)
    print('saved vel_mod_r_perp_dependence.png')
else:
    print('no bins available yet -- nothing to plot')