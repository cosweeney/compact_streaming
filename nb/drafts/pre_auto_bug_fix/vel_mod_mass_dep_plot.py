"""
vel_mod_mass_dep_plot.py

Fits the sigma_v joint model (unchanged from ASbase_compact_calibration_multiple.ipynb)
and the r_perp-dependent Delta model (flat-core amplitude x rise-peak-fall shape, with a
freely-varying exponent alpha on x -- shared across r_perp within a fit, not r_perp-dependent
itself) independently for every (phase, mass bin), then jackknifes the fitted parameters over
phases to check how severe the mass dependence is. Also fits b1 only (b2, bs fixed to 0) to
the phase-averaged xi_hh per mass bin, and plots the velocity model parameters against b1(M).

Usage:
    python vel_mod_mass_dep_plot.py
    python vel_mod_mass_dep_plot.py --n-phases 5   # quick sanity check before the full 15

Outputs:
    - <output-dir>/vel_mod_mass_dep_plot.png   (sigma_v/Delta params vs mass; local by default)
    - <output-dir>/b1_mass_dep_plot.png        (b1 vs mass; local by default)
    - <output-dir>/vel_mod_vs_b1_plot.png      (sigma_v/Delta params vs b1(M); local by default)
    - <cache-path>  (fitted sigma_v/Delta parameter arrays; spiffball by default, since these
      are worth keeping without re-running the fits. b1(M) is printed to stdout, not cached.)

Assumptions carried over from ASbase_compact_calibration_multiple.ipynb (flag if wrong):
    - REDSHIFT = 'z0.200', sims = c000_ph000..c000_ph014 (15 phases)
    - r_perp/r_los grid: 135 bins over [5, 140] h^-1 Mpc, matching ST_pars/meas_sigv shapes
    - ST_pars, meas_sigv already fit and stored in pdf_fits.h5 with shape (n_sims, n_mass, ...)
    - <v_r|r> profiles are NOT cached anywhere and must be rebuilt per phase per mass bin
      from the raw PairHist files
"""

import argparse
import sys

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
import psutil

from scipy.interpolate import InterpolatedUnivariateSpline

from compact.velocities.pairhist import PairHist
from compact.lpt import GetLPTInputs


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

PDF_FITS_PATH = '/spiffball/cosweeney/simulations/AbacusSummit_base/data/pdf_fits.h5'
PAIRHIST_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base/halos/'
                 'AbacusSummit_base_{sim}/halos/{redshift}/data/pairhist_allbins.h5')
CACHE_PATH_DEFAULT = '/spiffball/cosweeney/simulations/AbacusSummit_base/data/vel_mod_mass_dep.h5'
PAPER_CACHE_PATH_DEFAULT = 'paper_figs_cache.h5'  # relative to cwd, matching gather_paper_data.py's output

REDSHIFT = 'z0.200'

# fiducial cosmology/redshift for the b1-only xi_hh fit, matching the Fig 1 bias-fitting cell
FID_COSMO = [0.6736, 0.02237, 0.1200, 2.083e-9, 0.9649]
FID_REDSHIFT = 0.2
R_MIN_FIT = 20.0


# ------------------------------------------- <v_r|r> profiles (per phase, per mass bin)
# verbatim from ASbase_compact_calibration_multiple.ipynb -- not exported from compact.model
def vr_profile(h, k, r_edges=None, r_max_trust=None):
    """Pair-weighted <v_r>(r) and sigma_vr(r) for mass bin k.

    Returns (r_eff, mean_vr, sigma_vr, n_pairs), all length len(r_edges)-1.
    r_eff is the pair-weighted mean radius in each bin, not the bin centre.
    Units follow the accumulator -- km/s if conv was applied at accumulation.
    """
    Rc = 0.5 * (h.R_edges[1:] + h.R_edges[:-1])
    lc = 0.5 * (h.rlos_edges[1:] + h.rlos_edges[:-1])
    r3 = np.sqrt(Rc[:, None]**2 + lc[None, :]**2).ravel()

    n = h.n[k].ravel().astype(float)
    s1 = h.sum_vr[k].ravel()
    s2 = h.sum_vr2[k].ravel()

    if r_edges is None:
        # default: 2 Mpc/h bins out to the largest radius with full angular
        # coverage (beyond R_edges[-1] only the cylinder corners contribute)
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
    sd = np.sqrt(np.maximum(m2 - mu**2, 0.0))
    return r_eff, mu, sd, N.astype(np.int64)


def make_vr_interp(h, k, r_edges=None, r_max_trust=None, n_min=1000, ext=3):
    """Spline interpolator for <v_r>(r), analogous to mean_vr_r_interp.

    ext=3 clamps to the end values outside the fitted range rather than
    extrapolating or returning NaN -- your r_3d reaches ~198 Mpc/h at the grid
    corners, and a spline let loose out there will do something wild.
    """
    r, mu, _, N = vr_profile(h, k, r_edges, r_max_trust)
    good = np.isfinite(mu) & (N >= n_min)
    if good.sum() < 4:
        raise ValueError(f'mass bin {k}: only {good.sum()} usable radial bins')
    return InterpolatedUnivariateSpline(r[good], mu[good], k=3, ext=ext)
N_PHASES_DEFAULT = 15

LOG_MASS_EDGES = np.array([[12, 12.01], [12.5, 12.6], [13, 13.1],
                           [13.5, 13.6], [14.0, 14.2], [14.4, 14.7]])

SIG_FLOOR = 0.1
DEL_FLOOR = -1000.0
DEL_P0 = [125.0, 50.0, 10.0, 2.0, 1.0]  # A_delta0, r_delta, sigma_delta, beta_delta, alpha


# --------------------------------------------------------------------- grid
def build_grid():
    r_perp_edges = np.linspace(5, 140, 135 + 1)
    r_los_edges = np.linspace(5, 140, 135 + 1)
    r_p = 0.5 * (r_perp_edges[1:] + r_perp_edges[:-1])
    r_l = 0.5 * (r_los_edges[1:] + r_los_edges[:-1])
    r_p_2d, r_l_2d = np.meshgrid(r_p, r_l, indexing='ij')
    r = np.sqrt(r_p_2d**2 + r_l_2d**2)
    x_grid = r - r_p_2d
    return r_p, r_l_2d, r, x_grid


# --------------------------------------------------- sigma_v joint model
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


def fit_sigma_joint(x, r_p, sig):
    R = np.broadcast_to(r_p[:, None], x.shape)
    m = np.isfinite(x) & np.isfinite(sig) & (sig > SIG_FLOOR)
    X, RR, S = x[m], R[m], sig[m]
    R_piv = np.median(RR)
    popt, pcov = curve_fit(
        lambda c, s0, Ap, b, sr: sigma_mod_2d(c, s0, Ap, b, sr, R_piv),
        (X, RR), S, p0=_seed_sigma(X, RR, S, R_piv),
        bounds=([0.0, 1e-4, 1e-3, 1e-3], [np.inf, 1.0, np.inf, np.inf]), maxfev=20_000)
    return popt


# ------------------------------------------------------- Delta model, with free alpha
def A_delta_mod(R, A_delta0, r_delta):
    return A_delta0 / (1 + (R / r_delta)**2)


def delta_mod_alpha(x, A, sigma_delta, beta_delta, alpha):
    return A * x**alpha / (1 + (x / sigma_delta)**beta_delta)


def delta_mod_alpha_R(coords, A_delta0, r_delta, sigma_delta, beta_delta, alpha):
    x, R = coords
    return delta_mod_alpha(x, A_delta_mod(R, A_delta0, r_delta), sigma_delta, beta_delta, alpha)


def fit_delta_joint(x, r_p, D, p0):
    R = np.broadcast_to(r_p[:, None], x.shape)
    m = np.isfinite(x) & np.isfinite(D) & (D > DEL_FLOOR)
    X, RR, DD = x[m], R[m], D[m]
    popt, _ = curve_fit(delta_mod_alpha_R, (X, RR), DD, p0=p0, maxfev=20_000)
    return popt


# --------------------------------------------- b1-only xi_hh fit (Fig 1 bias model)
def model_xihh(r, b1, b2, bs, lpt):
    pars = np.array([b1, b2, bs, 0, 0, 0, 0, 0, 0])
    return lpt.xi_real(r, pars)


def fit_b1_only(k, paper_cache_path, lpt, r_min=R_MIN_FIT):
    """b1-only fit (b2, bs fixed to 0) to the phase-averaged xi_hh for mass bin k."""
    with h5.File(paper_cache_path, 'r') as hdf:
        gk = hdf['fig1'][f'mbin_{k}']
        r_xi = gk['r_cens'][()]
        xihh = gk['xi'][()]
        var_xi = np.diag(gk['cov'][()])

    m = r_xi > r_min
    r_xi_fit, xihh_fit, var_xi_fit = r_xi[m], xihh[m], var_xi[m]

    def loss(params):
        res = model_xihh(r_xi_fit, params[0], 0.0, 0.0, lpt) - xihh_fit
        c = np.sum(res**2 / var_xi_fit)
        return c if np.isfinite(c) else np.inf

    result = minimize(loss, x0=[1.0], bounds=[(-0.3, 5.0)], method='L-BFGS-B')
    return result.x[0]


# ------------------------------------------------------------- jackknife
def jackknife_mean_err(arr):
    n = arr.shape[0]
    total = arr.sum(axis=0)
    mean_vals = total / n
    loo_means = (total - arr) / (n - 1)
    err = np.sqrt((n - 1) / n * np.sum((loo_means - mean_vals)**2, axis=0))
    return mean_vals, err


# ------------------------------------------------------------------ main
def main(n_phases, cache_path, out_dir, paper_cache_path):
    psutil.Process().nice(10)

    sims = [f'c000_ph{i:03d}' for i in range(n_phases)]
    M_cens = (10**LOG_MASS_EDGES[:, 0] * 10**LOG_MASS_EDGES[:, 1])**0.5
    r_p, r_l_2d, r, x_grid = build_grid()

    print(f'loading ST_pars/meas_sigv for {len(sims)} phases from {PDF_FITS_PATH}')
    with h5.File(PDF_FITS_PATH, 'r') as hdf:
        ST_pars = hdf['ST_pars'][:n_phases]
        meas_sigv = hdf['meas_sigv'][:n_phases]

    n_M = len(M_cens)

    print('building <v_r|r> interpolators from raw pairhist files (this is the slow part)')
    interps_sims = []
    for sim in sims:
        out_tmpl = PAIRHIST_TMPL.format(sim=sim, redshift=REDSHIFT)
        h_pw = PairHist.load(out_tmpl)
        interps_sims.append({k: make_vr_interp(h_pw, k) for k in range(n_M)})
        print(f'  built interpolators for {sim}')

    sig_opts = np.full((n_phases, n_M, 4), np.nan)
    del_opts_full = np.full((n_phases, n_M, 5), np.nan)

    for si, sim in enumerate(sims):
        for k in range(n_M):
            sig = meas_sigv[si, k]
            Delta = ST_pars[si, k, :, :, 0] - interps_sims[si][k](r) * r_l_2d / r

            sig_opts[si, k] = fit_sigma_joint(x_grid, r_p, sig)

            try:
                del_opts_full[si, k] = fit_delta_joint(x_grid, r_p, Delta, p0=DEL_P0)
            except RuntimeError:
                print(f'  delta fit failed for {sim}, mass bin {k}')
        print(f'done {sim}')

    sig_mean, sig_err = jackknife_mean_err(sig_opts)
    del_mean, del_err = jackknife_mean_err(del_opts_full)

    parameters = np.concatenate((sig_opts, del_opts_full), axis=-1)
    mean_vals = np.concatenate((sig_mean, del_mean), axis=-1)
    jack_err = np.concatenate((sig_err, del_err), axis=-1)

    print(f'saving fitted parameter arrays to {cache_path}')
    with h5.File(cache_path, 'w') as hdf:
        hdf.create_dataset('sig_opts', data=sig_opts)
        hdf.create_dataset('del_opts_full', data=del_opts_full)
        hdf.create_dataset('mean_vals', data=mean_vals)
        hdf.create_dataset('jack_err', data=jack_err)
        hdf.create_dataset('M_cens', data=M_cens)
        hdf.attrs['sims'] = np.array(sims, dtype=h5.string_dtype())

    labels = [r"$\sigma_{v,0}$ [km/s]", r"$A_{\rho,0}$", r"$A_{\rho,p}$ [$h^{-1}$ Mpc]", r"$\sigma_\rho$ [$h^{-1}$ Mpc]",
            r"$A_{\Delta,0}$ [km/s / ($h^{-1}$ Mpc)$^\alpha$]", r"$r_\Delta$ [$h^{-1}$ Mpc]",
            r"$\sigma_\Delta$ [$h^{-1}$ Mpc]", r"$\beta_\Delta$", r"$\alpha$"]

    fig, ax = plt.subplots(1, 9, figsize=(29, 5), layout='constrained')
    for i, label in enumerate(labels):
        for si in range(n_phases):
            ax[i].plot(M_cens, parameters[si, :, i], marker='o', alpha=0.3)
        ax[i].errorbar(M_cens, mean_vals[:, i], yerr=jack_err[:, i], marker='o', color='k',
                        label=f'mean of {n_phases} sims')
        ax[i].axhline(np.mean(mean_vals[:, i]), c='k', ls=':', label='parameter average')
        ax[i].set_ylabel(label)
        ax[i].set_xlabel(r'$M\ [h^{-1} M_\odot]$')
        ax[i].set_xscale('log')
    ax[0].legend(fontsize=8)

    fig_path = f'{out_dir}/vel_mod_mass_dep_plot.png'
    fig.savefig(fig_path, dpi=150)
    print(f'saved figure to {fig_path}')

    # --- b1-only fit to xi_hh (Fig 1), per mass bin ---
    print(f'fitting b1 (b2=bs=0 fixed) to xi_hh for {n_M} mass bins from {paper_cache_path}')
    lpt = GetLPTInputs(redshift=FID_REDSHIFT, cosmo=FID_COSMO)
    lpt.get_pklin_gsm()

    b1_of_M = np.full(n_M, np.nan)
    for k in range(n_M):
        b1_of_M[k] = fit_b1_only(k, paper_cache_path, lpt)
        print(f'  mass bin {k} (M={M_cens[k]:.2e}): b1 = {b1_of_M[k]:.4f}')

    fig_b1, ax_b1 = plt.subplots(figsize=(6, 5))
    ax_b1.plot(M_cens, b1_of_M, marker='o', color='k')
    ax_b1.set_xlabel(r'$M\ [h^{-1} M_\odot]$')
    ax_b1.set_ylabel(r'$b_1(M)$')
    ax_b1.set_xscale('log')
    ax_b1.margins(x=0)
    fig_b1.tight_layout()
    b1_fig_path = f'{out_dir}/b1_mass_dep_plot.png'
    fig_b1.savefig(b1_fig_path, dpi=150)
    print(f'saved figure to {b1_fig_path}')

    # --- velocity model parameters vs b1(M) ---
    fig_vs_b1, ax_vs_b1 = plt.subplots(1, 9, figsize=(29, 5), layout='constrained')
    for i, label in enumerate(labels):
        ax_vs_b1[i].errorbar(b1_of_M, mean_vals[:, i], yerr=jack_err[:, i], marker='o', color='k')
        ax_vs_b1[i].set_xlabel(r'$b_1(M)$')
        ax_vs_b1[i].set_ylabel(label)
    vs_b1_fig_path = f'{out_dir}/vel_mod_vs_b1_plot.png'
    fig_vs_b1.savefig(vs_b1_fig_path, dpi=150)
    print(f'saved figure to {vs_b1_fig_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-phases', type=int, default=N_PHASES_DEFAULT,
                         help='number of phases (c000_ph000..) to include')
    parser.add_argument('--cache-path', type=str, default=CACHE_PATH_DEFAULT,
                         help='where to save fitted parameter arrays (spiffball by default)')
    parser.add_argument('--out-dir', type=str, default='.',
                         help='local directory to save the figures into')
    parser.add_argument('--paper-cache-path', type=str, default=PAPER_CACHE_PATH_DEFAULT,
                         help='path to gather_paper_data.py\'s paper_figs_cache.h5 (for the b1-only xi_hh fit)')
    args = parser.parse_args()

    main(n_phases=args.n_phases, cache_path=args.cache_path, out_dir=args.out_dir,
         paper_cache_path=args.paper_cache_path)