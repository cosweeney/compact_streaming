"""
Standalone tests of a variable-nu skew-t for the compact pairwise-velocity model.

Run the cells top to bottom (each '#%%' is a cell). Sections:
  setup / functions -- imports, paths, grid, fitting + model helpers
  A  b1(M), b2(M) from the joint xi_hh + v_r fit (cached)
  B  single-phase skew-t fits over the whole grid, nu fixed at 6 and nu free (cached); nu / Delta scatter plots
  C  functional forms for nu(r), fit to per-r-bin medians of the free-nu fits
  D  power-law nu(r) per (phase, mass bin), Fig-5-style summary vs mass (cached)
  E  3x3 compact-model predictions using nu(r), with Delta refit to the nu(r) calibration data

Conventions (match compact/model.py): Delta(x, r_perp) = A_delta0 exp(-r_perp/r_delta) x^alpha / (1 + (x/sigma_delta)^beta),
sigma_v with the fixed pivot R_PIV = 72.5 in r_perp. Delta = xi_ST - <v_r|r> r_los / r, with <v_r|r> the measured
profile (calibration) and the b1/b2 LPT model mean (prediction), as in the pipeline.
"""

#%% setup
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import h5py as h5
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, differential_evolution, least_squares
from scipy.stats import t as student_t
from scipy.special import gammaln
from scipy.interpolate import InterpolatedUnivariateSpline
from joblib import Parallel, delayed
import psutil
psutil.Process().nice(10)

matplotlib.rcParams.update({'text.usetex': True,
                            'font.family': 'Computer Modern Roman'})
plt.rcParams['axes.labelsize'] = 20

from compact.velocities.pairhist import PairHist
from compact.lpt import GetLPTInputs

VR_AUTO_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base/halos/'
                'AbacusSummit_base_{sim}/halos/z0.200/data/pairhist_auto_mbin{k}.h5')
XI_JK_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base/halos/'
              'AbacusSummit_base_{sim}/halos/z0.200/data/xi_hh_jk.npz')
ALL_PLANNED_PHASES = [f'c000_ph{i:03d}' for i in range(0, 25)]

CACHE_DIR = 'nu_tests_cache'
FIGURES_DIR = 'figures_nu'
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

LOG_MASS_EDGES = np.array([[12, 12.01], [12.5, 12.6], [13, 13.1],
                           [13.5, 13.6], [13.9, 14.3]])
N_MASS = len(LOG_MASS_EDGES)
M_CENS = (10**LOG_MASS_EDGES[:, 0] * 10**LOG_MASS_EDGES[:, 1])**0.5

VR_R_EDGES = np.arange(2.0, 142.0, 2.0)
R_MIN_FIT = 15.0
FID_COSMO = [0.6736, 0.02237, 0.1200, 2.083e-9, 0.9649]
FID_REDSHIFT_VAL = 0.2

# grid -- matches the accumulator's R / r_los edges
r_perp_edges = np.linspace(5, 140, 135 + 1)
r_los_edges = np.linspace(5, 140, 135 + 1)
r_p = 0.5 * (r_perp_edges[1:] + r_perp_edges[:-1])
r_l = 0.5 * (r_los_edges[1:] + r_los_edges[:-1])
r_p_2d, r_l_2d = np.meshgrid(r_p, r_l, indexing='ij')
r3_2d = np.sqrt(r_p_2d**2 + r_l_2d**2)
x_grid = r3_2d - r_p_2d

REBIN = 5
N_MIN = 500
NU_FIXED = 6
NU_BOUNDS = (2.1, 1000.0)   # free-nu bounds (variance needs nu > 2; ~1000 is effectively skew-normal)
SINGLE_BIN = 2              # mass bin used for the single-phase tests (B, C, E)


#%% functions
def vr_profile(h, k, r_edges):
    Rc = 0.5 * (h.R_edges[1:] + h.R_edges[:-1])
    lc = 0.5 * (h.rlos_edges[1:] + h.rlos_edges[:-1])
    r3 = np.sqrt(Rc[:, None]**2 + lc[None, :]**2).ravel()
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
    m2 = np.where(N > 0, acc(s2) / np.where(N > 0, N, 1), np.nan)
    sd = np.sqrt(np.maximum(m2 - mu**2, 0.0))
    return r_eff, mu, sd, N.astype(np.int64)


def meas_sigv_grid(h, k=0, n_min=500):
    n = h.n[k].astype(float)
    mu = np.where(n >= n_min, h.sum_v[k] / np.where(n > 0, n, 1), np.nan)
    m2 = np.where(n >= n_min, h.sum_v2[k] / np.where(n > 0, n, 1), np.nan)
    return np.sqrt(np.maximum(m2 - mu**2, 0.0))


def skewed_t_pdf(x, xi, omega, alpha, nu):
    z = (x - xi) / omega
    t_pdf = student_t.pdf(z, df=nu)
    t_cdf = student_t.cdf(alpha * z * np.sqrt((nu + 1) / (nu + z**2)), df=nu + 1)
    return (2 / omega) * t_pdf * t_cdf


def skewt_params_from_delta(delta, sig_v, nu):
    """(omega, alpha) of a nu-dof skew-t with standard deviation sig_v and location offset delta = xi - mean.
    Closed-form equivalent of compact.model.SkewTMeanInterpolator.get_skew_params, valid for any nu > 2."""
    b = np.sqrt(nu / np.pi) * np.exp(gammaln((nu - 1) / 2) - gammaln(nu / 2))
    omega = np.sqrt((sig_v**2 + delta**2) * (nu - 2) / nu)
    d = np.clip(-delta / (omega * b), -0.9999, 0.9999)
    return omega, d / np.sqrt(1 - d**2)


def fit_skewt_row(n_row, sv_row, sv2_row, counts_row, cent_fine, nu_row=None, keep=None):
    """[xi, omega, alpha, nu] per cell in one R row, fit to the binned v_los density (unweighted least squares,
    +/-2500 km/s window around the cell mean). nu_row: per-cell fixed nu, or None to fit nu freely
    (as ln nu, bounded by NU_BOUNDS). keep: optional boolean mask of cells to fit."""
    dv = (cent_fine[1] - cent_fine[0]) * REBIN
    cent = cent_fine.reshape(-1, REBIN).mean(axis=1)
    out = np.full((len(n_row), 4), np.nan)
    for i in range(len(n_row)):
        n = n_row[i]
        if n < N_MIN or (keep is not None and not keep[i]):
            continue
        mu = sv_row[i] / n
        width = np.sqrt(max(sv2_row[i] / n - mu**2, 0.0))
        counts = counts_row[i].reshape(-1, REBIN).sum(axis=1)
        sel = np.abs(cent - mu) <= 2500
        c_w, x_w = counts[sel], cent[sel]
        if c_w.sum() == 0:
            continue
        hist = c_w / (c_w.sum() * dv)
        try:
            if nu_row is None:
                p, _ = curve_fit(lambda x, xi_, om_, al_, lnnu_: skewed_t_pdf(x, xi_, om_, al_, np.exp(lnnu_)),
                                 x_w, hist, p0=[mu, width, -0.1, np.log(float(NU_FIXED))],
                                 bounds=([-np.inf, 1e-3, -50.0, np.log(NU_BOUNDS[0])],
                                         [np.inf, np.inf, 50.0, np.log(NU_BOUNDS[1])]), max_nfev=5000)
                out[i] = [p[0], p[1], p[2], np.exp(p[3])]
            else:
                nu_i = nu_row[i]
                p, _ = curve_fit(lambda x, xi_, om_, al_: skewed_t_pdf(x, xi_, om_, al_, nu_i),
                                 x_w, hist, p0=[mu, width, -0.1], method='lm', maxfev=5000)
                out[i] = [p[0], p[1], p[2], nu_i]
        except RuntimeError:
            continue
    return out


def fit_skewt_grid(h, nu=None, n_jobs=-1):
    """Skew-t fit to every cell of a PairHist grid (R rows in parallel). nu: None (free), a scalar, or an
    (n_R, n_l) array of per-cell values. Returns ST[j, i] = [xi, omega, alpha, nu]."""
    n_R, n_l = h.n[0].shape

    def nu_row(j):
        if nu is None:
            return None
        return np.full(n_l, float(nu)) if np.isscalar(nu) else nu[j]

    res = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(fit_skewt_row)(h.n[0][j], h.sum_v[0][j], h.sum_v2[0][j], h.counts[0][j],
                               h.v_centers, nu_row(j)) for j in range(n_R))
    return np.array(res)


# ---- sigma_v model (compact/model.py forms, fixed pivot) ----
SIG_FLOOR = 0.1
R_PIV = 72.5   # fixed r_perp pivot for A_rho (R_PIV_FIXED in compact/model.py)


def sigma_mod(x, sigma0, A_rho, sigma_rho):
    arg = 2.0 * sigma0**2 * (1.0 - A_rho * np.exp(-x / sigma_rho))
    return np.sqrt(np.clip(arg, 1e-8, None))


def A_rho_mod(R, A_piv, A_rho_p):
    return A_piv * np.exp(-(R - R_PIV) / A_rho_p)


def sigma_mod_2d(coords, sigma0, A_piv, A_rho_p, sigma_rho):
    x, R = coords
    return sigma_mod(x, sigma0, A_rho_mod(R, A_piv, A_rho_p), sigma_rho)


def _seed_sigma(x, R, sig):
    sigma0 = sig.max() / np.sqrt(2.0) * 1.02
    y = 2.0 * sigma0**2 - sig**2
    m = y > 0
    G = np.column_stack([np.ones(m.sum()), -(R[m] - R_PIV), -x[m]])
    c = np.linalg.lstsq(G, np.log(y[m]), rcond=None)[0]
    A_rho_p = 1.0 / c[1] if c[1] > 0 else np.ptp(R) + 1.0
    sigma_rho = 1.0 / c[2] if c[2] > 0 else 10.0
    A_piv = np.clip(np.exp(c[0]) / (2.0 * sigma0**2), 1e-3, 0.99)
    return [sigma0, A_piv, A_rho_p, sigma_rho]


def fit_sigma_joint(x, r_p_arr, sig):
    R = np.broadcast_to(r_p_arr[:, None], x.shape)
    m = np.isfinite(x) & np.isfinite(sig) & (sig > SIG_FLOOR)
    X, RR, S = x[m], R[m], sig[m]
    popt, _ = curve_fit(sigma_mod_2d, (X, RR), S, p0=_seed_sigma(X, RR, S),
                        bounds=([0.0, 1e-4, 1e-3, 1e-3], [np.inf, 1.0, np.inf, np.inf]), maxfev=20_000)
    return popt


# ---- Delta model (exponential A_delta, as in compact/model.py) ----
DEL_FLOOR = -1000.0
DEL_P0 = [125.0, 50.0, 10.0, 2.0, 1.0]


def A_delta_mod(R, A_delta0, r_delta):
    return A_delta0 * np.exp(-R / r_delta)


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


def jackknife_mean_err(arr):
    n = arr.shape[0]
    total = arr.sum(axis=0)
    mean_vals = total / n
    loo_means = (total - arr) / (n - 1)
    err = np.sqrt((n - 1) / n * np.sum((loo_means - mean_vals)**2, axis=0))
    return mean_vals, err


#%% A. b1(M), b2(M): joint (b1, b2) fit to phase-averaged xi_hh and v_r, with intrinsic scatter (cached)
lpt = GetLPTInputs(redshift=FID_REDSHIFT_VAL, cosmo=FID_COSMO)
lpt.get_pklin_gsm()


def model_xihh(r, b1, b2, bs=0.0):
    pars = np.array([b1, b2, bs, 0, 0, 0, 0, 0, 0])
    return lpt.xi_real(r, pars)


def model_vr(r, b1, b2, bs=0.0):
    pars = np.array([b1, b2, bs, 0, 0, 0, 0, 0, 0])
    # recomputed with compute_xi_real (not lpt.pwv_mean, which normalizes by a Tukey-windowed xi)
    lpt._ensure_cumulants(pars)
    v_int = np.interp(r, lpt.gsm.rint, lpt.gsm.veft)
    xi_consistent = lpt.gsm.compute_xi_real(r, *pars)
    return lpt.conv * lpt.f * v_int / (1 + xi_consistent)


def regularized_pinv_and_logdet(cov, max_rank):
    """Rank-truncated pseudo-inverse and log-determinant of the retained eigenvalues."""
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    max_rank = min(max_rank, len(vals))
    vals = np.clip(vals[:max_rank], 1e-300, None)
    vecs = vecs[:, :max_rank]
    return vecs @ np.diag(1.0 / vals) @ vecs.T, np.sum(np.log(vals))


def fit_bias_intrinsic(r_min, r_xi, xihh, cov_xi, max_rank_xi,
                       r_vr, vr, cov_vr, max_rank_vr,
                       popsize=15, maxiter=200, tol=1e-6, seed=0):
    """Joint (b1, b2, eps_xi, eps_v) fit to xi_hh and v_r with fractional intrinsic scatter."""
    m_xi = r_xi > r_min
    r_xi_fit, xihh_fit = r_xi[m_xi], xihh[m_xi]
    cov_xi_fit = cov_xi[np.ix_(m_xi, m_xi)]
    m_vr = r_vr > r_min
    r_vr_fit, vr_fit = r_vr[m_vr], vr[m_vr]
    cov_vr_fit = cov_vr[np.ix_(m_vr, m_vr)]

    def loss(params):
        b1, b2, eps_xi, eps_v = params
        res_xi = model_xihh(r_xi_fit, b1, b2) - xihh_fit
        C_xi = cov_xi_fit + np.diag(eps_xi**2 * xihh_fit**2)
        inv_xi, logdet_xi = regularized_pinv_and_logdet(C_xi, max_rank_xi)
        res_vr = model_vr(r_vr_fit, b1, b2) - vr_fit
        C_vr = cov_vr_fit + np.diag(eps_v**2 * vr_fit**2)
        inv_vr, logdet_vr = regularized_pinv_and_logdet(C_vr, max_rank_vr)
        total = res_xi @ inv_xi @ res_xi + logdet_xi + res_vr @ inv_vr @ res_vr + logdet_vr
        return total if np.isfinite(total) else np.inf

    bounds = [(-0.3, 5.0), (-10.0, 10.0), (0.0, 2.0), (0.0, 2.0)]
    return differential_evolution(loss, bounds, tol=tol, seed=seed, maxiter=maxiter,
                                  popsize=popsize, polish=True)


B1B2_CACHE = os.path.join(CACHE_DIR, 'b1_b2.npz')
if os.path.exists(B1B2_CACHE):
    print(f'b1, b2: loading cached fits from {B1B2_CACHE} (delete this file to force a re-fit)')
    _d = np.load(B1B2_CACHE)
    b1_bins, b2_bins = _d['b1'], _d['b2']
else:
    xi_by_phase = {k: [] for k in range(N_MASS)}
    vr_by_phase = {k: [] for k in range(N_MASS)}
    vr_r_ref = {k: None for k in range(N_MASS)}
    r_xi = None
    for sim in ALL_PLANNED_PHASES:
        if os.path.exists(XI_JK_TMPL.format(sim=sim)):
            d = np.load(XI_JK_TMPL.format(sim=sim))
            for k in range(N_MASS):
                xi_by_phase[k].append(d['xi_per_bin'][k])
            r_xi = d['r_cens']
        for k in range(N_MASS):
            vr_path = VR_AUTO_TMPL.format(sim=sim, k=k)
            if not os.path.exists(vr_path):
                continue
            r_eff, mu, sd, npairs = vr_profile(PairHist.load(vr_path), 0, VR_R_EDGES)
            vr_by_phase[k].append(np.where(np.isfinite(mu) & (npairs > 1000), mu, np.nan))
            if vr_r_ref[k] is None:
                vr_r_ref[k] = r_eff

    b1_bins, b2_bins = np.full(N_MASS, np.nan), np.full(N_MASS, np.nan)
    for k in range(N_MASS):
        n_xi, n_vr = len(xi_by_phase[k]), len(vr_by_phase[k])
        if n_xi < 2 or n_vr < 2:
            print(f'bin {k}: {n_xi} xi_hh phases, {n_vr} v_r phases -- skipping')
            continue
        xi_arr = np.array(xi_by_phase[k])
        xi_mean = xi_arr.mean(axis=0)
        dxi = xi_arr - xi_mean
        xi_cov = (dxi.T @ dxi) / (n_xi * (n_xi - 1))
        vr_arr = np.array(vr_by_phase[k])
        vr_mean = np.nanmean(vr_arr, axis=0)
        vr_var = np.nanvar(vr_arr, axis=0, ddof=1) / n_vr
        good = np.isfinite(vr_mean) & (vr_var > 0)
        cov_vr = np.diag(np.where(good, vr_var, 1e30))     # diagonal (phase-to-phase) v_r covariance
        res = fit_bias_intrinsic(R_MIN_FIT, r_xi, xi_mean, xi_cov, max(1, n_xi - 2),
                                 vr_r_ref[k], np.where(good, vr_mean, 0.0), cov_vr, max(1, n_vr - 2))
        b1_bins[k], b2_bins[k] = res.x[:2]
        print(f'bin {k}: success={res.success}, b1={res.x[0]:.3f}, b2={res.x[1]:.3f}, '
              f'eps_xi={res.x[2]:.3f}, eps_v={res.x[3]:.3f}')
    np.savez(B1B2_CACHE, b1=b1_bins, b2=b2_bins)


def mean_vr_model(k):
    """<v_r|r> [km/s] from the LPT model at bin k's best-fit (b1, b2), as a spline in r."""
    r_tab = np.linspace(1.0, 200.0, 400)
    return InterpolatedUnivariateSpline(r_tab, model_vr(r_tab, b1_bins[k], b2_bins[k]), k=3, ext=3)


#%% B. single phase, whole grid: skew-t fits with nu = 6 and with nu free (cached)
sim_single = next((s for s in ALL_PLANNED_PHASES
                   if os.path.exists(VR_AUTO_TMPL.format(sim=s, k=SINGLE_BIN))), None)
if sim_single is None:
    raise RuntimeError(f'no phase has bin {SINGLE_BIN} auto-stats')
h_s = PairHist.load(VR_AUTO_TMPL.format(sim=sim_single, k=SINGLE_BIN))

sig_s = meas_sigv_grid(h_s, 0)
r_vr_s, vr_s, _, npairs_s = vr_profile(h_s, 0, VR_R_EDGES)
good_s = np.isfinite(vr_s) & (npairs_s > 1000)
vr_interp_s = InterpolatedUnivariateSpline(r_vr_s[good_s], vr_s[good_s], k=3, ext=3)
mu_meas = vr_interp_s(r3_2d) * r_l_2d / r3_2d     # measured mean projected on the LOS (calibration reference)

fit_cache = os.path.join(CACHE_DIR, f'skewt_fits_bin{SINGLE_BIN}_{sim_single}.npz')
if os.path.exists(fit_cache):
    print(f'loading cached skew-t fits from {fit_cache} (delete this file to force a re-fit)')
    _d = np.load(fit_cache)
    ST_free, ST_6 = _d['ST_free'], _d['ST_6']
else:
    ST_free = fit_skewt_grid(h_s, nu=None)
    ST_6 = fit_skewt_grid(h_s, nu=NU_FIXED)
    np.savez(fit_cache, ST_free=ST_free, ST_6=ST_6)

nu_free = ST_free[:, :, 3]
fin = np.isfinite(nu_free)
r_all, x_all, nu_all = r3_2d[fin], x_grid[fin], nu_free[fin]
rp_all = np.broadcast_to(r_p[:, None], r3_2d.shape)[fin]
at_bound = (nu_all < 1.01 * NU_BOUNDS[0]) | (nu_all > 0.99 * NU_BOUNDS[1])
free_pt = ~at_bound
print(f'free-nu fits ({sim_single}, bin {SINGLE_BIN}): {fin.sum()} converged of {fin.size} cells; '
      f'median nu = {np.median(nu_all):.1f}; {at_bound.mean():.1%} at a bound')


#%% B. plots: Delta residual (free nu - fixed nu), nu(r|r_perp) with a power-law fit, nu(x|r_perp)
dxi = (ST_free[:, :, 0] - ST_6[:, :, 0])[fin]       # = Delta_free - Delta_6 (the mean cancels)
ok6 = np.isfinite(dxi)

# single power law nu = A r^p, least squares in log-log; points at a bound are censored, so excluded
p_pl, lnA_pl = np.polyfit(np.log(r_all[free_pt]), np.log(nu_all[free_pt]), 1)
print(f'nu(r) power-law fit: nu = {np.exp(lnA_pl):.3g} * r^{p_pl:.3f} '
      f'({free_pt.sum()} points used, {at_bound.sum()} at a bound excluded)')

fig, ax = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
kw = dict(s=3, c=rp_all, cmap='viridis', vmin=r_p.min(), vmax=r_p.max(), rasterized=True)
sc = ax[0].scatter(x_all[ok6], dxi[ok6], **{**kw, 'c': rp_all[ok6]})
ax[1].scatter(r_all, nu_all, **kw)
ax[2].scatter(x_all, nu_all, **kw)
rr = np.linspace(r_all.min(), r_all.max(), 200)
ax[1].plot(rr, np.exp(lnA_pl) * rr**p_pl, c='r', lw=2.5, zorder=6,
           label=rf'power law: $\nu={np.exp(lnA_pl):.2g}\,r^{{{p_pl:.2f}}}$')
ax[1].plot([], [], '--', c='k', label=rf'fixed $\nu={NU_FIXED}$')
ax[0].axhline(0, c='k', ls='--', lw=1.5)
for a_ in ax[1:]:
    a_.axhline(NU_FIXED, c='k', ls='--', lw=1.5)
    #a_.set_yscale('log')
    a_.set_ylim(0, 150)
ax[0].set_xlabel(r'$x = r - r_\perp$ [$h^{-1}$ Mpc]')
ax[0].set_ylabel(rf'$\Delta_{{\rm free\ \nu}} - \Delta_{{\nu={NU_FIXED}}}$ [km/s]')
ax[1].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
ax[1].set_ylabel(r'$\nu(r|r_\perp)$')
ax[2].set_xlabel(r'$x = r - r_\perp$ [$h^{-1}$ Mpc]')
ax[2].set_ylabel(r'$\nu(x|r_\perp)$')
for a_ in ax:
    a_.margins(x=0)
ax[1].legend(fontsize=9, loc='lower right')
fig.colorbar(sc, ax=ax, location='right', pad=0.02, label=r'$r_\perp$ [$h^{-1}$ Mpc]')
fig.savefig(os.path.join(FIGURES_DIR, 'nu_scan_free_vs_fixed.png'), dpi=150)
print(f'saved {os.path.join(FIGURES_DIR, "nu_scan_free_vs_fixed.png")}')


#%% C. nu(r) two-parameter forms with nu(r=0) >= 2, fit to per-r-bin medians of the free-nu fits
R_BIN = 6.0          # r-bin width for the medians [h^-1 Mpc]
MIN_BIN_PTS = 30     # minimum cells per bin

NU_FORMS = {     # key: (label, function, starting values, (lower bounds, upper bounds)); all have nu(0) >= 2 by construction
    'shifted': (r'$2 + A\,r^p$',
                lambda r, A, p: 2 + A * r**p, [1.0, 0.7], ([1e-6, 0.01], [1e3, 5])),
    'line':    (r'$a + b\,r$,  $a\geq2$',
                lambda r, a, b: a + b * r, [8.0, 0.15], ([2.0, -5], [100, 5])),
    'log1p':   (r'$2 + b\ln(1+r/r_0)$',
                lambda r, b, r0: 2 + b * np.log1p(r / r0), [8.0, 30.0], ([0.0, 0.1], [100, 500])),
    'exp':     (r'$2+(\nu_0-2)(1-e^{-r/r_c})$',
                lambda r, ni, rc: 2 + (ni - 2) * (1 - np.exp(-r / rc)), [30.0, 80.0],
                ([2.01, 1.0], [1e3, 1e4])),
    'hill2':   (r'$2+(\nu_0-2)/(1+(r_c/r)^2)$',
                lambda r, ni, rc: 2 + (ni - 2) / (1 + (rc / r)**2), [25.0, 40.0],
                ([2.01, 1.0], [1e3, 500.0])),
    'hyper':   (r'$2+(\nu_0-2)\,r/(r+r_c)$',
                lambda r, ni, rc: 2 + (ni - 2) * r / (r + rc), [40.0, 60.0],
                ([2.01, 1.0], [1e3, 1e4])),
}

edges_r = np.arange(r_all.min(), r_all.max() + R_BIN, R_BIN)
ib = np.digitize(r_all, edges_r) - 1
r_med, nu_med, f_pin = [], [], []
for b in range(len(edges_r) - 1):
    s = ib == b
    if s.sum() < MIN_BIN_PTS:
        continue
    r_med.append(r_all[s].mean())
    nu_med.append(np.median(nu_all[s]))
    f_pin.append(at_bound[s].mean())
r_med, nu_med, f_pin = np.array(r_med), np.array(nu_med), np.array(f_pin)
if (f_pin > 0.3).any():
    print(f'WARNING: {np.sum(f_pin > 0.3)} bin(s) have >30% of cells at a nu bound; their medians are unreliable '
          f'(r = {np.round(r_med[f_pin > 0.3], 0)})')

nu_form_fits = {}
rr = np.linspace(r_all.min(), r_all.max(), 300)
fig, ax = plt.subplots(figsize=(8.5, 5.5), layout='constrained')
sc = ax.scatter(r_all[free_pt], nu_all[free_pt], s=3, c=rp_all[free_pt], cmap='viridis',
                vmin=r_p.min(), vmax=r_p.max(), alpha=0.6, rasterized=True)
ax.plot(r_med, nu_med, 'ko', ms=4, zorder=7, label='bin medians')
print(f'{"form":8s} {"rms ln nu (medians)":>20s} {"rms ln nu (cells)":>18s} {"nu(0)":>7s}  parameters')
for n_, (key, (label, f, p0, bnds)) in enumerate(NU_FORMS.items()):
    sol = least_squares(lambda th: np.log(np.clip(f(r_med, *th), 1e-6, None)) - np.log(nu_med),
                        p0, bounds=bnds, loss='soft_l1', f_scale=0.1)
    th = sol.x
    nu0 = float(f(1e-9, *th))
    assert nu0 >= 2.0 - 1e-6, f'{key}: nu(0) = {nu0} < 2'
    nu_form_fits[key] = th
    rms_med = np.sqrt(np.mean((np.log(np.clip(f(r_med, *th), 1e-6, None)) - np.log(nu_med))**2))
    rms_cell = np.sqrt(np.mean((np.log(np.clip(f(r_all[free_pt], *th), 1e-6, None)) - np.log(nu_all[free_pt]))**2))
    print(f'{key:8s} {rms_med:20.3f} {rms_cell:18.3f} {nu0:7.2f}  ' + ', '.join(f'{v:.3g}' for v in th))
    ax.plot(rr, f(rr, *th), lw=2.5, c=f'C{n_}', zorder=6, label=label)
ax.axhline(NU_FIXED, c='k', ls='--', lw=1.5)
ax.set_yscale('log')
ax.set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
ax.set_ylabel(r'$\nu(r|r_\perp)$')
ax.legend(fontsize=8, loc='lower right')
fig.colorbar(sc, ax=ax, label=r'$r_\perp$ [$h^{-1}$ Mpc]')
fig.savefig(os.path.join(FIGURES_DIR, 'nu_r_forms.png'), dpi=150)
print(f'saved {os.path.join(FIGURES_DIR, "nu_r_forms.png")}')

#%% F-a. free-nu skew-t fits per (phase, mass bin) on a thinned grid; raw (r, nu) points cached
HILL_STRIDE = 3        # fit every HILL_STRIDE-th R row and r_los column (1 = all cells, ~9x more fits)
HILL_CACHE = os.path.join(CACHE_DIR, 'nu_free_points_phase_bin.h5')


def free_nu_points_phase_bin(sim, k):
    """Returns (sim, k, r, nu, err): the free-nu fits on the thinned grid for one (phase, bin)."""
    path = VR_AUTO_TMPL.format(sim=sim, k=k)
    if not os.path.exists(path):
        return sim, k, None, None, 'file missing'
    try:
        h = PairHist.load(path)
    except Exception as e:
        return sim, k, None, None, f'PairHist.load: {type(e).__name__}: {e}'
    keep_cols = (np.arange(len(r_l)) % HILL_STRIDE == 0)
    r_pts, nu_pts = [], []
    for j in np.arange(0, len(r_p), HILL_STRIDE):
        row = fit_skewt_row(h.n[0][j], h.sum_v[0][j], h.sum_v2[0][j], h.counts[0][j], h.v_centers,
                            nu_row=None, keep=keep_cols)
        ok = np.isfinite(row[:, 3])
        r_pts.append(r3_2d[j][ok])
        nu_pts.append(row[ok, 3])
    return sim, k, np.concatenate(r_pts), np.concatenate(nu_pts), None


nu_pts_by = {}      # (sim, k) -> (r, nu)
if os.path.exists(HILL_CACHE):
    print(f'loading cached free-nu points from {HILL_CACHE} (delete this file to force a re-fit)')
    with h5.File(HILL_CACHE, 'r') as hdf:
        for sim in hdf:
            for gk in hdf[sim]:
                nu_pts_by[(sim, int(gk.split('_')[1]))] = (hdf[sim][gk]['r'][()], hdf[sim][gk]['nu'][()])
else:
    tasks = [(sim, k) for sim in ALL_PLANNED_PHASES for k in range(N_MASS)
             if os.path.exists(VR_AUTO_TMPL.format(sim=sim, k=k))]
    print(f'free-nu points: {len(tasks)} (phase, bin) combinations -- fitting (will cache to {HILL_CACHE})')
    results = Parallel(n_jobs=-1, verbose=10)(delayed(free_nu_points_phase_bin)(s, k) for s, k in tasks)
    err_counts = {}
    for sim, k, r_, nu_, err in results:
        if err is not None:
            err_counts[err] = err_counts.get(err, 0) + 1
        else:
            nu_pts_by[(sim, k)] = (r_, nu_)
    if err_counts:
        print(f'{sum(err_counts.values())} task(s) failed:', err_counts)
    with h5.File(HILL_CACHE, 'w') as hdf:
        for (sim, k), (r_, nu_) in nu_pts_by.items():
            g_ = hdf.require_group(sim).create_group(f'bin_{k}')
            g_.create_dataset('r', data=r_)
            g_.create_dataset('nu', data=nu_)
print(f'{len(nu_pts_by)} (phase, bin) point sets available')


#%% F-b. hill2 parameters per (phase, mass bin) and their mass dependence (Fig-5 style)
HILL_BIN = 6.0           # r-bin width for the medians [h^-1 Mpc]
HILL_MIN_BIN_PTS = 10    # minimum cells per r-bin (per phase there are fewer cells than in cell C)
HILL_MIN_BINS = 6        # minimum number of r-bins to fit a (phase, bin)

# same form, starting values and bounds as NU_FORMS['hill2'] in cell C
hill2 = lambda r, ni, rc: 2 + (ni - 2) / (1 + (rc / r)**2)
H_P0, H_LB, H_UB = [25.0, 40.0], np.array([2.01, 1.0]), np.array([1e3, 500.0])
labels_h = [r'$\nu_0$', r'$r_c$ [$h^{-1}$ Mpc]']


def fit_hill2(r_pts, nu_pts):
    edges = np.arange(r_pts.min(), r_pts.max() + HILL_BIN, HILL_BIN)
    ib = np.digitize(r_pts, edges) - 1
    rm, nm = [], []
    for b in range(len(edges) - 1):
        s = ib == b
        if s.sum() >= HILL_MIN_BIN_PTS:
            rm.append(r_pts[s].mean())
            nm.append(np.median(nu_pts[s]))
    if len(rm) < HILL_MIN_BINS:
        return None
    rm, nm = np.array(rm), np.array(nm)
    sol = least_squares(lambda th: np.log(np.clip(hill2(rm, *th), 1e-6, None)) - np.log(nm),
                        H_P0, bounds=(H_LB, H_UB), loss='soft_l1', f_scale=0.1)
    return sol.x


by_bin_h = {k: {} for k in range(N_MASS)}
for (sim, k), (r_, nu_) in nu_pts_by.items():
    th = fit_hill2(r_, nu_)
    if th is not None:
        by_bin_h[k][sim] = th

params_h, sims_h, mean_h, jack_h, std_h = {}, {}, np.full((N_MASS, 2), np.nan), np.full((N_MASS, 2), np.nan), np.full((N_MASS, 2), np.nan)
for k in range(N_MASS):
    sims_k = sorted(by_bin_h[k])
    n_bnd = sum(bool(np.any((by_bin_h[k][s] <= H_LB * 1.001) | (by_bin_h[k][s] >= H_UB * 0.999))) for s in sims_k)
    print(f'bin {k}: {len(sims_k)} hill2 fits ({n_bnd} with a parameter at a bound)')
    if len(sims_k) >= 2:
        params_h[k] = np.array([by_bin_h[k][s] for s in sims_k])
        sims_h[k] = sims_k
        mean_h[k], jack_h[k] = jackknife_mean_err(params_h[k])
        std_h[k] = params_h[k].std(axis=0)
        print(f'        nu_inf = {mean_h[k, 0]:.2f} +/- {jack_h[k, 0]:.2f} (jackknife),  '
              f'r_c = {mean_h[k, 1]:.2f} +/- {jack_h[k, 1]:.2f}')

avail_h = sorted(params_h)
if avail_h:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for a_ in ax:
        a_.set_xscale('log')
    all_sims_h = sorted(set(s for k in avail_h for s in sims_h[k]))
    grand_h = np.concatenate([params_h[k] for k in avail_h], axis=0).mean(axis=0)
    for i, label in enumerate(labels_h):
        for sim in all_sims_h:
            ks_sim = [k for k in avail_h if sim in sims_h[k]]
            if ks_sim:
                ax[i].plot(M_CENS[ks_sim], [params_h[k][sims_h[k].index(sim), i] for k in ks_sim],
                           marker='o', alpha=0.3)
        ax[i].errorbar(M_CENS[avail_h], mean_h[avail_h, i], yerr=jack_h[avail_h, i],
                       marker='o', color='k', ls='none', capsize=3,
                       label='jackknife mean' if i == 0 else None, zorder=5)
        ax[i].plot(M_CENS[avail_h], mean_h[avail_h, i] + std_h[avail_h, i], 'k--', lw=2,
                   label=r'$\pm$ standard dev over phases' if i == 0 else None)
        ax[i].plot(M_CENS[avail_h], mean_h[avail_h, i] - std_h[avail_h, i], 'k--', lw=2)
        ax[i].axhline(grand_h[i], c='k', ls=':', lw=3,
                      label='mean over all phases, bins' if i == 0 else None)
        ax[i].set_ylabel(label)
        ax[i].set_xlabel(r'$M\ [h^{-1} M_\odot]$')
        ax[i].set_xlim(M_CENS[avail_h].min() / 1.5, M_CENS[avail_h].max() * 1.5)
    ax[0].legend(fontsize=9, loc='upper left')
    fig.savefig(os.path.join(FIGURES_DIR, 'nu_hill2_mass_dependence.png'), dpi=150)
    print(f'saved {os.path.join(FIGURES_DIR, "nu_hill2_mass_dependence.png")}')
else:
    print('hill2: no bin has enough phases yet -- nothing to plot')

#%% G-a. refit Delta with nu(r) enforced to the hill2 model (jackknife-mean and grand-mean parameters); same phase / mass bin as B
RP_MAX_PLOT, R3_MAX_PLOT = 30.0, 70.0      # display range of the Delta-fit plots (as in Fig 4)
NU_MIN_MODEL = 2.05                        # safety floor on nu(r) (the variance needs nu > 2)

if not np.all(np.isfinite(mean_h[SINGLE_BIN])):
    raise RuntimeError(f'no hill2 jackknife mean for mass bin {SINGLE_BIN} (cell F-b)')
HILL_VARIANTS = {                          # name: (file slug, (nu_inf, r_c))
    'jackknife mean': ('jackknife', mean_h[SINGLE_BIN]),
    'grand mean':     ('grand', grand_h),
}


def nu_hill(r, pars):
    return np.maximum(hill2(r, *pars), NU_MIN_MODEL)


sig_params = fit_sigma_joint(x_grid, r_p, sig_s)
mean_vr_g = mean_vr_model(SINGLE_BIN)      # predictions use the b1/b2 LPT mean (cell A)

# starting point for the Delta fits: the nu = 6 calibration data from cell B
del_6 = fit_delta_joint(x_grid, r_p, ST_6[:, :, 0] - mu_meas)
print('Delta params (A_delta0, r_delta, sigma_delta, beta_delta, alpha):')
print('  nu = 6            : ' + ', '.join(f'{v:.3g}' for v in del_6))

fits_g = {}
for name, (slug, pars) in HILL_VARIANTS.items():
    cache = os.path.join(CACHE_DIR, f'skewt_fits_hill2_bin{SINGLE_BIN}_{sim_single}_{pars[0]:.4f}_{pars[1]:.4f}.npz')
    if os.path.exists(cache):
        ST = np.load(cache)['ST']
    else:
        ST = fit_skewt_grid(h_s, nu=nu_hill(r3_2d, pars))      # xi, omega, alpha free; nu fixed per cell
        np.savez(cache, ST=ST)
    Delta = ST[:, :, 0] - mu_meas
    dpars = fit_delta_joint(x_grid, r_p, Delta, p0=del_6)
    fits_g[name] = dict(slug=slug, pars=pars, Delta=Delta, dpars=dpars)
    print(f'  {name:17s} : ' + ', '.join(f'{v:.3g}' for v in dpars)
          + f'   (nu_inf = {pars[0]:.2f}, r_c = {pars[1]:.2f})')


#%% G-b. plots: (A) Delta fits, restricted r_perp bins; (B, C) 4x4 v_los comparisons for each hill2 model
G_SEL = np.linspace(15.5, 139.5, 4)     # r_perp (columns) and r_los (rows): 4 bins spanning [15, 140]
G_V_LIM, G_V_PLOT = 2500.0, 2000.0      # curve range / displayed x-range [km/s]
G_MIN_COUNT = 1                          # rebinned v-bins with fewer pairs are masked
G_Y_FLOOR = 1e-5                         # log-y lower limit, as a fraction of each panel's peak

cent_g = h_s.v_centers.reshape(-1, REBIN).mean(axis=1)
dv_g = (h_s.v_centers[1] - h_s.v_centers[0]) * REBIN
edges_g = np.r_[cent_g - dv_g / 2, cent_g[-1] + dv_g / 2]
vv_g = np.linspace(-G_V_LIM, G_V_LIM, 600)
theta_g = np.linspace(0, np.pi / 2, 100)

g_cells = [(np.argmin(np.abs(r_p - rp)), np.argmin(np.abs(r_l - rl))) for rl in G_SEL for rp in G_SEL]
R_INSET_G = 10.0 * np.ceil((max(r3_2d[j, i] for j, i in g_cells) + 5.0) / 10.0)
INSET_PAD_G = 0.12 * R_INSET_G
fmt_var = lambda v: rf'${v / 1e5:.2f}\times10^5$'


def gauss_pdf(v, mu, sd):
    return np.exp(-0.5 * ((v - mu) / sd)**2) / (np.sqrt(2 * np.pi) * sd)


def model_pdf_g(pars_nu, dpars, rp, rl, v):
    """Compact model with nu = hill2(r): returns (pdf, sigma_v^2, nu)."""
    r = np.hypot(rp, rl)
    x = r - rp
    nu = float(nu_hill(r, pars_nu))
    mu = mean_vr_g(r) * rl / r
    sig = sigma_mod_2d((x, rp), *sig_params)
    delta = delta_mod_alpha_R((x, rp), *dpars)
    omega, alpha = skewt_params_from_delta(delta, sig, nu)
    return skewed_t_pdf(v, mu + delta, omega, alpha, nu), float(sig)**2, nu


def plot_delta_fits(name, fit):
    """Fig-4-style plot: Delta(r|r_perp) and Delta(x) for r_perp <= RP_MAX_PLOT, r <= R3_MAX_PLOT, with the fit."""
    D, dp, pars = fit['Delta'], fit['dpars'], fit['pars']
    rows = np.where(r_p <= RP_MAX_PLOT)[0]
    cmap = plt.cm.viridis(np.linspace(0, 1, len(rows)))
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for c, j in zip(cmap, rows):
        m = np.isfinite(D[j]) & (r3_2d[j] <= R3_MAX_PLOT)
        ax[0].plot(r3_2d[j][m], D[j][m], 'o', ms=3, c=c)
        ax[1].plot(x_grid[j][m], D[j][m], 'o', ms=3, c=c)
        if m.sum() < 2:
            continue
        xs = np.linspace(x_grid[j][m].min(), x_grid[j][m].max(), 500)
        ax[1].plot(xs, delta_mod_alpha(xs, A_delta_mod(r_p[j], dp[0], dp[1]), *dp[2:]), '--', c=c, lw=1)
    ax[0].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
    ax[0].set_ylabel(r'$\Delta(r|r_\perp)$ [km/s]')
    ax[1].set_xlabel(r'$x = r - r_\perp$ [$h^{-1}$ Mpc]')
    ax[1].set_ylabel(r'$\Delta(x)$ [km/s]')
    for a_ in ax:
        a_.margins(x=0)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=r_p[rows[0]], vmax=RP_MAX_PLOT))
    fig.colorbar(sm, ax=ax, location='right', pad=0.02, label=r'$r_\perp$ [$h^{-1}$ Mpc]')
    fig.suptitle(rf'hill2 $\nu(r)$, {name}: $\nu_0={pars[0]:.1f}$, $r_c={pars[1]:.1f}$', fontsize=12)
    fname = f'g_delta_fits_{fit["slug"]}.png'
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150)
    print(f'saved {os.path.join(FIGURES_DIR, fname)}')


def plot_vlos_grid(name, fit, logy):
    """4x4 Fig-7-style plot: histogram vs moment-matched Gaussian vs compact model with this hill2 nu(r)."""
    pars, dpars = fit['pars'], fit['dpars']
    n_ax = len(G_SEL)
    fig, axs = plt.subplots(n_ax, n_ax, figsize=(4.4 * n_ax, 3.7 * n_ax), layout='constrained', sharex=True)
    for a, rl_t in enumerate(G_SEL):
        for b_, rp_t in enumerate(G_SEL):
            ax_ = axs[a, b_]
            j, i = np.argmin(np.abs(r_p - rp_t)), np.argmin(np.abs(r_l - rl_t))
            n = h_s.n[0][j, i]
            if n < N_MIN:
                ax_.set_title('too few pairs')
                continue
            counts = h_s.counts[0][j, i].reshape(-1, REBIN).sum(axis=1)
            norm = counts.sum() * dv_g
            hist = counts / norm
            ok = counts >= G_MIN_COUNT
            ax_.stairs(np.where(ok, hist, 1e-30), edges_g, color='k', lw=1, label='simulation')
            err = np.sqrt(counts) / norm
            err_lo = np.minimum(err, 0.999 * hist) if logy else err
            ax_.errorbar(cent_g[ok], hist[ok], yerr=[err_lo[ok], err[ok]], fmt='none',
                         ecolor='k', elinewidth=0.8, capsize=0)

            # Gaussian matched to the histogram's mean and variance (pair sums of this phase)
            mean_hs = h_s.sum_v[0][j, i] / n
            var_hs = h_s.sum_v2[0][j, i] / n - mean_hs**2
            pdf_g = gauss_pdf(vv_g, mean_hs, np.sqrt(var_hs))
            pdf_c, var_c, nu_c = model_pdf_g(pars, dpars, r_p[j], r_l[i], vv_g)
            ax_.plot(vv_g, pdf_g, c='C1', lw=2, label=rf'Gaussian: $\sigma^2=$ {fmt_var(var_hs)}')
            ax_.plot(vv_g, pdf_c, c='C0', lw=2, label=rf'compact ($\nu={nu_c:.1f}$): $\sigma^2=$ {fmt_var(var_c)}')

            ax_.axvline(0, c='gray', ls=':', lw=1)
            ax_.set_xlim(-G_V_PLOT, G_V_PLOT)
            if logy:
                ax_.set_yscale('log')
                ax_.set_ylim(np.nanmax(hist) * G_Y_FLOOR, np.nanmax(hist) * 2)
            else:
                ax_.set_ylim(0, 1.1 * max(np.nanmax(hist), pdf_g.max(), pdf_c.max()))
            ax_.legend(fontsize=6.5, framealpha=0.85, labelspacing=0.3,
                       loc='lower center' if logy else 'lower right')
            ax_.set_title(rf'$r_\perp\in[{r_perp_edges[j]:.2f}, {r_perp_edges[j+1]:.2f}]$, '
                          rf'$r_{{\rm los}}\in[{r_los_edges[i]:.2f}, {r_los_edges[i+1]:.2f}]$ '
                          r'$h^{-1}$Mpc', fontsize=9)
            if a == n_ax - 1:
                ax_.set_xlabel(r'$v_{\rm los}$ [km/s]')

            # inset: where this bin sits on the (R, r_los) quarter-disk
            ins = ax_.inset_axes([0.82, 0.72, 0.16, 0.24])
            ins.fill(np.r_[-INSET_PAD_G, R_INSET_G, R_INSET_G * np.cos(theta_g), -INSET_PAD_G],
                     np.r_[-INSET_PAD_G, -INSET_PAD_G, R_INSET_G * np.sin(theta_g), R_INSET_G],
                     c='C1', alpha=0.5, lw=0)
            ins.axvspan(r_perp_edges[j], r_perp_edges[j + 1], color='C0', alpha=0.4, lw=0)
            ins.axhspan(r_los_edges[i], r_los_edges[i + 1], color='C0', alpha=0.4, lw=0)
            ins.axvline(r_p[j], c='gray', lw=1)
            ins.axhline(r_l[i], c='gray', lw=1)
            ins.set_xlim(-INSET_PAD_G, R_INSET_G)
            ins.set_ylim(-INSET_PAD_G, R_INSET_G)
            ins.set_xticks([])
            ins.set_yticks([])
            ins.set_xlabel(r'$R$', fontsize=6, labelpad=1)
            ins.set_ylabel(r'$r_{\rm LOS}$', fontsize=6, labelpad=1)

    fig.supylabel(r'$P(v_{\rm los}\,|\,r_\perp, r_{\rm los})$', fontsize=20)
    fig.suptitle(rf'hill2 $\nu(r)$, {name}: $\nu_0={pars[0]:.1f}$, $r_c={pars[1]:.1f}$', fontsize=13)
    fname = f'g_vlos_{fit["slug"]}_{"log" if logy else "linear"}.png'
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150)
    print(f'saved {os.path.join(FIGURES_DIR, fname)} ({sim_single}, bin {SINGLE_BIN})')


for name, fit in fits_g.items():
    plot_delta_fits(name, fit)                  # A
    for logy in (True, False):                  # B (jackknife mean), C (grand mean)
        plot_vlos_grid(name, fit, logy)

#%% G-d. side-by-side Delta(x) scatter: nu(r) = hill2 grand-mean model vs the free-nu skew-t fits (single phase / mass bin, as in B and G-a)
D_hill = fits_g['grand mean']['Delta']             # xi - mu_meas with nu fixed to hill2(r; grand-mean nu_0, r_c), cell G-a
D_free = ST_free[:, :, 0] - mu_meas                # xi - mu_meas with nu free per cell, cell B
nu0_g, rc_g = fits_g['grand mean']['pars']

rows = np.where(r_p <= RP_MAX_PLOT)[0]
cmap = plt.cm.viridis(np.linspace(0, 1, len(rows)))
fig, ax = plt.subplots(1, 2, figsize=(12, 5), layout='constrained', sharex=True, sharey=True)
for c, j in zip(cmap, rows):
    for a_, D in zip(ax, (D_hill, D_free)):
        m = np.isfinite(D[j]) & (r3_2d[j] <= R3_MAX_PLOT)
        a_.plot(x_grid[j][m], D[j][m], 'o', ms=3, c=c)
for a_ in ax:
    a_.set_xlabel(r'$x = r - r_\perp$ [$h^{-1}$ Mpc]')
    a_.margins(x=0)
ax[0].set_ylabel(r'$\Delta(x)$ [km/s]')
ax[0].set_title(rf'$\nu(r)$ hill2, grand mean ($\nu_0={nu0_g:.1f}$, $r_c={rc_g:.1f}$)', fontsize=12)
ax[1].set_title(r'free $\nu$ per cell', fontsize=12)
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=r_p[rows[0]], vmax=RP_MAX_PLOT))
fig.colorbar(sm, ax=ax, location='right', pad=0.02, label=r'$r_\perp$ [$h^{-1}$ Mpc]')
fname = 'g_delta_x_hill2_vs_free.png'
fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150)
print(f'saved {os.path.join(FIGURES_DIR, fname)}')

#%% G-e. percent difference between the free-nu and hill2-nu Delta values, per cell (same cells and data as G-d)
PCT_MIN_DELTA = 20.0       # cells with |Delta_hill| [km/s] below this are excluded (the percent difference blows up)
X_BIN = 4.0                # x-bin width for the median trend [h^-1 Mpc]

pct = 100.0 * (D_free - D_hill) / D_hill                  # D_hill, D_free from cell G-d
in_region = np.isfinite(pct) & (r3_2d <= R3_MAX_PLOT) & (r_p_2d <= RP_MAX_PLOT)
use = in_region & (np.abs(D_hill) >= PCT_MIN_DELTA)
q16, q50, q84 = np.percentile(pct[use], [16, 50, 84])
print(f'{use.sum()} cells plotted ({(in_region & ~use).sum()} excluded with |Delta_hill| < {PCT_MIN_DELTA:g} km/s); '
      f'percent difference: median = {q50:.2f}%, 16-84% = [{q16:.2f}, {q84:.2f}]%')

edges_x = np.arange(0.0, R3_MAX_PLOT + X_BIN, X_BIN)
xb, lo, med, hi = [], [], [], []
for a_, b_ in zip(edges_x[:-1], edges_x[1:]):
    s = use & (x_grid >= a_) & (x_grid < b_)
    if s.sum() >= 10:
        xb.append(x_grid[s].mean())
        ql, qm, qh = np.percentile(pct[s], [16, 50, 84])
        lo.append(ql); med.append(qm); hi.append(qh)

fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
sc = ax.scatter(x_grid[use], pct[use], s=6, c=r_p_2d[use], cmap='viridis',
                vmin=r_p.min(), vmax=RP_MAX_PLOT, rasterized=True)
ax.fill_between(xb, lo, hi, color='k', alpha=0.15, lw=0)
ax.plot(xb, med, 'k-', lw=2, label=r'median (16-84\% band) in $x$ bins')
ax.axhline(0, c='k', ls='--', lw=1.5)
lim = 1.2 * np.percentile(np.abs(pct[use]), 99)
ax.set_ylim(-lim, lim)
ax.set_xlabel(r'$x = r - r_\perp$ [$h^{-1}$ Mpc]')
ax.set_ylabel(r'$100\,(\Delta_{\rm free\ \nu}-\Delta_{\rm hill2})/\Delta_{\rm hill2}$ [\%]')
ax.margins(x=0)
ax.legend(fontsize=9, loc='upper right')
fig.colorbar(sc, ax=ax, label=r'$r_\perp$ [$h^{-1}$ Mpc]')
fname = 'g_delta_pct_free_vs_hill2.png'
fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150)
print(f'saved {os.path.join(FIGURES_DIR, fname)}')

#%% I-a. free-nu skew-t fits per (phase, mass bin): (r, r_perp, nu, Delta) per cell, then the Delta-model fit per (phase, bin); cached
I_STRIDE = 3               # fit every I_STRIDE-th R row and r_los column (1 = full grid, as in Fig 5)
EXCLUDE_PINNED = False     # True: drop cells whose fitted nu sits at a bound from the Delta-model fit
P0_I = fits_g['grand mean']['dpars'] if 'fits_g' in globals() else DEL_P0     # starting point for the Delta fits (cell G-a, if run)
I_CACHE = os.path.join(CACHE_DIR, f'free_nu_delta_stride{I_STRIDE}.h5')


def free_fit_phase_bin(sim, k):
    """Returns (sim, k, {r, rp, nu, delta} over converged cells or None, err string or None).
    Delta = xi_free - measured <v_r|r> r_los / r, with the measured profile of this (phase, bin)."""
    path = VR_AUTO_TMPL.format(sim=sim, k=k)
    if not os.path.exists(path):
        return sim, k, None, 'file missing'
    try:
        h = PairHist.load(path)
    except Exception as e:
        return sim, k, None, f'PairHist.load: {type(e).__name__}: {e}'
    r_vr, vr_, _, npairs = vr_profile(h, 0, VR_R_EDGES)
    good = np.isfinite(vr_) & (npairs > 1000)
    if good.sum() < 4:
        return sim, k, None, f'vr_profile: only {good.sum()} usable radial bins'
    vr_interp = InterpolatedUnivariateSpline(r_vr[good], vr_[good], k=3, ext=3)
    keep_cols = (np.arange(len(r_l)) % I_STRIDE == 0)
    ST = np.full(r3_2d.shape + (4,), np.nan)
    for j in np.arange(0, len(r_p), I_STRIDE):
        ST[j] = fit_skewt_row(h.n[0][j], h.sum_v[0][j], h.sum_v2[0][j], h.counts[0][j], h.v_centers,
                              nu_row=None, keep=keep_cols)
    fin = np.isfinite(ST[:, :, 3])
    Delta = ST[:, :, 0] - vr_interp(r3_2d) * r_l_2d / r3_2d
    return sim, k, dict(r=r3_2d[fin], rp=r_p_2d[fin], nu=ST[:, :, 3][fin], delta=Delta[fin]), None


def fit_delta_points(pts, p0):
    """Delta-model fit to a (phase, bin)'s per-cell points (same model and floor as fit_delta_joint)."""
    m = np.isfinite(pts['delta']) & (pts['delta'] > DEL_FLOOR)
    if EXCLUDE_PINNED:
        m &= (pts['nu'] > 1.01 * NU_BOUNDS[0]) & (pts['nu'] < 0.99 * NU_BOUNDS[1])
    x = pts['r'][m] - pts['rp'][m]
    popt, _ = curve_fit(delta_mod_alpha_R, (x, pts['rp'][m]), pts['delta'][m], p0=p0, maxfev=20_000)
    return popt


pts_by = {}      # (sim, k) -> {r, rp, nu, delta}
if os.path.exists(I_CACHE):
    print(f'loading cached free-nu / Delta points from {I_CACHE} (delete this file to force a re-fit)')
    with h5.File(I_CACHE, 'r') as hdf:
        for sim in hdf:
            for gk in hdf[sim]:
                pts_by[(sim, int(gk.split('_')[1]))] = {f: hdf[sim][gk][f][()] for f in ('r', 'rp', 'nu', 'delta')}
else:
    tasks = [(sim, k) for sim in ALL_PLANNED_PHASES for k in range(N_MASS)
             if os.path.exists(VR_AUTO_TMPL.format(sim=sim, k=k))]
    print(f'free-nu fits: {len(tasks)} (phase, bin) combinations -- fitting (will cache to {I_CACHE})')
    results = Parallel(n_jobs=-1, verbose=10)(delayed(free_fit_phase_bin)(s, k) for s, k in tasks)
    err_counts = {}
    for sim, k, pts, err in results:
        if err is not None:
            err_counts[err] = err_counts.get(err, 0) + 1
        else:
            pts_by[(sim, k)] = pts
    if err_counts:
        print(f'{sum(err_counts.values())} task(s) failed:', err_counts)
    with h5.File(I_CACHE, 'w') as hdf:
        for (sim, k), pts in pts_by.items():
            g_ = hdf.require_group(sim).create_group(f'bin_{k}')
            for f, v in pts.items():
                g_.create_dataset(f, data=v)
print(f'{len(pts_by)} (phase, bin) point sets available')

# the same fits give nu(r) points: fills nu_pts_by in the format cell F-b reads, so cell F-a can be skipped
nu_pts_by = {key: (p['r'], p['nu']) for key, p in pts_by.items()}

# Delta-model fit per (phase, bin) from the free-fit xi
by_bin_i, fail_i = {k: {} for k in range(N_MASS)}, {}
for (sim, k), pts in pts_by.items():
    try:
        by_bin_i[k][sim] = fit_delta_points(pts, P0_I)
    except Exception as e:
        msg = f'fit_delta_points: {type(e).__name__}: {e}'
        fail_i[msg] = fail_i.get(msg, 0) + 1
if fail_i:
    print(f'{sum(fail_i.values())} Delta fit(s) failed:', fail_i)

labels_i = [r'$A_{\Delta,0}$', r'$r_\Delta$ [$h^{-1}$ Mpc]', r'$\sigma_\Delta$ [$h^{-1}$ Mpc]',
            r'$\beta_\Delta$', r'$\alpha_\Delta$']
per_bin_i, per_bin_sims_i = {}, {}
mean_i, jack_i, std_i = (np.full((N_MASS, 5), np.nan) for _ in range(3))
for k in range(N_MASS):
    sims_k = sorted(by_bin_i[k])
    if len(sims_k) >= 2:
        per_bin_i[k] = np.array([by_bin_i[k][s] for s in sims_k])
        per_bin_sims_i[k] = sims_k
        mean_i[k], jack_i[k] = jackknife_mean_err(per_bin_i[k])
        std_i[k] = per_bin_i[k].std(axis=0)
        print(f'bin {k} ({len(sims_k)} phases): ' + ',  '.join(
            f'{lab.split(" [")[0]} = {mean_i[k, i]:.3g} +/- {jack_i[k, i]:.2g}' for i, lab in enumerate(labels_i)))
    else:
        print(f'bin {k}: fewer than 2 phases with a Delta fit -- skipping')


#%% I-b. mass dependence of the Delta-model parameters, Delta calibrated from the free-nu skew-t fits (Fig-5 style)
avail_i = sorted(per_bin_i)
if avail_i:
    fig, axs = plt.subplots(2, 3, figsize=(15, 9), layout='constrained')
    axs = axs.ravel()
    for a_ in axs[:5]:
        a_.set_xscale('log')
    all_sims_i = sorted(set(s for k in avail_i for s in per_bin_sims_i[k]))
    grand_i = np.concatenate([per_bin_i[k] for k in avail_i], axis=0).mean(axis=0)
    for i, label in enumerate(labels_i):
        for sim in all_sims_i:
            ks_sim = [k for k in avail_i if sim in per_bin_sims_i[k]]
            if ks_sim:
                axs[i].plot(M_CENS[ks_sim], [per_bin_i[k][per_bin_sims_i[k].index(sim), i] for k in ks_sim],
                            marker='o', alpha=0.3)
        axs[i].errorbar(M_CENS[avail_i], mean_i[avail_i, i], yerr=jack_i[avail_i, i],
                        marker='o', color='k', ls='none', capsize=3,
                        label='jackknife mean' if i == 0 else None, zorder=5)
        axs[i].plot(M_CENS[avail_i], mean_i[avail_i, i] + std_i[avail_i, i], 'k--', lw=2,
                    label=r'$\pm$ standard dev over phases' if i == 0 else None)
        axs[i].plot(M_CENS[avail_i], mean_i[avail_i, i] - std_i[avail_i, i], 'k--', lw=2)
        axs[i].axhline(grand_i[i], c='k', ls=':', lw=3,
                       label='mean over all phases, bins' if i == 0 else None)
        axs[i].set_ylabel(label)
        axs[i].set_xlim(M_CENS[avail_i].min() / 1.5, M_CENS[avail_i].max() * 1.5)
        if i >= 2:
            axs[i].set_xlabel(r'$M\ [h^{-1} M_\odot]$')
    handles, leg_labels = axs[0].get_legend_handles_labels()
    axs[5].axis('off')
    axs[5].legend(handles, leg_labels, fontsize=12, loc='center')
    fig.suptitle(r'$\Delta$ parameters calibrated from the free-$\nu$ skew-t fits', fontsize=13)
    fname = 'delta_params_free_nu_mass_dependence.png'
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150)
    print(f'saved {os.path.join(FIGURES_DIR, fname)}')
else:
    print('no bin has enough phases yet -- nothing to plot')


#%% I-c. 4x4 v_los plots: grand-mean Delta parameters from the free-nu fits (I-a) + grand-mean hill2 nu(r) (F-b)
DEL_GRAND_FREE = np.concatenate([per_bin_i[k] for k in sorted(per_bin_i)], axis=0).mean(axis=0)
print('grand-mean Delta params (A_delta0, r_delta, sigma_delta, beta_delta, alpha): '
      + ', '.join(f'{v:.3g}' for v in DEL_GRAND_FREE))
fit_grand_free = dict(slug='grand_free', pars=grand_h, dpars=DEL_GRAND_FREE)
for logy in (True, False):
    plot_vlos_grid('grand-mean $\\Delta$ (free-$\\nu$ calibration)', fit_grand_free, logy)



#%% J-a. sigma_v parameters per (phase, mass bin) (fixed pivot R_PIV = 72.5, as in model.py); cached
SIGMA_CACHE = os.path.join(CACHE_DIR, 'sigma_params_phase_bin.h5')


def fit_sigma_phase_bin(sim, k):
    """Returns (sim, k, [sigma0, A_rho0, A_rho_p, sigma_rho] or None, err string or None)."""
    path = VR_AUTO_TMPL.format(sim=sim, k=k)
    if not os.path.exists(path):
        return sim, k, None, 'file missing'
    try:
        h = PairHist.load(path)
    except Exception as e:
        return sim, k, None, f'PairHist.load: {type(e).__name__}: {e}'
    try:
        popt = fit_sigma_joint(x_grid, r_p, meas_sigv_grid(h, 0))
    except Exception as e:
        return sim, k, None, f'fit_sigma_joint: {type(e).__name__}: {e}'
    popt = popt[0] if isinstance(popt, tuple) else popt       # (popt, R_piv) if the paper-script version is loaded
    return sim, k, np.asarray(popt, dtype=float), None


per_bin_s, per_bin_sims_s = {}, {}
if os.path.exists(SIGMA_CACHE):
    print(f'sigma_v parameters: loading cached fits from {SIGMA_CACHE} (delete this file to force a re-fit)')
    with h5.File(SIGMA_CACHE, 'r') as hdf:
        for k in range(N_MASS):
            if f'bin_{k}' in hdf:
                per_bin_s[k] = hdf[f'bin_{k}']['params'][()]
                per_bin_sims_s[k] = [s.decode() if isinstance(s, bytes) else s for s in hdf[f'bin_{k}']['sims'][()]]
else:
    tasks = [(sim, k) for sim in ALL_PLANNED_PHASES for k in range(N_MASS)
             if os.path.exists(VR_AUTO_TMPL.format(sim=sim, k=k))]
    print(f'sigma_v parameters: {len(tasks)} (phase, bin) combinations -- fitting (will cache to {SIGMA_CACHE})')
    results = Parallel(n_jobs=-1, verbose=5)(delayed(fit_sigma_phase_bin)(s, k) for s, k in tasks)
    by_bin_s, err_counts = {k: {} for k in range(N_MASS)}, {}
    for sim, k, par, err in results:
        if par is not None:
            by_bin_s[k][sim] = par
        if err is not None:
            err_counts[err] = err_counts.get(err, 0) + 1
    if err_counts:
        print(f'{sum(err_counts.values())} task(s) failed:', err_counts)
    with h5.File(SIGMA_CACHE, 'w') as hdf:
        for k in range(N_MASS):
            sims_k = sorted(by_bin_s[k])
            if len(sims_k) >= 2:
                per_bin_s[k] = np.array([by_bin_s[k][s] for s in sims_k])
                per_bin_sims_s[k] = sims_k
                gk = hdf.create_group(f'bin_{k}')
                gk.create_dataset('params', data=per_bin_s[k])
                gk.create_dataset('sims', data=np.array(sims_k, dtype=h5.string_dtype()))
for k in sorted(per_bin_s):
    print(f'bin {k}: {len(per_bin_sims_s[k])} phases')


#%% J-b. all 11 velocity-model parameters vs mass (sigma_v, Delta from the free-nu fits, nu), Fig-5 order, no title
# groups: (per-bin parameter arrays, per-bin phase lists, column indices, labels)
groups = [
    (per_bin_s, per_bin_sims_s, [0, 1, 2, 3],
     [r'$\sigma_0$ [km/s]', r'$A_{\rho,0}$', r'$A_{\rho,p}$ [$h^{-1}$ Mpc]', r'$\sigma_\rho$ [$h^{-1}$ Mpc]']),
    (per_bin_i, per_bin_sims_i, [0, 1, 2, 3, 4],
     [r'$A_{\Delta,0}$', r'$r_\Delta$ [$h^{-1}$ Mpc]', r'$\sigma_\Delta$ [$h^{-1}$ Mpc]', r'$\beta_\Delta$', r'$\alpha_\Delta$']),
    (params_h, sims_h, [0, 1],
     [r'$\nu_0$', r'$r_c$ [$h^{-1}$ Mpc]']),
]
panels = [(pb, ps, col, lab) for pb, ps, cols, labs in groups for col, lab in zip(cols, labs)]     # 4 + 5 + 2 = 11
avail_all = sorted(set().union(*[set(pb) for pb, _, _, _ in panels]))
if not avail_all:
    raise RuntimeError('no parameter fits available (cells J-a, I-a/I-b, F-b)')

n_rows, n_cols = 3, 4
fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 13), layout='constrained')
axs = axs.ravel()
bottom_panels = {7, 8, 9, 10}      # lowest panel in each column (panel 7 sits above the empty legend slot)

for p, (pb, ps, col, label) in enumerate(panels):
    ax_ = axs[p]
    ks = sorted(pb)
    mean_, jack_ = np.full(N_MASS, np.nan), np.full(N_MASS, np.nan)
    std_ = np.full(N_MASS, np.nan)
    for k in ks:
        m_, j_ = jackknife_mean_err(pb[k])
        mean_[k], jack_[k], std_[k] = m_[col], j_[col], pb[k][:, col].std()
    grand = np.concatenate([pb[k] for k in ks], axis=0)[:, col].mean()

    for sim in sorted(set(s for k in ks for s in ps[k])):
        ks_sim = [k for k in ks if sim in ps[k]]
        ax_.plot(M_CENS[ks_sim], [pb[k][ps[k].index(sim), col] for k in ks_sim], marker='o', alpha=0.3)
    ax_.errorbar(M_CENS[ks], mean_[ks], yerr=jack_[ks], marker='o', color='k', ls='none', capsize=3,
                 label='jackknife mean' if p == 0 else None, zorder=5)
    ax_.plot(M_CENS[ks], mean_[ks] + std_[ks], 'k--', lw=2,
             label=r'$\pm$ standard dev over phases' if p == 0 else None)
    ax_.plot(M_CENS[ks], mean_[ks] - std_[ks], 'k--', lw=2)
    ax_.axhline(grand, c='k', ls=':', lw=3, label='mean over all phases, bins' if p == 0 else None)

    ax_.set_xscale('log')
    ax_.set_xlim(M_CENS[avail_all].min() / 1.5, M_CENS[avail_all].max() * 1.5)
    ax_.set_ylabel(label)
    if p in bottom_panels:
        ax_.set_xlabel(r'$M\ [h^{-1} M_\odot]$')
    else:
        ax_.tick_params(labelbottom=False)

handles, leg_labels = axs[0].get_legend_handles_labels()
axs[11].axis('off')
axs[11].legend(handles, leg_labels, fontsize=20, loc='center')

fname = 'all_params_mass_dependence.png'
fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150)
print(f'saved {os.path.join(FIGURES_DIR, fname)}')

#%%
print('_MEAN_PARAMS = np.array([' + ', '.join(f'{np.concatenate([pb[k] for k in sorted(pb)], axis=0)[:, col].mean():.6g}' for pb, ps, col, lab in panels) + '])')


#%% H. mass dependence of the Delta-model parameters with nu(r) = hill2(r; grand-mean nu_0, r_c) enforced, all phases and mass bins
D_STRIDE = 1               # fit every D_STRIDE-th R row and r_los column (1 = full grid, as in Fig 5; 3 is ~9x faster)
NU_MIN_MODEL = 2.05        # safety floor on nu(r) (the variance needs nu > 2)
NU_PARS_H = grand_h        # (nu_0, r_c): grand mean over all phases and mass bins (cell F-b)
P0_D = fits_g['grand mean']['dpars'] if 'fits_g' in globals() else DEL_P0   # starting point for the Delta fits (cell G-a, if run)
DELTA_CACHE = os.path.join(CACHE_DIR, f'delta_params_hill2_{NU_PARS_H[0]:.4f}_{NU_PARS_H[1]:.4f}_stride{D_STRIDE}.h5')


def fit_delta_phase_bin(sim, k, nu_pars, p0):
    """Returns (sim, k, Delta-model parameters or None, err string or None): skew-t fit per cell with
    nu = hill2(r), Delta = xi - measured mean (as in Fig 5), then the joint Delta-model fit."""
    path = VR_AUTO_TMPL.format(sim=sim, k=k)
    if not os.path.exists(path):
        return sim, k, None, 'file missing'
    try:
        h = PairHist.load(path)
    except Exception as e:
        return sim, k, None, f'PairHist.load: {type(e).__name__}: {e}'
    r_vr, vr_, _, npairs = vr_profile(h, 0, VR_R_EDGES)
    good = np.isfinite(vr_) & (npairs > 1000)
    if good.sum() < 4:
        return sim, k, None, f'vr_profile: only {good.sum()} usable radial bins'
    vr_interp = InterpolatedUnivariateSpline(r_vr[good], vr_[good], k=3, ext=3)
    nu_grid = np.maximum(2 + (nu_pars[0] - 2) / (1 + (nu_pars[1] / r3_2d)**2), NU_MIN_MODEL)   # hill2
    keep_cols = (np.arange(len(r_l)) % D_STRIDE == 0)
    ST = np.full(r3_2d.shape + (4,), np.nan)
    for j in np.arange(0, len(r_p), D_STRIDE):
        ST[j] = fit_skewt_row(h.n[0][j], h.sum_v[0][j], h.sum_v2[0][j], h.counts[0][j], h.v_centers,
                              nu_row=nu_grid[j], keep=keep_cols)
    Delta = ST[:, :, 0] - vr_interp(r3_2d) * r_l_2d / r3_2d
    try:
        return sim, k, fit_delta_joint(x_grid, r_p, Delta, p0=p0), None
    except Exception as e:
        return sim, k, None, f'fit_delta_joint: {type(e).__name__}: {e}'


labels_d = [r'$A_{\Delta,0}$', r'$r_\Delta$ [$h^{-1}$ Mpc]', r'$\sigma_\Delta$ [$h^{-1}$ Mpc]',
            r'$\beta_\Delta$', r'$\alpha_\Delta$']
mean_d = np.full((N_MASS, 5), np.nan)
jack_d = np.full((N_MASS, 5), np.nan)
std_d = np.full((N_MASS, 5), np.nan)
per_bin_d, per_bin_sims_d = {}, {}

if os.path.exists(DELTA_CACHE):
    print(f'Delta parameters: loading cached fits from {DELTA_CACHE} (delete this file to force a re-fit)')
    with h5.File(DELTA_CACHE, 'r') as hdf:
        for k in range(N_MASS):
            if f'bin_{k}' not in hdf:
                continue
            per_bin_d[k] = hdf[f'bin_{k}']['params'][()]
            per_bin_sims_d[k] = [s.decode() if isinstance(s, bytes) else s for s in hdf[f'bin_{k}']['sims'][()]]
else:
    tasks = [(sim, k) for sim in ALL_PLANNED_PHASES for k in range(N_MASS)
             if os.path.exists(VR_AUTO_TMPL.format(sim=sim, k=k))]
    print(f'Delta parameters: {len(tasks)} (phase, bin) combinations, nu_0 = {NU_PARS_H[0]:.2f}, r_c = {NU_PARS_H[1]:.2f} '
          f'-- fitting (will cache to {DELTA_CACHE})')
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(fit_delta_phase_bin)(sim, k, NU_PARS_H, P0_D) for sim, k in tasks)
    by_bin, err_counts = {k: {} for k in range(N_MASS)}, {}
    for sim, k, par, err in results:
        if par is not None:
            by_bin[k][sim] = par
        if err is not None:
            err_counts[err] = err_counts.get(err, 0) + 1
    if err_counts:
        print(f'\n{sum(err_counts.values())} task(s) failed; most common reasons:')
        for msg, n in sorted(err_counts.items(), key=lambda kv: -kv[1])[:10]:
            print(f'  {n:4d}x  {msg}')
    with h5.File(DELTA_CACHE, 'w') as hdf:
        for k in range(N_MASS):
            sims_k = sorted(by_bin[k])
            if len(sims_k) >= 2:
                params = np.array([by_bin[k][s] for s in sims_k])
                per_bin_d[k], per_bin_sims_d[k] = params, sims_k
                gk = hdf.create_group(f'bin_{k}')
                gk.create_dataset('params', data=params)
                gk.create_dataset('sims', data=np.array(sims_k, dtype=h5.string_dtype()))
            else:
                print(f'Delta parameters, bin {k}: fewer than 2 phases fit -- skipping')

for k in sorted(per_bin_d):
    mean_d[k], jack_d[k] = jackknife_mean_err(per_bin_d[k])
    std_d[k] = per_bin_d[k].std(axis=0)
    print(f'bin {k} ({len(per_bin_d[k])} phases): ' + ',  '.join(
        f'{lab.split(" [")[0]} = {mean_d[k, i]:.3g} +/- {jack_d[k, i]:.2g}' for i, lab in enumerate(labels_d)))


#%% H. plot (fast to re-run once cached)
avail_d = sorted(per_bin_d.keys())
if avail_d:
    fig, axs = plt.subplots(2, 3, figsize=(15, 9), layout='constrained')
    axs = axs.ravel()
    for a_ in axs[:5]:
        a_.set_xscale('log')
    all_sims_d = sorted(set(s for k in avail_d for s in per_bin_sims_d[k]))
    grand_d = np.concatenate([per_bin_d[k] for k in avail_d], axis=0).mean(axis=0)

    for i, label in enumerate(labels_d):
        for sim in all_sims_d:
            ks_sim = [k for k in avail_d if sim in per_bin_sims_d[k]]
            if ks_sim:
                axs[i].plot(M_CENS[ks_sim], [per_bin_d[k][per_bin_sims_d[k].index(sim), i] for k in ks_sim],
                            marker='o', alpha=0.3)
        axs[i].errorbar(M_CENS[avail_d], mean_d[avail_d, i], yerr=jack_d[avail_d, i],
                        marker='o', color='k', ls='none', capsize=3,
                        label='jackknife mean' if i == 0 else None, zorder=5)
        axs[i].plot(M_CENS[avail_d], mean_d[avail_d, i] + std_d[avail_d, i], 'k--', lw=2,
                    label=r'$\pm$ standard dev over phases' if i == 0 else None)
        axs[i].plot(M_CENS[avail_d], mean_d[avail_d, i] - std_d[avail_d, i], 'k--', lw=2)
        axs[i].axhline(grand_d[i], c='k', ls=':', lw=3,
                       label='mean over all phases, bins' if i == 0 else None)
        axs[i].set_ylabel(label)
        axs[i].set_xlim(M_CENS[avail_d].min() / 1.5, M_CENS[avail_d].max() * 1.5)
        if i >= 2:
            axs[i].set_xlabel(r'$M\ [h^{-1} M_\odot]$')
    handles, leg_labels = axs[0].get_legend_handles_labels()
    axs[5].axis('off')
    axs[5].legend(handles, leg_labels, fontsize=12, loc='center')
    fig.suptitle(rf'$\Delta$ parameters with hill2 $\nu(r)$: $\nu_0={NU_PARS_H[0]:.1f}$, $r_c={NU_PARS_H[1]:.1f}$ (grand mean)',
                 fontsize=13)
    fname = 'delta_params_hill2_mass_dependence.png'
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150)
    print(f'saved {os.path.join(FIGURES_DIR, fname)}')
else:
    print('Delta parameters: no bin has enough phases yet -- nothing to plot')


#%% G-c. 4x4 v_los plots with the grand-mean Delta parameters (cell H) and grand-mean nu(r) parameters (cells F-b / H)
if not per_bin_d:
    raise RuntimeError('no Delta-parameter fits available (cell H)')
DEL_GRAND = np.concatenate([per_bin_d[k] for k in sorted(per_bin_d)], axis=0).mean(axis=0)   # grand mean over all phases and mass bins
print('grand-mean Delta params (A_delta0, r_delta, sigma_delta, beta_delta, alpha): '
      + ', '.join(f'{v:.3g}' for v in DEL_GRAND))
print(f'grand-mean nu(r): nu_0 = {NU_PARS_H[0]:.2f}, r_c = {NU_PARS_H[1]:.2f}')

fit_grand_all = dict(slug='grand_all', pars=NU_PARS_H, dpars=DEL_GRAND)
for logy in (True, False):
    plot_vlos_grid('grand-mean $\\Delta$ parameters', fit_grand_all, logy)
    
#%% D. power-law nu(r) per (phase, mass bin), Fig-5-style summary vs mass (cached)
R_NU_PIV = 30.0        # pivot for nu = nu_piv * (r / R_NU_PIV)^p; set 1.0 for nu = A r^p
D_RP_MAX = 30.0        # cells used per (phase, bin): r_perp <= D_RP_MAX and r <= D_R_MAX
D_R_MAX = 70.0         # (np.inf for all cells -- roughly 10x more fits)
MIN_NU_PTS = 20        # minimum unbounded nu values needed to fit a (phase, bin)
NU_CACHE = os.path.join(CACHE_DIR, 'nu_powerlaw_phase_bin.h5')


def fit_nu_powerlaw_phase_bin(sim, k):
    """Returns (sim, k, [nu_piv, p] or None, fraction of cells at a nu bound, err string or None)."""
    path = VR_AUTO_TMPL.format(sim=sim, k=k)
    if not os.path.exists(path):
        return sim, k, None, np.nan, 'file missing'
    try:
        h = PairHist.load(path)
    except Exception as e:
        return sim, k, None, np.nan, f'PairHist.load: {type(e).__name__}: {e}'
    r_pts, nu_pts = [], []
    for j in np.where(r_p <= D_RP_MAX)[0]:
        row = fit_skewt_row(h.n[0][j], h.sum_v[0][j], h.sum_v2[0][j], h.counts[0][j], h.v_centers,
                            nu_row=None, keep=r3_2d[j] <= D_R_MAX)
        ok = np.isfinite(row[:, 3])
        r_pts.append(r3_2d[j][ok])
        nu_pts.append(row[ok, 3])
    r_pts, nu_pts = np.concatenate(r_pts), np.concatenate(nu_pts)
    free = (nu_pts > 1.01 * NU_BOUNDS[0]) & (nu_pts < 0.99 * NU_BOUNDS[1])
    if free.sum() < MIN_NU_PTS:
        return sim, k, None, 1 - free.mean() if len(free) else np.nan, f'only {free.sum()} unbounded nu values'
    p, ln_nu_piv = np.polyfit(np.log(r_pts[free] / R_NU_PIV), np.log(nu_pts[free]), 1)
    return sim, k, np.array([np.exp(ln_nu_piv), p]), 1 - free.mean(), None


labels_nu = [rf'$\nu(r={R_NU_PIV:g}\,h^{{-1}}\,{{\rm Mpc}})$', r'$p_\nu$']
mean_nu = np.full((N_MASS, 2), np.nan)
jack_nu = np.full((N_MASS, 2), np.nan)
per_bin_nu, per_bin_sims_nu = {}, {}

if os.path.exists(NU_CACHE):
    print(f'nu power law: loading cached fits from {NU_CACHE} (delete this file to force a re-fit)')
    with h5.File(NU_CACHE, 'r') as hdf:
        for k in range(N_MASS):
            if f'bin_{k}' not in hdf:
                continue
            per_bin_nu[k] = hdf[f'bin_{k}']['params'][()]
            per_bin_sims_nu[k] = [s.decode() if isinstance(s, bytes) else s for s in hdf[f'bin_{k}']['sims'][()]]
            mean_nu[k], jack_nu[k] = jackknife_mean_err(per_bin_nu[k])
else:
    nu_tasks = [(sim, k) for sim in ALL_PLANNED_PHASES for k in range(N_MASS)
                if os.path.exists(VR_AUTO_TMPL.format(sim=sim, k=k))]
    print(f'nu power law: {len(nu_tasks)} (phase, bin) combinations -- fitting (will cache to {NU_CACHE})')
    nu_results = Parallel(n_jobs=-1, verbose=10)(
        delayed(fit_nu_powerlaw_phase_bin)(sim, k) for sim, k in nu_tasks)

    by_bin, err_counts, frac_bound = {k: {} for k in range(N_MASS)}, {}, {k: [] for k in range(N_MASS)}
    for sim, k, par, fb, err in nu_results:
        if par is not None:
            by_bin[k][sim] = par
        if np.isfinite(fb):
            frac_bound[k].append(fb)
        if err is not None:
            err_counts[err] = err_counts.get(err, 0) + 1
    for k in range(N_MASS):
        print(f'bin {k}: {len(by_bin[k])} power-law fits; median fraction of cells at a nu bound '
              f'= {np.median(frac_bound[k]) if frac_bound[k] else np.nan:.1%}')
    if err_counts:
        print(f'\n{sum(err_counts.values())} task(s) failed; most common reasons:')
        for msg, n in sorted(err_counts.items(), key=lambda kv: -kv[1])[:10]:
            print(f'  {n:4d}x  {msg}')

    with h5.File(NU_CACHE, 'w') as hdf:
        for k in range(N_MASS):
            sims_k = sorted(by_bin[k])
            if len(sims_k) >= 2:
                params = np.array([by_bin[k][s] for s in sims_k])
                mean_nu[k], jack_nu[k] = jackknife_mean_err(params)
                per_bin_nu[k], per_bin_sims_nu[k] = params, sims_k
                gk = hdf.create_group(f'bin_{k}')
                gk.create_dataset('params', data=params)
                gk.create_dataset('sims', data=np.array(sims_k, dtype=h5.string_dtype()))
            else:
                print(f'nu power law, bin {k}: fewer than 2 phases fit -- skipping')


#%% D. plot (fast to re-run once cached)
avail_nu = sorted(per_bin_nu.keys())
if avail_nu:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for a_ in ax:
        a_.set_xscale('log')
    all_sims_nu = sorted(set(s for k in avail_nu for s in per_bin_sims_nu[k]))
    std_nu = np.full((N_MASS, 2), np.nan)
    for k in avail_nu:
        std_nu[k] = per_bin_nu[k].std(axis=0)
    grand_nu = np.concatenate([per_bin_nu[k] for k in avail_nu], axis=0).mean(axis=0)

    for i, label in enumerate(labels_nu):
        for sim in all_sims_nu:
            ks_sim = [k for k in avail_nu if sim in per_bin_sims_nu[k]]
            if ks_sim:
                ax[i].plot(M_CENS[ks_sim], [per_bin_nu[k][per_bin_sims_nu[k].index(sim), i] for k in ks_sim],
                           marker='o', alpha=0.3)
        ax[i].errorbar(M_CENS[avail_nu], mean_nu[avail_nu, i], yerr=jack_nu[avail_nu, i],
                       marker='o', color='k', ls='none', capsize=3,
                       label='jackknife mean' if i == 0 else None, zorder=5)
        mu, sd = mean_nu[avail_nu, i], std_nu[avail_nu, i]
        ax[i].plot(M_CENS[avail_nu], mu + sd, 'k--', lw=2,
                   label=r'$\pm$ standard dev over phases' if i == 0 else None)
        ax[i].plot(M_CENS[avail_nu], mu - sd, 'k--', lw=2)
        ax[i].axhline(grand_nu[i], c='k', ls=':', lw=3,
                      label='mean over all phases, bins' if i == 0 else None)
        ax[i].set_ylabel(label)
        ax[i].set_xlabel(r'$M\ [h^{-1} M_\odot]$')
        ax[i].set_xlim(M_CENS[avail_nu].min() / 1.5, M_CENS[avail_nu].max() * 1.5)
    ax[0].legend(fontsize=9, loc='upper left')
    fig.savefig(os.path.join(FIGURES_DIR, 'nu_powerlaw_mass_dependence.png'), dpi=150)
    print(f'saved {os.path.join(FIGURES_DIR, "nu_powerlaw_mass_dependence.png")}')
else:
    print('nu power law: no bin has enough phases yet -- nothing to plot')


#%% E. compact model with nu(r) and Delta refit to the nu(r) calibration data, single phase / mass bin
NU_FORM = 'power'              # key into NU_FORMS (cell C), e.g. 'power', 'shifted', 'hill', 'exp'
NU_CLIP = (3.0, 200.0)         # nu(r) is clipped to this range (calibrated for r <~ 70 only)
RP_SEL = [5.5, 15.5, 29.5]     # columns (r_perp)
RL_SEL = [5.5, 30.5, 60.5]     # rows (r_los)
LOGY = False
V_LIM = 2500.0                 # range over which the model curves are computed
V_PLOT = 2000.0                # displayed x-range
MIN_COUNT = 1                  # rebinned v-bins with fewer pairs are masked
Y_FLOOR = 1e-5                 # log-y lower limit, as a fraction of each panel's peak


def nu_of_r(r):
    f = NU_FORMS[NU_FORM][1]
    return np.clip(f(np.asarray(r, dtype=float), *nu_form_fits[NU_FORM]), *NU_CLIP)


# calibration data: skew-t refit per cell with nu = nu(r); Delta = xi - measured mean (as in the pipeline)
nu_grid = nu_of_r(r3_2d)
ST_nu = fit_skewt_grid(h_s, nu=nu_grid)
Delta_6 = ST_6[:, :, 0] - mu_meas
Delta_nu = ST_nu[:, :, 0] - mu_meas

sig_params = fit_sigma_joint(x_grid, r_p, sig_s)
del_6 = fit_delta_joint(x_grid, r_p, Delta_6)
del_nu = fit_delta_joint(x_grid, r_p, Delta_nu, p0=del_6)
print(f'nu(r) form "{NU_FORM}": parameters {np.round(nu_form_fits[NU_FORM], 4)}, clipped to {NU_CLIP}')
print('Delta params (A_delta0, r_delta, sigma_delta, beta_delta, alpha):')
print('  nu = 6 : ' + ', '.join(f'{v:.3g}' for v in del_6))
print('  nu(r)  : ' + ', '.join(f'{v:.3g}' for v in del_nu))

mean_vr_m = mean_vr_model(SINGLE_BIN)      # prediction uses the b1/b2 LPT mean


def model_pdf(nu, dpars, rp, rl, v):
    r = np.hypot(rp, rl)
    x = r - rp
    mu = mean_vr_m(r) * rl / r
    sig = sigma_mod_2d((x, rp), *sig_params)
    delta = delta_mod_alpha_R((x, rp), *dpars)
    omega, alpha = skewt_params_from_delta(delta, sig, nu)
    return skewed_t_pdf(v, mu + delta, omega, alpha, nu)


cent_h = h_s.v_centers.reshape(-1, REBIN).mean(axis=1)
dv_h = (h_s.v_centers[1] - h_s.v_centers[0]) * REBIN
edges_h = np.r_[cent_h - dv_h / 2, cent_h[-1] + dv_h / 2]
vv = np.linspace(-V_LIM, V_LIM, 600)

# inset quarter-disk radius: just beyond the farthest selected cell, rounded up to 10
r3_sel = [np.hypot(r_p[np.argmin(np.abs(r_p - rp))], r_l[np.argmin(np.abs(r_l - rl))])
          for rp in RP_SEL for rl in RL_SEL]
R_INSET = 10.0 * np.ceil((max(r3_sel) + 5.0) / 10.0)
INSET_PAD = 0.12 * R_INSET
theta = np.linspace(0, np.pi / 2, 100)

nr, nc = len(RL_SEL), len(RP_SEL)
fig, axs = plt.subplots(nr, nc, figsize=(5 * nc, 4 * nr), layout='constrained', sharex=True)
axs = np.atleast_2d(axs)
for a, rl_t in enumerate(RL_SEL):
    for b_, rp_t in enumerate(RP_SEL):
        ax_ = axs[a, b_]
        j, i = np.argmin(np.abs(r_p - rp_t)), np.argmin(np.abs(r_l - rl_t))
        if h_s.n[0][j, i] < N_MIN:
            ax_.set_title('too few pairs')
            continue
        counts = h_s.counts[0][j, i].reshape(-1, REBIN).sum(axis=1)
        norm = counts.sum() * dv_h
        hist = counts / norm
        ok = counts >= MIN_COUNT
        ax_.stairs(np.where(ok, hist, 1e-30), edges_h, color='k', lw=1, label='simulation')
        err = np.sqrt(counts) / norm
        err_lo = np.minimum(err, 0.999 * hist) if LOGY else err
        ax_.errorbar(cent_h[ok], hist[ok], yerr=[err_lo[ok], err[ok]], fmt='none',
                     ecolor='k', elinewidth=0.8, capsize=0)

        nu_c = float(nu_of_r(r3_2d[j, i]))
        pdf_ref = model_pdf(NU_FIXED, del_6, r_p[j], r_l[i], vv)
        pdf_new = model_pdf(nu_c, del_nu, r_p[j], r_l[i], vv)
        ax_.plot(vv, pdf_ref, c='C1', lw=1.8, ls='--', label=rf'$\nu={NU_FIXED}$, its $\Delta$ fit')
        ax_.plot(vv, pdf_new, c='C0', lw=2.2, label=r'$\nu(r)$, refit $\Delta$')

        ax_.axvline(0, c='gray', ls=':', lw=1)
        ax_.set_xlim(-V_PLOT, V_PLOT)
        if LOGY:
            ax_.set_yscale('log')
            ax_.set_ylim(np.nanmax(hist) * Y_FLOOR, np.nanmax(hist) * 2)
        else:
            ax_.set_ylim(0, 1.1 * max(np.nanmax(hist), pdf_ref.max(), pdf_new.max()))
        ax_.text(0.03, 0.05, rf'$\nu(r)={nu_c:.1f}$', transform=ax_.transAxes, fontsize=9)
        ax_.set_title(rf'$r_\perp\in[{r_perp_edges[j]:.2f}, {r_perp_edges[j+1]:.2f}]$, '
                      rf'$r_{{\rm los}}\in[{r_los_edges[i]:.2f}, {r_los_edges[i+1]:.2f}]$ '
                      r'$h^{-1}$Mpc', fontsize=12)
        if a == nr - 1:
            ax_.set_xlabel(r'$v_{\rm los}$ [km/s]')

        # inset: where this bin sits on the (R, r_los) quarter-disk
        ins = ax_.inset_axes([0.82, 0.72, 0.16, 0.24])
        ins.fill(np.r_[-INSET_PAD, R_INSET, R_INSET * np.cos(theta), -INSET_PAD],
                 np.r_[-INSET_PAD, -INSET_PAD, R_INSET * np.sin(theta), R_INSET],
                 c='C1', alpha=0.5, lw=0)
        ins.axvspan(r_perp_edges[j], r_perp_edges[j + 1], color='C0', alpha=0.4, lw=0)
        ins.axhspan(r_los_edges[i], r_los_edges[i + 1], color='C0', alpha=0.4, lw=0)
        ins.axvline(r_p[j], c='gray', lw=1)
        ins.axhline(r_l[i], c='gray', lw=1)
        ins.set_xlim(-INSET_PAD, R_INSET)
        ins.set_ylim(-INSET_PAD, R_INSET)
        ins.set_xticks([])
        ins.set_yticks([])
        ins.set_xlabel(r'$R$', fontsize=7, labelpad=1)
        ins.set_ylabel(r'$r_{\rm LOS}$', fontsize=7, labelpad=1)

fig.supylabel(r'$P(v_{\rm los}\,|\,r_\perp, r_{\rm los})$', fontsize=20)
axs[0, 0].legend(fontsize=8, loc='lower center' if LOGY else 'upper left')
fname = f'nu_function_refit_delta_{NU_FORM}.png'
fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150)
print(f'saved {os.path.join(FIGURES_DIR, fname)} ({sim_single}, bin {SINGLE_BIN})')