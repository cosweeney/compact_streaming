"""
cosmo_grid_vel_fit_out.py

Streamlined version. Fits the 11-parameter compact velocity model
(sigma_v: 4 params, Delta: 5 params, nu(r): 2 params; see compact.model) to
every available (sim, redshift, mass bin) in the AbacusSummit cosmology grid,
on the native (r_perp, r_los) grid -- no re-binning or special handling for
any bin -- builds LambdaCDM-only ridge emulators per (mass bin, redshift),
extrapolates the high-mass bin to high redshift, and produces three sets of
plots:
  1. cosmo_dep_emulators_z*.png   -- per-redshift cosmology-dependence
     scatter + per-bin/average emulator lines, rho-annotated, green-shaded
     by |rho|.
  2. cosmo_dep_residuals_vs_average_z*.png -- per-redshift residuals of
     each bin's data against the bin-averaged emulator, one combined
     stat/sys percentage per panel.
  3. emulator_intercept_vs_redshift.png / emulator_slope_vs_redshift.png
     -- ridge intercept and per-LambdaCDM-column slope vs z, with
     propagated error bars and robust (MAD-based) y-limits.

Per (sim, z, bin) the fit is:
  - FREE skew-t (xi, omega, alpha, nu) per (r_perp, r_los) cell, ln nu bounded
    to [2.1, 1000];
  - Delta = xi_free - <v_r|r> r_los/r from those free fits (independent of the
    nu(r) model), then a bounded multi-start 5-parameter Delta fit;
  - sigma_v from the per-cell moments, then a 4-parameter fit (R_piv = 72.5);
  - (nu_0, r_c): hill2 nu(r) = 2 + (nu_0 - 2)/(1 + (r_c/r)^2) fit to the
    per-r-bin medians of the free-fit nu (6 h^-1 Mpc bins, >= 30 cells/bin,
    soft-L1 loss on ln nu).

One cell threshold (N_MIN = 50 pairs) is used for everything: the sigma_v grid and
the free skew-t fits. Each parameter family (sigma_v, Delta, nu) is stored
separately per (sim, z, bin), so a failed Delta or nu fit does not discard that
sim's sigma_v; a missing family is NaN and is skipped in the emulator fit.

High-mass bin at high z: for the parameters listed in EXTRAPOLATE_ROWS (the Delta
shape parameters by default), bin 1's emulator at z > Z_ANCHOR_MAX is NOT built from
bin 1's own fits. Instead, for each emulator coefficient (the intercept and each
LambdaCDM slope), bin 1's value at z is bin 0's value at z plus the (bin 1 - bin 0)
shift averaged over the redshifts z <= Z_ANCHOR_MAX where both bins are fit. All other
parameters use bin 1's own fits at every z. The intercept/slope-vs-z plots are saved
twice: raw fits only (*_raw.png), and with the extrapolation overlaid for the
EXTRAPOLATE_ROWS parameters (*_extrap.png).
"""

#%% imports and setup
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
from scipy.stats import pearsonr
from scipy.interpolate import InterpolatedUnivariateSpline
from joblib import Parallel, delayed
from matplotlib.colors import to_rgba

import matplotlib
matplotlib.rcParams.update({'text.usetex': True,
                            'font.family': 'Computer Modern Roman'})
plt.rcParams['axes.labelsize'] = 20

import nb.calibration.runs.auto_runs.abacus_cosmo_core_auto as core   # safe: no pycorr import in this module
from compact.velocities.pairhist import PairHist
from compact.model import nu_of_r, skewed_t_pdf

FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

CANDIDATE_REDSHIFTS = [0.2, 0.3, 0.4, 0.5, 0.8, 1.1, 1.4, 1.7]
N_MASS = len(core.BIN_NPART)   # 2: low, high

VR_R_EDGES = np.arange(2.0, 142.0, 2.0)

r_p = 0.5 * (core.r_perp[1:] + core.r_perp[:-1])
r_l = 0.5 * (core.r_los[1:] + core.r_los[:-1])
r_p_2d, r_l_2d = np.meshgrid(r_p, r_l, indexing='ij')
r3_2d = np.sqrt(r_p_2d**2 + r_l_2d**2)
x_grid = r3_2d - r_p_2d


# ---------------------------------------------------------------- 11-param model
SIG_FLOOR = 0.1
DEL_FLOOR = -1000.0
DEL_P0 = [125.0, 50.0, 10.0, 2.0, 1.0]   # A_delta0, r_delta, sigma_delta, beta_delta, alpha



def sigma_mod(x, sigma0, A_rho, sigma_rho):
    arg = 2.0 * sigma0**2 * (1.0 - A_rho * np.exp(-x / sigma_rho))
    return np.sqrt(np.clip(arg, 1e-8, None))


def A_rho_mod(R, A_piv, A_rho_p, R_piv):
    return A_piv * np.exp(-(R - R_piv) / A_rho_p)


def sigma_mod_2d(coords, sigma0, A_piv, A_rho_p, sigma_rho, R_piv):
    x, R = coords
    return sigma_mod(x, sigma0, A_rho_mod(R, A_piv, A_rho_p, R_piv), sigma_rho)


def _seed_sigma(x, R, sig, R_piv):
    sigma0 = sig.max() / np.sqrt(2.0) * 1.02
    y = 2.0 * sigma0**2 - sig**2
    m = y > 0
    G = np.column_stack([np.ones(m.sum()), -(R[m] - R_piv), -x[m]])
    c = np.linalg.lstsq(G, np.log(y[m]), rcond=None)[0]
    A_rho_p = 1.0 / c[1] if c[1] > 0 else np.ptp(R) + 1.0
    sigma_rho = 1.0 / c[2] if c[2] > 0 else 10.0
    A_piv = np.clip(np.exp(c[0]) / (2.0 * sigma0**2), 1e-3, 0.99)
    return [sigma0, A_piv, A_rho_p, sigma_rho]


def fit_sigma_joint(x, r_p_arr, sig):
    R = np.broadcast_to(r_p_arr[:, None], x.shape)
    m = np.isfinite(x) & np.isfinite(sig) & (sig > SIG_FLOOR)
    X, RR, S = x[m], R[m], sig[m]
    R_piv = 72.5 #np.median(RR)
    popt, _ = curve_fit(
        lambda c, s0, Ap, Arp, sr: sigma_mod_2d(c, s0, Ap, Arp, sr, R_piv),
        (X, RR), S, p0=_seed_sigma(X, RR, S, R_piv),
        bounds=([0.0, 1e-4, 1e-3, 1e-3], [np.inf, 1.0, np.inf, np.inf]), maxfev=20_000)
    return popt


def A_delta_mod(R, A_delta0, r_delta):
    return A_delta0 * np.exp(-R / r_delta)


def delta_mod_alpha(x, A, sigma_delta, beta_delta, alpha):
    return A * x**alpha / (1 + (x / sigma_delta)**beta_delta)


def delta_mod_alpha_R(coords, A_delta0, r_delta, sigma_delta, beta_delta, alpha):
    x, R = coords
    return delta_mod_alpha(x, A_delta_mod(R, A_delta0, r_delta), sigma_delta, beta_delta, alpha)


def fit_delta_joint(x, r_p_arr, D, p0=DEL_P0, n_restarts=8, seed=0):
    """Bounded + multi-start replacement for the original unconstrained
    fit. A_delta0 is sign-unconstrained (the model allows either sign);
    r_delta, sigma_delta, beta_delta are constrained > 0 (a negative
    lengthscale or shape exponent is degenerate, not a legitimate
    alternative fit); alpha is bounded away from negative values, both to
    block the runaway-to-~30 regime seen in a real fraction of bin-1 sims
    (10/31 at z=1.1, checked directly) and to sidestep the documented x=0
    singularity in delta_mod_alpha for negative alpha. n_restarts random
    perturbations of p0 are tried in addition to p0 itself; whichever
    converged fit has the lowest residual sum of squares wins."""
    R = np.broadcast_to(r_p_arr[:, None], x.shape)
    m = np.isfinite(x) & np.isfinite(D) & (D > DEL_FLOOR)
    X, RR, DD = x[m], R[m], D[m]

    lower = [-np.inf, 1e-3, 1e-3, 1e-3, 0.0]
    upper = [np.inf, np.inf, np.inf, 20.0, 10.0]

    rng = np.random.default_rng(seed)
    p0_arr = np.clip(np.array(p0, dtype=float), lower, upper)
    best_popt, best_cost = None, np.inf

    for trial in range(n_restarts + 1):
        p0_trial = p0_arr if trial == 0 else np.clip(
            p0_arr * rng.uniform(0.3, 3.0, size=len(p0_arr)), lower, upper)
        try:
            popt, _ = curve_fit(delta_mod_alpha_R, (X, RR), DD, p0=p0_trial,
                               bounds=(lower, upper), maxfev=20_000)
        except RuntimeError:
            continue
        resid = delta_mod_alpha_R((X, RR), *popt) - DD
        cost = np.sum(resid ** 2)
        if cost < best_cost:
            best_cost, best_popt = cost, popt

    if best_popt is None:
        raise RuntimeError('fit_delta_joint: all restarts failed to converge')
    return best_popt


# free skew-t cell fits: nu is fit as ln(nu), bounded
NU_P0 = 6.0
NU_BOUNDS = (2.1, 1000.0)

# nu(r) = hill2 fit to per-r-bin medians of the free-fit nu
NU_BIN_WIDTH = 6.0                 # h^-1 Mpc
NU_MIN_CELLS = 30                  # cells per r bin
NU_FIT_P0 = [25.0, 40.0]           # nu_0, r_c
NU_FIT_LOWER = [2.01, 1.0]
NU_FIT_UPPER = [1000.0, 500.0]

REBIN = 5
N_MIN = 50    # minimum pairs per (r_perp, r_los) cell, for both the sigma_v grid and the
              # free skew-t fits.


def meas_sigv_grid(h, k, n_min=N_MIN):
    n = h.n[k].astype(float)
    mu = np.where(n >= n_min, h.sum_v[k] / np.where(n > 0, n, 1), np.nan)
    m2 = np.where(n >= n_min, h.sum_v2[k] / np.where(n > 0, n, 1), np.nan)
    return np.sqrt(np.maximum(m2 - mu**2, 0.0))


def fit_skewt_grid(h, k, n_min=N_MIN):
    """Free skew-t fit per (r_perp, r_los) cell. Returns an array of shape
    (n_R, n_l, 4) holding (xi, omega, alpha, nu); NaN where the cell is thin
    or the fit failed. nu is fit as ln(nu) with bounds NU_BOUNDS. Unweighted
    least squares on the density histogram, rebinned x REBIN, restricted to
    +-2500 km/s around the cell mean; start values (mu, std, -0.1, NU_P0)."""
    n_k, sum_v_k, sum_v2_k, counts_k = h.n[k], h.sum_v[k], h.sum_v2[k], h.counts[k]
    cent_fine = h.v_centers
    dv = (cent_fine[1] - cent_fine[0]) * REBIN
    cent = cent_fine.reshape(-1, REBIN).mean(axis=1)

    lower = [-np.inf, 1e-3, -np.inf, np.log(NU_BOUNDS[0])]
    upper = [np.inf, np.inf, np.inf, np.log(NU_BOUNDS[1])]

    def model(xx, xi_, omega_, alpha_, lnnu_):
        return skewed_t_pdf(xx, xi_, omega_, alpha_, np.exp(lnnu_))

    n_R, n_l = n_k.shape
    ST_pars = np.full((n_R, n_l, 4), np.nan)
    for j in range(n_R):
        for i in range(n_l):
            n = n_k[j, i]
            if n < n_min:
                continue
            mu = sum_v_k[j, i] / n
            width = np.sqrt(max(sum_v2_k[j, i] / n - mu**2, 0.0))
            if width <= 0:
                continue

            counts = counts_k[j, i].reshape(-1, REBIN).sum(axis=1)
            sel = np.abs(cent - mu) <= 2500
            c_w, x_w = counts[sel], cent[sel]
            if c_w.sum() == 0:
                continue
            hist = c_w / (c_w.sum() * dv)

            try:
                p, _ = curve_fit(model, x_w, hist,
                                 p0=[mu, width, -0.1, np.log(NU_P0)],
                                 bounds=(lower, upper), method='trf', maxfev=5_000)
                ST_pars[j, i] = [p[0], p[1], p[2], np.exp(p[3])]
            except (RuntimeError, ValueError):
                continue
    return ST_pars


def fit_nu_profile(nu_cells, r3):
    """Fit nu(r) = 2 + (nu_0 - 2)/(1 + (r_c/r)^2) (compact.model.nu_of_r) to
    the per-r-bin medians of the free-fit nu. nu_cells and r3 are same-shape
    arrays (r3 = 3D separation of each cell). Bins are NU_BIN_WIDTH wide with
    >= NU_MIN_CELLS finite cells; cells pinned at the nu bounds stay in the
    median. Soft-L1 loss on ln nu, start (nu_0, r_c) = NU_FIT_P0. Returns
    np.array([nu_0, r_c]). Raises RuntimeError if too few bins qualify."""
    ok = np.isfinite(nu_cells)
    nu_v, r_v = nu_cells[ok], r3[ok]
    edges = np.arange(0.0, r_v.max() + NU_BIN_WIDTH, NU_BIN_WIDTH) if r_v.size else np.array([0.0])
    idx = np.digitize(r_v, edges) - 1
    r_med, nu_med = [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() >= NU_MIN_CELLS:
            r_med.append(r_v[sel].mean())
            nu_med.append(np.median(nu_v[sel]))
    if len(r_med) < 3:
        raise RuntimeError('fit_nu_profile: fewer than 3 r bins with enough cells')
    r_med, nu_med = np.array(r_med), np.array(nu_med)

    res = least_squares(lambda p: np.log(nu_of_r(r_med, p[0], p[1])) - np.log(nu_med),
                        x0=NU_FIT_P0, bounds=(NU_FIT_LOWER, NU_FIT_UPPER), loss='soft_l1')
    return res.x


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



#%% scan for available (sim, z, bin) output and fit each, in parallel
def out_path(n, z, k):
    return os.path.join(core.OUT_DIR.format(sim=core.sim_name(n), zdir=core.zdir(z)),
                        f'pairhist_auto_mbin{k}.h5')


tasks = [(n, z, k) for n in core.SIM_NUMS for z in CANDIDATE_REDSHIFTS
         for k in range(N_MASS) if os.path.exists(out_path(n, z, k))]
print(f'{len(tasks)} (sim, z, bin) combinations available')


def fit_one(n, z, k):
    """(n, z, k, sigma_v params, Delta params, nu params); each family is None on failure,
    independently of the others."""
    try:
        h = PairHist.load(out_path(n, z, k))
    except Exception:
        return n, z, k, None, None, None

    sig_params = del_params = nu_params = None
    try:
        sig_params = fit_sigma_joint(x_grid, r_p, meas_sigv_grid(h, 0))
    except Exception:
        pass

    try:
        ST_pars = fit_skewt_grid(h, 0)   # free skew-t: (xi, omega, alpha, nu) per cell
    except Exception:
        return n, z, k, sig_params, None, None

    try:
        nu_params = fit_nu_profile(ST_pars[:, :, 3], r3_2d)
    except Exception:
        pass

    try:
        r_vr, vr_mean, npairs_vr = vr_profile(h, 0, VR_R_EDGES)
        good_vr = np.isfinite(vr_mean) & (npairs_vr > 100)
        if good_vr.sum() >= 4:
            vr_interp = InterpolatedUnivariateSpline(r_vr[good_vr], vr_mean[good_vr], k=3, ext=3)
            Delta = ST_pars[:, :, 0] - vr_interp(r3_2d) * r_l_2d / r3_2d
            del_params = fit_delta_joint(x_grid, r_p, Delta, p0=DEL_P0)
    except Exception:
        pass

    return n, z, k, sig_params, del_params, nu_params


results = Parallel(n_jobs=-1, verbose=10)(delayed(fit_one)(n, z, k) for n, z, k in tasks)

params = {k: {z: {} for z in CANDIDATE_REDSHIFTS} for k in range(N_MASS)}
for n, z, k, sig_p, del_p, nu_p in results:
    vec = np.full(11, np.nan)
    for sl, p in ((slice(0, 4), sig_p), (slice(4, 9), del_p), (slice(9, 11), nu_p)):
        if p is not None:
            vec[sl] = p
    if np.isfinite(vec).any():
        params[k][z][n] = vec

for k in range(N_MASS):
    for z in CANDIDATE_REDSHIFTS:
        if params[k][z]:
            print(f'bin {k}, z={z}: {len(params[k][z])} sims fit successfully')

labels = [r"$\sigma_0$ [km/s]", r"$A_{\rho,0}$", r"$A_{\rho,p}$ [$h^{-1}$ Mpc]", r"$\sigma_\rho$ [$h^{-1}$ Mpc]",
          r"$A_{\Delta,0}$ [km/s]", r"$r_\Delta$ [$h^{-1}$ Mpc]", r"$\sigma_\Delta$ [$h^{-1}$ Mpc]",
          r"$\beta_\Delta$", r"$\alpha_\Delta$",
          r"$\nu_0$", r"$r_c$ [$h^{-1}$ Mpc]"]
n_labels = len(labels)


#%% pull cosmological parameters per sim (redshift-independent)
COSMO_COLS = ['omega_b', 'omega_cdm', 'h', 'n_s', 'alpha_s', 'w0_fld', 'wa_fld', 'sigma8_m']

all_sims_used = sorted(set(n for k in range(N_MASS) for z in CANDIDATE_REDSHIFTS
                          for n in params[k][z]))
cosmo_pars = {}
cosmo_As = {}
for n in all_sims_used:
    z_avail = next((z for z in CANDIDATE_REDSHIFTS if n in params[0].get(z, {})
                    or n in params[min(N_MASS - 1, 1)].get(z, {})), CANDIDATE_REDSHIFTS[0])
    par = core.read_params(n, z_avail)
    cosmo_pars[n] = [float(par[c]) for c in COSMO_COLS]
    cosmo_As[n] = float(par['A_s'])
print(f'cosmology parameters pulled for {len(cosmo_pars)} sims')

LCDM_LABELS = [r"$\omega_b$", r"$\omega_{\rm cdm}$", r"$\log(10^{10}A_s)$", r"$n_s$"]
LCDM_NAMES = ['omega_b', 'omega_cdm', 'ln_1e10_A_s', 'n_s']
n_lcdm = len(LCDM_LABELS)

_ob, _oc, _ns = COSMO_COLS.index('omega_b'), COSMO_COLS.index('omega_cdm'), COSMO_COLS.index('n_s')
lcdm_pars = {n: np.array([cosmo_pars[n][_ob], cosmo_pars[n][_oc],
                          np.log(1e10 * cosmo_As[n]), cosmo_pars[n][_ns]])
             for n in all_sims_used}

FIDUCIAL_LCDM = np.array([0.02237, 0.1200, np.log(1e10 * 2.0830e-9), 0.9649])
assert len(FIDUCIAL_LCDM) == n_lcdm


#%% save fit results to disk
PARAM_NAMES = ['sigma0', 'A_rho0', 'A_rho_p', 'sigma_rho',
              'A_delta0', 'r_delta', 'sigma_delta', 'beta_delta', 'alpha',
              'nu_0', 'r_c']
FIT_RESULTS_PATH = os.path.join(FIGURES_DIR, 'cosmo_grid_fit_results.h5')

rows_sim, rows_z, rows_bin, rows_params, rows_cosmo = [], [], [], [], []
for k in range(N_MASS):
    for z in CANDIDATE_REDSHIFTS:
        for n, p in params[k][z].items():
            rows_sim.append(n)
            rows_z.append(z)
            rows_bin.append(k)
            rows_params.append(p)
            rows_cosmo.append(cosmo_pars[n])

with h5.File(FIT_RESULTS_PATH, 'w') as hdf:
    hdf.create_dataset('sim', data=np.array(rows_sim, dtype=np.int32))
    hdf.create_dataset('z', data=np.array(rows_z, dtype=np.float64))
    hdf.create_dataset('bin', data=np.array(rows_bin, dtype=np.int32))
    hdf.create_dataset('params', data=np.array(rows_params, dtype=np.float64))
    hdf.create_dataset('cosmo', data=np.array(rows_cosmo, dtype=np.float64))
    hdf.attrs['param_names'] = PARAM_NAMES
    hdf.attrs['cosmo_names'] = COSMO_COLS
print(f'saved {len(rows_sim)} (sim, z, bin) fit results to {FIT_RESULTS_PATH}')



#%% LambdaCDM ridge emulators, one per (mass bin, redshift, parameter). All
# n_lcdm columns are always fit, shrinking rather than dropping a column's
# slope when the data doesn't support it. Each bin uses its own sims.
MIN_SIMS_FOR_FIT = 4
ALPHA_GRID = np.logspace(-3, 3, 13)


def ridge_fit_beta(delta, y, alpha):
    n_sim, n_col = delta.shape
    scale = np.std(delta, axis=0)
    scale[scale == 0] = 1.0
    delta_s = delta / scale
    X = np.column_stack([np.ones(n_sim), delta_s])
    Gamma = np.diag([0.0] + [alpha] * n_col)
    beta_s = np.linalg.solve(X.T @ X + Gamma, X.T @ y)
    return np.concatenate([[beta_s[0]], beta_s[1:] / scale])


def ridge_cv_alpha(delta, y, alpha_grid=ALPHA_GRID):
    n_sim = len(y)
    if n_sim < 2 * MIN_SIMS_FOR_FIT:
        return alpha_grid[len(alpha_grid) // 2]
    best_alpha, best_err = alpha_grid[0], np.inf
    for alpha in alpha_grid:
        se = 0.0
        for m in range(n_sim):
            mask = np.ones(n_sim, dtype=bool)
            mask[m] = False
            beta = ridge_fit_beta(delta[mask], y[mask], alpha)
            pred = beta[0] + delta[m] @ beta[1:]
            se += (pred - y[m]) ** 2
        if se < best_err:
            best_err, best_alpha = se, alpha
    return best_alpha


def ridge_beta_cov(delta, y, alpha):
    """Ridge coefficients and covariance (in original units) at a given alpha."""
    n_sim, n_col = delta.shape
    scale = np.std(delta, axis=0)
    scale[scale == 0] = 1.0
    delta_s = delta / scale
    X = np.column_stack([np.ones(n_sim), delta_s])
    Gamma = np.diag([0.0] + [alpha] * n_col)
    XtX = X.T @ X
    A_inv = np.linalg.inv(XtX + Gamma)
    beta_s = A_inv @ (X.T @ y)

    H = X @ A_inv @ X.T
    edf = np.trace(H)
    resid = y - X @ beta_s
    dof = max(n_sim - edf, 1.0)
    sigma2 = np.sum(resid ** 2) / dof

    cov_s = sigma2 * A_inv @ XtX @ A_inv
    D = np.diag(np.concatenate([[1.0], 1.0 / scale]))
    return D @ beta_s, D @ cov_s @ D.T


def eval_emulator(beta, theta_query):
    """beta[0] + sum_j beta[1+j] (FIDUCIAL_j - theta_j); theta_query is (n_lcdm,) or (m, n_lcdm)."""
    return beta[0] + (FIDUCIAL_LCDM - theta_query) @ beta[1:]


# robust outlier exclusion on the already-fit parameters (median + scaled MAD
# on the y-value alone). Cheap to tune without re-running the velocity fits.
def robust_outlier_mask(y, mad_k=6.0):
    y = np.asarray(y, dtype=float)
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    if mad == 0:
        return np.ones_like(y, dtype=bool)
    return np.abs(y - med) <= mad_k * 1.4826 * mad


def finite_sims(sims_z, k, z, i):
    """Sims whose fit for parameter i succeeded (the value is not NaN)."""
    return [n for n in sims_z if np.isfinite(params[k][z][n][i])]


def filtered_sims_theta_y(sims_z, k, z, i, mad_k=6.0):
    """(sims_kept, theta_kept, y_kept, dropped_sims). Keeps everyone if the
    mask would leave fewer than MIN_SIMS_FOR_FIT sims."""
    sims_z = finite_sims(sims_z, k, z, i)
    sims_arr = np.array(sims_z)
    theta_full = np.array([lcdm_pars[n] for n in sims_z])
    y_full = np.array([params[k][z][n][i] for n in sims_z])
    keep = robust_outlier_mask(y_full, mad_k=mad_k)
    dropped = sims_arr[~keep].tolist()
    if keep.sum() < MIN_SIMS_FOR_FIT:
        keep = np.ones_like(keep, dtype=bool)
        dropped = []
    return sims_arr[keep], theta_full[keep], y_full[keep], dropped


def fit_one_bin_ridge(sims_z, k, z, i, mad_k=6.0):
    """(beta, alpha_used, rho_diag, dropped_sims); beta is None if too few sims."""
    sims_z = finite_sims(sims_z, k, z, i)
    if len(sims_z) < MIN_SIMS_FOR_FIT:
        return None, None, {}, []
    _, theta, y, dropped = filtered_sims_theta_y(sims_z, k, z, i, mad_k=mad_k)
    if len(y) < MIN_SIMS_FOR_FIT:
        return None, None, {}, dropped
    delta = FIDUCIAL_LCDM[None, :] - theta
    alpha = ridge_cv_alpha(delta, y)
    beta = ridge_fit_beta(delta, y, alpha)
    rho_diag = {j: pearsonr(theta[:, j], y)[0] for j in range(n_lcdm)}
    return beta, alpha, rho_diag, dropped


sims_by = {k: {z: sorted(params[k][z]) for z in CANDIDATE_REDSHIFTS} for k in range(N_MASS)}
print('sims with at least one successful fit family, per (z, bin):')
for z in CANDIDATE_REDSHIFTS:
    print(f'  z={z}: bin 0: {len(sims_by[0][z]):>3d}   bin 1: {len(sims_by[1][z]):>3d}')

BETA_RAW = {0: {}, 1: {}}   # (i, z) -> beta, length 1 + n_lcdm, from the bin's OWN fits (or None)
SE_RAW = {0: {}, 1: {}}     # (i, z) -> standard errors, same length
rhos_bin = {0: {}, 1: {}}
dropped_sims_log = {0: {}, 1: {}}

for k in (0, 1):
    for i in range(n_labels):
        for z in CANDIDATE_REDSHIFTS:
            sims_z = sims_by[k][z]
            beta, alpha, rhos, drop = fit_one_bin_ridge(sims_z, k, z, i)
            BETA_RAW[k][(i, z)] = beta
            rhos_bin[k][(i, z)] = rhos
            dropped_sims_log[k][(i, z)] = drop
            if beta is None:
                continue
            _, theta, y, _ = filtered_sims_theta_y(sims_z, k, z, i)
            _, cov = ridge_beta_cov(FIDUCIAL_LCDM[None, :] - theta, y, alpha)
            SE_RAW[k][(i, z)] = np.sqrt(np.diag(cov))

n_raw = {k: sum(b is not None for b in BETA_RAW[k].values()) for k in (0, 1)}
print(f'ridge emulators from own fits: bin 0: {n_raw[0]}, bin 1: {n_raw[1]} '
      f'of {n_labels * len(CANDIDATE_REDSHIFTS)} (parameter, redshift) combinations each')


#%% high-mass bin at z > Z_ANCHOR_MAX: bin 0 + the average (bin 1 - bin 0) shift
# over z <= Z_ANCHOR_MAX, applied to the intercept and to every LambdaCDM slope --
# only for the parameters in EXTRAPOLATE_ROWS; the rest keep bin 1's own fits.
Z_ANCHOR_MAX = 0.5
EXTRAPOLATE_ROWS = ['r_delta', 'sigma_delta', 'beta_delta', 'alpha']
EXTRAP_IDX = [PARAM_NAMES.index(p) for p in EXTRAPOLATE_ROWS]


def bin1_shift(i):
    """Mean over z <= Z_ANCHOR_MAX of (beta_bin1 - beta_bin0), elementwise over the
    1 + n_lcdm coefficients, and its standard error (scatter/sqrt(N); falls back to
    the quadrature ridge errors if only one redshift is available)."""
    diffs, errs = [], []
    for z in CANDIDATE_REDSHIFTS:
        if z > Z_ANCHOR_MAX:
            continue
        b0, b1 = BETA_RAW[0].get((i, z)), BETA_RAW[1].get((i, z))
        if b0 is None or b1 is None:
            continue
        diffs.append(b1 - b0)
        errs.append(np.sqrt(SE_RAW[0][(i, z)] ** 2 + SE_RAW[1][(i, z)] ** 2))
    if not diffs:
        return None, None
    diffs = np.array(diffs)
    shift = diffs.mean(axis=0)
    shift_err = (diffs.std(axis=0, ddof=1) / np.sqrt(len(diffs)) if len(diffs) > 1
                 else np.mean(errs, axis=0))
    return shift, shift_err


BETA = {0: dict(BETA_RAW[0]), 1: {}}        # what the emulator actually uses
SE = {0: dict(SE_RAW[0]), 1: {}, 'avg': {}}
AVG = {}                                    # (i, z) -> (beta_bin0 + beta_bin1) / 2

for i in range(n_labels):
    shift, shift_err = bin1_shift(i)
    for z in CANDIDATE_REDSHIFTS:
        if z <= Z_ANCHOR_MAX or i not in EXTRAP_IDX:
            BETA[1][(i, z)] = BETA_RAW[1].get((i, z))
            if (i, z) in SE_RAW[1]:
                SE[1][(i, z)] = SE_RAW[1][(i, z)]
        else:
            b0 = BETA_RAW[0].get((i, z))
            if b0 is None or shift is None:
                BETA[1][(i, z)] = None
            else:
                BETA[1][(i, z)] = b0 + shift
                SE[1][(i, z)] = np.sqrt(SE_RAW[0][(i, z)] ** 2 + shift_err ** 2)

        b0, b1 = BETA[0].get((i, z)), BETA[1].get((i, z))
        if b0 is None or b1 is None:
            AVG[(i, z)] = None
        else:
            AVG[(i, z)] = (b0 + b1) / 2.0
            SE['avg'][(i, z)] = np.sqrt(SE[0][(i, z)] ** 2 + SE[1][(i, z)] ** 2) / 2.0

print(f'bin 1 at z > {Z_ANCHOR_MAX} for {EXTRAPOLATE_ROWS}: bin 0 + mean(bin 1 - bin 0) shift over '
      f'z <= {Z_ANCHOR_MAX}; {sum(AVG[(i, z)] is not None for i in range(n_labels) for z in CANDIDATE_REDSHIFTS)} '
      f'of {n_labels * len(CANDIDATE_REDSHIFTS)} average emulators built')
missing = [(PARAM_NAMES[i], z) for i in range(n_labels) for z in CANDIDATE_REDSHIFTS
           if BETA[0].get((i, z)) is not None and BETA[1].get((i, z)) is None]
if missing:
    print('no bin-1 emulator (so no average) for (parameter, z):', missing)


#%% save emulator parameters for config.py generation (bin 1 is the used,
# i.e. extrapolated-where-applicable, version)
EMULATOR_PATH = os.path.join(FIGURES_DIR, 'cosmo_emulator_params.h5')
with h5.File(EMULATOR_PATH, 'w') as hdf:
    hdf.attrs['param_names'] = PARAM_NAMES
    hdf.attrs['cosmo_names'] = LCDM_NAMES
    hdf.attrs['fiducial_cosmo'] = FIDUCIAL_LCDM
    hdf.attrs['bin1_extrapolated_above_z'] = Z_ANCHOR_MAX
    hdf.create_dataset('redshifts', data=np.array(CANDIDATE_REDSHIFTS))

    for name, source in (('bin0', BETA[0]), ('bin1', BETA[1]), ('average', AVG)):
        grp = hdf.create_group(name)
        for i in range(n_labels):
            for iz, z in enumerate(CANDIDATE_REDSHIFTS):
                beta = source.get((i, z))
                if beta is None:
                    continue
                sub = grp.create_group(f'param{i}_z{iz}')
                sub.attrs['redshift'] = z
                sub.attrs['param_index'] = i
                sub.create_dataset('dependent_cols', data=np.arange(n_lcdm, dtype=np.int32))
                sub.create_dataset('beta', data=beta)
print(f'saved emulator parameters to {EMULATOR_PATH}')


#%% ============================== PLOT SET 1 ==============================
# cosmology dependence: per-bin emulators (dashed) + average (solid), one
# figure per redshift, rho annotation + green shading. Bin-1 points at
# z > Z_ANCHOR_MAX are open markers: measured, but not used by the emulator.
# Per-row y-limits come from the data (median + scaled MAD) so a few extreme
# sims can't set the scale.
ALPHA_MAX = 0.45


def robust_ylim(vals_list, pad_frac=0.15, mad_k=4.0):
    vals = np.concatenate([np.atleast_1d(v) for v in vals_list if v is not None and len(np.atleast_1d(v))])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    if mad == 0:
        spread = np.std(vals)
        if spread == 0:
            spread = abs(med) * 0.1 + 1e-6
        lo, hi = med - spread, med + spread
    else:
        robust_sigma = 1.4826 * mad
        lo, hi = med - mad_k * robust_sigma, med + mad_k * robust_sigma
    lo, hi = max(lo, vals.min()), min(hi, vals.max())
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


for z in CANDIDATE_REDSHIFTS:
    if not any(len(sims_by[k][z]) >= MIN_SIMS_FOR_FIT for k in (0, 1)):
        continue
    theta_k = {k: np.array([lcdm_pars[n] for n in sims_by[k][z]]) for k in (0, 1)}
    theta_all = np.concatenate([theta_k[k] for k in (0, 1) if len(theta_k[k])])

    fig, ax = plt.subplots(n_labels, n_lcdm, figsize=(3.2 * n_lcdm, 2.6 * n_labels),
                           sharex='col', layout='constrained')
    for i in range(n_labels):
        y_k = {k: np.array([params[k][z][n][i] for n in sims_by[k][z]]) for k in (0, 1)}
        row_ylim = robust_ylim([y_k[0], y_k[1]], mad_k=5.0)
        beta0, beta1, beta_avg = BETA[0].get((i, z)), BETA[1].get((i, z)), AVG.get((i, z))

        for j in range(n_lcdm):
            rho = {k: rhos_bin[k].get((i, z), {}).get(j) for k in (0, 1)}
            rho_show = max(abs(rho[0] or 0.0), abs(rho[1] or 0.0))
            has_fit = beta0 is not None or beta1 is not None
            ax[i, j].set_facecolor(to_rgba('tab:green', ALPHA_MAX * min(rho_show, 1.0))
                                   if has_fit else to_rgba('white', 0.0))
            for k in (0, 1):
                if rho[k] is not None:
                    ax[i, j].annotate(rf'$\rho={rho[k]:.2f}$', xy=(0.05, 0.85 - 0.15 * k),
                                      xycoords='axes fraction', fontsize=7, color=f'C{k}')
                if len(y_k[k]) == 0:
                    continue
                unused = (k == 1 and z > Z_ANCHOR_MAX and i in EXTRAP_IDX)
                ax[i, j].plot(theta_k[k][:, j], y_k[k], 'o', ms=4, alpha=0.6, c=f'C{k}',
                              mfc='none' if unused else f'C{k}')
                drop_idx = [sims_by[k][z].index(n) for n in dropped_sims_log[k].get((i, z), [])]
                if drop_idx:
                    ax[i, j].plot(theta_k[k][drop_idx, j], y_k[k][drop_idx], 'x', ms=8, c=f'C{k}',
                                  mew=2, zorder=5)

            if beta_avg is not None:
                x_smooth = np.linspace(theta_all[:, j].min(), theta_all[:, j].max(), 50)
                q = np.tile(np.nanmean(theta_all, axis=0), (50, 1))
                q[:, j] = x_smooth
                ax[i, j].plot(x_smooth, eval_emulator(beta0, q), '--', c='C0', lw=1)
                ax[i, j].plot(x_smooth, eval_emulator(beta1, q), '--', c='C1', lw=1)
                ax[i, j].plot(x_smooth, eval_emulator(beta_avg, q), '-', c='purple', lw=2)

            if row_ylim is not None:
                ax[i, j].set_ylim(*row_ylim)
            if i == n_labels - 1:
                ax[i, j].set_xlabel(LCDM_LABELS[j])
        ax[i, 0].set_ylabel(labels[i])

    extrap = r' (extrapolated parameters: bin-1 open markers unused)' if z > Z_ANCHOR_MAX else ''
    fig.suptitle(rf'cosmology dependence at $z={z:g}$ ($\Lambda$CDM only) -- '
                 rf'dashed: per-bin emulators, solid: average' + extrap)
    fname = os.path.join(FIGURES_DIR, f'cosmo_dep_emulators_z{z:.3f}'.replace('.', 'p') + '.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f'saved {fname}')


#%% ============================== PLOT SET 2 ==============================
# residuals of each bin's data w.r.t. the AVERAGE emulator, one figure per
# redshift; one combined stat + one combined sys percentage per panel;
# per-row y-limits percentile-based, centered on 1.00, clamped.
Y_HALF_RANGE_MIN = 0.05
Y_HALF_RANGE_MAX = 0.5
Y_PERCENTILE = 1

for z in CANDIDATE_REDSHIFTS:
    theta_k = {k: np.array([lcdm_pars[n] for n in sims_by[k][z]]) for k in (0, 1)}

    panel = {}   # (i, k) -> (theta, ratio, sys)
    for i in range(n_labels):
        beta_avg = AVG.get((i, z))
        if beta_avg is None:
            continue
        for k in (0, 1):
            if len(sims_by[k][z]) == 0 or BETA[k].get((i, z)) is None:
                continue
            actual = np.array([params[k][z][n][i] for n in sims_by[k][z]])
            pred_avg = eval_emulator(beta_avg, theta_k[k])
            pred_k = eval_emulator(BETA[k][(i, z)], theta_k[k])
            panel[(i, k)] = (theta_k[k], pred_avg / actual, (pred_k - pred_avg) / actual)

    rows_shown = sorted({i for i, _ in panel})
    if not rows_shown:
        continue

    row_ylim = {}
    for i in rows_shown:
        vals = np.concatenate([panel[(i, k)][1] for k in (0, 1) if (i, k) in panel])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        lo, hi = np.nanpercentile(vals, [Y_PERCENTILE, 100 - Y_PERCENTILE])
        half = float(np.clip(max(abs(1.0 - lo), abs(hi - 1.0)) * 1.15, Y_HALF_RANGE_MIN, Y_HALF_RANGE_MAX))
        row_ylim[i] = (1.0 - half, 1.0 + half)

    fig2, ax2 = plt.subplots(n_labels, n_lcdm, figsize=(3.2 * n_lcdm, 2.6 * n_labels),
                             sharex='col', layout='constrained')
    for i in range(n_labels):
        for j in range(n_lcdm):
            ax2[i, j].axis('off')

    for i in rows_shown:
        ratio_cat = np.concatenate([panel[(i, k)][1] for k in (0, 1) if (i, k) in panel])
        sys_cat = np.concatenate([panel[(i, k)][2] for k in (0, 1) if (i, k) in panel])
        stat_rms = np.sqrt(np.nanmean((ratio_cat - 1) ** 2))
        sys_rms = np.sqrt(np.nanmean(sys_cat ** 2))
        txt = (rf'stat$={stat_rms:.1%}$'.replace('%', r'\%') + ', ' + rf'sys$={sys_rms:.1%}$'.replace('%', r'\%'))

        for j in range(n_lcdm):
            ax2[i, j].axis('on')
            for k in (0, 1):
                if (i, k) not in panel:
                    continue
                th, ratio, sys_frac = panel[(i, k)]
                unused = (k == 1 and z > Z_ANCHOR_MAX and i in EXTRAP_IDX)
                ax2[i, j].errorbar(th[:, j], ratio, yerr=np.abs(sys_frac), fmt='o', c=f'C{k}', ms=4,
                                   capsize=2, elinewidth=1, alpha=0.6, mfc='none' if unused else f'C{k}')
            ax2[i, j].text(0.95, 0.95, txt, transform=ax2[i, j].transAxes, va='top', ha='right', fontsize=6)
            ax2[i, j].axhline(1, c='k', ls='--', lw=1)
            for band, ls in [(0.05, ':'), (0.10, '-.')]:
                ax2[i, j].axhline(1 + band, c='gray', ls=ls, lw=1)
                ax2[i, j].axhline(1 - band, c='gray', ls=ls, lw=1)
            if i in row_ylim:
                ax2[i, j].set_ylim(*row_ylim[i])
        ax2[i, 0].set_ylabel(labels[i])

    for j in range(n_lcdm):
        ax2[-1, j].set_xlabel(LCDM_LABELS[j])

    extrap = r' (extrapolated parameters: bin-1 open markers not used)' if z > Z_ANCHOR_MAX else ''
    fig2.suptitle(rf'residuals w.r.t. the AVERAGE $\Lambda$CDM emulator, $z={z:g}$ (C0: bin 0, C1: bin 1)' + extrap)
    fname = os.path.join(FIGURES_DIR, f'cosmo_dep_residuals_vs_average_z{z:.3f}'.replace('.', 'p') + '.png')
    fig2.savefig(fname, dpi=150)
    plt.close(fig2)
    print(f'saved {fname}')


#%% ============================== PLOT SET 3 ==============================
# intercept and slope vs redshift with error bars and robust y-limits, saved twice:
#   *_raw.png    each bin's own ridge fits, plus the average where both bins have a fit;
#   *_extrap.png the same, with the extrapolated bin-1 (orange) and average (purple)
#                points overlaid for the EXTRAPOLATE_ROWS parameters. For those rows
#                the y-limits ignore bin 1's own fits above Z_ANCHOR_MAX, so a runaway
#                fit does not set the scale.
def raw_series(k, i, col):
    zs, vals, errs = [], [], []
    for z in CANDIDATE_REDSHIFTS:
        beta = BETA_RAW[k].get((i, z))
        if beta is None:
            continue
        zs.append(z)
        vals.append(beta[col])
        errs.append(SE_RAW[k][(i, z)][col])
    return np.array(zs), np.array(vals), np.array(errs)


def raw_avg_series(i, col):
    zs, vals, errs = [], [], []
    for z in CANDIDATE_REDSHIFTS:
        b0, b1 = BETA_RAW[0].get((i, z)), BETA_RAW[1].get((i, z))
        if b0 is None or b1 is None:
            continue
        zs.append(z)
        vals.append((b0[col] + b1[col]) / 2.0)
        errs.append(np.sqrt(SE_RAW[0][(i, z)][col] ** 2 + SE_RAW[1][(i, z)][col] ** 2) / 2.0)
    return np.array(zs), np.array(vals), np.array(errs)


def extrap_series(key, i, col):
    """Extrapolated bin 1 (key=1) or average (key='avg') for z > Z_ANCHOR_MAX, starting from
    the raw value at Z_ANCHOR_MAX so the dashed line joins the measured one."""
    beta_src, se_src = (BETA[1], SE[1]) if key == 1 else (AVG, SE['avg'])
    start = raw_series(1, i, col) if key == 1 else raw_avg_series(i, col)
    zs, vals, errs = [], [], []
    if Z_ANCHOR_MAX in start[0]:
        m = list(start[0]).index(Z_ANCHOR_MAX)
        zs.append(Z_ANCHOR_MAX); vals.append(start[1][m]); errs.append(start[2][m])
    for z in CANDIDATE_REDSHIFTS:
        if z <= Z_ANCHOR_MAX or beta_src.get((i, z)) is None:
            continue
        zs.append(z)
        vals.append(beta_src[(i, z)][col])
        errs.append(se_src[(i, z)][col])
    return np.array(zs), np.array(vals), np.array(errs)


def plot_vs_z(axis, i, col, extrap, add_legend=False):
    show_extrap = extrap and i in EXTRAP_IDX
    ylim_vals = []
    for key, label, c, ls in ((0, 'bin 0', 'C0', '--'), (1, 'bin 1', 'C1', '--'), ('avg', 'average', 'purple', '-')):
        zs, vals, errs = raw_avg_series(i, col) if key == 'avg' else raw_series(key, i, col)
        if zs.size == 0:
            continue
        axis.errorbar(zs, vals, yerr=errs, marker='o', ms=4, capsize=2, c=c, ls=ls,
                      label=label if add_legend else None)
        keep = (zs <= Z_ANCHOR_MAX) if (show_extrap and key != 0) else np.ones_like(zs, dtype=bool)
        ylim_vals.append(vals[keep])
    if show_extrap:
        for key, c in ((1, 'C1'), ('avg', 'purple')):
            zs, vals, errs = extrap_series(key, i, col)
            if zs.size > 1:
                axis.errorbar(zs, vals, yerr=errs, marker='o', ms=4, capsize=2, c=c, ls='--', lw=1,
                              markerfacecolor='none')
                ylim_vals.append(vals)
        axis.axvline(Z_ANCHOR_MAX, c='gray', ls=':', lw=1, alpha=0.5)
    if add_legend and extrap and EXTRAP_IDX:
        axis.plot([], [], 'o--', c='k', mfc='none', lw=1, label='extrapolated (bin 0 + shift)')
    return ylim_vals


for extrap, suffix in ((False, 'raw'), (True, 'extrap')):
    fig_int, ax_int = plt.subplots(n_labels, 1, figsize=(5, 2.2 * n_labels), sharex=True, layout='constrained')
    for i in range(n_labels):
        ylim = robust_ylim(plot_vs_z(ax_int[i], i, 0, extrap, add_legend=(i == 0)), 0.25)
        if ylim is not None:
            ax_int[i].set_ylim(*ylim)
        ax_int[i].set_ylabel(labels[i])
    ax_int[0].legend(fontsize=7)
    ax_int[-1].set_xlabel(r'$z$')
    fig_int.suptitle('Compact-model parameter value at fiducial cosmology vs redshift'
                     + (' (raw fits)' if not extrap else ' (with extrapolation)'))
    fname = os.path.join(FIGURES_DIR, f'emulator_intercept_vs_redshift_{suffix}.png')
    fig_int.savefig(fname, dpi=150)
    print(f'saved {fname}')

    fig_slope, ax_slope = plt.subplots(n_labels, n_lcdm, figsize=(3.2 * n_lcdm, 2.2 * n_labels),
                                       sharex='col', layout='constrained')
    for i in range(n_labels):
        for j in range(n_lcdm):
            vals = plot_vs_z(ax_slope[i, j], i, 1 + j, extrap)
            ax_slope[i, j].axhline(0, c='gray', ls='--', lw=1)
            ylim = robust_ylim(vals + [np.array([0.0])], 0.25)
            if ylim is not None:
                ax_slope[i, j].set_ylim(*ylim)
            if i == n_labels - 1:
                ax_slope[i, j].set_xlabel(r'$z$')
        ax_slope[i, 0].set_ylabel(labels[i])
    for j in range(n_lcdm):
        ax_slope[0, j].set_title(LCDM_LABELS[j])
    fig_slope.suptitle(r'Cosmology-dependence slope ($\partial\,\mathrm{param}/\partial\Delta\theta$) vs redshift'
                       + (' (raw fits)' if not extrap else ' (with extrapolation)'))
    fname = os.path.join(FIGURES_DIR, f'emulator_slope_vs_redshift_{suffix}.png')
    fig_slope.savefig(fname, dpi=150)
    print(f'saved {fname}')
# %%
# Needs the main script run through the extrapolation and plot-set-3 cells (BETA_RAW, SE_RAW, build_emulators, plot_vs_z, ...).
#%% new extrapolations: raw bin-0 / bin-1 / average fits plus the extrapolated bin-1 (orange)
# and average (purple) points above Z_ANCHOR_MAX, for only the parameters in the list below.
# Uses the ridge fits already in memory (BETA_RAW, SE_RAW) -- no refitting. Rebuilds BETA, SE
# and AVG, so re-run the emulator-saving cell afterwards if you want the h5 to match.

def build_emulators(extrap_idx):
    """(BETA, SE, AVG) actually used by the emulator: bin 1 is bin 0 + the averaged shift above
    Z_ANCHOR_MAX for the parameter indices in extrap_idx, and its own fit otherwise."""
    BETA = {0: dict(BETA_RAW[0]), 1: {}}        # what the emulator actually uses
    SE = {0: dict(SE_RAW[0]), 1: {}, 'avg': {}}
    AVG = {}                                    # (i, z) -> (beta_bin0 + beta_bin1) / 2
 
    for i in range(n_labels):
        shift, shift_err = bin1_shift(i)
        for z in CANDIDATE_REDSHIFTS:
            if z <= Z_ANCHOR_MAX or i not in extrap_idx:
                BETA[1][(i, z)] = BETA_RAW[1].get((i, z))
                if (i, z) in SE_RAW[1]:
                    SE[1][(i, z)] = SE_RAW[1][(i, z)]
            else:
                b0 = BETA_RAW[0].get((i, z))
                if b0 is None or shift is None:
                    BETA[1][(i, z)] = None
                else:
                    BETA[1][(i, z)] = b0 + shift
                    SE[1][(i, z)] = np.sqrt(SE_RAW[0][(i, z)] ** 2 + shift_err ** 2)
 
            b0, b1 = BETA[0].get((i, z)), BETA[1].get((i, z))
            if b0 is None or b1 is None:
                AVG[(i, z)] = None
            else:
                AVG[(i, z)] = (b0 + b1) / 2.0
                SE['avg'][(i, z)] = np.sqrt(SE[0][(i, z)] ** 2 + SE[1][(i, z)] ** 2) / 2.0
    return BETA, SE, AVG


EXTRAPOLATE_ROWS = ['r_delta', 'beta_delta', 'alpha', 'r_c']
EXTRAP_IDX = [PARAM_NAMES.index(p) for p in EXTRAPOLATE_ROWS]
BETA, SE, AVG = build_emulators(EXTRAP_IDX)

fig_x, ax_x = plt.subplots(len(EXTRAP_IDX), 1 + n_lcdm, figsize=(3.0 * (1 + n_lcdm), 2.4 * len(EXTRAP_IDX)),
                           sharex=True, layout='constrained')
for r, i in enumerate(EXTRAP_IDX):
    for col in range(1 + n_lcdm):          # col 0 = intercept, 1..n_lcdm = slopes
        vals = plot_vs_z(ax_x[r, col], i, col, extrap=True, add_legend=(r == 0 and col == 0))
        if col > 0:
            ax_x[r, col].axhline(0, c='gray', ls='--', lw=1)
            vals = vals + [np.array([0.0])]
        ylim = robust_ylim(vals, 0.25)
        if ylim is not None:
            ax_x[r, col].set_ylim(*ylim)
        if r == len(EXTRAP_IDX) - 1:
            ax_x[r, col].set_xlabel(r'$z$')
    ax_x[r, 0].set_ylabel(labels[i])
ax_x[0, 0].set_title('intercept')
for j in range(n_lcdm):
    ax_x[0, 1 + j].set_title(f'slope: {LCDM_LABELS[j]}')
ax_x[0, 0].legend(fontsize=7)
fig_x.suptitle(r'Extrapolated parameters: bin 1 above $z=%g$ = bin 0 + mean shift' % Z_ANCHOR_MAX)

# Needs the main script run through the extrapolation and plot-set-3 cells (BETA_RAW, SE_RAW, build_emulators, plot_vs_z, robust_ylim, ...).
#%% all 11 parameters x (intercept + 4 slopes) in one figure: raw bin 0 / bin 1 / average for every
# parameter, plus the extrapolated bin-1 (orange) and average (purple) points above Z_ANCHOR_MAX for
# the EXTRAPOLATE_ROWS parameters only. No refitting.
EXTRAPOLATE_ROWS = ['r_delta', 'beta_delta', 'alpha', 'r_c']
EXTRAP_IDX = [PARAM_NAMES.index(p) for p in EXTRAPOLATE_ROWS]
BETA, SE, AVG = build_emulators(EXTRAP_IDX)

fig_all, ax_all = plt.subplots(n_labels, 1 + n_lcdm, figsize=(3.0 * (1 + n_lcdm), 2.0 * n_labels),
                               sharex=True, layout='constrained')
for i in range(n_labels):
    for col in range(1 + n_lcdm):          # col 0 = intercept, 1..n_lcdm = slopes
        vals = plot_vs_z(ax_all[i, col], i, col, extrap=True, add_legend=(i == 0 and col == 0))
        if col > 0:
            ax_all[i, col].axhline(0, c='gray', ls='--', lw=1)
            vals = vals + [np.array([0.0])]
        ylim = robust_ylim(vals, 0.25)
        if ylim is not None:
            ax_all[i, col].set_ylim(*ylim)
        if i == n_labels - 1:
            ax_all[i, col].set_xlabel(r'$z$')
    ax_all[i, 0].set_ylabel(labels[i])
ax_all[0, 0].set_title('intercept')
for j in range(n_lcdm):
    ax_all[0, 1 + j].set_title(f'slope: {LCDM_LABELS[j]}')
ax_all[0, 0].legend(fontsize=6)
fig_all.suptitle(r'All parameters; extrapolated above $z=%g$ only for %s'
                 % (Z_ANCHOR_MAX, ', '.join(EXTRAPOLATE_ROWS)))
fig_all.savefig(os.path.join(FIGURES_DIR, 'emulator_all_params_vs_redshift.png'), dpi=150)

#%% re-save the emulator h5 for the final extrapolation scheme, including which rows were extrapolated
EXTRAPOLATE_ROWS = ['r_delta', 'beta_delta', 'alpha', 'r_c']
EXTRAP_IDX = [PARAM_NAMES.index(p) for p in EXTRAPOLATE_ROWS]
BETA, SE, AVG = build_emulators(EXTRAP_IDX)

EMULATOR_PATH = os.path.join(FIGURES_DIR, 'cosmo_emulator_params.h5')
with h5.File(EMULATOR_PATH, 'w') as hdf:
    hdf.attrs['param_names'] = PARAM_NAMES
    hdf.attrs['cosmo_names'] = LCDM_NAMES
    hdf.attrs['fiducial_cosmo'] = FIDUCIAL_LCDM
    hdf.attrs['bin1_extrapolated_above_z'] = Z_ANCHOR_MAX
    hdf.attrs['bin1_extrapolated_rows'] = EXTRAPOLATE_ROWS
    hdf.create_dataset('redshifts', data=np.array(CANDIDATE_REDSHIFTS))

    for name, source in (('bin0', BETA[0]), ('bin1', BETA[1]), ('average', AVG)):
        grp = hdf.create_group(name)
        for i in range(n_labels):
            for iz, z in enumerate(CANDIDATE_REDSHIFTS):
                beta = source.get((i, z))
                if beta is None:
                    continue
                sub = grp.create_group(f'param{i}_z{iz}')
                sub.attrs['redshift'] = z
                sub.attrs['param_index'] = i
                sub.create_dataset('dependent_cols', data=np.arange(n_lcdm, dtype=np.int32))
                sub.create_dataset('beta', data=beta)
print(f'saved {EMULATOR_PATH}')

