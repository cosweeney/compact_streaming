"""
cosmo_grid_vel_fit_out.py

Fits the 9-parameter compact velocity model (sigma_v: 4 params, Delta: 5
params) across every available (sim, redshift, mass bin) in the
AbacusSummit cosmology grid, builds LambdaCDM-only ridge emulators per
(mass bin, redshift), and produces three sets of plots:
  1. cosmo_dep_emulators_z*.png   -- per-redshift cosmology-dependence
     scatter + per-bin/average fit lines, rho-annotated, green-shaded
     where either bin's fit shows real dependence.
  2. cosmo_dep_residuals_vs_average_z*.png -- per-redshift residuals of
     each bin's own fit against the bin-averaged emulator, one combined
     stat/sys percentage per panel.
  3. emulator_intercept_vs_redshift.png / emulator_slope_vs_redshift.png
     -- ridge-fit intercept and per-LambdaCDM-column slope vs z, with
     propagated error bars and robust (MAD-based) y-limits.

Includes the bounded + multi-start fit_delta_joint fix: the original
Delta-model fit (A_delta0, r_delta, sigma_delta, beta_delta, alpha) had no
bounds at all, unlike the sigma_v fit, and a real fraction of bin-1 sims
(10/31 at z=1.1, checked directly) converged to degenerate, unphysical
optima (r_delta < 0, beta_delta/alpha ~20-30) as a result. Bounding the
shape parameters to physically sensible ranges and giving curve_fit
several randomized restarts fixes this at the source, for every sim at
every (z, bin), rather than patching outliers after the fact.

I have not run this myself -- no cluster access -- so treat this as a
structurally-complete draft to check over, same as everything else in
this pipeline.
"""

#%% imports and setup
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import t as student_t, pearsonr
from scipy.interpolate import InterpolatedUnivariateSpline
from joblib import Parallel, delayed
from types import SimpleNamespace
from matplotlib.colors import to_rgba

import matplotlib
matplotlib.rcParams.update({'text.usetex': True,
                            'font.family': 'Computer Modern Roman'})
plt.rcParams['axes.labelsize'] = 20

import nb.calibration.runs.auto_runs.abacus_cosmo_core_auto as core   # safe: no pycorr import in this module
from compact.velocities.pairhist import PairHist

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


# ---------------------------------------------------------------- 9-param model
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


def fit_sigma_joint(x, r_p_arr, sig, weights=None):
    R = np.broadcast_to(r_p_arr[:, None], x.shape)
    m = np.isfinite(x) & np.isfinite(sig) & (sig > SIG_FLOOR)
    X, RR, S = x[m], R[m], sig[m]
    W = weights[m] if weights is not None else np.ones_like(S)
    R_piv = 72.5 #np.median(RR)
    popt, _ = curve_fit(
        lambda c, s0, Ap, Arp, sr: sigma_mod_2d(c, s0, Ap, Arp, sr, R_piv),
        (X, RR), S, p0=_seed_sigma(X, RR, S, R_piv), sigma=1.0 / np.sqrt(W),
        bounds=([0.0, 1e-4, 1e-3, 1e-3], [np.inf, 1.0, np.inf, np.inf]), maxfev=20_000)
    return popt


def A_delta_mod(R, A_delta0, r_delta):
    return A_delta0 * np.exp(-R / r_delta)


def delta_mod_alpha(x, A, sigma_delta, beta_delta, alpha):
    return A * x**alpha / (1 + (x / sigma_delta)**beta_delta)


def delta_mod_alpha_R(coords, A_delta0, r_delta, sigma_delta, beta_delta, alpha):
    x, R = coords
    return delta_mod_alpha(x, A_delta_mod(R, A_delta0, r_delta), sigma_delta, beta_delta, alpha)


def fit_delta_joint(x, r_p_arr, D, p0=DEL_P0, weights=None, n_restarts=8, seed=0):
    """Bounded + multi-start replacement for the original unconstrained
    fit. A_delta0 is sign-unconstrained (the model allows either sign);
    r_delta, sigma_delta, beta_delta are constrained > 0 (a negative
    lengthscale or shape exponent is degenerate, not a legitimate
    alternative fit); alpha is bounded away from negative values, both to
    block the runaway-to-~30 regime seen in a real fraction of bin-1 sims
    (10/31 at z=1.1, checked directly) and to sidestep the documented x=0
    singularity in delta_mod_alpha for negative alpha. n_restarts random
    perturbations of p0 are tried in addition to p0 itself; whichever
    converged fit has the lowest weighted residual sum of squares wins."""
    R = np.broadcast_to(r_p_arr[:, None], x.shape)
    m = np.isfinite(x) & np.isfinite(D) & (D > DEL_FLOOR)
    X, RR, DD = x[m], R[m], D[m]
    W = weights[m] if weights is not None else np.ones_like(DD)
    sigma_w = 1.0 / np.sqrt(W)

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
                               sigma=sigma_w, bounds=(lower, upper), maxfev=20_000)
        except RuntimeError:
            continue
        resid = (delta_mod_alpha_R((X, RR), *popt) - DD) / sigma_w
        cost = np.sum(resid ** 2)
        if cost < best_cost:
            best_cost, best_popt = cost, popt

    if best_popt is None:
        raise RuntimeError('fit_delta_joint: all restarts failed to converge')
    return best_popt


def skewed_t_pdf(x, xi, omega, alpha, nu):
    z = (x - xi) / omega
    t_pdf = student_t.pdf(z, df=nu)
    t_cdf = student_t.cdf(alpha * z * np.sqrt((nu + 1) / (nu + z**2)), df=nu + 1)
    return (2 / omega) * t_pdf * t_cdf


NU_FIXED = 6
REBIN = 5
N_MIN = 50   # was 500 -- lowered so weighting can use thin-but-nonzero cells
             # instead of discarding them outright; see fit_sigma_joint /
             # fit_delta_joint's weights= argument.


def meas_sigv_grid(h, k, n_min=N_MIN):
    n = h.n[k].astype(float)
    mu = np.where(n >= n_min, h.sum_v[k] / np.where(n > 0, n, 1), np.nan)
    m2 = np.where(n >= n_min, h.sum_v2[k] / np.where(n > 0, n, 1), np.nan)
    return np.sqrt(np.maximum(m2 - mu**2, 0.0))


def fit_skewt_grid(h, k, n_min=N_MIN):
    n_k, sum_v_k, sum_v2_k, counts_k = h.n[k], h.sum_v[k], h.sum_v2[k], h.counts[k]
    cent_fine = h.v_centers
    dv = (cent_fine[1] - cent_fine[0]) * REBIN
    cent = cent_fine.reshape(-1, REBIN).mean(axis=1)

    n_R, n_l = n_k.shape
    ST_pars = np.full((n_R, n_l, 3), np.nan)
    for j in range(n_R):
        for i in range(n_l):
            n = n_k[j, i]
            if n < n_min:
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


#%% scan for available (sim, z, bin) output and fit each, in parallel
def out_path(n, z, k):
    return os.path.join(core.OUT_DIR.format(sim=core.sim_name(n), zdir=core.zdir(z)),
                        f'pairhist_auto_mbin{k}.h5')


tasks = [(n, z, k) for n in core.SIM_NUMS for z in CANDIDATE_REDSHIFTS
         for k in range(N_MASS) if os.path.exists(out_path(n, z, k))]
print(f'{len(tasks)} (sim, z, bin) combinations available')


def fit_one(n, z, k):
    path = out_path(n, z, k)
    try:
        h = PairHist.load(path)
        sig = meas_sigv_grid(h, 0)
        ST_pars = fit_skewt_grid(h, 0)

        r_vr, vr_mean, npairs_vr = vr_profile(h, 0, VR_R_EDGES)
        good_vr = np.isfinite(vr_mean) & (npairs_vr > 100)
        if good_vr.sum() < 4:
            return n, z, k, None, None
        vr_interp = InterpolatedUnivariateSpline(r_vr[good_vr], vr_mean[good_vr], k=3, ext=3)
        Delta = ST_pars[:, :, 0] - vr_interp(r3_2d) * r_l_2d / r3_2d

        sig_params = fit_sigma_joint(x_grid, r_p, sig)
        try:
            del_params = fit_delta_joint(x_grid, r_p, Delta, p0=DEL_P0)
        except RuntimeError:
            del_params = None

        return n, z, k, sig_params, del_params
    except Exception:
        return n, z, k, None, None


results = Parallel(n_jobs=-1, verbose=10)(delayed(fit_one)(n, z, k) for n, z, k in tasks)

params = {k: {z: {} for z in CANDIDATE_REDSHIFTS} for k in range(N_MASS)}
for n, z, k, sig_p, del_p in results:
    if sig_p is not None and del_p is not None:
        params[k][z][n] = np.concatenate([sig_p, del_p])

for k in range(N_MASS):
    for z in CANDIDATE_REDSHIFTS:
        if params[k][z]:
            print(f'bin {k}, z={z}: {len(params[k][z])} sims fit successfully')

labels = [r"$\sigma_0$ [km/s]", r"$A_{\rho,0}$", r"$A_{\rho,p}$ [$h^{-1}$ Mpc]", r"$\sigma_\rho$ [$h^{-1}$ Mpc]",
          r"$A_{\Delta,0}$ [km/s]", r"$r_\Delta$ [$h^{-1}$ Mpc]", r"$\sigma_\Delta$ [$h^{-1}$ Mpc]",
          r"$\beta_\Delta$", r"$\alpha_\Delta$"]
n_labels = len(labels)


#%% r_p-coarsening rescue path for bin 1 @ z=1.4. Uses the same
# fit_delta_joint defined above, so the bounds/multistart fix applies here
# too automatically.
def merge_pairhist_rp_variable(h, group_edges):
    n_rp = h.n[0].shape[0]
    if group_edges[0] != 0 or group_edges[-1] != n_rp:
        raise ValueError(f'group_edges must span [0, {n_rp}], got {group_edges}')
    starts = np.array(group_edges[:-1])

    def merge(arr):
        return np.add.reduceat(arr, starts, axis=0)

    R_edges_new = h.R_edges[np.array(group_edges)]

    return SimpleNamespace(
        n=[merge(h.n[0])], sum_v=[merge(h.sum_v[0])], sum_v2=[merge(h.sum_v2[0])],
        sum_vr=[merge(h.sum_vr[0])], counts=[merge(h.counts[0])],
        R_edges=R_edges_new, rlos_edges=h.rlos_edges, v_centers=h.v_centers,
    )


def build_group_edges(n_stack, n_min, target_coverage):
    n_rp = n_stack.shape[1]
    edges = [n_rp]
    end = n_rp
    coverage_achieved = []
    while end > 0:
        start = end
        pooled = np.zeros(n_stack.shape[::2])
        cov = 0.0
        while start > 0 and cov < target_coverage:
            start -= 1
            pooled = pooled + n_stack[:, start, :]
            cov = (pooled >= n_min).mean()
        edges.append(start)
        coverage_achieved.append(cov)
        end = start
    edges = sorted(set(edges))
    return edges, list(reversed(coverage_achieved))


N_MIN_MERGE = 100
TARGET_COVERAGE = 0.5

_z_rescue, _k_rescue = 1.4, 1
_sims_for_edges = [n for n in core.SIM_NUMS if os.path.exists(out_path(n, _z_rescue, _k_rescue))]
_n_stack = np.stack([PairHist.load(out_path(n, _z_rescue, _k_rescue)).n[0] for n in _sims_for_edges])
group_edges_z1p4_bin1, _coverages = build_group_edges(_n_stack, N_MIN_MERGE, TARGET_COVERAGE)
print(f'z={_z_rescue}, bin={_k_rescue}: {len(group_edges_z1p4_bin1) - 1} r_p groups '
      f'from {len(_sims_for_edges)} sims, '
      f'widths (small-r_p first) {np.diff(group_edges_z1p4_bin1).tolist()}')


def fit_one_rescue_variable(n, z, k, group_edges, n_min):
    path = out_path(n, z, k)
    try:
        h_raw = PairHist.load(path)
        h = merge_pairhist_rp_variable(h_raw, group_edges)

        r_p_c = 0.5 * (h.R_edges[1:] + h.R_edges[:-1])
        r_l_c = 0.5 * (h.rlos_edges[1:] + h.rlos_edges[:-1])
        rp2d, rl2d = np.meshgrid(r_p_c, r_l_c, indexing='ij')
        r3_2d_c = np.sqrt(rp2d**2 + rl2d**2)
        x_grid_c = r3_2d_c - rp2d

        sig = meas_sigv_grid(h, 0, n_min=n_min)
        ST_pars = fit_skewt_grid(h, 0, n_min=n_min)

        r_vr, vr_mean, npairs_vr = vr_profile(h, 0, VR_R_EDGES)
        good_vr = np.isfinite(vr_mean) & (npairs_vr > 100)
        if good_vr.sum() < 4:
            return n, z, k, None, None
        vr_interp = InterpolatedUnivariateSpline(r_vr[good_vr], vr_mean[good_vr], k=3, ext=3)
        Delta = ST_pars[:, :, 0] - vr_interp(r3_2d_c) * rl2d / r3_2d_c

        n_cells = h.n[0].astype(float)
        sig_params = fit_sigma_joint(x_grid_c, r_p_c, sig, weights=n_cells)
        try:
            del_params = fit_delta_joint(x_grid_c, r_p_c, Delta, p0=DEL_P0, weights=n_cells)
        except RuntimeError:
            del_params = None

        return n, z, k, sig_params, del_params
    except Exception:
        return n, z, k, None, None


RESCUE_JOBS = [(n, _z_rescue, _k_rescue) for n in core.SIM_NUMS
              if os.path.exists(out_path(n, _z_rescue, _k_rescue))]
rescue_results_z1p4_bin1 = Parallel(n_jobs=-1, verbose=10)(
    delayed(fit_one_rescue_variable)(n, z, k, group_edges_z1p4_bin1, N_MIN_MERGE)
    for n, z, k in RESCUE_JOBS)

n_rescued = sum(1 for (_, _, _, sp, dp) in rescue_results_z1p4_bin1 if sp is not None and dp is not None)
print(f'z=1.4, bin 1: {n_rescued}/{len(RESCUE_JOBS)} sims fit successfully with r_p rescue')

for n, z, k, sig_p, del_p in rescue_results_z1p4_bin1:
    if sig_p is not None and del_p is not None:
        params[k][z][n] = np.concatenate([sig_p, del_p])

# z=1.7 bin 1: pair-starved across the ENTIRE r_p range -- excluded rather
# than rescued (see the earlier diagnostics for why).
if 1.7 in params[1]:
    params[1][1.7] = {}

print('\nfinal per-(bin, z) sim counts:')
for k in range(N_MASS):
    for z in CANDIDATE_REDSHIFTS:
        if params[k][z]:
            print(f'bin {k}, z={z}: {len(params[k][z])} sims fit successfully')

#%% undo rescue: rebuild params from the original fit_one results only
params = {k: {z: {} for z in CANDIDATE_REDSHIFTS} for k in range(N_MASS)}
for n, z, k, sig_p, del_p in results:
    if sig_p is not None and del_p is not None:
        params[k][z][n] = np.concatenate([sig_p, del_p])

print('rebuilt params from native-grid results (rescue undone, z=1.7 bin 1 restored):')
for k in range(N_MASS):
    for z in CANDIDATE_REDSHIFTS:
        if params[k][z]:
            print(f'bin {k}, z={z}: {len(params[k][z])} sims')


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
              'A_delta0', 'r_delta', 'sigma_delta', 'beta_delta', 'alpha']
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


#%% LambdaCDM ridge emulators: always fits all n_lcdm columns, shrinking
# rather than dropping a column's slope when the data doesn't support it.
MIN_SIMS_FOR_FIT = 4

common_sims_by_z = {z: sorted(set(params[0].get(z, {})) & set(params[1].get(z, {})))
                    for z in CANDIDATE_REDSHIFTS}


def ridge_fit_beta(delta, y, alpha):
    n_sim, n_col = delta.shape
    scale = np.std(delta, axis=0)
    scale[scale == 0] = 1.0
    delta_s = delta / scale
    X = np.column_stack([np.ones(n_sim), delta_s])
    Gamma = np.diag([0.0] + [alpha] * n_col)
    beta_s = np.linalg.solve(X.T @ X + Gamma, X.T @ y)
    return np.concatenate([[beta_s[0]], beta_s[1:] / scale])


ALPHA_GRID = np.logspace(-3, 3, 13)


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


def fit_one_bin_ridge(sims_z, k, z, i, alpha_grid=ALPHA_GRID):
    if len(sims_z) < MIN_SIMS_FOR_FIT:
        return None, None, None, {}
    theta = np.array([lcdm_pars[n] for n in sims_z])
    delta = FIDUCIAL_LCDM[None, :] - theta
    y = np.array([params[k][z][n][i] for n in sims_z])

    alpha = ridge_cv_alpha(delta, y, alpha_grid)
    beta = ridge_fit_beta(delta, y, alpha)
    rho_diag = {j: pearsonr(theta[:, j], y)[0] for j in range(n_lcdm)}
    return list(range(n_lcdm)), beta, alpha, rho_diag


emulator_bin = {0: {}, 1: {}}
emulator_avg = {}
rhos_bin = {0: {}, 1: {}}
alpha_bin = {0: {}, 1: {}}

for i in range(n_labels):
    for z in CANDIDATE_REDSHIFTS:
        sims_z = common_sims_by_z[z]
        cols0, beta0, a0, rhos0 = fit_one_bin_ridge(sims_z, 0, z, i)
        cols1, beta1, a1, rhos1 = fit_one_bin_ridge(sims_z, 1, z, i)
        emulator_bin[0][(i, z)] = (cols0, beta0)
        emulator_bin[1][(i, z)] = (cols1, beta1)
        rhos_bin[0][(i, z)] = rhos0
        rhos_bin[1][(i, z)] = rhos1
        alpha_bin[0][(i, z)] = a0
        alpha_bin[1][(i, z)] = a1

        if beta0 is None or beta1 is None:
            emulator_avg[(i, z)] = (None, None)
            continue
        emulator_avg[(i, z)] = (list(range(n_lcdm)), (beta0 + beta1) / 2.0)

n_fit = sum(1 for i in range(n_labels) for z in CANDIDATE_REDSHIFTS if emulator_avg[(i, z)][1] is not None)
print(f'ridge-fit LambdaCDM emulators for {n_fit} of '
      f'{n_labels * len(CANDIDATE_REDSHIFTS)} (parameter, redshift) combinations')


def eval_emulator(dependent_cols, beta, theta_query, hold_at_fiducial=None):
    if beta is None:
        return np.nan
    val = beta[0]
    for idx, j in enumerate(dependent_cols):
        if hold_at_fiducial and j in hold_at_fiducial:
            continue
        val += beta[1 + idx] * (FIDUCIAL_LCDM[j] - theta_query[j])
    return val


#%% ridge covariance at the already-selected alpha -- for error bars in
# the intercept/slope-vs-z plots below.
def ridge_beta_cov(delta, y, alpha):
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


INTERCEPT_ERR = {0: {}, 1: {}, 'avg': {}}
SLOPE_ERR = {0: {}, 1: {}, 'avg': {}}

for i in range(n_labels):
    for z in CANDIDATE_REDSHIFTS:
        sims_z = common_sims_by_z[z]
        if len(sims_z) < MIN_SIMS_FOR_FIT:
            continue
        theta = np.array([lcdm_pars[n] for n in sims_z])
        delta = FIDUCIAL_LCDM[None, :] - theta

        se_by_bin = {}
        for k in (0, 1):
            _, beta_k = emulator_bin[k][(i, z)]
            if beta_k is None:
                se_by_bin[k] = None
                continue
            y = np.array([params[k][z][n][i] for n in sims_z])
            _, cov = ridge_beta_cov(delta, y, alpha_bin[k][(i, z)])
            se = np.sqrt(np.diag(cov))
            se_by_bin[k] = se
            INTERCEPT_ERR[k][(i, z)] = se[0]
            SLOPE_ERR[k][(i, z)] = se[1:]

        if se_by_bin.get(0) is not None and se_by_bin.get(1) is not None:
            se_avg = np.sqrt(se_by_bin[0] ** 2 + se_by_bin[1] ** 2) / 2.0
            INTERCEPT_ERR['avg'][(i, z)] = se_avg[0]
            SLOPE_ERR['avg'][(i, z)] = se_avg[1:]

BIN_STYLE = {0: dict(c='C0', ls='--', label='low-mass fit'),
             1: dict(c='C1', ls='--', label='high-mass fit'),
             'avg': dict(c='purple', ls='-', label='simultaneous fit')}


#%% robust outlier exclusion for the emulator fitting stage -- operates on
# already-fit params, NOT on the velocity-model fits themselves, so this
# is cheap to apply/tune without re-running fit_one/fit_one_rescue_variable.
# Same median + scaled-MAD approach as robust_ylim/robust_ylim_rowdata,
# now used to EXCLUDE points from a fit rather than just clip an axis.
#
# Caveat worth stating plainly: this flags based on the y-VALUE alone, not
# on a residual from an initial fit -- it can't in principle distinguish
# "unphysical fit blowup" from "a real, extreme cosmology-dependence point
# that happens to sit far from the bulk." Given the values actually seen
# (sigma0 ~7500 km/s against a ~300 km/s bulk, A_Delta,0 ~ -1.6e8) that
# distinction isn't really in question here -- these are degenerate
# curve_fit failures, not real signal -- but if this is used on parameters
# with more legitimately heavy tails, a residual-based (fit -> flag ->
# refit) version would be the more careful choice.
def robust_outlier_mask(y, mad_k=6.0):
    y = np.asarray(y, dtype=float)
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    if mad == 0:
        return np.ones_like(y, dtype=bool)
    robust_sigma = 1.4826 * mad
    return np.abs(y - med) <= mad_k * robust_sigma


def filtered_sims_theta_y(sims_z, k, z, i, mad_k=6.0, min_keep=None):
    """Returns (sims_kept, theta_kept, y_kept, dropped_sims). Falls back to
    keeping everyone (dropped_sims=[]) if the mask would leave fewer than
    min_keep sims -- refusing to drop anything rather than risk fitting on
    too few points; that's a sign the threshold was too aggressive for
    this (i, k, z), not that there's truly nothing to trim."""
    if min_keep is None:
        min_keep = MIN_SIMS_FOR_FIT
    sims_arr = np.array(sims_z)
    theta_full = np.array([lcdm_pars[n] for n in sims_z])
    y_full = np.array([params[k][z][n][i] for n in sims_z])

    keep = robust_outlier_mask(y_full, mad_k=mad_k)
    dropped = sims_arr[~keep].tolist()
    if keep.sum() < min_keep:
        keep = np.ones_like(keep, dtype=bool)
        dropped = []
    return sims_arr[keep], theta_full[keep], y_full[keep], dropped


def fit_one_bin_ridge(sims_z, k, z, i, alpha_grid=ALPHA_GRID, mad_k=6.0):
    """Same as before, but fits on the outlier-filtered subset. Returns a
    5-tuple now: (dependent_cols, beta, alpha_used, rho_diag, dropped_sims)
    -- the extra dropped_sims lets the fitting loop log exactly which
    sims got excluded, per (parameter, bin, redshift)."""
    if len(sims_z) < MIN_SIMS_FOR_FIT:
        return None, None, None, {}, []
    _, theta, y, dropped = filtered_sims_theta_y(sims_z, k, z, i, mad_k=mad_k)
    if len(y) < MIN_SIMS_FOR_FIT:
        return None, None, None, {}, dropped

    delta = FIDUCIAL_LCDM[None, :] - theta
    alpha = ridge_cv_alpha(delta, y, alpha_grid)
    beta = ridge_fit_beta(delta, y, alpha)
    rho_diag = {j: pearsonr(theta[:, j], y)[0] for j in range(n_lcdm)}
    return list(range(n_lcdm)), beta, alpha, rho_diag, dropped


#%% re-run the emulator fitting loop with outlier exclusion, logging which
# sims got dropped so it's possible to check whether it's a small,
# consistent set of bad sims (as suspected) rather than scattered per-row noise
from collections import Counter

emulator_bin = {0: {}, 1: {}}
emulator_avg = {}
rhos_bin = {0: {}, 1: {}}
alpha_bin = {0: {}, 1: {}}
dropped_sims_log = {0: {}, 1: {}}

for i in range(n_labels):
    for z in CANDIDATE_REDSHIFTS:
        sims_z = common_sims_by_z[z]
        cols0, beta0, a0, rhos0, drop0 = fit_one_bin_ridge(sims_z, 0, z, i)
        cols1, beta1, a1, rhos1, drop1 = fit_one_bin_ridge(sims_z, 1, z, i)
        emulator_bin[0][(i, z)] = (cols0, beta0)
        emulator_bin[1][(i, z)] = (cols1, beta1)
        rhos_bin[0][(i, z)] = rhos0
        rhos_bin[1][(i, z)] = rhos1
        alpha_bin[0][(i, z)] = a0
        alpha_bin[1][(i, z)] = a1
        dropped_sims_log[0][(i, z)] = drop0
        dropped_sims_log[1][(i, z)] = drop1

        if beta0 is None or beta1 is None:
            emulator_avg[(i, z)] = (None, None)
            continue
        emulator_avg[(i, z)] = (list(range(n_lcdm)), (beta0 + beta1) / 2.0)

n_fit = sum(1 for i in range(n_labels) for z in CANDIDATE_REDSHIFTS if emulator_avg[(i, z)][1] is not None)
print(f'ridge-fit LambdaCDM emulators (outlier-filtered) for {n_fit} of '
      f'{n_labels * len(CANDIDATE_REDSHIFTS)} (parameter, redshift) combinations')

for k in (0, 1):
    drop_counts = Counter(n for sims in dropped_sims_log[k].values() for n in sims)
    print(f"\nbin {k}: sims most frequently excluded as outliers (across all {n_labels} params x "
          f"{len(CANDIDATE_REDSHIFTS)} redshifts = up to {n_labels * len(CANDIDATE_REDSHIFTS)} fits):")
    for n, c in drop_counts.most_common(15):
        print(f'  sim {n}: excluded in {c} fits')

#%% ridge covariance at the already-selected alpha -- now computed on the
# SAME outlier-filtered subset used by fit_one_bin_ridge, so the error
# bars stay consistent with what was actually fit
for i in range(n_labels):
    for z in CANDIDATE_REDSHIFTS:
        sims_z = common_sims_by_z[z]
        if len(sims_z) < MIN_SIMS_FOR_FIT:
            continue

        se_by_bin = {}
        for k in (0, 1):
            _, beta_k = emulator_bin[k][(i, z)]
            if beta_k is None:
                se_by_bin[k] = None
                continue
            _, theta, y, _ = filtered_sims_theta_y(sims_z, k, z, i)
            delta = FIDUCIAL_LCDM[None, :] - theta
            _, cov = ridge_beta_cov(delta, y, alpha_bin[k][(i, z)])
            se = np.sqrt(np.diag(cov))
            se_by_bin[k] = se
            INTERCEPT_ERR[k][(i, z)] = se[0]
            SLOPE_ERR[k][(i, z)] = se[1:]

        if se_by_bin.get(0) is not None and se_by_bin.get(1) is not None:
            se_avg = np.sqrt(se_by_bin[0] ** 2 + se_by_bin[1] ** 2) / 2.0
            INTERCEPT_ERR['avg'][(i, z)] = se_avg[0]
            SLOPE_ERR['avg'][(i, z)] = se_avg[1:]



#%% ============================== PLOT SET 1 ==============================
# cosmology-dependence plot: per-bin fits (dashed) + average (solid), one
# figure per redshift, with rho annotation + green shading.
ALPHA_MAX = 0.45

for z in CANDIDATE_REDSHIFTS:
    sims_z = common_sims_by_z[z]
    if len(sims_z) < MIN_SIMS_FOR_FIT:
        continue
    theta_z = np.array([lcdm_pars[n] for n in sims_z])

    fig, ax = plt.subplots(n_labels, n_lcdm, figsize=(3.2 * n_lcdm, 2.6 * n_labels),
                           sharex='col', layout='constrained')
    for i in range(n_labels):
        cols0, beta0 = emulator_bin[0][(i, z)]
        cols1, beta1 = emulator_bin[1][(i, z)]
        cols_avg, beta_avg = emulator_avg[(i, z)]
        rhos0 = rhos_bin[0].get((i, z), {})
        rhos1 = rhos_bin[1].get((i, z), {})

        for j in range(n_lcdm):
            y0 = np.array([params[0][z][n][i] for n in sims_z])
            y1 = np.array([params[1][z][n][i] for n in sims_z])

            dependent = bool((cols0 and j in cols0) or (cols1 and j in cols1))
            rho_show = max(abs(rhos0.get(j, 0.0)), abs(rhos1.get(j, 0.0)))
            ax[i, j].set_facecolor(to_rgba('tab:green', ALPHA_MAX * min(rho_show, 1.0))
                                   if dependent else to_rgba('white', 0.0))
            for k, rhos_k in ((0, rhos0), (1, rhos1)):
                rho_j = rhos_k.get(j)
                if rho_j is not None:
                    ax[i, j].annotate(rf'$\rho={rho_j:.2f}$', xy=(0.05, 0.85 - 0.15 * k),
                                      xycoords='axes fraction', fontsize=7, color=f'C{k}')

            ax[i, j].plot(theta_z[:, j], y0, 'o', ms=4, alpha=0.6, c='C0')
            ax[i, j].plot(theta_z[:, j], y1, 'o', ms=4, alpha=0.6, c='C1')

            for k, drop_list in ((0, dropped_sims_log[0].get((i, z), [])), (1, dropped_sims_log[1].get((i, z), []))):
                if not drop_list:
                    continue
                drop_idx = [sims_z.index(n) for n in drop_list if n in sims_z]
                if drop_idx:
                    y_arr = y0 if k == 0 else y1
                    ax[i, j].plot(theta_z[drop_idx, j], y_arr[drop_idx], 'x', ms=8, c=f'C{k}', mew=2, zorder=5)

            if dependent and beta_avg is not None:
                x_smooth = np.linspace(theta_z[:, j].min(), theta_z[:, j].max(), 50)
                q = np.tile(np.nanmean(theta_z, axis=0), (50, 1))
                q[:, j] = x_smooth
                y_avg_line = [eval_emulator(cols_avg, beta_avg, q[m]) for m in range(50)]
                y_b0_line = [eval_emulator(cols0, beta0, q[m]) for m in range(50)]
                y_b1_line = [eval_emulator(cols1, beta1, q[m]) for m in range(50)]
                ax[i, j].plot(x_smooth, y_b0_line, '--', c='C0', lw=1)
                ax[i, j].plot(x_smooth, y_b1_line, '--', c='C1', lw=1)
                ax[i, j].plot(x_smooth, y_avg_line, '-', c='purple', lw=2)

            if i == n_labels - 1:
                ax[i, j].set_xlabel(LCDM_LABELS[j])
        ax[i, 0].set_ylabel(labels[i])

    fig.suptitle(rf'cosmology dependence at $z={z:g}$ ($\Lambda$CDM only) -- '
                rf'dashed: per-bin fits, solid: average')
    fname = os.path.join(FIGURES_DIR, f'cosmo_dep_emulators_z{z:.3f}'.replace('.', 'p') + '.png')
    fig.savefig(fname, dpi=150)
    print(f'saved {fname}')

#%% ============================== PLOT SET 1 (robust y-limits) ==============
# Same cosmology-dependence plot as before, but with per-row y-limits
# computed from the actual data points (median + scaled MAD, same
# approach as Plot Set 3's robust_ylim), so a handful of extreme sims --
# e.g. the sigma0~7500, A_rho_p~3e5 outliers visible at z=1.7 -- can't set
# the scale for the whole row. Fit lines are still drawn at full extent
# and simply clipped by the axis; that clipping is the point.
def robust_ylim_rowdata(vals_list, pad_frac=0.15, mad_k=5.0):
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
    sims_z = common_sims_by_z[z]
    if len(sims_z) < MIN_SIMS_FOR_FIT:
        continue
    theta_z = np.array([lcdm_pars[n] for n in sims_z])

    fig, ax = plt.subplots(n_labels, n_lcdm, figsize=(3.2 * n_lcdm, 2.6 * n_labels),
                           sharex='col', layout='constrained')
    for i in range(n_labels):
        cols0, beta0 = emulator_bin[0][(i, z)]
        cols1, beta1 = emulator_bin[1][(i, z)]
        cols_avg, beta_avg = emulator_avg[(i, z)]
        rhos0 = rhos_bin[0].get((i, z), {})
        rhos1 = rhos_bin[1].get((i, z), {})

        y0 = np.array([params[0][z][n][i] for n in sims_z])
        y1 = np.array([params[1][z][n][i] for n in sims_z])
        # one shared y-range for the whole row (all n_lcdm columns plot
        # the same y0/y1 against different x's), computed from the DATA
        # points only -- fit-line values are never used to set the range
        row_ylim = robust_ylim_rowdata([y0, y1])

        for j in range(n_lcdm):
            dependent = bool((cols0 and j in cols0) or (cols1 and j in cols1))
            rho_show = max(abs(rhos0.get(j, 0.0)), abs(rhos1.get(j, 0.0)))
            ax[i, j].set_facecolor(to_rgba('tab:green', ALPHA_MAX * min(rho_show, 1.0))
                                   if dependent else to_rgba('white', 0.0))
            for k, rhos_k in ((0, rhos0), (1, rhos1)):
                rho_j = rhos_k.get(j)
                if rho_j is not None:
                    ax[i, j].annotate(rf'$\rho={rho_j:.2f}$', xy=(0.05, 0.85 - 0.15 * k),
                                      xycoords='axes fraction', fontsize=7, color=f'C{k}')

            ax[i, j].plot(theta_z[:, j], y0, 'o', ms=4, alpha=0.6, c='C0')
            ax[i, j].plot(theta_z[:, j], y1, 'o', ms=4, alpha=0.6, c='C1')

            if dependent and beta_avg is not None:
                x_smooth = np.linspace(theta_z[:, j].min(), theta_z[:, j].max(), 50)
                q = np.tile(np.nanmean(theta_z, axis=0), (50, 1))
                q[:, j] = x_smooth
                y_avg_line = [eval_emulator(cols_avg, beta_avg, q[m]) for m in range(50)]
                y_b0_line = [eval_emulator(cols0, beta0, q[m]) for m in range(50)]
                y_b1_line = [eval_emulator(cols1, beta1, q[m]) for m in range(50)]
                ax[i, j].plot(x_smooth, y_b0_line, '--', c='C0', lw=1)
                ax[i, j].plot(x_smooth, y_b1_line, '--', c='C1', lw=1)
                ax[i, j].plot(x_smooth, y_avg_line, '-', c='purple', lw=2)

            if row_ylim is not None:
                ax[i, j].set_ylim(*row_ylim)
            if i == n_labels - 1:
                ax[i, j].set_xlabel(LCDM_LABELS[j])
        ax[i, 0].set_ylabel(labels[i])

    fig.suptitle(rf'cosmology dependence at $z={z:g}$ ($\Lambda$CDM only) -- '
                rf'dashed: per-bin fits, solid: average (y-limited)')
    fname = os.path.join(FIGURES_DIR, f'cosmo_dep_emulators_z{z:.3f}'.replace('.', 'p') + '_ylim.png')
    fig.savefig(fname, dpi=150)
    print(f'saved {fname}')

#%% ============================== PLOT SET 2 ==============================
# residuals with respect to the AVERAGE emulator, one figure per redshift,
# one combined stat + one combined sys percentage per panel, y-limits
# per-row, percentile-based, centered on 1.00, clamped.
Y_HALF_RANGE_MIN = 0.05
Y_HALF_RANGE_MAX = 0.5
Y_PERCENTILE = 1

for z in CANDIDATE_REDSHIFTS:
    sims_z = common_sims_by_z[z]
    if len(sims_z) < MIN_SIMS_FOR_FIT:
        continue
    theta_z = np.array([lcdm_pars[n] for n in sims_z])
    n_sim = len(sims_z)

    panel_ratio, panel_sys = {}, {}
    cols_shown_by_i = {i: set() for i in range(n_labels)}

    for i in range(n_labels):
        cols_avg, beta_avg = emulator_avg[(i, z)]
        if beta_avg is None:
            continue
        cols0, beta0 = emulator_bin[0][(i, z)]
        cols1, beta1 = emulator_bin[1][(i, z)]
        pred_avg = np.array([eval_emulator(cols_avg, beta_avg, theta_z[m]) for m in range(n_sim)])

        ratio_both, sys_both = [], []
        for cols_k, beta_k, kbin in ((cols0, beta0, 0), (cols1, beta1, 1)):
            actual_k = np.array([params[kbin][z][n][i] for n in sims_z])
            pred_k = np.array([eval_emulator(cols_k, beta_k, theta_z[m]) for m in range(n_sim)])
            ratio_both.append(pred_avg / actual_k)
            sys_both.append((pred_k - pred_avg) / actual_k)

        for j in cols_avg:
            cols_shown_by_i[i].add(j)
            panel_ratio[(i, j)] = np.concatenate(ratio_both)
            panel_sys[(i, j)] = np.concatenate(sys_both)

    if not any(cols_shown_by_i.values()):
        continue

    row_ylim = {}
    for i in range(n_labels):
        if not cols_shown_by_i[i]:
            continue
        vals = np.concatenate([panel_ratio[(i, j)] for j in cols_shown_by_i[i]])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        lo, hi = np.nanpercentile(vals, [Y_PERCENTILE, 100 - Y_PERCENTILE])
        half_range = max(abs(1.0 - lo), abs(hi - 1.0))
        half_range = float(np.clip(half_range * 1.15, Y_HALF_RANGE_MIN, Y_HALF_RANGE_MAX))
        row_ylim[i] = (1.0 - half_range, 1.0 + half_range)

    fig2, ax2 = plt.subplots(n_labels, n_lcdm, figsize=(3.2 * n_lcdm, 2.6 * n_labels),
                             sharex='col', layout='constrained')
    for i in range(n_labels):
        for j in range(n_lcdm):
            ax2[i, j].axis('off')

    for i in range(n_labels):
        for j in cols_shown_by_i[i]:
            ratio, sys_frac = panel_ratio[(i, j)], panel_sys[(i, j)]
            ratio0, ratio1 = ratio[:n_sim], ratio[n_sim:]
            sys0, sys1 = sys_frac[:n_sim], sys_frac[n_sim:]

            ax2[i, j].axis('on')
            for k, (r_k, s_k) in enumerate(((ratio0, sys0), (ratio1, sys1))):
                ax2[i, j].errorbar(theta_z[:, j], r_k, yerr=np.abs(s_k),
                                   fmt='o', c=f'C{k}', ms=4, capsize=2, elinewidth=1, alpha=0.6)

            stat_rms = np.sqrt(np.nanmean((ratio - 1) ** 2))
            sys_rms = np.sqrt(np.nanmean(sys_frac ** 2))
            stat_str = rf'stat$={stat_rms:.1%}$'.replace('%', r'\%')
            sys_str = rf'sys$={sys_rms:.1%}$'.replace('%', r'\%')
            ax2[i, j].text(0.95, 0.95, stat_str + ', ' + sys_str,
                          transform=ax2[i, j].transAxes, va='top', ha='right', fontsize=6)

            ax2[i, j].axhline(1, c='k', ls='--', lw=1)
            for band, ls in [(0.05, ':'), (0.10, '-.')]:
                ax2[i, j].axhline(1 + band, c='gray', ls=ls, lw=1)
                ax2[i, j].axhline(1 - band, c='gray', ls=ls, lw=1)
            if i in row_ylim:
                ax2[i, j].set_ylim(*row_ylim[i])

        if cols_shown_by_i[i]:
            ax2[i, min(cols_shown_by_i[i])].set_ylabel(labels[i])

    for j in range(n_lcdm):
        ax2[-1, j].set_xlabel(LCDM_LABELS[j])

    fig2.suptitle(rf'residuals w.r.t. the AVERAGE $\Lambda$CDM emulator, $z={z:g}$ (C0: bin 0, C1: bin 1)')
    fname = os.path.join(FIGURES_DIR, f'cosmo_dep_residuals_vs_average_z{z:.3f}'.replace('.', 'p') + '.png')
    fig2.savefig(fname, dpi=150)
    print(f'saved {fname}')


#%% ============================== PLOT SET 3 ==============================
# intercept and slope vs redshift, with error bars and robust (MAD-based)
# y-limits.
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


fig_int, ax_int = plt.subplots(n_labels, 1, figsize=(5, 2.2 * n_labels), sharex=True, layout='constrained')

for i in range(n_labels):
    row_vals = []
    for key in (0, 1, 'avg'):
        zs, vals, errs = [], [], []
        for z in CANDIDATE_REDSHIFTS:
            beta = (emulator_avg[(i, z)][1] if key == 'avg' else emulator_bin[key][(i, z)][1])
            if beta is None:
                continue
            zs.append(z)
            vals.append(beta[0])
            errs.append(INTERCEPT_ERR[key].get((i, z), np.nan))
        if not zs:
            continue
        order = np.argsort(zs)
        zs_s, vals_s, errs_s = np.array(zs)[order], np.array(vals)[order], np.array(errs)[order]
        row_vals.append(vals_s)
        style = BIN_STYLE[key]
        ax_int[i].errorbar(zs_s, vals_s, yerr=errs_s, marker='o', ms=4, capsize=2,
                           c=style['c'], ls=style['ls'], label=style['label'] if i == 0 else None)
    ylim = robust_ylim(row_vals, 0.25)
    if ylim is not None:
        ax_int[i].set_ylim(*ylim)
    ax_int[i].set_ylabel(labels[i])

ax_int[0].legend(fontsize=8)
ax_int[-1].set_xlabel(r'$z$')
fig_int.suptitle('Compact-model parameter value at fiducial cosmology vs redshift')
fname = os.path.join(FIGURES_DIR, 'emulator_intercept_vs_redshift.png')
fig_int.savefig(fname, dpi=150)
print(f'saved {fname}')


fig_slope, ax_slope = plt.subplots(n_labels, n_lcdm, figsize=(3.2 * n_lcdm, 2.2 * n_labels),
                                   sharex='col', layout='constrained')

for i in range(n_labels):
    for j in range(n_lcdm):
        panel_vals = []
        for key in (0, 1, 'avg'):
            zs, vals, errs = [], [], []
            for z in CANDIDATE_REDSHIFTS:
                beta = (emulator_avg[(i, z)][1] if key == 'avg' else emulator_bin[key][(i, z)][1])
                if beta is None:
                    continue
                err_arr = SLOPE_ERR[key].get((i, z))
                zs.append(z)
                vals.append(beta[1 + j])
                errs.append(err_arr[j] if err_arr is not None else np.nan)
            if not zs:
                continue
            order = np.argsort(zs)
            zs_s, vals_s, errs_s = np.array(zs)[order], np.array(vals)[order], np.array(errs)[order]
            panel_vals.append(vals_s)
            style = BIN_STYLE[key]
            ax_slope[i, j].errorbar(zs_s, vals_s, yerr=errs_s, marker='o', ms=4, capsize=2,
                                    c=style['c'], ls=style['ls'])
        ax_slope[i, j].axhline(0, c='gray', ls='--', lw=1)
        ylim = robust_ylim(panel_vals + [np.array([0.0])], 0.25)
        if ylim is not None:
            ax_slope[i, j].set_ylim(*ylim)
        if i == n_labels - 1:
            ax_slope[i, j].set_xlabel(r'$z$')
    ax_slope[i, 0].set_ylabel(labels[i])

for j in range(n_lcdm):
    ax_slope[0, j].set_title(LCDM_LABELS[j])

fig_slope.suptitle(r'Cosmology-dependence slope ($\partial\,\mathrm{param}/\partial\Delta\theta$) vs redshift')
fname = os.path.join(FIGURES_DIR, 'emulator_slope_vs_redshift.png')
fig_slope.savefig(fname, dpi=150)
print(f'saved {fname}')

#%%
# testing intercept plot axes changes 


fig_int, ax_int = plt.subplots(n_labels, 1, figsize=(5, 2.2 * n_labels), sharex=True, layout='constrained')

for i in range(n_labels):
    row_vals = []
    for key in (0, 1, 'avg'):
        zs, vals, errs = [], [], []
        for z in CANDIDATE_REDSHIFTS:
            beta = (emulator_avg[(i, z)][1] if key == 'avg' else emulator_bin[key][(i, z)][1])
            if beta is None:
                continue
            zs.append(z)
            vals.append(beta[0])
            errs.append(INTERCEPT_ERR[key].get((i, z), np.nan))
        if not zs:
            continue
        order = np.argsort(zs)
        zs_s, vals_s, errs_s = np.array(zs)[order], np.array(vals)[order], np.array(errs)[order]
        row_vals.append(vals_s)
        style = BIN_STYLE[key]
        ax_int[i].errorbar(zs_s, vals_s, yerr=errs_s, marker='o', ms=4, capsize=2,
                           c=style['c'], ls=style['ls'], label=style['label'] if i == 0 else None)
    ylim = robust_ylim(row_vals, 0.25)
    # if ylim is not None:
    #     ax_int[i].set_ylim(*ylim)
    ax_int[i].set_ylabel(labels[i])

ax_int[0].legend(fontsize=8)
ax_int[-1].set_xlabel(r'$z$')
fig_int.suptitle('Compact-model parameter value at fiducial cosmology vs redshift')
# fname = os.path.join(FIGURES_DIR, 'emulator_intercept_vs_redshift.png')
# fig_int.savefig(fname, dpi=150)
# print(f'saved {fname}')

#%% formal check: does each row's z<=1.1 trend predict its z=1.4/1.7
# intercept, or do those points represent a genuine break from it?
# Reuses fit_weighted_shift_trend's machinery but applied directly to
# bin 1's OWN intercept vs z (not a bin0-bin1 difference), since the
# question here is about bin 1's internal consistency across z, not
# about tracking bin 0.
def fit_weighted_value_trend(get_val, get_err, z_max=1.1):
    zs, vals, errs = [], [], []
    for z in CANDIDATE_REDSHIFTS:
        if z > z_max:
            continue
        v = get_val(z)
        if v is None:
            continue
        zs.append(z)
        vals.append(v)
        errs.append(get_err(z) or 0.0)
    if len(zs) < 3:
        return None
    zs, vals, errs = np.array(zs), np.array(vals), np.array(errs)
    errs = np.where(errs > 0, errs, np.median(errs[errs > 0]) if np.any(errs > 0) else 1.0)
    z0 = zs.mean()
    X = np.column_stack([np.ones_like(zs), zs - z0])
    W = np.diag(1.0 / errs ** 2)
    cov = np.linalg.inv(X.T @ W @ X)
    coeffs = cov @ (X.T @ W @ vals)
    return dict(coeffs=coeffs, cov=cov, z0=z0)


def eval_value_trend(trend, z_query):
    if trend is None:
        return None, None
    x = np.array([1.0, z_query - trend['z0']])
    return x @ trend['coeffs'], np.sqrt(max(x @ trend['cov'] @ x, 0.0))


print(f"{'param':>10s}  {'z':>4s}  {'actual':>10s}  {'trend_pred':>10s}  {'z-score':>8s}  verdict")
for i in range(n_labels):
    def get_val(z, k=1, i=i):
        beta = emulator_bin[k][(i, z)][1]
        return beta[0] if beta is not None else None
    def get_err(z, k=1, i=i):
        return INTERCEPT_ERR[k].get((i, z))

    trend = fit_weighted_value_trend(get_val, get_err, z_max=1.1)
    if trend is None:
        continue
    for z in (1.4, 1.7):
        actual = get_val(z)
        actual_err = get_err(z)
        if actual is None:
            continue
        pred, pred_err = eval_value_trend(trend, z)
        zscore = (actual - pred) / np.sqrt((actual_err or 0.0) ** 2 + (pred_err or 0.0) ** 2)
        verdict = 'consistent with z<=1.1 trend' if abs(zscore) < 2 else 'BREAKS from z<=1.1 trend'
        print(f"{PARAM_NAMES[i]:>10s}  {z:>4}  {actual:>10.3g}  {pred:>10.3g}  {zscore:>8.2f}  {verdict}")

#%% quadratic-in-z trend, same weighted-least-squares structure as
# fit_weighted_value_trend, just with an extra (z-z0)^2 term. A genuinely
# smooth, accelerating physical trend should now extrapolate consistently;
# a genuine data-quality discontinuity should still fail even with the
# extra degree of freedom.
def fit_weighted_value_trend_quad(get_val, get_err, z_max=1.1):
    zs, vals, errs = [], [], []
    for z in CANDIDATE_REDSHIFTS:
        if z > z_max:
            continue
        v = get_val(z)
        if v is None:
            continue
        zs.append(z); vals.append(v); errs.append(get_err(z) or 0.0)
    if len(zs) < 4:   # need more points than parameters (3) to fit meaningfully
        return None
    zs, vals, errs = np.array(zs), np.array(vals), np.array(errs)
    errs = np.where(errs > 0, errs, np.median(errs[errs > 0]) if np.any(errs > 0) else 1.0)
    z0 = zs.mean()
    dz = zs - z0
    X = np.column_stack([np.ones_like(zs), dz, dz ** 2])
    W = np.diag(1.0 / errs ** 2)
    cov = np.linalg.inv(X.T @ W @ X)
    coeffs = cov @ (X.T @ W @ vals)
    return dict(coeffs=coeffs, cov=cov, z0=z0)


def eval_value_trend_quad(trend, z_query):
    if trend is None:
        return None, None
    dz = z_query - trend['z0']
    x = np.array([1.0, dz, dz ** 2])
    return x @ trend['coeffs'], np.sqrt(max(x @ trend['cov'] @ x, 0.0))


print(f"{'param':>10s}  {'z':>4s}  {'actual':>10s}  {'lin_pred':>9s}  {'lin_z':>6s}  "
      f"{'quad_pred':>9s}  {'quad_z':>7s}  verdict")
for i in range(n_labels):
    def get_val(z, k=1, i=i):
        beta = emulator_bin[k][(i, z)][1]
        return beta[0] if beta is not None else None
    def get_err(z, k=1, i=i):
        return INTERCEPT_ERR[k].get((i, z))

    trend_lin = fit_weighted_value_trend(get_val, get_err, z_max=1.1)
    trend_quad = fit_weighted_value_trend_quad(get_val, get_err, z_max=1.1)
    for z in (1.4, 1.7):
        actual = get_val(z)
        actual_err = get_err(z)
        if actual is None:
            continue
        pred_lin, err_lin = eval_value_trend(trend_lin, z)
        pred_quad, err_quad = eval_value_trend_quad(trend_quad, z)
        zscore_lin = (actual - pred_lin) / np.sqrt((actual_err or 0.0) ** 2 + (err_lin or 0.0) ** 2)
        zscore_quad = ((actual - pred_quad) / np.sqrt((actual_err or 0.0) ** 2 + (err_quad or 0.0) ** 2)
                      if pred_quad is not None else np.nan)
        verdict = ('quadratic resolves it' if abs(zscore_lin) >= 2 and abs(zscore_quad) < 2
                  else 'still breaks (quad too)' if abs(zscore_quad) >= 2 else 'fine either way')
        print(f"{PARAM_NAMES[i]:>10s}  {z:>4}  {actual:>10.3g}  {pred_lin:>9.3g}  {zscore_lin:>6.2f}  "
              f"{pred_quad if pred_quad is not None else float('nan'):>9.3g}  {zscore_quad:>7.2f}  {verdict}")

#%% direct check: did the outlier filter actually fire for the rows/z's
# in the z-score table? If dropped_sims_log is empty everywhere below,
# the exclusion fitting loop either wasn't re-run, or genuinely found
# nothing to drop at mad_k=6.0 for these specific (row, z) combos --
# this tells you which, rather than inferring from downstream numbers.
CHECK_ROWS = ['sigma0', 'A_rho0', 'A_rho_p', 'sigma_delta', 'beta_delta', 'alpha']
for pname in CHECK_ROWS:
    i = PARAM_NAMES.index(pname)
    for z in (1.4, 1.7):
        dropped = dropped_sims_log[1].get((i, z), 'NEVER POPULATED -- fitting loop not re-run with exclusion')
        print(f'{pname:>12s}  z={z}:  dropped = {dropped}')

#%% z-score check, annotated with n_sim used per fit -- lets you read
# "breaks from trend" alongside "how much data was that break based on"
for i in range(n_labels):
    def get_val(z, k=1, i=i):
        beta = emulator_bin[k][(i, z)][1]
        return beta[0] if beta is not None else None
    def get_err(z, k=1, i=i):
        return INTERCEPT_ERR[k].get((i, z))

    trend = fit_weighted_value_trend(get_val, get_err, z_max=1.1)
    if trend is None:
        continue
    for z in (1.4, 1.7):
        actual = get_val(z)
        if actual is None:
            continue
        n_used = len(common_sims_by_z.get(z, [])) - len(dropped_sims_log[1].get((i, z), []))
        pred, pred_err = eval_value_trend(trend, z)
        zscore = (actual - pred) / np.sqrt((get_err(z) or 0.0) ** 2 + (pred_err or 0.0) ** 2)
        print(f'{PARAM_NAMES[i]:>12s}  z={z}  n_sim_used={n_used:>3d}  '
              f'actual={actual:>10.3g}  pred={pred:>10.3g}  z={zscore:>7.2f}')
        
#%% quantify, per (bin, z), how "small-r_perp deprived" the data actually
# is -- does the fraction of pairs/cells at LOW r_perp collapse for bin 1
# at high z, the way this plot suggests? Reuses the raw pair-count
# histograms already on disk; no re-fitting needed.
R_PERP_LOW_CUT = 20.0   # h^-1 Mpc -- matches where the plot shows bin 4 running out of data

for z in CANDIDATE_REDSHIFTS:
    for k in (0, 1):
        sims_here = [n for n in core.SIM_NUMS if os.path.exists(out_path(n, z, k))][:8]
        if not sims_here:
            continue
        n_stack = np.stack([PairHist.load(out_path(n, z, k)).n[0] for n in sims_here])
        rp_c = 0.5 * (PairHist.load(out_path(sims_here[0], z, k)).R_edges[1:]
                     + PairHist.load(out_path(sims_here[0], z, k)).R_edges[:-1])
        low_mask = rp_c < R_PERP_LOW_CUT
        frac_pairs_low = n_stack[:, low_mask, :].sum() / n_stack.sum()
        print(f'z={z}, bin={k}: {frac_pairs_low:.1%} of all pairs sit at r_perp < {R_PERP_LOW_CUT}')


#%% same check, but normalized per unit r_perp-annulus AREA, not raw pair
# fraction -- controls for the fact that small-r_perp annuli have
# inherently less "room" for pairs regardless of halo abundance
for z in CANDIDATE_REDSHIFTS:
    for k in (0, 1):
        sims_here = [n for n in core.SIM_NUMS if os.path.exists(out_path(n, z, k))][:8]
        if not sims_here:
            continue
        h0 = PairHist.load(out_path(sims_here[0], z, k))
        rp_edges = h0.R_edges
        rp_c = 0.5 * (rp_edges[1:] + rp_edges[:-1])
        annulus_area = np.pi * (rp_edges[1:] ** 2 - rp_edges[:-1] ** 2)

        n_stack = np.stack([PairHist.load(out_path(n, z, k)).n[0] for n in sims_here])
        pairs_per_rp = n_stack.sum(axis=(0, 2))          # total pairs per r_perp bin, summed over sims+r_los
        density_per_rp = pairs_per_rp / annulus_area       # pairs per unit annulus area -- the density check

        low_mask = rp_c < R_PERP_LOW_CUT
        # ratio of typical low-r_perp DENSITY to typical high-r_perp DENSITY,
        # rather than a raw pair-count fraction
        density_ratio = np.median(density_per_rp[low_mask]) / np.median(density_per_rp[~low_mask])
        print(f'z={z}, bin={k}: low/high r_perp PAIR DENSITY ratio = {density_ratio:.3f}')

#%% reproduce the attached plot's style directly from THIS script's own
# bin=1 pairhist data, to check by eye whether the same r_perp desert is
# present here -- rather than infer it from a density ratio or from
# possibly-mismatched mass-bin labels
import matplotlib.cm as cm

def plot_sigma_delta_scatter(n, z, k, ax_top, ax_bot, title):
    h = PairHist.load(out_path(n, z, k))
    sig = meas_sigv_grid(h, 0)
    ST_pars = fit_skewt_grid(h, 0)
    r_vr, vr_mean, npairs_vr = vr_profile(h, 0, VR_R_EDGES)
    good_vr = np.isfinite(vr_mean) & (npairs_vr > 100)
    vr_interp = InterpolatedUnivariateSpline(r_vr[good_vr], vr_mean[good_vr], k=3, ext=3)
    Delta = ST_pars[:, :, 0] - vr_interp(r3_2d) * r_l_2d / r3_2d

    finite = np.isfinite(sig) & np.isfinite(Delta)
    x_flat, rp_flat = x_grid[finite], r_p_2d[finite]
    sig2_flat, del_flat = (sig[finite]) ** 2, Delta[finite]

    sc = ax_top.scatter(x_flat, sig2_flat, c=rp_flat, cmap='viridis', s=8)
    ax_top.set_title(title)
    ax_top.set_ylabel(r'$\sigma_v^2(x)$ [km/s]$^2$')
    ax_bot.scatter(x_flat, del_flat, c=rp_flat, cmap='viridis', s=8)
    ax_bot.axhline(0, c='k', lw=1)
    ax_bot.set_ylabel(r'$\Delta(x)$ [km/s]')
    ax_bot.set_xlabel(r'$x = r-r_\perp$ [$h^{-1}$ Mpc]')
    return sc


for z_check in (0.2, 1.4, 1.7):
    sims_here = [n for n in core.SIM_NUMS if os.path.exists(out_path(n, z_check, 1))]
    if not sims_here:
        continue
    n_check = sims_here[0]   # c000-equivalent if present, else whatever's first
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    plot_sigma_delta_scatter(n_check, z_check, 0, ax[0, 0], ax[1, 0], f'bin 0, z={z_check}, sim {n_check}')
    sc = plot_sigma_delta_scatter(n_check, z_check, 1, ax[0, 1], ax[1, 1], f'bin 1, z={z_check}, sim {n_check}')
    fig.colorbar(sc, ax=ax, label=r'$r_\perp$ [$h^{-1}$ Mpc]')
    # fname = os.path.join(FIGURES_DIR, f'sigma_delta_scatter_check_z{z_check:.3f}'.replace('.', 'p') + '.png')
    # fig.savefig(fname, dpi=150)
    # print(f'saved {fname}')

#%% for the two suspicious rows, plot the actual bin-1 alpha/beta_delta
# distribution at z=1.4 with dropped sims marked, to see whether the
# 14/10 exclusions are trimming a genuine long tail (fine) or splitting
# what's actually a bimodal/two-cluster distribution (a sign MAD-around-
# one-median is the wrong tool here)
for pname in ('alpha', 'beta_delta'):
    i = PARAM_NAMES.index(pname)
    sims_z = common_sims_by_z[1.4]
    y = np.array([params[1][1.4][n][i] for n in sims_z])
    dropped = set(dropped_sims_log[1].get((i, 1.4), []))
    is_dropped = np.array([n in dropped for n in sims_z])

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(y[~is_dropped], bins=15, alpha=0.7, label='kept', color='C1')
    ax.hist(y[is_dropped], bins=15, alpha=0.7, label='dropped', color='gray')
    ax.axvline(np.median(y[~is_dropped]), c='k', ls='--', lw=1)
    ax.set_title(f'{pname} @ (bin 1, z=1.4): {is_dropped.sum()}/{len(y)} dropped')
    ax.legend()
    # fig.savefig(os.path.join(FIGURES_DIR, f'{pname}_z1p4_bin1_distribution.png'), dpi=150)

#%% do the secondary-cluster sims in alpha/beta_delta's "kept" distribution
# overlap with the sims already flagged as problematic elsewhere (the
# original r_delta/beta_delta/alpha outlier union from the pre-rescue
# check, and each other)?
def secondary_cluster_sims(pname, z, k=1, low_cut=None, high_cut=None):
    i = PARAM_NAMES.index(pname)
    sims_z = common_sims_by_z[z]
    dropped = set(dropped_sims_log[k].get((i, z), []))
    kept_sims = [n for n in sims_z if n not in dropped]
    y = np.array([params[k][z][n][i] for n in kept_sims])
    mask = (y >= low_cut) & (y <= high_cut) if low_cut is not None else np.ones_like(y, dtype=bool)
    return [n for n, m in zip(kept_sims, mask) if m]

alpha_secondary = secondary_cluster_sims('alpha', 1.4, low_cut=2.0, high_cut=8.0)
beta_secondary = secondary_cluster_sims('beta_delta', 1.4, low_cut=8.0, high_cut=14.0)
print('alpha secondary-cluster sims:', alpha_secondary)
print('beta_delta secondary-cluster sims:', beta_secondary)
print('overlap:', sorted(set(alpha_secondary) & set(beta_secondary)))

#%% consolidated prerequisites for the scheme-comparison plots -- gathers
# everything introduced piecemeal across earlier turns: bin 0's own
# z=1.7 fit (dropped by common_sims_by_z's both-bins requirement),
# the get_val_*/get_err_* accessors, the anchor/trend-fitting helpers,
# the family-driver definitions, and the fixed real-data cutoffs.

PARAM_NAMES = ['sigma0', 'A_rho0', 'A_rho_p', 'sigma_rho',
              'A_delta0', 'r_delta', 'sigma_delta', 'beta_delta', 'alpha']
FIT_RESULTS_PATH = os.path.join(FIGURES_DIR, 'cosmo_grid_fit_results.h5')

#--- bin 0's own z=1.7 fit (bin 1 has none there, so common_sims_by_z
# drops z=1.7 entirely; this recovers bin 0's side of it specifically) ---
EXTRA_Z_BIN0 = [z for z in CANDIDATE_REDSHIFTS
               if len(common_sims_by_z.get(z, [])) < MIN_SIMS_FOR_FIT
               and len(params[0].get(z, {})) >= MIN_SIMS_FOR_FIT]

bin0_extra_beta, bin0_extra_err_int, bin0_extra_err_slope = {}, {}, {}
for z in EXTRA_Z_BIN0:
    sims_z0 = sorted(params[0].get(z, {}))
    for i in range(n_labels):
        cols, beta, alpha, _, _ = fit_one_bin_ridge(sims_z0, 0, z, i)
        bin0_extra_beta[(i, z)] = beta
        if beta is not None:
            theta = np.array([lcdm_pars[n] for n in sims_z0])
            delta = FIDUCIAL_LCDM[None, :] - theta
            y = np.array([params[0][z][n][i] for n in sims_z0])
            _, cov = ridge_beta_cov(delta, y, alpha)
            se = np.sqrt(np.diag(cov))
            bin0_extra_err_int[(i, z)] = se[0]
            bin0_extra_err_slope[(i, z)] = se[1:]
print(f'recovered bin-0-only fits at z={EXTRA_Z_BIN0}')


#--- accessors: uniform (bin, z, row[, lcdm-column]) -> value/error lookup,
# transparently using bin0_extra_* for the z's only bin 0 covers ---
def get_val_int(k, z, i):
    beta = bin0_extra_beta.get((i, z)) if (k == 0 and z in EXTRA_Z_BIN0) else emulator_bin[k][(i, z)][1]
    return beta[0] if beta is not None else None

def get_err_int(k, z, i):
    if k == 0 and z in EXTRA_Z_BIN0:
        return bin0_extra_err_int.get((i, z))
    return INTERCEPT_ERR[k].get((i, z))

def get_val_slope(k, z, i, j):
    beta = bin0_extra_beta.get((i, z)) if (k == 0 and z in EXTRA_Z_BIN0) else emulator_bin[k][(i, z)][1]
    return beta[1 + j] if beta is not None else None

def get_err_slope(k, z, i, j):
    if k == 0 and z in EXTRA_Z_BIN0:
        err_arr = bin0_extra_err_slope.get((i, z))
    else:
        err_arr = SLOPE_ERR[k].get((i, z))
    return err_arr[j] if err_arr is not None else None


#--- anchor (flat, z<=0.5) shift between bin 1 and bin 0 ---
Z_ANCHOR_MAX = 0.5

def anchor_shift_and_err(get_val, get_err):
    diffs, errs = [], []
    for z in CANDIDATE_REDSHIFTS:
        if z > Z_ANCHOR_MAX:
            continue
        v0, v1 = get_val(0, z), get_val(1, z)
        e0, e1 = get_err(0, z), get_err(1, z)
        if v0 is not None and v1 is not None:
            diffs.append(v1 - v0)
            errs.append(np.sqrt((e0 or 0.0) ** 2 + (e1 or 0.0) ** 2))
    if not diffs:
        return None, None
    diffs = np.array(diffs)
    shift = diffs.mean()
    shift_err = diffs.std(ddof=1) / np.sqrt(len(diffs)) if len(diffs) > 1 else np.mean(errs)
    return shift, shift_err


#--- weighted-least-squares trend fits, two flavors:
#    "shift" trend = bin1-bin0 DIFFERENCE vs z (used by the family scheme)
#    "value" trend = a single series' OWN value vs z (used by 'linear'/
#    'quadratic' in the scheme comparison, and the sigma0-instability check)
def fit_weighted_shift_trend(get_val, get_err, z_max=1.1):
    zs, diffs, errs = [], [], []
    for z in CANDIDATE_REDSHIFTS:
        if z > z_max:
            continue
        v0, v1 = get_val(0, z), get_val(1, z)
        e0, e1 = get_err(0, z), get_err(1, z)
        if v0 is None or v1 is None:
            continue
        zs.append(z)
        diffs.append(v1 - v0)
        errs.append(np.sqrt((e0 or 0.0) ** 2 + (e1 or 0.0) ** 2))
    if len(zs) < 2:
        return None
    zs, diffs, errs = np.array(zs), np.array(diffs), np.array(errs)
    errs = np.where(errs > 0, errs, np.median(errs[errs > 0]) if np.any(errs > 0) else 1.0)
    z0 = zs.mean()
    X = np.column_stack([np.ones_like(zs), zs - z0])
    W = np.diag(1.0 / errs ** 2)
    cov = np.linalg.inv(X.T @ W @ X)
    coeffs = cov @ (X.T @ W @ diffs)
    return dict(coeffs=coeffs, cov=cov, z0=z0)

def eval_shift_trend(trend, z_query):
    if trend is None:
        return None, None
    x = np.array([1.0, z_query - trend['z0']])
    val = x @ trend['coeffs']
    var = x @ trend['cov'] @ x
    return val, np.sqrt(max(var, 0.0))


def fit_weighted_value_trend(get_val, get_err, z_max=1.1):
    zs, vals, errs = [], [], []
    for z in CANDIDATE_REDSHIFTS:
        if z > z_max:
            continue
        v = get_val(z)
        if v is None:
            continue
        zs.append(z)
        vals.append(v)
        errs.append(get_err(z) or 0.0)
    if len(zs) < 3:
        return None
    zs, vals, errs = np.array(zs), np.array(vals), np.array(errs)
    errs = np.where(errs > 0, errs, np.median(errs[errs > 0]) if np.any(errs > 0) else 1.0)
    z0 = zs.mean()
    X = np.column_stack([np.ones_like(zs), zs - z0])
    W = np.diag(1.0 / errs ** 2)
    cov = np.linalg.inv(X.T @ W @ X)
    coeffs = cov @ (X.T @ W @ vals)
    return dict(coeffs=coeffs, cov=cov, z0=z0)

def eval_value_trend(trend, z_query):
    if trend is None:
        return None, None
    x = np.array([1.0, z_query - trend['z0']])
    return x @ trend['coeffs'], np.sqrt(max(x @ trend['cov'] @ x, 0.0))


#--- family definitions: sigma_v family driven by sigma0, Delta family
# driven by A_delta0 ---
_SIGMA_ROWS = ['sigma0', 'A_rho0', 'A_rho_p', 'sigma_rho']
_DELTA_ROWS = ['A_delta0', 'r_delta', 'sigma_delta', 'beta_delta', 'alpha']

FAMILY_DRIVER_ROW = {'sigma': PARAM_NAMES.index('sigma0'), 'delta': PARAM_NAMES.index('A_delta0')}
ROW_FAMILY = {}
for _name in _SIGMA_ROWS:
    ROW_FAMILY[PARAM_NAMES.index(_name)] = 'sigma'
for _name in _DELTA_ROWS:
    ROW_FAMILY[PARAM_NAMES.index(_name)] = 'delta'

MIN_ANCHOR_MAG = 1e-8


#--- fixed real-data cutoffs: driver rows trusted through z<=1.1,
# everything else only through z<=0.5 ---
DRIVER_REAL_CUTOFF = 1.1

def get_real_cutoff(row_idx):
    family = ROW_FAMILY.get(row_idx)
    is_driver = family is not None and row_idx == FAMILY_DRIVER_ROW[family]
    return DRIVER_REAL_CUTOFF if is_driver else Z_ANCHOR_MAX


#--- family-scaled reconstruction: driver rows use their own fitted
# trend; non-driver rows use their own z<=0.5 anchor shift, scaled by
# the family driver's fractional trend growth relative to the driver's
# own z<=0.5 anchor ---
def build_bin1_reconstruction_family2(get_val, get_err, get_val_driver, get_err_driver, row_idx):
    real_cutoff = get_real_cutoff(row_idx)
    is_driver = real_cutoff == DRIVER_REAL_CUTOFF

    recon_vals, recon_errs = {}, {}
    for z in CANDIDATE_REDSHIFTS:
        if z > real_cutoff:
            continue
        v1 = get_val(1, z)
        if v1 is not None:
            recon_vals[z] = v1
            recon_errs[z] = get_err(1, z) or 0.0

    future_zs = [z for z in CANDIDATE_REDSHIFTS if z > real_cutoff and get_val(0, z) is not None]
    if not future_zs:
        return recon_vals, recon_errs

    if is_driver:
        trend = fit_weighted_shift_trend(get_val, get_err, z_max=DRIVER_REAL_CUTOFF)
        for z in future_zs:
            v0 = get_val(0, z)
            shift, shift_err = eval_shift_trend(trend, z)
            if shift is None:
                continue
            recon_vals[z] = v0 + shift
            recon_errs[z] = np.sqrt((get_err(0, z) or 0.0) ** 2 + shift_err ** 2)
        return recon_vals, recon_errs

    anchor_shift, anchor_err = anchor_shift_and_err(get_val, get_err)
    family = ROW_FAMILY.get(row_idx)

    if family is None or anchor_shift is None:
        for z in future_zs:
            v0 = get_val(0, z)
            shift = anchor_shift if anchor_shift is not None else 0.0
            recon_vals[z] = v0 + shift
            recon_errs[z] = np.sqrt((get_err(0, z) or 0.0) ** 2 + (anchor_err or 0.0) ** 2)
        return recon_vals, recon_errs

    driver_anchor_shift, driver_anchor_err = anchor_shift_and_err(get_val_driver, get_err_driver)
    driver_trend = fit_weighted_shift_trend(get_val_driver, get_err_driver, z_max=DRIVER_REAL_CUTOFF)
    use_family_scaling = (driver_anchor_shift is not None and abs(driver_anchor_shift) > MIN_ANCHOR_MAG
                         and driver_trend is not None)

    for z in future_zs:
        v0 = get_val(0, z)
        e0 = get_err(0, z) or 0.0

        if use_family_scaling:
            driver_val, driver_err = eval_shift_trend(driver_trend, z)
            if driver_val is None:
                shift, shift_err = anchor_shift, anchor_err
            else:
                g = driver_val / driver_anchor_shift
                shift = anchor_shift * g
                var = (g ** 2) * (anchor_err ** 2) \
                    + ((anchor_shift / driver_anchor_shift) ** 2) * (driver_err ** 2) \
                    + ((anchor_shift * driver_val / driver_anchor_shift ** 2) ** 2) * (driver_anchor_err ** 2)
                shift_err = np.sqrt(max(var, 0.0))
        else:
            shift, shift_err = anchor_shift, anchor_err

        recon_vals[z] = v0 + shift
        recon_errs[z] = np.sqrt(e0 ** 2 + shift_err ** 2)

    return recon_vals, recon_errs

#%% fold EXTRA_Z_BIN0 ridge fits into emulator_bin[0] so the save cell
# below includes them -- otherwise bin 0's z=1.7 entries exist only in
# the plotting accessors and never reach the h5
for z in EXTRA_Z_BIN0:
    for i in range(n_labels):
        beta = bin0_extra_beta.get((i, z))
        if beta is not None:
            emulator_bin[0][(i, z)] = (list(range(n_lcdm)), beta)


#%% save emulator parameters (ridge, outlier-excluded) for config.py generation
EMULATOR_PATH = os.path.join(FIGURES_DIR, 'cosmo_emulator_params.h5')
with h5.File(EMULATOR_PATH, 'w') as hdf:
    hdf.attrs['param_names'] = PARAM_NAMES
    hdf.attrs['cosmo_names'] = LCDM_NAMES
    hdf.attrs['fiducial_cosmo'] = FIDUCIAL_LCDM
    hdf.create_dataset('redshifts', data=np.array(CANDIDATE_REDSHIFTS))

    for name, source in (('bin0', emulator_bin[0]), ('bin1', emulator_bin[1]), ('average', emulator_avg)):
        grp = hdf.create_group(name)
        for i in range(n_labels):
            for iz, z in enumerate(CANDIDATE_REDSHIFTS):
                entry = source.get((i, z))
                if entry is None:
                    continue
                cols, beta = entry
                if beta is None:
                    continue
                sub = grp.create_group(f'param{i}_z{iz}')
                sub.attrs['redshift'] = z
                sub.attrs['param_index'] = i
                sub.create_dataset('dependent_cols', data=np.array(cols, dtype=np.int32))
                sub.create_dataset('beta', data=beta)
print(f'saved emulator parameters to {EMULATOR_PATH}')

#%% scheme comparison: for each row, compute every candidate high-z
# handling scheme's value at every z, and plot each scheme's DIFFERENCE
# from the current (raw bin-1 ridge fit) value. "current" is therefore
# always exactly zero -- everything else shows how far that scheme would
# move you away from trusting the direct fit as-is.
#
# quadratic trend helper (reused from the sigma0-instability check earlier)
def fit_weighted_value_trend_quad(get_val, get_err, z_max=1.1):
    zs, vals, errs = [], [], []
    for z in CANDIDATE_REDSHIFTS:
        if z > z_max:
            continue
        v = get_val(z)
        if v is None:
            continue
        zs.append(z); vals.append(v); errs.append(get_err(z) or 0.0)
    if len(zs) < 4:
        return None
    zs, vals, errs = np.array(zs), np.array(vals), np.array(errs)
    errs = np.where(errs > 0, errs, np.median(errs[errs > 0]) if np.any(errs > 0) else 1.0)
    z0 = zs.mean()
    dz = zs - z0
    X = np.column_stack([np.ones_like(zs), dz, dz ** 2])
    W = np.diag(1.0 / errs ** 2)
    cov = np.linalg.inv(X.T @ W @ X)
    coeffs = cov @ (X.T @ W @ vals)
    return dict(coeffs=coeffs, cov=cov, z0=z0)

def eval_value_trend_quad(trend, z_query):
    if trend is None:
        return None, None
    dz = z_query - trend['z0']
    x = np.array([1.0, dz, dz ** 2])
    return x @ trend['coeffs'], np.sqrt(max(x @ trend['cov'] @ x, 0.0))


SCHEME_STYLES = {
    'frozen':    dict(c='tab:gray',   ls=':',  marker='s', label='frozen @ last good $z$'),
    'shared':    dict(c='tab:green',  ls='-.', marker='^', label='shared (use bin 0)'),
    'family':    dict(c='tab:red',    ls='--', marker='D', label='family-driven'),
    'linear':    dict(c='tab:blue',   ls='-',  marker='o', label='linear extrap.'),
    'quadratic': dict(c='tab:purple', ls='-',  marker='v', label='quadratic extrap.'),
}


def scheme_value(scheme, i, z, j=None, TREND_ZMAX=1.1):
    """j=None -> intercept; j=int -> slope on LCDM column j. Returns the
    scheme's predicted value at (row i, redshift z), or None if that
    scheme has nothing to say at this z (e.g. z is at/before the anchor,
    where 'frozen'/'family' just return the real value and 'linear'/
    'quadratic' aren't meaningfully different from a regression residual)."""
    get_val = (lambda k, zz: get_val_int(k, zz, i)) if j is None else (lambda k, zz: get_val_slope(k, zz, i, j))
    get_err = (lambda k, zz: get_err_int(k, zz, i)) if j is None else (lambda k, zz: get_err_slope(k, zz, i, j))

    if scheme == 'current':
        return get_val(1, z)

    if scheme == 'shared':
        return get_val(0, z)

    if scheme == 'frozen':
        cutoff = get_real_cutoff(i)
        anchor_z = max((zz for zz in CANDIDATE_REDSHIFTS if zz <= cutoff and get_val(1, zz) is not None),
                       default=None)
        return get_val(1, anchor_z) if anchor_z is not None else None

    if scheme == 'family':
        family = ROW_FAMILY.get(i)
        driver_row = FAMILY_DRIVER_ROW[family] if family is not None else i
        get_val_driver = ((lambda k, zz: get_val_int(k, zz, driver_row)) if j is None
                         else (lambda k, zz: get_val_slope(k, zz, driver_row, j)))
        get_err_driver = ((lambda k, zz: get_err_int(k, zz, driver_row)) if j is None
                         else (lambda k, zz: get_err_slope(k, zz, driver_row, j)))
        recon_vals, _ = build_bin1_reconstruction_family2(get_val, get_err, get_val_driver, get_err_driver, row_idx=i)
        return recon_vals.get(z)

    if scheme in ('linear', 'quadratic'):
        get_val1 = lambda zz: get_val(1, zz)
        get_err1 = lambda zz: get_err(1, zz)
        if scheme == 'linear':
            trend = fit_weighted_value_trend(get_val1, get_err1, z_max=TREND_ZMAX)
            val, _ = eval_value_trend(trend, z)
        else:
            trend = fit_weighted_value_trend_quad(get_val1, get_err1, z_max=TREND_ZMAX)
            val, _ = eval_value_trend_quad(trend, z)
        return val

    raise ValueError(scheme)

#%% update scheme_value to take an explicit z_max for linear/quadratic,
# rather than a hardcoded 1.1 -- so it can match each row's own anchor
def scheme_value(scheme, i, z, j=None, z_max=1.1):
    get_val = (lambda k, zz: get_val_int(k, zz, i)) if j is None else (lambda k, zz: get_val_slope(k, zz, i, j))
    get_err = (lambda k, zz: get_err_int(k, zz, i)) if j is None else (lambda k, zz: get_err_slope(k, zz, i, j))

    if scheme == 'current':
        return get_val(1, z)
    if scheme == 'shared':
        return get_val(0, z)

    if scheme == 'frozen':
        anchor_z = max((zz for zz in CANDIDATE_REDSHIFTS if zz <= z_max and get_val(1, zz) is not None),
                       default=None)
        return get_val(1, anchor_z) if anchor_z is not None else None

    if scheme == 'family':
        family = ROW_FAMILY.get(i)
        driver_row = FAMILY_DRIVER_ROW[family] if family is not None else i
        get_val_driver = ((lambda k, zz: get_val_int(k, zz, driver_row)) if j is None
                         else (lambda k, zz: get_val_slope(k, zz, driver_row, j)))
        get_err_driver = ((lambda k, zz: get_err_int(k, zz, driver_row)) if j is None
                         else (lambda k, zz: get_err_slope(k, zz, driver_row, j)))
        recon_vals, _ = build_bin1_reconstruction_family2(get_val, get_err, get_val_driver, get_err_driver, row_idx=i)
        return recon_vals.get(z)

    if scheme in ('linear', 'quadratic'):
        get_val1 = lambda zz: get_val(1, zz)
        get_err1 = lambda zz: get_err(1, zz)
        if scheme == 'linear':
            trend = fit_weighted_value_trend(get_val1, get_err1, z_max=z_max)
            val, _ = eval_value_trend(trend, z)
        else:
            trend = fit_weighted_value_trend_quad(get_val1, get_err1, z_max=z_max)
            val, _ = eval_value_trend_quad(trend, z)
        return val

    raise ValueError(scheme)

#%% updated real-data cutoffs, reflecting THIS session's actual findings:
# the rescue + outlier exclusion fixed 7/9 rows through z=1.4 (confirmed
# by the z-score/quadratic checks and the sane parameter values in the
# plot); only beta_delta and alpha remain genuinely unconstrained there
# (bound-pegged failures + real, non-overlapping multi-modality in their
# "kept" distributions). Replaces the old family-driver-based cutoff.
UNSTABLE_ROWS = [PARAM_NAMES.index(p) for p in ('beta_delta', 'alpha')]
STABLE_CUTOFF = 1.4
UNSTABLE_CUTOFF = 1.1   # last z where these two showed a clean, unimodal fit

def get_real_cutoff(row_idx):
    return UNSTABLE_CUTOFF if row_idx in UNSTABLE_ROWS else STABLE_CUTOFF


#%% intercepts vs z: measurements unchanged, schemes now branch from each
# row's OWN designated anchor (get_real_cutoff: 1.1 for the two family
# drivers, 0.5 for everything else) rather than from the last available
# (but not necessarily trustworthy) bin-1 fit point -- so for non-driver
# rows you can now see each scheme's prediction against the REAL z=0.8/
# 1.1/1.4 measurements too, not just against z=1.7's unknown.
fig_int_cmp, ax_int_cmp = plt.subplots(n_labels, 1, figsize=(6, 2.4 * n_labels), sharex=True, layout='constrained')

for i in range(n_labels):
    row_vals_for_ylim = []

    for key in (0, 1, 'avg'):
        zs, vals, errs = [], [], []
        for z in CANDIDATE_REDSHIFTS:
            if key == 'avg':
                beta = emulator_avg[(i, z)][1]
                val = beta[0] if beta is not None else None
                err = INTERCEPT_ERR['avg'].get((i, z))
            else:
                val = get_val_int(key, z, i)
                err = get_err_int(key, z, i)
            if val is None:
                continue
            zs.append(z); vals.append(val); errs.append(err or 0.0)
        if not zs:
            continue
        order = np.argsort(zs)
        zs_s, vals_s, errs_s = np.array(zs)[order], np.array(vals)[order], np.array(errs)[order]
        row_vals_for_ylim.append(vals_s)
        style = BIN_STYLE[key]
        ax_int_cmp[i].errorbar(zs_s, vals_s, yerr=errs_s, marker='o', ms=4, capsize=2,
                               c=style['c'], ls=style['ls'], label=style['label'] if i == 0 else None)

    anchor_z = get_real_cutoff(i)
    v_anchor = get_val_int(1, anchor_z, i)
    if v_anchor is None:
        continue

    future_zs = [z for z in CANDIDATE_REDSHIFTS if z > anchor_z]
    for scheme, style in SCHEME_STYLES.items():
        zs_plot, vals_plot = [anchor_z], [v_anchor]
        for z in future_zs:
            val = scheme_value(scheme, i, z, z_max=anchor_z)
            if val is not None:
                zs_plot.append(z)
                vals_plot.append(val)
        if len(zs_plot) > 1:
            row_vals_for_ylim.append(np.array(vals_plot[1:]))
            ax_int_cmp[i].plot(zs_plot, vals_plot, marker=style['marker'], ms=4, c=style['c'], ls=style['ls'],
                              alpha=0.85, label=style['label'] if i == 0 else None)

    ax_int_cmp[i].axvline(anchor_z, c='gray', ls=':', lw=1, alpha=0.4)
    ylim = robust_ylim(row_vals_for_ylim)
    if ylim is not None:
        ax_int_cmp[i].set_ylim(*ylim)
    ax_int_cmp[i].set_ylabel(labels[i])

ax_int_cmp[0].legend(fontsize=6, ncol=2)
ax_int_cmp[-1].set_xlabel(r'$z$')
fig_int_cmp.suptitle(r'Compact-model parameter value vs redshift: measurements + extrapolation schemes, '
                    'branched from each row\'s own anchor ($z{=}1.1$ for drivers, $z{=}0.5$ otherwise)')
fname = os.path.join(FIGURES_DIR, 'emulator_intercept_vs_redshift_with_schemes.png')
fig_int_cmp.savefig(fname, dpi=150)
print(f'saved {fname}')


#%% slopes vs z: same per-row anchor treatment
fig_slope_cmp, ax_slope_cmp = plt.subplots(n_labels, n_lcdm, figsize=(3.2 * n_lcdm, 2.4 * n_labels),
                                          sharex='col', layout='constrained')

for i in range(n_labels):
    anchor_z = get_real_cutoff(i)
    for jcol in range(n_lcdm):
        panel_vals_for_ylim = []

        for key in (0, 1, 'avg'):
            zs, vals, errs = [], [], []
            for z in CANDIDATE_REDSHIFTS:
                if key == 'avg':
                    beta = emulator_avg[(i, z)][1]
                    val = beta[1 + jcol] if beta is not None else None
                    err_arr = SLOPE_ERR['avg'].get((i, z))
                    err = err_arr[jcol] if err_arr is not None else None
                else:
                    val = get_val_slope(key, z, i, jcol)
                    err = get_err_slope(key, z, i, jcol)
                if val is None:
                    continue
                zs.append(z); vals.append(val); errs.append(err or 0.0)
            if not zs:
                continue
            order = np.argsort(zs)
            zs_s, vals_s, errs_s = np.array(zs)[order], np.array(vals)[order], np.array(errs)[order]
            panel_vals_for_ylim.append(vals_s)
            style = BIN_STYLE[key]
            ax_slope_cmp[i, jcol].errorbar(zs_s, vals_s, yerr=errs_s, marker='o', ms=4, capsize=2,
                                          c=style['c'], ls=style['ls'])

        v_anchor = get_val_slope(1, anchor_z, i, jcol)
        if v_anchor is not None:
            future_zs = [z for z in CANDIDATE_REDSHIFTS if z > anchor_z]
            for scheme, style in SCHEME_STYLES.items():
                zs_plot, vals_plot = [anchor_z], [v_anchor]
                for z in future_zs:
                    val = scheme_value(scheme, i, z, j=jcol, z_max=anchor_z)
                    if val is not None:
                        zs_plot.append(z)
                        vals_plot.append(val)
                if len(zs_plot) > 1:
                    panel_vals_for_ylim.append(np.array(vals_plot[1:]))
                    ax_slope_cmp[i, jcol].plot(zs_plot, vals_plot, marker=style['marker'], ms=3,
                                              c=style['c'], ls=style['ls'], alpha=0.85)
            ax_slope_cmp[i, jcol].axvline(anchor_z, c='gray', ls=':', lw=1, alpha=0.4)

        ax_slope_cmp[i, jcol].axhline(0, c='gray', ls='--', lw=1)
        ylim = robust_ylim(panel_vals_for_ylim + [np.array([0.0])])
        if ylim is not None:
            ax_slope_cmp[i, jcol].set_ylim(*ylim)
        if i == n_labels - 1:
            ax_slope_cmp[i, jcol].set_xlabel(r'$z$')
    ax_slope_cmp[i, 0].set_ylabel(labels[i])

for jcol in range(n_lcdm):
    ax_slope_cmp[0, jcol].set_title(LCDM_LABELS[jcol])

fig_slope_cmp.suptitle(r'Cosmology-dependence slope vs redshift: measurements + extrapolation schemes, '
                      'branched from each row\'s own anchor')
fname = os.path.join(FIGURES_DIR, 'emulator_slope_vs_redshift_with_schemes.png')
fig_slope_cmp.savefig(fname, dpi=150)
print(f'saved {fname}')

#%% diagnostic: what R_piv did each (sim, z, bin) fit ACTUALLY use? Reuses
# the exact same mask as fit_sigma_joint (isfinite(x) & isfinite(sig) &
# sig > SIG_FLOOR), but stops short of calling curve_fit -- R_piv only
# depends on which (x, R) cells survive that mask, not on the fit itself.
def get_actual_r_piv(n, z, k, coarsened_group_edges=None, n_min=N_MIN):
    """coarsened_group_edges: pass group_edges_z1p4_bin1 (or equivalent)
    to check the RESCUED fit's R_piv instead of the native-grid one --
    the r_p array (and therefore the median) differs between the two."""
    h = PairHist.load(out_path(n, z, k))
    if coarsened_group_edges is not None:
        h = merge_pairhist_rp_variable(h, coarsened_group_edges)
        r_p_use = 0.5 * (h.R_edges[1:] + h.R_edges[:-1])
        r_l_use = 0.5 * (h.rlos_edges[1:] + h.rlos_edges[:-1])
        rp2d, rl2d = np.meshgrid(r_p_use, r_l_use, indexing='ij')
        r3_2d_use = np.sqrt(rp2d**2 + rl2d**2)
        x_grid_use = r3_2d_use - rp2d
    else:
        r_p_use, x_grid_use = r_p, x_grid

    sig = meas_sigv_grid(h, 0, n_min=n_min)
    R = np.broadcast_to(r_p_use[:, None], x_grid_use.shape)
    m = np.isfinite(x_grid_use) & np.isfinite(sig) & (sig > SIG_FLOOR)
    return np.median(R[m])


#%% scan across a representative sample of (z, bin) combos -- native grid
# for everything except z=1.4 bin=1, which uses the rescued grid
print(f"{'z':>5s}  {'bin':>3s}  {'n_sims_checked':>14s}  {'R_piv median':>12s}  {'R_piv std':>10s}  {'min':>8s}  {'max':>8s}")
for z in CANDIDATE_REDSHIFTS:
    for k in range(N_MASS):
        if z == 1.4 and k == 1:
            continue   # handled separately below, uses the rescued grid
        sims_here = [n for n in core.SIM_NUMS if os.path.exists(out_path(n, z, k))][:15]
        if not sims_here:
            continue
        pivs = [get_actual_r_piv(n, z, k) for n in sims_here]
        print(f"{z:>5}  {k:>3d}  {len(pivs):>14d}  {np.median(pivs):>12.2f}  "
              f"{np.std(pivs):>10.3f}  {min(pivs):>8.2f}  {max(pivs):>8.2f}")

# z=1.4, bin=1 specifically -- rescued grid
sims_here = [n for n in core.SIM_NUMS if os.path.exists(out_path(n, 1.4, 1))][:15]
pivs_rescued = [get_actual_r_piv(n, 1.4, 1, coarsened_group_edges=group_edges_z1p4_bin1, n_min=N_MIN_MERGE)
               for n in sims_here]
print(f"\nz=1.4  bin=1 (RESCUED grid)  n={len(pivs_rescued)}  "
      f"median={np.median(pivs_rescued):.2f}  std={np.std(pivs_rescued):.3f}  "
      f"min={min(pivs_rescued):.2f}  max={max(pivs_rescued):.2f}")
