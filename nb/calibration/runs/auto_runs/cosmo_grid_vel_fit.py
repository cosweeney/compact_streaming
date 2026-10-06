"""
Fits the 9-parameter compact velocity model (sigma_v: 4 params, Delta: 5
params) across every available (sim, redshift, mass bin) in the
AbacusSummit cosmology grid.

This version stops at the point of producing the final, rescued `params`
dict -- i.e. through the r_p-coarsened rescue fit for bin 1 @ z=1.4. The
cosmology-dependence / LambdaCDM emulator cells that consume `params`
downstream are a separate stage and not included here.

Scoped narrowly to what was asked, NOT a full port of
check_abacus_cosmo_runs_redshift.ipynb: that notebook additionally does
CAMB-based sigma8_nl corrections, linear cosmology-emulator regression fits,
Gaussian Process fits (george), and Quijote cross-checks -- none of that is
reproduced here. It also predates this project's model updates: it fits
the OLD 6-parameter model (2-parameter, flat-core Delta) against the OLD
pairhist_2bins.h5 output, whereas this script fits the CURRENT 9-parameter
model (5-parameter, exponential-amplitude Delta) against the CURRENT
pairhist_auto_mbin{k}.h5 auto-statistics output.

Model-choice note: A_delta_mod here uses the EXPONENTIAL form
(A_delta0 * exp(-R/r_delta)), matching the form found to fit better than
the original flat-core form -- but that comparison was done on base-suite
data (a different mass-bin scheme), not this cosmology grid. Treat this as
carried over by consistency, not independently re-validated here.

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
from types import SimpleNamespace

import matplotlib
matplotlib.rcParams.update({'text.usetex': True,
                            'font.family': 'Computer Modern Roman'})
plt.rcParams['axes.labelsize'] = 20

import nb.calibration.runs.auto_runs.abacus_cosmo_core_auto as core   # safe: no pycorr import in this module
from compact.velocities.pairhist import PairHist

FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# scan for whatever redshifts actually have output, rather than hardcoding
# REDSHIFTS -- run_abacus_cosmo_rest.py's own list may not all be done yet
CANDIDATE_REDSHIFTS = [0.2, 0.3, 0.4, 0.5, 0.8, 1.1, 1.4, 1.7]
N_MASS = len(core.BIN_NPART)   # 2: low, high -- matches abacus_cosmo_core.py

VR_R_EDGES = np.arange(2.0, 142.0, 2.0)

# grid -- matches abacus_cosmo_core.py's R_edges/rlos_edges exactly
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
    R_piv = np.median(RR)
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


def fit_delta_joint(x, r_p_arr, D, p0=DEL_P0, weights=None):
    R = np.broadcast_to(r_p_arr[:, None], x.shape)
    m = np.isfinite(x) & np.isfinite(D) & (D > DEL_FLOOR)
    X, RR, DD = x[m], R[m], D[m]
    W = weights[m] if weights is not None else np.ones_like(DD)
    popt, _ = curve_fit(delta_mod_alpha_R, (X, RR), DD, p0=p0, sigma=1.0 / np.sqrt(W), maxfev=20_000)
    return popt


def skewed_t_pdf(x, xi, omega, alpha, nu):
    z = (x - xi) / omega
    t_pdf = student_t.pdf(z, df=nu)
    t_cdf = student_t.cdf(alpha * z * np.sqrt((nu + 1) / (nu + z**2)), df=nu + 1)
    return (2 / omega) * t_pdf * t_cdf


NU_FIXED = 6
REBIN = 5
N_MIN = 50   # was 500 -- lowered so weighting (below) can use thin-but-nonzero
             # cells instead of discarding them outright; see fit_sigma_joint /
             # fit_delta_joint's weights= argument. NOTE: this is a floor for
             # "measurable at all," not "trustworthy enough to weight equally,"
             # and it changes the default cell-selection for EVERY fit_one
             # call, not just the rescue path below -- worth spot-checking
             # that the previously-passing low-z bins haven't shifted.


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
        good_vr = np.isfinite(vr_mean) & (npairs_vr > 100)   # matches this
                                                                # project's MIN_HALOS-era
                                                                # floor, not the older 1000
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

# params[k][z][n] = concatenated 9-vector, only where BOTH fits succeeded
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


#%% r_p-coarsening rescue path for bin 1 @ z=1.4, diagnosed as pair-starved
# specifically at small r_p (unlike z=1.7 bin 1, which is thin across the
# ENTIRE r_p range and is excluded rather than rescued -- see the note at
# the bottom of this cell). Scoped to a local, coarser r_p grid; does not
# touch the shared module-level r_p/x_grid used by fit_one, so every other
# (z, bin) combination is unaffected.

def merge_pairhist_rp_variable(h, group_edges):
    """Merge r_p bins into groups defined by group_edges (bin INDICES into
    the original R_edges array, e.g. [0, 40, 70, 95, 115, 135] merges bins
    [0:40), [40:70), [70:95), [95:115), [115:135) -- coarser groups where
    r_p is small/coverage is thin, finer groups where r_p is large/coverage
    is already fine. group_edges must start at 0 and end at n_rp. Sums the
    additive per-pair quantities (n, sum_v, sum_v2, sum_vr, counts);
    leaves r_l/the velocity-histogram axis untouched since the coverage
    gap is in r_p specifically."""
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
    """n_stack: (n_sim, n_rp, n_rl) raw pair counts, native grid.
    Returns group_edges (bin indices, ascending, spanning [0, n_rp]) built
    by walking from the LARGE-r_p end backward, growing each group until
    its pooled (summed) per-cell coverage >= target_coverage. The final
    (smallest-r_p) group takes whatever bins are left, however thin --
    it is not guaranteed to clear target_coverage, and that's a signal
    worth reading off `coverage_achieved`, not a bug to fix here."""
    n_rp = n_stack.shape[1]
    edges = [n_rp]
    end = n_rp
    coverage_achieved = []
    while end > 0:
        start = end
        pooled = np.zeros(n_stack.shape[::2])   # (n_sim, n_rl)
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

# built from EVERY available sim at (z=1.4, bin=1), not the 8-sim sample
# used during exploration -- this schedule now gets applied to the actual
# production fit for all of them, so it should be based on all of them
_z_rescue, _k_rescue = 1.4, 1
_sims_for_edges = [n for n in core.SIM_NUMS if os.path.exists(out_path(n, _z_rescue, _k_rescue))]
_n_stack = np.stack([PairHist.load(out_path(n, _z_rescue, _k_rescue)).n[0] for n in _sims_for_edges])
group_edges_z1p4_bin1, _coverages = build_group_edges(_n_stack, N_MIN_MERGE, TARGET_COVERAGE)
print(f'z={_z_rescue}, bin={_k_rescue}: {len(group_edges_z1p4_bin1) - 1} r_p groups '
      f'from {len(_sims_for_edges)} sims, '
      f'widths (small-r_p first) {np.diff(group_edges_z1p4_bin1).tolist()}')


def fit_one_rescue_variable(n, z, k, group_edges, n_min):
    """Same pipeline as fit_one, but on a variable-width r_p-merged grid
    (group_edges from build_group_edges) and an explicit n_min, with
    pair-count weighting in the joint fits. Use only for (z, k) combos
    diagnosed as needing this -- not a general replacement for fit_one."""
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
        params[k][z][n] = np.concatenate([sig_p, del_p])   # slot into the existing params dict

# z=1.7 bin 1: diagnosed as pair-starved across the ENTIRE r_p range (not
# just small r_p), unlike z=1.4 -- even heavy r_p-merging (width up to 15)
# only reached 33-73% coverage with no well-sampled region left to anchor
# a shape fit. Excluded from downstream emulator use rather than rescued;
# left in as a finalization step here so `params` is in its complete,
# ready-for-the-next-stage state. Remove this if you'd rather keep the
# raw (mostly-empty) z=1.7 bin-1 entries around for inspection.
if 1.7 in params[1]:
    params[1][1.7] = {}

print('\nfinal per-(bin, z) sim counts:')
for k in range(N_MASS):
    for z in CANDIDATE_REDSHIFTS:
        if params[k][z]:
            print(f'bin {k}, z={z}: {len(params[k][z])} sims fit successfully')

#%% pull cosmological parameters per sim (redshift-independent)
COSMO_COLS = ['omega_b', 'omega_cdm', 'h', 'n_s', 'alpha_s', 'w0_fld', 'wa_fld', 'sigma8_m']
c_labels = [r"$\omega_b$", r"$\omega_{\rm cdm}$", r"$h$", r"$n_s$",
           r"$\alpha_s$", r"$w_0$", r"$w_a$", r"$\sigma_{8,m}$"]

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

# LambdaCDM-only subset used by the emulator-related cells below: omega_b,
# omega_cdm, log(10^10 A_s), n_s -- dropping alpha_s, w0_fld, wa_fld (dark
# energy / running extensions beyond flat LCDM) and the derived sigma8_m
# (redundant with A_s). Matches check_abacus_cosmo_runs_redshift.ipynb's
# lcdm_arr exactly, including the natural-log A_s transform.
LCDM_LABELS = [r"$\omega_b$", r"$\omega_{\rm cdm}$", r"$\log(10^{10}A_s)$", r"$n_s$"]
LCDM_NAMES = ['omega_b', 'omega_cdm', 'ln_1e10_A_s', 'n_s']
n_lcdm = len(LCDM_LABELS)

_ob, _oc, _ns = COSMO_COLS.index('omega_b'), COSMO_COLS.index('omega_cdm'), COSMO_COLS.index('n_s')
lcdm_pars = {n: np.array([cosmo_pars[n][_ob], cosmo_pars[n][_oc],
                          np.log(1e10 * cosmo_As[n]), cosmo_pars[n][_ns]])
             for n in all_sims_used}

# c000's own fiducial values, in LCDM_NAMES' exact order
FIDUCIAL_LCDM = np.array([0.02237, 0.1200, np.log(1e10 * 2.0830e-9), 0.9649])
assert len(FIDUCIAL_LCDM) == n_lcdm


#%% save fit results to disk (flat, reload-friendly format -- both the 9
# velocity-model parameters and the 8 cosmological parameters, per row)
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
    hdf.create_dataset('params', data=np.array(rows_params, dtype=np.float64))   # (N, 9)
    hdf.create_dataset('cosmo', data=np.array(rows_cosmo, dtype=np.float64))     # (N, 8)
    hdf.attrs['param_names'] = PARAM_NAMES
    hdf.attrs['cosmo_names'] = COSMO_COLS
print(f'saved {len(rows_sim)} (sim, z, bin) fit results to {FIT_RESULTS_PATH}')

#%% to reload later without redoing the fit:
PARAM_NAMES = ['sigma0', 'A_rho0', 'A_rho_p', 'sigma_rho',
              'A_delta0', 'r_delta', 'sigma_delta', 'beta_delta', 'alpha']
FIT_RESULTS_PATH = os.path.join(FIGURES_DIR, 'cosmo_grid_fit_results.h5')

with h5.File(FIT_RESULTS_PATH, 'r') as hdf:
    sim, z, bin_ = hdf['sim'][()], hdf['z'][()], hdf['bin'][()]
    params_arr, cosmo_arr = hdf['params'][()], hdf['cosmo'][()]
    param_names, cosmo_names = list(hdf.attrs['param_names']), list(hdf.attrs['cosmo_names'])

#%% resume from saved fits: rebuild `params` and everything the LambdaCDM
# emulator stage needs from cosmo_grid_fit_results.h5, skipping fit_one /
# the diagnostic passes / the rescue fit entirely. Run this INSTEAD of
# everything from '#%% scan for available (sim, z, bin) output' through
# the z=1.7 bin-1 zeroing cell -- not in addition to it. Because the save
# cell runs AFTER both the z=1.4 rescue merge and the z=1.7 bin-1 zeroing
# in the original script, the reloaded `params` already reflects that
# final, corrected state -- no need to redo either step.
import os
import h5py as h5
import numpy as np

FIGURES_DIR = 'figures'
FIT_RESULTS_PATH = os.path.join(FIGURES_DIR, 'cosmo_grid_fit_results.h5')

with h5.File(FIT_RESULTS_PATH, 'r') as hdf:
    sim_arr = hdf['sim'][()]
    z_arr = hdf['z'][()]
    bin_arr = hdf['bin'][()]
    params_arr = hdf['params'][()]      # (N, 9)
    cosmo_arr = hdf['cosmo'][()]        # (N, 8) -- COSMO_COLS order; does NOT include raw A_s
    PARAM_NAMES = list(hdf.attrs['param_names'])
    COSMO_COLS = list(hdf.attrs['cosmo_names'])

CANDIDATE_REDSHIFTS = sorted(set(z_arr.tolist()))
N_MASS = int(bin_arr.max()) + 1

labels = [r"$\sigma_0$ [km/s]", r"$A_{\rho,0}$", r"$A_{\rho,p}$ [$h^{-1}$ Mpc]", r"$\sigma_\rho$ [$h^{-1}$ Mpc]",
          r"$A_{\Delta,0}$ [km/s]", r"$r_\Delta$ [$h^{-1}$ Mpc]", r"$\sigma_\Delta$ [$h^{-1}$ Mpc]",
          r"$\beta_\Delta$", r"$\alpha_\Delta$"]
n_labels = len(labels)
assert n_labels == len(PARAM_NAMES), "labels/PARAM_NAMES length mismatch -- check save/reload are in sync"

params = {k: {z: {} for z in CANDIDATE_REDSHIFTS} for k in range(N_MASS)}
cosmo_pars = {}
for row in range(len(sim_arr)):
    n, z, k = int(sim_arr[row]), float(z_arr[row]), int(bin_arr[row])
    params[k][z][n] = params_arr[row]
    cosmo_pars[n] = cosmo_arr[row].tolist()   # same cosmo repeated per (z,bin) row for a given sim

print(f'reloaded {len(sim_arr)} (sim, z, bin) fits from {FIT_RESULTS_PATH}')
for k in range(N_MASS):
    for z in CANDIDATE_REDSHIFTS:
        if params[k][z]:
            print(f'bin {k}, z={z}: {len(params[k][z])} sims')

#%% re-pull A_s per sim (cheap: reads a param file, no fitting) to
# reconstruct lcdm_pars, since it isn't in the saved cosmo array
import nb.calibration.runs.auto_runs.abacus_cosmo_core_auto as core

all_sims_used = sorted(set(n for k in range(N_MASS) for z in CANDIDATE_REDSHIFTS for n in params[k][z]))
cosmo_As = {}
for n in all_sims_used:
    z_avail = next((z for z in CANDIDATE_REDSHIFTS if n in params[0].get(z, {})
                    or n in params[min(N_MASS - 1, 1)].get(z, {})), CANDIDATE_REDSHIFTS[0])
    par = core.read_params(n, z_avail)
    cosmo_As[n] = float(par['A_s'])

_ob, _oc, _ns = COSMO_COLS.index('omega_b'), COSMO_COLS.index('omega_cdm'), COSMO_COLS.index('n_s')
LCDM_LABELS = [r"$\omega_b$", r"$\omega_{\rm cdm}$", r"$\log(10^{10}A_s)$", r"$n_s$"]
LCDM_NAMES = ['omega_b', 'omega_cdm', 'ln_1e10_A_s', 'n_s']
n_lcdm = len(LCDM_LABELS)

lcdm_pars = {n: np.array([cosmo_pars[n][_ob], cosmo_pars[n][_oc],
                          np.log(1e10 * cosmo_As[n]), cosmo_pars[n][_ns]])
             for n in all_sims_used}

FIDUCIAL_LCDM = np.array([0.02237, 0.1200, np.log(1e10 * 2.0830e-9), 0.9649])
assert len(FIDUCIAL_LCDM) == n_lcdm
            
#%% identify the outlier sims driving r_delta/beta_delta/alpha's wild
# scatter at (k=1, z=1.1) -- MAD-based, same robustness logic as
# robust_ylim, but returning the actual sim IDs rather than just axis limits
def find_outlier_sims(k, z, param_name, mad_k=5.0):
    idx = PARAM_NAMES.index(param_name)
    sims = sorted(params[k][z])
    vals = np.array([params[k][z][n][idx] for n in sims])
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    thresh = mad_k * 1.4826 * mad if mad > 0 else np.inf
    return [(n, v) for n, v in zip(sims, vals) if abs(v - med) > thresh]

for pname in ('r_delta', 'beta_delta', 'alpha'):
    outliers = find_outlier_sims(1, 1.1, pname)
    print(f'{pname}: {len(outliers)} outlier sim(s) -> {outliers}')


def fit_delta_joint(x, r_p_arr, D, p0=DEL_P0, weights=None, n_restarts=5, seed=0):
    """Same interface as before, now with:
    (1) physically motivated bounds -- A_delta0 unconstrained in sign (the
        model's sign convention allows either), but r_delta, sigma_delta,
        beta_delta > 0 (a negative lengthscale or negative shape exponent
        isn't a different fit, it's degenerate/unphysical), and alpha
        bounded away from negative values, which also sidesteps the
        documented x=0 singularity in delta_mod_alpha for negative alpha.
    (2) multi-start: p0 plus n_restarts random perturbations, keeping
        whichever converged fit has the lowest residual sum of squares --
        a single fixed seed is exactly what let a handful of noisier
        bin-1 fits wander to a degenerate optimum uncontested."""
    R = np.broadcast_to(r_p_arr[:, None], x.shape)
    m = np.isfinite(x) & np.isfinite(D) & (D > DEL_FLOOR)
    X, RR, DD = x[m], R[m], D[m]
    W = weights[m] if weights is not None else np.ones_like(DD)
    sigma_w = 1.0 / np.sqrt(W)

    lower = [-np.inf, 1e-3, 1e-3, 1e-3, 0.0]
    upper = [np.inf, np.inf, np.inf, 20.0, 10.0]

    rng = np.random.default_rng(seed)
    best_popt, best_cost = None, np.inf
    p0_arr = np.clip(np.array(p0, dtype=float), lower, upper)

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

#%% refit the full union of outlier sims (10, not just per-parameter
# overlaps) using the bounded + multi-start fit_delta_joint from before.
# Bumping n_restarts up from the default given how high the failure rate
# turned out to be (10/31, ~32%) -- worth giving the optimizer more chances
# to escape whatever degenerate basin it's landing in.
outlier_sims_z1p1_bin1 = sorted({130, 137, 142, 148, 152, 159, 170, 172, 175, 181})
print(f'refitting {len(outlier_sims_z1p1_bin1)} outlier sims at (bin=1, z=1.1): {outlier_sims_z1p1_bin1}')

def refit_delta_only(n, z, k, n_restarts=15):
    path = out_path(n, z, k)
    h = PairHist.load(path)
    sig = meas_sigv_grid(h, 0)
    ST_pars = fit_skewt_grid(h, 0)
    r_vr, vr_mean, npairs_vr = vr_profile(h, 0, VR_R_EDGES)
    good_vr = np.isfinite(vr_mean) & (npairs_vr > 100)
    vr_interp = InterpolatedUnivariateSpline(r_vr[good_vr], vr_mean[good_vr], k=3, ext=3)
    Delta = ST_pars[:, :, 0] - vr_interp(r3_2d) * r_l_2d / r3_2d
    return fit_delta_joint(x_grid, r_p, Delta, p0=DEL_P0, n_restarts=n_restarts)

refit_log = []
for n in outlier_sims_z1p1_bin1:
    old = params[1][1.1][n][4:].copy()   # A_delta0, r_delta, sigma_delta, beta_delta, alpha
    try:
        new_del = refit_delta_only(n, 1.1, 1)
        params[1][1.1][n][4:] = new_del
        refit_log.append((n, old, new_del, 'ok'))
    except RuntimeError as e:
        refit_log.append((n, old, None, f'failed: {e}'))

print(f"{'sim':>5s}  {'r_delta (old->new)':>28s}  {'beta_delta':>18s}  {'alpha':>14s}  status")
for n, old, new, status in refit_log:
    if new is None:
        print(f'{n:>5d}  {old[1]:>10.2f} -> {"N/A":<10s}  {old[3]:>8.2f} -> N/A       {old[4]:>6.2f} -> N/A  {status}')
    else:
        print(f'{n:>5d}  {old[1]:>10.2f} -> {new[1]:<10.2f}  {old[3]:>8.2f} -> {new[3]:<8.2f}  '
              f'{old[4]:>6.2f} -> {new[4]:<6.2f}  {status}')
        
#%% cosmology dependence: one figure per redshift, params vs ALL raw
# cosmological parameters (not LambdaCDM-restricted) -- exploratory scan,
# separate from the LambdaCDM emulator built below
from scipy.stats import spearmanr

for z in CANDIDATE_REDSHIFTS:
    sims_here = sorted(set(n for k in range(N_MASS) for n in params[k].get(z, {})))
    if not sims_here:
        continue

    fig, ax = plt.subplots(len(labels), len(COSMO_COLS),
                           figsize=(3.2 * len(COSMO_COLS), 2.6 * len(labels)),
                           sharex='col', layout='constrained')
    for i, plabel in enumerate(labels):
        for j, clabel in enumerate(c_labels):
            for k in range(N_MASS):
                sims_k = sorted(params[k].get(z, {}))
                if not sims_k:
                    continue
                cvals = np.array([cosmo_pars[n][j] for n in sims_k])
                pvals = np.array([params[k][z][n][i] for n in sims_k])
                ax[i, j].plot(cvals, pvals, 'o', ms=4, alpha=0.6, c=f'C{k}',
                              label=f'bin {k}' if i == 0 and j == 0 else None)
                if len(cvals) > 3:
                    rho, _ = spearmanr(cvals, pvals)
                    ax[i, j].annotate(rf'$\rho={rho:.2f}$', xy=(0.05, 0.85 - 0.15*k),
                                      xycoords='axes fraction', fontsize=7, color=f'C{k}')
            if i == len(labels) - 1:
                ax[i, j].set_xlabel(clabel)
            if j == 0:
                ax[i, j].set_ylabel(plabel)
    ax[0, 0].legend(fontsize=7)
    fig.suptitle(rf'cosmology dependence at $z={z:g}$ ({len(sims_here)} sims)')
    fname = os.path.join(FIGURES_DIR, f'cosmo_dep_z{z:.3f}'.replace('.', 'p') + '.png')
    fig.savefig(fname, dpi=150)
    print(f'saved {fname}')


#%% redshift dependence: one figure, 9 panels, one line per sim per mass bin
fig, ax = plt.subplots(2, 5, figsize=(22, 8), layout='constrained')
ax = ax.ravel()
for i, plabel in enumerate(labels):
    for k in range(N_MASS):
        for n in all_sims_used:
            zs_for_n = [z for z in CANDIDATE_REDSHIFTS if n in params[k].get(z, {})]
            if len(zs_for_n) < 2:
                continue
            vals = [params[k][z][n][i] for z in zs_for_n]
            ax[i].plot(zs_for_n, vals, marker='o', alpha=0.3,
                      c=f'C{k}', ms=3, lw=1)
    ax[i].set_xlabel(r'$z$')
    ax[i].set_ylabel(plabel)
ax[-1].axis('off')
handles = [plt.Line2D([0], [0], marker='o', color=f'C{k}', label=f'bin {k}')
          for k in range(N_MASS)]
ax[0].legend(handles=handles, fontsize=9)
fig.suptitle('redshift dependence, one line per sim (colored by mass bin)')
fname = os.path.join(FIGURES_DIR, 'redshift_dependence.png')
fig.savefig(fname, dpi=150)
print(f'saved {fname}')


#%% cosmology-dependence linear emulators: two INDEPENDENT fits (one per
# mass bin) + their average, restricted to the LambdaCDM columns only.
#
# Two deliberate departures from the old (check_abacus_cosmo_runs_redshift.
# ipynb) notebook, stated explicitly rather than left implicit:
#   1. The cosmology-parameter SUBSET (which ones get a slope at all) is
#      selected INDEPENDENTLY per bin here. The old notebook used only its
#      bin-0 correlation test to decide the subset for every bin; since
#      you want the two emulators to be genuinely independent, each bin's
#      own correlation test decides its own subset.
#   2. Since the two bins can therefore select different subsets, the
#      AVERAGE emulator's slope for a cosmological parameter that only one
#      bin selected treats the OTHER bin's contribution as exactly zero --
#      a defined convention, not an oversight.
from scipy.stats import pearsonr

RHO_MIN = 0.3
P_MAX = 0.05 / (len(labels) * n_lcdm)   # Bonferroni-corrected over the 4 LCDM columns
MIN_SIMS_FOR_FIT = 4

common_sims_by_z = {z: sorted(set(params[0].get(z, {})) & set(params[1].get(z, {})))
                    for z in CANDIDATE_REDSHIFTS}


def fit_one_bin(sims_z, k, z, i):
    """Independent subset-selection + linear fit for ONE bin, ONE velocity
    parameter, ONE redshift, restricted to the LambdaCDM columns
    (LCDM_LABELS/lcdm_pars) only. Returns (dependent_cols, beta, rhos),
    where rhos is {column index: pearson rho} for every LCDM column tested
    (not just the selected ones), so the plotting cell can annotate and
    shade panels consistently with the actual selection criterion.
    Returns (None, None, {}) if there's too little data to fit anything."""
    if len(sims_z) < MIN_SIMS_FOR_FIT:
        return None, None, {}
    theta = np.array([lcdm_pars[n] for n in sims_z])            # (n_sim, 4), raw LCDM theta
    delta = FIDUCIAL_LCDM[None, :] - theta                       # Delta_theta = theta_fid - theta
    y = np.array([params[k][z][n][i] for n in sims_z])

    dependent_cols, rhos = [], {}
    for j in range(n_lcdm):
        rho, p = pearsonr(theta[:, j], y)
        rhos[j] = rho
        if abs(rho) >= RHO_MIN and p < P_MAX:
            dependent_cols.append(j)

    X = np.column_stack([np.ones(len(sims_z))] + [delta[:, j] for j in dependent_cols])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return dependent_cols, beta, rhos


n_labels = len(labels)

# emulator_bin[k][(i, z)] = (dependent_cols, beta); emulator_avg[(i, z)] likewise
emulator_bin = {0: {}, 1: {}}
emulator_avg = {}
rhos_bin = {0: {}, 1: {}}

for i in range(n_labels):
    for z in CANDIDATE_REDSHIFTS:
        sims_z = common_sims_by_z[z]
        cols0, beta0, rhos0 = fit_one_bin(sims_z, 0, z, i)
        cols1, beta1, rhos1 = fit_one_bin(sims_z, 1, z, i)
        emulator_bin[0][(i, z)] = (cols0, beta0)
        emulator_bin[1][(i, z)] = (cols1, beta1)
        rhos_bin[0][(i, z)] = rhos0
        rhos_bin[1][(i, z)] = rhos1

        if beta0 is None or beta1 is None:
            emulator_avg[(i, z)] = (None, None)
            continue

        union_cols = sorted(set(cols0) | set(cols1))
        intercept_avg = (beta0[0] + beta1[0]) / 2.0
        slope_avg = []
        for j in union_cols:
            s0 = beta0[1 + cols0.index(j)] if j in cols0 else 0.0
            s1 = beta1[1 + cols1.index(j)] if j in cols1 else 0.0
            slope_avg.append((s0 + s1) / 2.0)
        emulator_avg[(i, z)] = (union_cols, np.array([intercept_avg] + slope_avg))

n_fit = sum(1 for i in range(n_labels) for z in CANDIDATE_REDSHIFTS if emulator_avg[(i, z)][1] is not None)
print(f'fit independent per-bin LambdaCDM emulators + average for {n_fit} of '
      f'{n_labels * len(CANDIDATE_REDSHIFTS)} (parameter, redshift) combinations')


def eval_emulator(dependent_cols, beta, theta_query, hold_at_fiducial=None):
    """beta[0] + sum of beta[1+idx]*(theta_fid[j]-theta_query[j]) over the
    fitted LambdaCDM columns j."""
    if beta is None:
        return np.nan
    val = beta[0]
    for idx, j in enumerate(dependent_cols):
        if hold_at_fiducial and j in hold_at_fiducial:
            continue
        val += beta[1 + idx] * (FIDUCIAL_LCDM[j] - theta_query[j])
    return val

#%% ridge-regularized replacement for fit_one_bin: ALWAYS fits all n_lcdm
# columns (no RHO_MIN/P_MAX significance gate), shrinking a column's slope
# toward zero when the data doesn't support it rather than dropping it
# from the model outright. This is what makes SLOPE(z) well-defined at
# every z for every column -- no column silently present at one z-node
# and absent at its neighbor purely from a sample-size-driven p-value.
def ridge_fit_beta(delta, y, alpha):
    """delta: (n_sim, n_lcdm) raw Delta_theta = theta_fid - theta.
    Returns beta of length n_lcdm+1: [intercept, slope_0, ..., slope_{n-1}].
    Slope columns are standardized by their std before the ridge penalty
    is applied (so one alpha penalizes all 4 LCDM columns comparably
    despite their very different physical scales -- omega_b ~0.02 vs.
    log(1e10*A_s) ~3), then converted back to per-unit-Delta_theta slopes
    on return. The intercept is NOT penalized and stays exactly the
    model's value at Delta_theta=0 (fiducial cosmology), matching the old
    hard-selection beta[0]'s meaning."""
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
    """Leave-one-out CV over alpha_grid; brute-force (cheap here -- at
    most ~50 sims, 5x5 linear solves per fold). Falls back to a mid-grid
    alpha when there are too few sims for a meaningful CV split."""
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
    """Ridge replacement for fit_one_bin. Returns
    (dependent_cols, beta, alpha_used, rho_diag):
      - dependent_cols is always list(range(n_lcdm)) -- kept only for
        format compatibility with eval_emulator / predictions.py's
        evaluation formula, which just sums over whatever list it's given.
      - rho_diag is {column: pearson rho}, for plotting/diagnostics only;
        it no longer plays any role in what gets fit.
    Returns (None, None, None, {}) if there's too little data to fit."""
    if len(sims_z) < MIN_SIMS_FOR_FIT:
        return None, None, None, {}
    theta = np.array([lcdm_pars[n] for n in sims_z])
    delta = FIDUCIAL_LCDM[None, :] - theta
    y = np.array([params[k][z][n][i] for n in sims_z])

    alpha = ridge_cv_alpha(delta, y, alpha_grid)
    beta = ridge_fit_beta(delta, y, alpha)
    rho_diag = {j: pearsonr(theta[:, j], y)[0] for j in range(n_lcdm)}
    return list(range(n_lcdm)), beta, alpha, rho_diag


# rebuild emulator_bin / emulator_avg with the ridge fits. Since both bins
# now always use dependent_cols = [0, 1, 2, 3], averaging is a plain
# elementwise mean of beta -- the old union-of-selected-columns bookkeeping
# is gone because there's no longer a "selected vs. not" distinction.
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
print(f'ridge-fit LambdaCDM emulators (all {n_lcdm} columns, every populated z) for {n_fit} of '
      f'{n_labels * len(CANDIDATE_REDSHIFTS)} (parameter, redshift) combinations')

#%% mean_params fallback -- per PARAM_NAMES row, pooled across all
# bins/z/sims. Only used by predictions.py if some row at some z-node
# genuinely has no fit at all; with ridge fitting whenever len(sims_z) >=
# MIN_SIMS_FOR_FIT, that shouldn't happen for any populated z-node, but
# the config format expects the key regardless.
mean_params_fallback = np.full(n_labels, np.nan)
for i in range(n_labels):
    vals = [params[k][z][n][i] for k in range(N_MASS) for z in CANDIDATE_REDSHIFTS
            for n in params[k][z]]
    if vals:
        mean_params_fallback[i] = np.nanmean(vals)
print('mean_params fallback:', dict(zip(PARAM_NAMES, mean_params_fallback)))


#%% generate the compact/config.py-format `compact_emulator` block as
# literal Python source, for manual review + paste. NOT written directly
# into config.py: this replaces the old 4-node, 6-parameter, hard-selected
# emulator wholesale, and fits_indiv's mass-bin keys ('low_M'/'high_M'
# strings, confirmed directly from the current config.py) are easy to get
# subtly wrong in a scripted rewrite.
BIN_KEY = {0: 'low_M', 1: 'high_M'}

def _fmt_array(arr):
    return 'np.array([' + ', '.join(f'{v:.8g}' for v in arr) + '])'

def _fmt_fits(fit_dict, z, indent=12):
    pad = ' ' * indent
    lines = []
    for i in range(n_labels):
        cols, beta = fit_dict.get((i, z), (None, None))
        if beta is None:
            continue
        lines.append(f'{pad}({i}, {cols}, {_fmt_array(beta)}),')
    return '\n'.join(lines)

_ns_idx = LCDM_NAMES.index('n_s')
out_lines = [
    "# predictor column order: [omega_b, omega_cdm, log(10^10 As), n_s]",
    f"# row order: {PARAM_NAMES}",
    "# fits are calibrated in Delta_theta = theta_fiducial - theta, NOT raw theta",
    f"FIDUCIAL_COSMO = {_fmt_array(FIDUCIAL_LCDM)}  # AbacusSummit c000",
    f"_MEAN_PARAMS = {_fmt_array(mean_params_fallback)}  # fallback only; ridge fits every row at every populated z",
    f"_NS_FID = {FIDUCIAL_LCDM[_ns_idx]!r}",
    "",
    "compact_emulator = {",
]
for z in CANDIDATE_REDSHIFTS:
    if len(common_sims_by_z[z]) < MIN_SIMS_FOR_FIT:
        continue   # no node written for z's excluded upstream (e.g. z=1.7, bin 1 zeroed out)
    out_lines.append(f"    {z}: {{")
    out_lines.append("        'ns_fiducial': _NS_FID,")
    out_lines.append("        'mean_params': _MEAN_PARAMS,")
    out_lines.append("        'fits': [")
    out_lines.append(_fmt_fits(emulator_avg, z, indent=12))
    out_lines.append("        ],")
    out_lines.append("        'fits_indiv': {")
    for k, key in BIN_KEY.items():
        out_lines.append(f"            '{key}': [")
        out_lines.append(_fmt_fits(emulator_bin[k], z, indent=16))
        out_lines.append("            ],")
    out_lines.append("        },")
    out_lines.append("    },")
out_lines.append("}")

config_src = '\n'.join(out_lines)
config_path = os.path.join(FIGURES_DIR, 'compact_emulator_config_block.py')
with open(config_path, 'w') as f:
    f.write(config_src)
print(f'wrote generated compact_emulator block to {config_path}')
print('review it, then paste it into compact/config.py in place of the '
      'existing FIDUCIAL_COSMO / _MEAN_PARAMS / compact_emulator definitions')

#%% cosmology-dependence plot: per-bin fits (dashed) + average (solid), one
# figure per redshift, restricted to LambdaCDM columns only, with rho
# annotation + green shading for panels where either bin's fit selected
# that column -- matching check_abacus_cosmo_runs_redshift.ipynb's style
from matplotlib.colors import to_rgba
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


#%% residuals with respect to the AVERAGE emulator (LambdaCDM columns only),
# one figure per redshift. Each panel reports one COMBINED statistical
# number (spread of the average emulator's prediction vs. the actual
# value, pooling both mass bins together) and one COMBINED systematic
# number (how much the mass-bin-specific fit differs from the average,
# also pooled). Y-limits are per-row, percentile-based, centered on 1.00,
# and clamped so a single blown-up outlier fit can't compress the rest of
# the row to a flat line.
Y_HALF_RANGE_MIN = 0.05    # don't zoom in tighter than +-5% even for a good fit
Y_HALF_RANGE_MAX = 0.5     # cap so a single blown-up outlier can't dominate the row
Y_PERCENTILE = 1           # use the [1, 99] percentile range, not the raw min/max

for z in CANDIDATE_REDSHIFTS:
    sims_z = common_sims_by_z[z]
    if len(sims_z) < MIN_SIMS_FOR_FIT:
        continue
    theta_z = np.array([lcdm_pars[n] for n in sims_z])
    n_sim = len(sims_z)

    # -- pass 1: gather combined (both-bin) ratios per (i, j) so y-limits
    # can be set per row before any plotting happens --
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
            panel_ratio[(i, j)] = np.concatenate(ratio_both)   # len 2*n_sim: bin0 then bin1
            panel_sys[(i, j)] = np.concatenate(sys_both)

    if not any(cols_shown_by_i.values()):
        continue

    # per-row y-limits: robust, percentile-based, centered on 1.00, clamped
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


#%% save emulator parameters for use in the pipeline later
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
                cols, beta = source[(i, z)]
                if beta is None:
                    continue
                sub = grp.create_group(f'param{i}_z{iz}')
                sub.attrs['redshift'] = z
                sub.attrs['param_index'] = i
                sub.create_dataset('dependent_cols', data=np.array(cols, dtype=np.int32))
                sub.create_dataset('beta', data=beta)
print(f'saved emulator parameters to {EMULATOR_PATH}')


#%% redshift dependence of the emulators at fiducial cosmology, interpolated
z_arr = np.array(CANDIDATE_REDSHIFTS)
fig_z, ax_z = plt.subplots(n_labels, 1, figsize=(5, 2.2 * n_labels), sharex=True, layout='constrained')
z_smooth = np.linspace(z_arr.min(), z_arr.max(), 200)

for i in range(n_labels):
    intercepts = {0: [], 1: [], 'avg': []}
    zs_available = []
    for z in CANDIDATE_REDSHIFTS:
        c0, b0 = emulator_bin[0][(i, z)]
        c1, b1 = emulator_bin[1][(i, z)]
        ca, ba = emulator_avg[(i, z)]
        if b0 is None or b1 is None:
            continue
        zs_available.append(z)
        intercepts[0].append(b0[0])
        intercepts[1].append(b1[0])
        intercepts['avg'].append(ba[0])

    zs_available = np.array(zs_available)
    if len(zs_available) < 2:
        continue
    order = np.argsort(zs_available)
    zs_sorted = zs_available[order]

    for key, color, label in ((0, 'C0', 'bin 0'), (1, 'C1', 'bin 1'), ('avg', 'purple', 'average')):
        vals = np.array(intercepts[key])[order]
        ax_z[i].plot(zs_sorted, vals, 'o', c=color, ms=5, label=label if i == 0 else None)
        if len(zs_sorted) >= 3:
            spline = InterpolatedUnivariateSpline(zs_sorted, vals, k=min(3, len(zs_sorted) - 1))
            in_range = (z_smooth >= zs_sorted.min()) & (z_smooth <= zs_sorted.max())
            ax_z[i].plot(z_smooth[in_range], spline(z_smooth[in_range]), '-', c=color, lw=1, alpha=0.7)
        else:
            ax_z[i].plot(zs_sorted, vals, '-', c=color, lw=1, alpha=0.7)
    ax_z[i].set_ylabel(labels[i])

ax_z[0].legend(fontsize=8)
ax_z[-1].set_xlabel(r'$z$')
fig_z.suptitle('emulator value at fiducial cosmology vs redshift (interpolated)')
fname = os.path.join(FIGURES_DIR, 'emulator_redshift_dependence.png')
fig_z.savefig(fname, dpi=150)
print(f'saved {fname}')

#%% redshift dependence of the emulators at fiducial cosmology, bin 0 only
# -- same data/intercepts as the previous cell, just isolating bin 0 so its
# trend can be read without bin 1's noisier line (especially relevant at
# z=1.4, where bin 1 came from the r_p-coarsened rescue rather than a
# native-resolution fit) or the average overlapping it
fig_z0, ax_z0 = plt.subplots(n_labels, 1, figsize=(5, 2.2 * n_labels), sharex=True, layout='constrained')

for i in range(n_labels):
    zs_available, vals0 = [], []
    for z in CANDIDATE_REDSHIFTS:
        c0, b0 = emulator_bin[0][(i, z)]
        if b0 is None:
            continue
        zs_available.append(z)
        vals0.append(b0[0])

    zs_available = np.array(zs_available)
    vals0 = np.array(vals0)
    if len(zs_available) < 2:
        continue
    order = np.argsort(zs_available)
    zs_sorted, vals_sorted = zs_available[order], vals0[order]

    ax_z0[i].plot(zs_sorted, vals_sorted, 'o', c='C0', ms=5)
    if len(zs_sorted) >= 3:
        spline = InterpolatedUnivariateSpline(zs_sorted, vals_sorted, k=min(3, len(zs_sorted) - 1))
        in_range = (z_smooth >= zs_sorted.min()) & (z_smooth <= zs_sorted.max())
        ax_z0[i].plot(z_smooth[in_range], spline(z_smooth[in_range]), '-', c='C0', lw=1, alpha=0.7)
    else:
        ax_z0[i].plot(zs_sorted, vals_sorted, '-', c='C0', lw=1, alpha=0.7)
    ax_z0[i].set_ylabel(labels[i])

ax_z0[-1].set_xlabel(r'$z$')
fig_z0.suptitle('emulator value at fiducial cosmology vs redshift, bin 0 only')
fname = os.path.join(FIGURES_DIR, 'emulator_redshift_dependence_bin0.png')
fig_z0.savefig(fname, dpi=150)
print(f'saved {fname}')

#%% ridge covariance at the already-selected alpha (reuses alpha_bin from
# the fitting cell -- does NOT re-run the CV search). Needed because
# neither fit_one_bin nor fit_one_bin_ridge ever computed an uncertainty;
# this adds it after the fact, purely for the plots below.
def ridge_beta_cov(delta, y, alpha):
    """Sandwich covariance of the ridge estimator at a fixed alpha. Also
    returns beta as a byproduct (should match what fit_one_bin_ridge
    already produced for this alpha -- not used, just a consistency
    cross-check available if needed)."""
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
    edf = np.trace(H)                        # effective dof used by the ridge fit (<= n_col+1)
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

#%% shared helper: robust y-limits via median + scaled MAD (median absolute
# deviation), not percentiles. With only ~21 points per panel (3 series x
# 7 redshifts), a 2nd/98th percentile cutoff sits almost exactly on the
# extreme points themselves -- percentiles only become robust once the
# tails are sparse relative to the sample, which isn't true here. MAD
# instead measures distance from the bulk directly, so a single wildly
# divergent point (e.g. the z=1.4 high-mass fits) gets excluded regardless
# of how few points there are.
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
        robust_sigma = 1.4826 * mad   # MAD -> std-equivalent for a normal distribution
        lo, hi = med - mad_k * robust_sigma, med + mad_k * robust_sigma

    # never expand PAST the actual data range, only ever tighten into it
    lo, hi = max(lo, vals.min()), min(hi, vals.max())
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


#%% intercepts vs z, with error bars, y-limited per row
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
    ylim = robust_ylim(row_vals)
    if ylim is not None:
        ax_int[i].set_ylim(*ylim)
    ax_int[i].set_ylabel(labels[i])

ax_int[0].legend(fontsize=8)
ax_int[-1].set_xlabel(r'$z$')
fig_int.suptitle('Compact-model parameter value at fiducial cosmology vs redshift')
fname = os.path.join(FIGURES_DIR, 'emulator_intercept_vs_redshift.png')
fig_int.savefig(fname, dpi=150)
print(f'saved {fname}')


#%% slopes vs z, y-limited per panel (each (param, cosmo-column) panel can
# have a very different natural scale, so limits are set per-panel rather
# than per-row here, unlike the intercept plot above)
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
        ylim = robust_ylim(panel_vals + [np.array([0.0])])   # include 0 so the reference line stays visible
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

#%% bin 0's own z=1.7 fit, independent of common_sims_by_z's both-bins
# requirement -- recovers a point that's been silently missing from
# every emulator/plot so far. Purely additive to what's already computed;
# doesn't touch emulator_bin/emulator_avg.
EXTRA_Z_BIN0 = [z for z in CANDIDATE_REDSHIFTS
               if len(common_sims_by_z.get(z, [])) < MIN_SIMS_FOR_FIT
               and len(params[0].get(z, {})) >= MIN_SIMS_FOR_FIT]

bin0_extra_beta, bin0_extra_err_int, bin0_extra_err_slope = {}, {}, {}
for z in EXTRA_Z_BIN0:
    sims_z0 = sorted(params[0].get(z, {}))
    for i in range(n_labels):
        cols, beta, alpha, _ = fit_one_bin_ridge(sims_z0, 0, z, i)
        bin0_extra_beta[(i, z)] = beta
        if beta is not None:
            theta = np.array([lcdm_pars[n] for n in sims_z0])
            delta = FIDUCIAL_LCDM[None, :] - theta
            y = np.array([params[0][z][n][i] for n in sims_z0])
            _, cov = ridge_beta_cov(delta, y, alpha)
            se = np.sqrt(np.diag(cov))
            bin0_extra_err_int[(i, z)] = se[0]
            bin0_extra_err_slope[(i, z)] = se[1:]
print(f'recovered bin-0-only fits at z={EXTRA_Z_BIN0} '
      f'(previously dropped everywhere by the common_sims_by_z intersection)')


#%% reconstruction helper: bin 1 unchanged for z<=0.5, tracks bin 0's
# z-to-z changes beyond that at a fixed offset. This is a STATED
# EXTRAPOLATION ASSUMPTION -- bin 1 evolves with z the way bin 0 does,
# at a roughly constant offset -- not a measurement of bin 1 at high z.
# Treat it as illustrative, especially at z=1.7 where it's now the ONLY
# thing standing in for a genuine bin-1 fit.
Z_ANCHOR_MAX = 0.5

def build_bin1_reconstruction(get_val, get_err):
    z_low = [z for z in CANDIDATE_REDSHIFTS if z <= Z_ANCHOR_MAX]
    diffs = []
    for z in z_low:
        v0, v1 = get_val(0, z), get_val(1, z)
        if v0 is not None and v1 is not None:
            diffs.append(v1 - v0)
    if not diffs:
        return {}, {}
    shift = np.mean(diffs)
    shift_err = (np.std(diffs, ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else 0.0

    recon_vals, recon_errs = {}, {}
    for z in CANDIDATE_REDSHIFTS:
        if z <= Z_ANCHOR_MAX:
            v1 = get_val(1, z)
            if v1 is not None:
                recon_vals[z] = v1
                recon_errs[z] = get_err(1, z) or 0.0
        else:
            v0 = get_val(0, z)
            if v0 is not None:
                recon_vals[z] = v0 + shift
                e0 = get_err(0, z) or 0.0
                recon_errs[z] = np.sqrt(e0 ** 2 + shift_err ** 2)   # propagates the shift's own uncertainty too
    return recon_vals, recon_errs


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

#%% intercepts vs z: bin 0 unchanged (now including its own z=1.7),
# bin 1 reconstructed beyond z=0.5
fig_int2, ax_int2 = plt.subplots(n_labels, 1, figsize=(5, 2.2 * n_labels), sharex=True, layout='constrained')

for i in range(n_labels):
    zs0 = [z for z in CANDIDATE_REDSHIFTS if get_val_int(0, z, i) is not None]
    vals0 = [get_val_int(0, z, i) for z in zs0]
    errs0 = [get_err_int(0, z, i) or 0.0 for z in zs0]
    ax_int2[i].errorbar(zs0, vals0, yerr=errs0, marker='o', ms=4, capsize=2,
                        c='C0', ls='--', label='bin 0 (unchanged)' if i == 0 else None)

    recon_vals, recon_errs = build_bin1_reconstruction(lambda k, z: get_val_int(k, z, i),
                                                        lambda k, z: get_err_int(k, z, i))
    zs1_real = sorted(z for z in recon_vals if z <= Z_ANCHOR_MAX)
    zs1_recon = sorted(z for z in recon_vals if z > Z_ANCHOR_MAX)
    if zs1_real:
        ax_int2[i].errorbar(zs1_real, [recon_vals[z] for z in zs1_real],
                            yerr=[recon_errs[z] for z in zs1_real],
                            marker='o', ms=4, capsize=2, c='C1', ls='-',
                            label='bin 1 (measured, $z\\leq0.5$)' if i == 0 else None)
    if zs1_recon:
        anchor = [max(zs1_real)] if zs1_real else []
        zs_plot = anchor + zs1_recon
        ax_int2[i].errorbar(zs_plot, [recon_vals[z] for z in zs_plot],
                            yerr=[recon_errs[z] for z in zs_plot],
                            marker='o', ms=4, mfc='none', capsize=2, c='C1', ls=':',
                            label='bin 1 (reconstructed, $z>0.5$)' if i == 0 else None)

    ax_int2[i].axvline(Z_ANCHOR_MAX, c='gray', ls=':', lw=1, alpha=0.5)
    ylim = robust_ylim([np.array(vals0), np.array(list(recon_vals.values()) or [0.0])])
    if ylim is not None:
        ax_int2[i].set_ylim(*ylim)
    ax_int2[i].set_ylabel(labels[i])

ax_int2[0].legend(fontsize=7)
ax_int2[-1].set_xlabel(r'$z$')
fig_int2.suptitle('Compact-model parameter value at fiducial cosmology vs redshift\n'
                  '(bin 1 reconstructed beyond $z=0.5$ from bin 0 slope + fixed shift)')
fname = os.path.join(FIGURES_DIR, 'emulator_intercept_vs_redshift_bin1_reconstructed.png')
fig_int2.savefig(fname, dpi=150)
print(f'saved {fname}')

#%% slopes vs z: same bin-0-unchanged / bin-1-reconstructed treatment,
# per (i, j) panel -- shift is fit independently per cosmological column
fig_slope2, ax_slope2 = plt.subplots(n_labels, n_lcdm, figsize=(3.2 * n_lcdm, 2.2 * n_labels),
                                     sharex='col', layout='constrained')

for i in range(n_labels):
    for j in range(n_lcdm):
        zs0 = [z for z in CANDIDATE_REDSHIFTS if get_val_slope(0, z, i, j) is not None]
        vals0 = [get_val_slope(0, z, i, j) for z in zs0]
        errs0 = [get_err_slope(0, z, i, j) or 0.0 for z in zs0]
        ax_slope2[i, j].errorbar(zs0, vals0, yerr=errs0, marker='o', ms=4, capsize=2, c='C0', ls='--')

        recon_vals, recon_errs = build_bin1_reconstruction(lambda k, z: get_val_slope(k, z, i, j),
                                                            lambda k, z: get_err_slope(k, z, i, j))
        zs1_real = sorted(z for z in recon_vals if z <= Z_ANCHOR_MAX)
        zs1_recon = sorted(z for z in recon_vals if z > Z_ANCHOR_MAX)
        if zs1_real:
            ax_slope2[i, j].errorbar(zs1_real, [recon_vals[z] for z in zs1_real],
                                     yerr=[recon_errs[z] for z in zs1_real],
                                     marker='o', ms=4, capsize=2, c='C1', ls='-')
        if zs1_recon:
            anchor = [max(zs1_real)] if zs1_real else []
            zs_plot = anchor + zs1_recon
            ax_slope2[i, j].errorbar(zs_plot, [recon_vals[z] for z in zs_plot],
                                     yerr=[recon_errs[z] for z in zs_plot],
                                     marker='o', ms=4, mfc='none', capsize=2, c='C1', ls=':')

        ax_slope2[i, j].axhline(0, c='gray', ls='--', lw=1)
        ax_slope2[i, j].axvline(Z_ANCHOR_MAX, c='gray', ls=':', lw=1, alpha=0.5)
        ylim = robust_ylim([np.array(vals0), np.array(list(recon_vals.values()) or [0.0]), np.array([0.0])])
        if ylim is not None:
            ax_slope2[i, j].set_ylim(*ylim)
        if i == n_labels - 1:
            ax_slope2[i, j].set_xlabel(r'$z$')
    ax_slope2[i, 0].set_ylabel(labels[i])

for j in range(n_lcdm):
    ax_slope2[0, j].set_title(LCDM_LABELS[j])

fig_slope2.suptitle(r'Cosmology-dependence slope ($\partial\,\mathrm{param}/\partial\Delta\theta$) vs redshift'
                    '\n(bin 1 reconstructed beyond $z=0.5$ from bin 0 slope + fixed shift)')
fname = os.path.join(FIGURES_DIR, 'emulator_slope_vs_redshift_bin1_reconstructed.png')
fig_slope2.savefig(fname, dpi=150)
print(f'saved {fname}')

#%% formal check: is the z<=0.5 anchor shift consistent with the REAL
# (non-reconstructed) offset at z=0.8/1.1, given both uncertainties?
# Replaces eyeballing "does this point look off-trend" with an actual
# z-score, so a wide error bar correctly counts as "uninformative" rather
# than either "confirms" or "falsifies" the tracking assumption.
CHECK_Z = [0.8, 1.1]

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

print(f"{'param':>10s}  {'z':>4s}  {'offset':>10s}  {'shift(z<=0.5)':>14s}  {'z-score':>8s}  verdict")
for i in range(n_labels):
    shift, shift_err = anchor_shift_and_err(lambda k, z: get_val_int(k, z, i),
                                            lambda k, z: get_err_int(k, z, i))
    if shift is None:
        continue
    for z in CHECK_Z:
        v0, v1 = get_val_int(0, z, i), get_val_int(1, z, i)
        e0, e1 = get_err_int(0, z, i), get_err_int(1, z, i)
        if v0 is None or v1 is None:
            continue
        offset = v1 - v0
        offset_err = np.sqrt((e0 or 0.0) ** 2 + (e1 or 0.0) ** 2)
        zscore = (offset - shift) / np.sqrt(offset_err ** 2 + shift_err ** 2)
        verdict = ('uninformative (err too large)' if offset_err > 3 * abs(shift_err) + abs(shift)
                  else 'consistent' if abs(zscore) < 2 else 'inconsistent')
        print(f"{PARAM_NAMES[i]:>10s}  {z:>4}  {offset:>10.3g}  {shift:>14.3g}  {zscore:>8.2f}  {verdict}")

#%% redshift-dependent shift trend, fit from ALL real bin-1 data through
# z=1.1 (weighted by combined uncertainty), not just the z<=0.5 anchor --
# this is what lets A_rho0/A_delta0's real, tightening drift show up in
# the extrapolation instead of being averaged away by a flat mean.
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


# rows with a real, resolvable trend through z=1.1 (per the z-score
# check); A_rho_p was too mixed at 1.1 to include by default, everything
# else was uninformative or inconsistent even near the anchor
TREND_ROWS = [PARAM_NAMES.index(p) for p in ('sigma0', 'A_rho0', 'sigma_rho', 'A_delta0')]


def build_bin1_reconstruction_v2(get_val, get_err, row_idx):
    """Same output shape as build_bin1_reconstruction. For TREND_ROWS:
    real bin-1 data is kept through z=1.1 (not just 0.5), and z>1.1 is
    reconstructed from the FITTED TREND line, not a flat mean. For every
    other row: unchanged flat z<=0.5 anchor scheme -- their own data
    wasn't informative enough to support fitting a trend at all."""
    use_trend = row_idx in TREND_ROWS
    trend = fit_weighted_shift_trend(get_val, get_err, z_max=1.1) if use_trend else None
    real_cutoff = 1.1 if use_trend else Z_ANCHOR_MAX

    recon_vals, recon_errs = {}, {}
    for z in CANDIDATE_REDSHIFTS:
        if z <= real_cutoff:
            v1 = get_val(1, z)
            if v1 is not None:
                recon_vals[z] = v1
                recon_errs[z] = get_err(1, z) or 0.0
            continue
        v0 = get_val(0, z)
        if v0 is None:
            continue
        e0 = get_err(0, z) or 0.0
        if use_trend:
            shift, shift_err = eval_shift_trend(trend, z)
        else:
            shift, shift_err = anchor_shift_and_err(get_val, get_err)   # old flat scheme
        recon_vals[z] = v0 + shift
        recon_errs[z] = np.sqrt(e0 ** 2 + shift_err ** 2)
    return recon_vals, recon_errs
#%% intercepts vs z: bin 0 unchanged, bin 1 real data kept through its own
# reliable cutoff (z<=1.1 for the trend-fit rows, z<=0.5 otherwise),
# reconstructed beyond that using build_bin1_reconstruction_v2
fig_int3, ax_int3 = plt.subplots(n_labels, 1, figsize=(5, 2.2 * n_labels), sharex=True, layout='constrained')

for i in range(n_labels):
    zs0 = [z for z in CANDIDATE_REDSHIFTS if get_val_int(0, z, i) is not None]
    vals0 = [get_val_int(0, z, i) for z in zs0]
    errs0 = [get_err_int(0, z, i) or 0.0 for z in zs0]
    ax_int3[i].errorbar(zs0, vals0, yerr=errs0, marker='o', ms=4, capsize=2,
                        c='C0', ls='--', label='bin 0 (unchanged)' if i == 0 else None)

    real_cutoff = 1.1 if i in TREND_ROWS else Z_ANCHOR_MAX
    recon_vals, recon_errs = build_bin1_reconstruction_v2(lambda k, z: get_val_int(k, z, i),
                                                          lambda k, z: get_err_int(k, z, i), row_idx=i)
    zs1_real = sorted(z for z in recon_vals if z <= real_cutoff)
    zs1_recon = sorted(z for z in recon_vals if z > real_cutoff)
    if zs1_real:
        ax_int3[i].errorbar(zs1_real, [recon_vals[z] for z in zs1_real],
                            yerr=[recon_errs[z] for z in zs1_real],
                            marker='o', ms=4, capsize=2, c='C1', ls='-',
                            label='bin 1 (measured)' if i == 0 else None)
    if zs1_recon:
        anchor = [max(zs1_real)] if zs1_real else []
        zs_plot = anchor + zs1_recon
        ax_int3[i].errorbar(zs_plot, [recon_vals[z] for z in zs_plot],
                            yerr=[recon_errs[z] for z in zs_plot],
                            marker='o', ms=4, mfc='none', capsize=2, c='C1', ls=':',
                            label='bin 1 (reconstructed)' if i == 0 else None)

    ax_int3[i].axvline(real_cutoff, c='gray', ls=':', lw=1, alpha=0.5)
    ylim = robust_ylim([np.array(vals0), np.array(list(recon_vals.values()) or [0.0])])
    if ylim is not None:
        ax_int3[i].set_ylim(*ylim)
    ax_int3[i].set_ylabel(labels[i])

ax_int3[0].legend(fontsize=7)
ax_int3[-1].set_xlabel(r'$z$')
fig_int3.suptitle('Compact-model parameter value at fiducial cosmology vs redshift\n'
                  r'(bin 1: trend-extrapolated for $\sigma_0, A_{\rho,0}, \sigma_\rho, A_{\Delta,0}$'
                  ' past $z{=}1.1$; flat-shift for the rest past $z{=}0.5$)')
fname = os.path.join(FIGURES_DIR, 'emulator_intercept_vs_redshift_bin1_trend.png')
fig_int3.savefig(fname, dpi=150)
print(f'saved {fname}')


#%% slopes vs z: same treatment, per (i, j) panel
fig_slope3, ax_slope3 = plt.subplots(n_labels, n_lcdm, figsize=(3.2 * n_lcdm, 2.2 * n_labels),
                                     sharex='col', layout='constrained')

for i in range(n_labels):
    real_cutoff = 1.1 if i in TREND_ROWS else Z_ANCHOR_MAX
    for j in range(n_lcdm):
        zs0 = [z for z in CANDIDATE_REDSHIFTS if get_val_slope(0, z, i, j) is not None]
        vals0 = [get_val_slope(0, z, i, j) for z in zs0]
        errs0 = [get_err_slope(0, z, i, j) or 0.0 for z in zs0]
        ax_slope3[i, j].errorbar(zs0, vals0, yerr=errs0, marker='o', ms=4, capsize=2, c='C0', ls='--')

        recon_vals, recon_errs = build_bin1_reconstruction_v2(
            lambda k, z: get_val_slope(k, z, i, j), lambda k, z: get_err_slope(k, z, i, j), row_idx=i)
        zs1_real = sorted(z for z in recon_vals if z <= real_cutoff)
        zs1_recon = sorted(z for z in recon_vals if z > real_cutoff)
        if zs1_real:
            ax_slope3[i, j].errorbar(zs1_real, [recon_vals[z] for z in zs1_real],
                                     yerr=[recon_errs[z] for z in zs1_real],
                                     marker='o', ms=4, capsize=2, c='C1', ls='-')
        if zs1_recon:
            anchor = [max(zs1_real)] if zs1_real else []
            zs_plot = anchor + zs1_recon
            ax_slope3[i, j].errorbar(zs_plot, [recon_vals[z] for z in zs_plot],
                                     yerr=[recon_errs[z] for z in zs_plot],
                                     marker='o', ms=4, mfc='none', capsize=2, c='C1', ls=':')

        ax_slope3[i, j].axhline(0, c='gray', ls='--', lw=1)
        ax_slope3[i, j].axvline(real_cutoff, c='gray', ls=':', lw=1, alpha=0.5)
        ylim = robust_ylim([np.array(vals0), np.array(list(recon_vals.values()) or [0.0]), np.array([0.0])])
        if ylim is not None:
            ax_slope3[i, j].set_ylim(*ylim)
        if i == n_labels - 1:
            ax_slope3[i, j].set_xlabel(r'$z$')
    ax_slope3[i, 0].set_ylabel(labels[i])

for j in range(n_lcdm):
    ax_slope3[0, j].set_title(LCDM_LABELS[j])

fig_slope3.suptitle(r'Cosmology-dependence slope ($\partial\,\mathrm{param}/\partial\Delta\theta$) vs redshift'
                    '\n(bin 1: trend-extrapolated amplitude rows past $z{=}1.1$, flat-shift rest past $z{=}0.5$)')
fname = os.path.join(FIGURES_DIR, 'emulator_slope_vs_redshift_bin1_trend.png')
fig_slope3.savefig(fname, dpi=150)
print(f'saved {fname}')

#%% family-informed reconstruction, v2: FIXED cutoffs per row type.
# Driver rows (sigma0 for sigma_v family, A_delta0 for Delta family): real
# data trusted through z<=1.1, trend fit over that range, extrapolated
# beyond. All other rows: real data trusted only through z<=0.5 (bin-1's
# own 0.8/1.1 fits are DISCARDED here even when they exist, per the
# z-score check showing those rows are inconsistent-or-uninformative past
# the anchor) -- reconstructed from 0.5 onward using the driver's
# fractional trend growth relative to the driver's own z<=0.5 anchor.
DRIVER_REAL_CUTOFF = 1.1

def get_real_cutoff(row_idx):
    family = ROW_FAMILY.get(row_idx)
    is_driver = family is not None and row_idx == FAMILY_DRIVER_ROW[family]
    return DRIVER_REAL_CUTOFF if is_driver else Z_ANCHOR_MAX


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

    # non-driver: this row's own z<=0.5 anchor, scaled by the family
    # driver's fractional trend growth relative to the DRIVER's z<=0.5 anchor
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
#%% intercepts vs z: bin 0 unchanged; bin 1 real through each row's own
# fixed cutoff (1.1 for the two family drivers, 0.5 for everything else),
# family-scaled reconstruction beyond that
fig_int5, ax_int5 = plt.subplots(n_labels, 1, figsize=(5, 2.2 * n_labels), sharex=True, layout='constrained')

for i in range(n_labels):
    zs0 = [z for z in CANDIDATE_REDSHIFTS if get_val_int(0, z, i) is not None]
    vals0 = [get_val_int(0, z, i) for z in zs0]
    errs0 = [get_err_int(0, z, i) or 0.0 for z in zs0]
    ax_int5[i].errorbar(zs0, vals0, yerr=errs0, marker='o', ms=4, capsize=2,
                        c='C0', ls='--', label='bin 0 (unchanged)' if i == 0 else None)

    family = ROW_FAMILY.get(i)
    driver_row = FAMILY_DRIVER_ROW[family] if family is not None else i
    recon_vals, recon_errs = build_bin1_reconstruction_family2(
        lambda k, z: get_val_int(k, z, i), lambda k, z: get_err_int(k, z, i),
        lambda k, z: get_val_int(k, z, driver_row), lambda k, z: get_err_int(k, z, driver_row), row_idx=i)

    cutoff = get_real_cutoff(i)
    zs1_real = sorted(z for z in recon_vals if z <= cutoff)
    zs1_recon = sorted(z for z in recon_vals if z > cutoff)
    if zs1_real:
        ax_int5[i].errorbar(zs1_real, [recon_vals[z] for z in zs1_real],
                            yerr=[recon_errs[z] for z in zs1_real],
                            marker='o', ms=4, capsize=2, c='C1', ls='-',
                            label='bin 1 (measured)' if i == 0 else None)
    if zs1_recon:
        anchor = [max(zs1_real)] if zs1_real else []
        zs_plot = anchor + zs1_recon
        ax_int5[i].errorbar(zs_plot, [recon_vals[z] for z in zs_plot],
                            yerr=[recon_errs[z] for z in zs_plot],
                            marker='o', ms=4, mfc='none', capsize=2, c='C1', ls=':',
                            label='bin 1 (reconstructed)' if i == 0 else None)
    ax_int5[i].axvline(cutoff, c='gray', ls=':', lw=1, alpha=0.5)

    ylim = robust_ylim([np.array(vals0), np.array(list(recon_vals.values()) or [0.0])])
    if ylim is not None:
        ax_int5[i].set_ylim(*ylim)
    ax_int5[i].set_ylabel(labels[i])

ax_int5[0].legend(fontsize=7)
ax_int5[-1].set_xlabel(r'$z$')
fig_int5.suptitle(r'Compact-model parameter value at fiducial cosmology vs redshift'
                  '\n($\\sigma_0$/$A_{\\Delta,0}$: real through $z{=}1.1$; other rows: real through '
                  '$z{=}0.5$, scaled by their family driver beyond it)')
fname = os.path.join(FIGURES_DIR, 'emulator_intercept_vs_redshift_bin1_family_v2.png')
fig_int5.savefig(fname, dpi=150)
print(f'saved {fname}')


#%% slopes vs z: same fixed-cutoff, family-scaled treatment, per (i, j) panel
fig_slope5, ax_slope5 = plt.subplots(n_labels, n_lcdm, figsize=(3.2 * n_lcdm, 2.2 * n_labels),
                                     sharex='col', layout='constrained')

for i in range(n_labels):
    family = ROW_FAMILY.get(i)
    driver_row = FAMILY_DRIVER_ROW[family] if family is not None else i
    cutoff = get_real_cutoff(i)
    for j in range(n_lcdm):
        zs0 = [z for z in CANDIDATE_REDSHIFTS if get_val_slope(0, z, i, j) is not None]
        vals0 = [get_val_slope(0, z, i, j) for z in zs0]
        errs0 = [get_err_slope(0, z, i, j) or 0.0 for z in zs0]
        ax_slope5[i, j].errorbar(zs0, vals0, yerr=errs0, marker='o', ms=4, capsize=2, c='C0', ls='--')

        recon_vals, recon_errs = build_bin1_reconstruction_family2(
            lambda k, z: get_val_slope(k, z, i, j), lambda k, z: get_err_slope(k, z, i, j),
            lambda k, z: get_val_slope(k, z, driver_row, j), lambda k, z: get_err_slope(k, z, driver_row, j),
            row_idx=i)

        zs1_real = sorted(z for z in recon_vals if z <= cutoff)
        zs1_recon = sorted(z for z in recon_vals if z > cutoff)
        if zs1_real:
            ax_slope5[i, j].errorbar(zs1_real, [recon_vals[z] for z in zs1_real],
                                     yerr=[recon_errs[z] for z in zs1_real],
                                     marker='o', ms=4, capsize=2, c='C1', ls='-')
        if zs1_recon:
            anchor = [max(zs1_real)] if zs1_real else []
            zs_plot = anchor + zs1_recon
            ax_slope5[i, j].errorbar(zs_plot, [recon_vals[z] for z in zs_plot],
                                     yerr=[recon_errs[z] for z in zs_plot],
                                     marker='o', ms=4, mfc='none', capsize=2, c='C1', ls=':')
        ax_slope5[i, j].axvline(cutoff, c='gray', ls=':', lw=1, alpha=0.5)

        ax_slope5[i, j].axhline(0, c='gray', ls='--', lw=1)
        ylim = robust_ylim([np.array(vals0), np.array(list(recon_vals.values()) or [0.0]), np.array([0.0])])
        if ylim is not None:
            ax_slope5[i, j].set_ylim(*ylim)
        if i == n_labels - 1:
            ax_slope5[i, j].set_xlabel(r'$z$')
    ax_slope5[i, 0].set_ylabel(labels[i])

for j in range(n_lcdm):
    ax_slope5[0, j].set_title(LCDM_LABELS[j])

fig_slope5.suptitle(r'Cosmology-dependence slope ($\partial\,\mathrm{param}/\partial\Delta\theta$) vs redshift'
                    '\n(driver rows real through $z{=}1.1$; others real through $z{=}0.5$, family-scaled beyond)')
fname = os.path.join(FIGURES_DIR, 'emulator_slope_vs_redshift_bin1_family_v2.png')
fig_slope5.savefig(fname, dpi=150)
print(f'saved {fname}')

#%% bin-1 line source for the cosmology-dependence-style check plot: reuses
# build_bin1_reconstruction_family2 exactly as validated in the previous
# cells -- NOT a separate implementation -- and just packages its (per-z)
# intercept + per-column slopes into the same beta format eval_emulator
# already expects, so the line can be drawn across a panel's full x-range.

def get_bin1_beta_for_plot(i, z, driver_row):
    """Returns (beta, is_real). beta is None if there's nothing to plot
    at this (i, z) at all. is_real reflects whether z is within this
    row's own real-data cutoff (from get_real_cutoff) -- used purely to
    pick the line style below, same real/reconstructed distinction as the
    earlier reconstruction plots."""
    recon_int, _ = build_bin1_reconstruction_family2(
        lambda k, zz: get_val_int(k, zz, i), lambda k, zz: get_err_int(k, zz, i),
        lambda k, zz: get_val_int(k, zz, driver_row), lambda k, zz: get_err_int(k, zz, driver_row), row_idx=i)
    if z not in recon_int:
        return None, False
    slopes = np.full(n_lcdm, np.nan)
    for j in range(n_lcdm):
        recon_slope, _ = build_bin1_reconstruction_family2(
            lambda k, zz: get_val_slope(k, zz, i, j), lambda k, zz: get_err_slope(k, zz, i, j),
            lambda k, zz: get_val_slope(k, zz, driver_row, j), lambda k, zz: get_err_slope(k, zz, driver_row, j),
            row_idx=i)
        if z in recon_slope:
            slopes[j] = recon_slope[z]
    beta = np.concatenate([[recon_int[z]], slopes])
    return beta, z <= get_real_cutoff(i)


def get_bin0_beta_for_plot(i, z):
    """bin 0's own beta at z, falling back to the separately-recovered
    z=1.7 fit (bin0_extra_beta) for the one z outside common_sims_by_z."""
    beta = emulator_bin[0].get((i, z), (None, None))[1]
    return beta if beta is not None else bin0_extra_beta.get((i, z))

#%% family definitions -- sigma_v family driven by sigma0, Delta family
# driven by A_delta0. These were defined but never actually pasted into
# the consolidated script; reconstructing here from PARAM_NAMES so it's
# not order-dependent on n_labels having been set elsewhere.
n_labels = len(PARAM_NAMES)   # harmless no-op if already defined identically

_SIGMA_ROWS = ['sigma0', 'A_rho0', 'A_rho_p', 'sigma_rho']
_DELTA_ROWS = ['A_delta0', 'r_delta', 'sigma_delta', 'beta_delta', 'alpha']

FAMILY_DRIVER_ROW = {'sigma': PARAM_NAMES.index('sigma0'), 'delta': PARAM_NAMES.index('A_delta0')}
ROW_FAMILY = {}
for _name in _SIGMA_ROWS:
    ROW_FAMILY[PARAM_NAMES.index(_name)] = 'sigma'
for _name in _DELTA_ROWS:
    ROW_FAMILY[PARAM_NAMES.index(_name)] = 'delta'

MIN_ANCHOR_MAG = 1e-8   # guard against dividing by a near-zero driver anchor shift


LCDM_LABELS = [r"$\omega_b$", r"$\omega_{\rm cdm}$", r"$\log(10^{10}A_s)$", r"$n_s$"]
LCDM_NAMES = ['omega_b', 'omega_cdm', 'ln_1e10_A_s', 'n_s']
n_lcdm = len(LCDM_LABELS)
#%% cosmology-dependence check plot at high z: real sim-level data points
# (wherever they exist, independent of common_sims_by_z or the anchor
# cutoffs) overlaid with each row's ACTUAL line source -- real per-bin
# fits where available, family-scaled reconstruction otherwise -- so the
# reconstruction can be judged visually against whatever real bin-1 data
# survives at that z, not just trusted by construction.
HIGH_Z_CHECK = [1.1, 1.4, 1.7]

for z in HIGH_Z_CHECK:
    fig, ax = plt.subplots(n_labels, n_lcdm, figsize=(3.2 * n_lcdm, 2.6 * n_labels),
                           sharex='col', layout='constrained')

    for i in range(n_labels):
        family = ROW_FAMILY.get(i)
        driver_row = FAMILY_DRIVER_ROW[family] if family is not None else i
        beta0 = get_bin0_beta_for_plot(i, z)
        beta1, bin1_is_real = get_bin1_beta_for_plot(i, z, driver_row)
        beta_avg = (beta0 + beta1) / 2.0 if (beta0 is not None and beta1 is not None) else None

        sims0 = sorted(params[0].get(z, {}))
        sims1 = sorted(params[1].get(z, {}))
        theta0 = np.array([lcdm_pars[n] for n in sims0]) if sims0 else None
        theta1 = np.array([lcdm_pars[n] for n in sims1]) if sims1 else None
        y0 = np.array([params[0][z][n][i] for n in sims0]) if sims0 else None
        y1 = np.array([params[1][z][n][i] for n in sims1]) if sims1 else None
        theta_ref = theta0 if theta0 is not None else theta1

        for j in range(n_lcdm):
            axij = ax[i, j]
            if theta0 is not None:
                axij.plot(theta0[:, j], y0, 'o', ms=4, alpha=0.5, c='C0')
            if theta1 is not None:
                axij.plot(theta1[:, j], y1, 'o', ms=4, alpha=0.5, c='C1')

            if theta0 is not None and len(theta0) > 3:
                axij.annotate(rf'$\rho={pearsonr(theta0[:, j], y0)[0]:.2f}$', xy=(0.05, 0.92),
                             xycoords='axes fraction', fontsize=6, color='C0')
            if theta1 is not None and len(theta1) > 3:
                axij.annotate(rf'$\rho={pearsonr(theta1[:, j], y1)[0]:.2f}$', xy=(0.05, 0.82),
                             xycoords='axes fraction', fontsize=6, color='C1')

            if theta_ref is not None:
                x_smooth = np.linspace(theta_ref[:, j].min(), theta_ref[:, j].max(), 50)
                q = np.tile(np.nanmean(theta_ref, axis=0), (50, 1))
                q[:, j] = x_smooth
                if beta0 is not None:
                    y_b0 = [eval_emulator(list(range(n_lcdm)), beta0, q[m]) for m in range(50)]
                    axij.plot(x_smooth, y_b0, '--', c='C0', lw=1)
                if beta1 is not None:
                    bin1_style = '--' if bin1_is_real else ':'
                    y_b1 = [eval_emulator(list(range(n_lcdm)), beta1, q[m]) for m in range(50)]
                    axij.plot(x_smooth, y_b1, bin1_style, c='C1', lw=1 if bin1_is_real else 1.5)
                if beta_avg is not None:
                    y_avg = [eval_emulator(list(range(n_lcdm)), beta_avg, q[m]) for m in range(50)]
                    axij.plot(x_smooth, y_avg, '-', c='purple', lw=2)

            if i == n_labels - 1:
                axij.set_xlabel(LCDM_LABELS[j])

        row_label = labels[i] + ('' if bin1_is_real else '\n(bin1 reconstructed)')
        ax[i, 0].set_ylabel(row_label, fontsize=8)

    fig.suptitle(rf'cosmology dependence at $z={z:g}$ ($\Lambda$CDM only) -- '
                rf'dashed: real per-bin fits, dotted: reconstructed bin 1, solid: average')
    fname = os.path.join(FIGURES_DIR, f'cosmo_dep_bin1_reconstruction_check_z{z:.3f}'.replace('.', 'p') + '.png')
    fig.savefig(fname, dpi=150)
    print(f'saved {fname}')