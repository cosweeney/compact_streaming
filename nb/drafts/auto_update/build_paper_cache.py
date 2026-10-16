"""
Builds the `auto/` groups of paper_figs_cache.h5 that make_paper_figs.py reads. RUN ON THE CLUSTER.
Not run by me (no access to /spiffball) -- treat as a first draft to check.

    python build_paper_cache.py                 # all sections, in order
    python build_paper_cache.py fig7            # just one (each section overwrites only its own group)

Sections (fig7 needs fig1 done first, for b1/b2):
  fig0   stacked P(v_los), mass bin K_FIG0
  fig1   phase-averaged xi_hh and v_r per mass bin, (b1, b2, eps_xi, eps_v) fits, model curves
  fig23  single phase / mass bin: sigma_v grid, free-nu Delta grid, joint fits, per-r_perp amplitudes
  fig5   per (phase, mass bin) 11 parameters: sigma_v (4) + Delta (5, from free-nu skew-t fits) + hill2 nu(r) (2)
  fig6   z=0.2 cosmology-grid fits (cosmo_grid_fit_results_numod.h5) + each sim's LCDM cosmology, for Fig 6
  lpt_sigma  measured <v_los^2> grid (mass bin K_SINGLE, same phase as fig23) and the velocileptors prediction of it
  xi_s   redshift-space xi^s(s_perp, s_par) maps AND multipoles ell=0,2,4 out to s=130, mass bin K_XIS: simulation read from the
         per-phase files of measure_xi_s_direct.py (run that first, in the pycorr environment); skew-t and Gaussian from the streaming integral
  fit_multipoles  fit the 13 parameters (b1, b2 + 11 compact) to the simulation xi_0, xi_2 (needs section xi_s), then predict xi_0, xi_2, xi_4
  fig7   histograms (stacked over phases) for mass bin K_FIG7 at the selected cells, plus the LPT mean v_r

The 11 parameters follow the recipe of the nu-test script: free-nu skew-t per cell (every STRIDE-th row/column),
Delta = xi_free - measured <v_r|r> r_los / r, hill2 nu(r) fit to per-r-bin medians of the free nu values.
All model forms come from compact.model (cm); the fixed pivot is cm.R_PIV_FIXED.
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')
from types import SimpleNamespace

import numpy as np
import h5py as h5
from scipy.optimize import curve_fit, differential_evolution, least_squares
from scipy.interpolate import InterpolatedUnivariateSpline
from joblib import Parallel, delayed
import psutil
psutil.Process().nice(10)

import compact.model as cm
from compact.config import Config
from compact.velocities.pairhist import PairHist
from compact.lpt import GetLPTInputs
from compact.streaming import simps_integrate

# ------------------------------------------------------------------ configuration
CACHE = 'paper_figs_cache.h5'
VR_AUTO_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base/halos/'
                'AbacusSummit_base_{sim}/halos/z0.200/data/pairhist_auto_mbin{k}.h5')
XI_JK_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base/halos/'
              'AbacusSummit_base_{sim}/halos/z0.200/data/xi_hh_jk.npz')
ALL_PLANNED_PHASES = [f'c000_ph{i:03d}' for i in range(0, 25)]

LOG_MASS_EDGES = np.array([[12, 12.01], [12.5, 12.6], [13, 13.1], [13.5, 13.6], [13.9, 14.3]])
N_MASS = len(LOG_MASS_EDGES)
VR_R_EDGES = np.arange(2.0, 142.0, 2.0)
R_MIN_FIT = 15.0
FID_COSMO = [0.6736, 0.02237, 0.1200, 2.083e-9, 0.9649]
FID_REDSHIFT_VAL = 0.2

r_perp_edges = np.linspace(5, 140, 135 + 1)
r_los_edges = np.linspace(5, 140, 135 + 1)
r_p = 0.5 * (r_perp_edges[1:] + r_perp_edges[:-1])
r_l = 0.5 * (r_los_edges[1:] + r_los_edges[:-1])
r_p_2d, r_l_2d = np.meshgrid(r_p, r_l, indexing='ij')
r3_2d = np.sqrt(r_p_2d**2 + r_l_2d**2)
x_grid = r3_2d - r_p_2d

REBIN, N_MIN = 5, 500
NU_FIXED, NU_BOUNDS = 6, (2.1, 1000.0)
SIG_FLOOR, DEL_FLOOR = 0.1, -1000.0
K_FIG0, K_SINGLE = 0, 2
K_XIS = 2                                    # mass bin for the xi^s maps / multipoles
K_FIG7 = 2                                   # mass bin for Fig 7 (same as K_SINGLE)
STRIDE = 3                                   # free-nu fits on every STRIDE-th R row and r_los column (fig5)
HILL_BIN, HILL_MIN_BIN_PTS, HILL_MIN_BINS = 6.0, 10, 6
FIG7_RP_SEL = [15.5, 30.5, 60.5]     # bin centres containing r = 15, 30, 60 h^-1 Mpc
FIG7_RL_SEL = [15.5, 30.5, 60.5]

PARS = np.asarray(Config._MEAN_PARAMS, dtype=float)    # starting values for the Delta / nu fits


# ------------------------------------------------------------------ cache io
def put(f, path, d):
    """(Re)write group `path` from a nested dict of arrays / scalars / string lists."""
    if path in f:
        del f[path]
    g = f.require_group(path)
    for k, v in d.items():
        if isinstance(v, dict):
            put(f, f'{path}/{k}', v)
        elif isinstance(v, (list, tuple)) and v and isinstance(v[0], str):
            g.create_dataset(k, data=np.array(v, dtype=h5.string_dtype()))
        else:
            g.create_dataset(k, data=v)


def jackknife_mean_err(arr):
    n = arr.shape[0]
    total = arr.sum(axis=0)
    mean_vals = total / n
    loo = (total - arr) / (n - 1)
    return mean_vals, np.sqrt((n - 1) / n * np.sum((loo - mean_vals)**2, axis=0))


def phases_for(k):
    return [s for s in ALL_PLANNED_PHASES if os.path.exists(VR_AUTO_TMPL.format(sim=s, k=k))]


# ------------------------------------------------------------------ measurements
def vr_profile(h, k, r_edges):
    Rc = 0.5 * (h.R_edges[1:] + h.R_edges[:-1])
    lc = 0.5 * (h.rlos_edges[1:] + h.rlos_edges[:-1])
    r3 = np.sqrt(Rc[:, None]**2 + lc[None, :]**2).ravel()
    n = h.n[k].ravel().astype(float)
    s1, s2 = h.sum_vr[k].ravel(), h.sum_vr2[k].ravel()
    nb = len(r_edges) - 1
    idx = np.searchsorted(r_edges, r3, side='right') - 1
    ok = (idx >= 0) & (idx < nb) & (n > 0)
    acc = lambda w: np.bincount(idx[ok], weights=w[ok], minlength=nb)
    N = acc(n)
    Nz = np.where(N > 0, N, 1)
    r_eff = np.where(N > 0, acc(n * r3) / Nz, np.nan)
    mu = np.where(N > 0, acc(s1) / Nz, np.nan)
    m2 = np.where(N > 0, acc(s2) / Nz, np.nan)
    return r_eff, mu, np.sqrt(np.maximum(m2 - mu**2, 0.0)), N.astype(np.int64)


def meas_sigv_grid(h, k=0, n_min=500):
    n = h.n[k].astype(float)
    nz = np.where(n > 0, n, 1)
    mu = np.where(n >= n_min, h.sum_v[k] / nz, np.nan)
    m2 = np.where(n >= n_min, h.sum_v2[k] / nz, np.nan)
    return np.sqrt(np.maximum(m2 - mu**2, 0.0))


def good_vr_interp(h):
    r_vr, vr, sd, npairs = vr_profile(h, 0, VR_R_EDGES)
    good = np.isfinite(vr) & (npairs > 1000)
    if good.sum() < 4:
        raise RuntimeError(f'only {good.sum()} usable v_r bins')
    return InterpolatedUnivariateSpline(r_vr[good], vr[good], k=3, ext=3), (r_vr, vr, sd, good)


def fit_skewt_row_free(n_row, sv_row, sv2_row, counts_row, cent_fine, keep=None):
    """[xi, omega, alpha, nu] per cell in one R row; nu free (fitted as ln nu, bounded by NU_BOUNDS)."""
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
            p, _ = curve_fit(lambda x, xi_, om_, al_, lnnu_: cm.skewed_t_pdf(x, xi_, om_, al_, np.exp(lnnu_)),
                             x_w, hist, p0=[mu, width, -0.1, np.log(float(NU_FIXED))],
                             bounds=([-np.inf, 1e-3, -50.0, np.log(NU_BOUNDS[0])],
                                     [np.inf, np.inf, 50.0, np.log(NU_BOUNDS[1])]), max_nfev=5000)
            out[i] = [p[0], p[1], p[2], np.exp(p[3])]
        except RuntimeError:
            continue
    return out


# ------------------------------------------------------------------ model fits (forms from compact.model)
def sigma_2d(coords, sigma0, A_piv, A_rho_p, sigma_rho):
    x, R = coords
    return cm.sigma_mod(x, sigma0, cm.A_rho_mod(R, A_piv, A_rho_p), sigma_rho)


def _seed_sigma(x, R, sig):
    sigma0 = sig.max() / np.sqrt(2.0) * 1.02
    y = 2.0 * sigma0**2 - sig**2
    m = y > 0
    G = np.column_stack([np.ones(m.sum()), -(R[m] - cm.R_PIV_FIXED), -x[m]])
    c = np.linalg.lstsq(G, np.log(y[m]), rcond=None)[0]
    A_rho_p = 1.0 / c[1] if c[1] > 0 else np.ptp(R) + 1.0
    sigma_rho = 1.0 / c[2] if c[2] > 0 else 10.0
    A_piv = np.clip(np.exp(c[0]) / (2.0 * sigma0**2), 1e-3, 0.99)
    return [sigma0, A_piv, A_rho_p, sigma_rho]


def fit_sigma_joint(sig):
    R = np.broadcast_to(r_p[:, None], x_grid.shape)
    m = np.isfinite(sig) & (sig > SIG_FLOOR)
    X, RR, S = x_grid[m], R[m], sig[m]
    popt, _ = curve_fit(sigma_2d, (X, RR), S, p0=_seed_sigma(X, RR, S),
                        bounds=([0.0, 1e-4, 1e-3, 1e-3], [np.inf, 1.0, np.inf, np.inf]), maxfev=20_000)
    return popt


def fit_A_rho_per_R(sig, sigma0, sigma_rho):
    A, Aerr = np.full(len(r_p), np.nan), np.full(len(r_p), np.nan)
    for j in range(len(r_p)):
        m = np.isfinite(sig[j]) & (sig[j] > SIG_FLOOR)
        if m.sum() < 5:
            continue
        try:
            p, c = curve_fit(lambda xx, a: cm.sigma_mod(xx, sigma0, a, sigma_rho), x_grid[j][m], sig[j][m],
                             p0=[0.5], bounds=(1e-4, 1.0), maxfev=5000)
            A[j], Aerr[j] = p[0], np.sqrt(c[0, 0])
        except RuntimeError:
            continue
    return A, Aerr


def delta_2d(coords, A_delta0, r_delta, sigma_delta, beta_delta, alpha):
    x, R = coords
    return cm.delta_mod_alpha(x, cm.A_delta_mod(R, A_delta0, r_delta), sigma_delta, beta_delta, alpha)


def fit_delta(x, R, D, p0):
    m = np.isfinite(x) & np.isfinite(D) & (D > DEL_FLOOR)
    popt, _ = curve_fit(delta_2d, (x[m], R[m]), D[m], p0=p0, maxfev=20_000)
    return popt


def fit_A_delta_per_R(D, dp):
    A, Aerr = np.full(len(r_p), np.nan), np.full(len(r_p), np.nan)
    for j in range(len(r_p)):
        m = np.isfinite(D[j]) & (D[j] > DEL_FLOOR)
        if m.sum() < 5:
            continue
        try:
            p, c = curve_fit(lambda xx, a: cm.delta_mod_alpha(xx, a, *dp[2:]), x_grid[j][m], D[j][m],
                             p0=[dp[0]], maxfev=5000)
            A[j], Aerr[j] = p[0], np.sqrt(c[0, 0])
        except RuntimeError:
            continue
    return A, Aerr


def fit_hill2(r_pts, nu_pts):
    """(nu_0, r_c) of cm.nu_of_r, fit to per-r-bin medians of the free-nu values (soft-L1 in ln nu)."""
    edges = np.arange(r_pts.min(), r_pts.max() + HILL_BIN, HILL_BIN)
    ib = np.digitize(r_pts, edges) - 1
    rm, nm = [], []
    for b in range(len(edges) - 1):
        s = ib == b
        if s.sum() >= HILL_MIN_BIN_PTS:
            rm.append(r_pts[s].mean())
            nm.append(np.median(nu_pts[s]))
    if len(rm) < HILL_MIN_BINS:
        raise RuntimeError(f'only {len(rm)} r-bins for the hill2 fit')
    rm, nm = np.array(rm), np.array(nm)
    sol = least_squares(lambda th: np.log(cm.nu_of_r(rm, *th)) - np.log(nm), PARS[9:11],
                        bounds=([2.01, 1.0], [1e3, 500.0]), loss='soft_l1', f_scale=0.1)
    return sol.x


# ------------------------------------------------------------------ LPT (b1, b2) fit, as in the nu-test script
_LPT = None


def lpt():
    global _LPT
    if _LPT is None:
        _LPT = GetLPTInputs(redshift=FID_REDSHIFT_VAL, cosmo=FID_COSMO)
        _LPT.get_pklin_gsm()
    return _LPT


def model_xihh(r, b1, b2, bs=0.0):
    return lpt().xi_real(r, np.array([b1, b2, bs, 0, 0, 0, 0, 0, 0]))


def model_vr(r, b1, b2, bs=0.0):
    L = lpt()
    pars = np.array([b1, b2, bs, 0, 0, 0, 0, 0, 0])
    L._ensure_cumulants(pars)
    v_int = np.interp(r, L.gsm.rint, L.gsm.veft)
    return L.conv * L.f * v_int / (1 + L.gsm.compute_xi_real(r, *pars))   # not pwv_mean (Tukey-windowed xi)


def regularized_pinv_and_logdet(cov, max_rank):
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    max_rank = min(max_rank, len(vals))
    vals = np.clip(vals[:max_rank], 1e-300, None)
    vecs = vecs[:, :max_rank]
    return vecs @ np.diag(1.0 / vals) @ vecs.T, np.sum(np.log(vals))


def fit_bias_intrinsic(r_xi, xihh, cov_xi, rank_xi, r_vr, vr, cov_vr, rank_vr, seed=0):
    m_xi, m_vr = r_xi > R_MIN_FIT, r_vr > R_MIN_FIT
    rx, xh, cx = r_xi[m_xi], xihh[m_xi], cov_xi[np.ix_(m_xi, m_xi)]
    rv, vv, cv = r_vr[m_vr], vr[m_vr], cov_vr[np.ix_(m_vr, m_vr)]

    def loss(p):
        b1, b2, e_xi, e_v = p
        res_xi = model_xihh(rx, b1, b2) - xh
        inv_xi, ld_xi = regularized_pinv_and_logdet(cx + np.diag(e_xi**2 * xh**2), rank_xi)
        res_vr = model_vr(rv, b1, b2) - vv
        inv_vr, ld_vr = regularized_pinv_and_logdet(cv + np.diag(e_v**2 * vv**2), rank_vr)
        tot = res_xi @ inv_xi @ res_xi + ld_xi + res_vr @ inv_vr @ res_vr + ld_vr
        return tot if np.isfinite(tot) else np.inf

    return differential_evolution(loss, [(-0.3, 5.0), (-10.0, 10.0), (0.0, 2.0), (0.0, 2.0)],
                                  tol=1e-6, seed=seed, maxiter=200, popsize=15, polish=True)


# ------------------------------------------------------------------ sections
def build_fig0(f):
    sim = phases_for(K_FIG0)[0]
    h = PairHist.load(VR_AUTO_TMPL.format(sim=sim, k=K_FIG0))
    R_mask, l_mask = r_p < 60, r_l < 60
    counts = h.counts[0][R_mask][:, l_mask].sum(axis=(0, 1))
    put(f, 'auto/fig0', dict(v_centers=h.v_centers, counts=counts, mass_bin=K_FIG0,
                             n_pairs=int(h.n[0][R_mask][:, l_mask].sum()), sim=[sim]))


def build_fig1(f):
    xi_by, vr_by, vr_r_ref, r_xi = ({k: [] for k in range(N_MASS)}, {k: [] for k in range(N_MASS)},
                                    {k: None for k in range(N_MASS)}, None)
    for sim in ALL_PLANNED_PHASES:
        if os.path.exists(XI_JK_TMPL.format(sim=sim)):
            d = np.load(XI_JK_TMPL.format(sim=sim))
            for k in range(N_MASS):
                xi_by[k].append(d['xi_per_bin'][k])
            r_xi = d['r_cens']
        for k in range(N_MASS):
            path = VR_AUTO_TMPL.format(sim=sim, k=k)
            if os.path.exists(path):
                r_eff, mu, _, npairs = vr_profile(PairHist.load(path), 0, VR_R_EDGES)
                vr_by[k].append(np.where(np.isfinite(mu) & (npairs > 1000), mu, np.nan))
                if vr_r_ref[k] is None:
                    vr_r_ref[k] = r_eff
    for k in range(N_MASS):
        n_xi, n_vr = len(xi_by[k]), len(vr_by[k])
        if n_xi < 2 or n_vr < 2:
            print(f'fig1 bin {k}: {n_xi} xi_hh phases, {n_vr} v_r phases -- skipping')
            continue
        xi_arr = np.array(xi_by[k])
        xi_mean = xi_arr.mean(axis=0)
        dxi = xi_arr - xi_mean
        xi_cov = (dxi.T @ dxi) / (n_xi * (n_xi - 1))
        vr_arr = np.array(vr_by[k])
        vr_mean = np.nanmean(vr_arr, axis=0)
        vr_var = np.nanvar(vr_arr, axis=0, ddof=1) / n_vr
        good = np.isfinite(vr_mean) & (vr_var > 0)
        res = fit_bias_intrinsic(r_xi, xi_mean, xi_cov, max(1, n_xi - 2), vr_r_ref[k],
                                 np.where(good, vr_mean, 0.0), np.diag(np.where(good, vr_var, 1e30)), max(1, n_vr - 2))
        b1, b2, e_xi, e_v = res.x
        print(f'fig1 bin {k}: success={res.success}, b1={b1:.3f}, b2={b2:.3f}, eps_xi={e_xi:.3f}, eps_v={e_v:.3f}')
        put(f, f'auto/fig1/bin_{k}', dict(
            b1=b1, b2=b2, eps_xi=e_xi, eps_v=e_v, loss=res.fun, success=bool(res.success),
            n_xi_phases=n_xi, n_vr_phases=n_vr,
            r_xi=r_xi, xihh=xi_mean, xi_err=np.sqrt(np.diag(xi_cov)), xi_model=model_xihh(r_xi, b1, b2),
            r_vr=vr_r_ref[k], vr=np.where(good, vr_mean, np.nan), vr_err=np.sqrt(np.where(good, vr_var, np.nan)),
            vr_model=model_vr(vr_r_ref[k], b1, b2)))


def build_fig23(f):
    sim = phases_for(K_SINGLE)[0]
    h = PairHist.load(VR_AUTO_TMPL.format(sim=sim, k=K_SINGLE))
    sig = meas_sigv_grid(h, 0)
    vr_interp, _ = good_vr_interp(h)
    mu_meas = vr_interp(r3_2d) * r_l_2d / r3_2d

    sp = fit_sigma_joint(sig)
    A_rho_R, A_rho_R_err = fit_A_rho_per_R(sig, sp[0], sp[3])

    n_R = len(r_p)
    ST = np.array(Parallel(n_jobs=-1, verbose=5)(
        delayed(fit_skewt_row_free)(h.n[0][j], h.sum_v[0][j], h.sum_v2[0][j], h.counts[0][j], h.v_centers)
        for j in range(n_R)))
    Delta = ST[:, :, 0] - mu_meas                                  # free-nu calibration, as for the 11 parameters
    dp = fit_delta(x_grid, np.broadcast_to(r_p[:, None], x_grid.shape), Delta, list(PARS[4:9]))
    A_delta_R, A_delta_R_err = fit_A_delta_per_R(Delta, dp)
    print(f'fig23 ({sim}, bin {K_SINGLE}): sigma pars {np.round(sp, 3)}, Delta pars {np.round(dp, 3)}')
    put(f, 'auto/fig23', dict(sim=[sim], mass_bin=K_SINGLE, sig=sig, sig_params=sp, A_rho_R=A_rho_R,
                              A_rho_R_err=A_rho_R_err, Delta=Delta, nu_free=ST[:, :, 3], del_params=dp,
                              A_delta_R=A_delta_R, A_delta_R_err=A_delta_R_err))


def fit_phase_bin(sim, k):
    """(sim, k, 11 parameters or None, error string or None)."""
    try:
        h = PairHist.load(VR_AUTO_TMPL.format(sim=sim, k=k))
        sp = fit_sigma_joint(meas_sigv_grid(h, 0))
        vr_interp, _ = good_vr_interp(h)
        keep_cols = np.arange(len(r_l)) % STRIDE == 0
        ST = np.full(r3_2d.shape + (4,), np.nan)
        for j in np.arange(0, len(r_p), STRIDE):
            ST[j] = fit_skewt_row_free(h.n[0][j], h.sum_v[0][j], h.sum_v2[0][j], h.counts[0][j], h.v_centers,
                                       keep=keep_cols)
        fin = np.isfinite(ST[:, :, 3])
        Delta = (ST[:, :, 0] - vr_interp(r3_2d) * r_l_2d / r3_2d)[fin]
        dp = fit_delta(x_grid[fin], r_p_2d[fin], Delta, list(PARS[4:9]))
        hill = fit_hill2(r3_2d[fin], ST[:, :, 3][fin])
        return sim, k, np.concatenate([sp, dp, hill]), None
    except Exception as e:
        return sim, k, None, f'{type(e).__name__}: {e}'


def build_fig5(f):
    tasks = [(s, k) for k in range(N_MASS) for s in phases_for(k)]
    print(f'fig5: {len(tasks)} (phase, bin) fits')
    results = Parallel(n_jobs=-1, verbose=10)(delayed(fit_phase_bin)(s, k) for s, k in tasks)
    by_bin, errs = {k: {} for k in range(N_MASS)}, {}
    for sim, k, par, err in results:
        if par is not None:
            by_bin[k][sim] = par
        else:
            errs[err] = errs.get(err, 0) + 1
    if errs:
        print('failures:', errs)
    if 'auto/fig5' in f:
        del f['auto/fig5']
    for k in range(N_MASS):
        sims_k = sorted(by_bin[k])
        if len(sims_k) >= 2:
            put(f, f'auto/fig5/bin_{k}', dict(params=np.array([by_bin[k][s] for s in sims_k]), sims=sims_k))
        else:
            print(f'fig5 bin {k}: fewer than 2 phases -- skipped')
    allp = np.concatenate([np.array(list(by_bin[k].values())) for k in range(N_MASS) if by_bin[k]], axis=0)
    print('grand-mean 11 parameters (compare Config._MEAN_PARAMS):')
    print('np.array([' + ', '.join(f'{v:.6g}' for v in allp.mean(axis=0)) + '])')


def load_stacked(sims, k):
    fields = ('n', 'sum_v', 'sum_v2', 'counts', 'sum_vr', 'sum_vr2')
    acc, first = None, None
    for s in sims:
        h = PairHist.load(VR_AUTO_TMPL.format(sim=s, k=k))
        if acc is None:
            first, acc = h, {fl: np.array(getattr(h, fl)[0], dtype=np.float64) for fl in fields}
        else:
            for fl in fields:
                acc[fl] += getattr(h, fl)[0]
    return SimpleNamespace(R_edges=first.R_edges, rlos_edges=first.rlos_edges, v_centers=first.v_centers,
                           **{fl: [acc[fl]] for fl in fields})


def build_fig7(f):
    cells = sorted({(int(np.argmin(np.abs(r_p - rp))), int(np.argmin(np.abs(r_l - rl))))
                    for rp in FIG7_RP_SEL for rl in FIG7_RL_SEL})
    r_tab = np.linspace(1.0, 200.0, 400)
    b1, b2 = f[f'auto/fig1/bin_{K_FIG7}/b1'][()], f[f'auto/fig1/bin_{K_FIG7}/b2'][()]
    sims = phases_for(K_FIG7)
    h = load_stacked(sims, K_FIG7)
    g = dict(k=K_FIG7, n_phases=len(sims), r_tab=r_tab, vr_model=model_vr(r_tab, b1, b2))
    for (j, i) in cells:
        g[f'cell_{j}_{i}'] = dict(counts=h.counts[0][j, i], n=h.n[0][j, i])
    put(f, 'auto/fig7', dict(rp_sel=np.array(FIG7_RP_SEL), rl_sel=np.array(FIG7_RL_SEL), v_centers=h.v_centers, bin=g))


def build_lpt_sigma(f):
    """<v_los^2> about ZERO (no mean subtracted): measured per cell, and the velocileptors prediction
    conv^2 f^2 (s0 + P2(mu) s2) / (1 + xi), i.e. GetLPTInputs.pwv_variance without its `- v_r^2` term.
    xi is compute_xi_real at the bin's (b1, b2), as in model_vr (not the Tukey-windowed xieft that pwv_variance uses)."""
    sim = phases_for(K_SINGLE)[0]
    h = PairHist.load(VR_AUTO_TMPL.format(sim=sim, k=K_SINGLE))
    n = h.n[0].astype(float)
    m2_meas = np.where(n >= N_MIN, h.sum_v2[0] / np.where(n > 0, n, 1), np.nan)
    b1, b2 = f[f'auto/fig1/bin_{K_SINGLE}/b1'][()], f[f'auto/fig1/bin_{K_SINGLE}/b2'][()]

    L = lpt()
    r = r3_2d.ravel()
    P2 = 0.5 * (3 * (r_l_2d.ravel() / r)**2 - 1)
    K = L.conv**2 * L.f**2

    def m2_for(s2fog):
        pars = np.array([b1, b2, 0.0, 0, 0, 0, 0, 0, s2fog])         # all other counterterms zero
        L._ensure_cumulants(pars)
        s0 = np.interp(r, L.gsm.rint, L.gsm.s0eft)                     # s0eft already includes s2fog
        s2 = np.interp(r, L.gsm.rint, L.gsm.s2eft)
        xi = L.gsm.compute_xi_real(r, *pars)
        return (K * (s0 + P2 * s2) / (1 + xi)).reshape(r3_2d.shape), (K / (1 + xi)).reshape(r3_2d.shape)

    m2_lpt, w = m2_for(0.0)
    # s2fog adds a constant c to s0eft, so m2 = m2(0) + c K/(1+xi): linear in c -> closed-form least squares over 15 <= r <= 130
    use = np.isfinite(m2_meas) & (r3_2d >= 15.0) & (r3_2d <= 130.0)
    s2fog = float(np.sum(w[use] * (m2_meas - m2_lpt)[use]) / np.sum(w[use]**2))     # [(Mpc/h)^2]
    m2_lpt_fit, _ = m2_for(s2fog)                                       # recomputed through velocileptors as a check
    chk = np.nanmax(np.abs(m2_lpt_fit - (m2_lpt + s2fog * w)) / m2_lpt_fit)
    print(f'lpt_sigma: fitted s2fog = {s2fog:.4g} (Mpc/h)^2 -> offset of {s2fog * K:.4g} (km/s)^2 at xi=0; '
          f'closed form vs velocileptors recompute: max rel diff {chk:.2e}')

    r_tab = np.linspace(1.0, 200.0, 400)
    put(f, 'auto/lpt_sigma', dict(sim=[sim], mass_bin=K_SINGLE, b1=b1, b2=b2, m2_meas=m2_meas, m2_lpt=m2_lpt, m2_lpt_fit=m2_lpt_fit, s2fog=s2fog,
                                  r_tab=r_tab, vr_model=model_vr(r_tab, b1, b2)))


XIS_DIRECT_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base/halos/'
                   'AbacusSummit_base_{sim}/halos/z0.200/data/xi_s_direct_mbin{k}.npz')   # written by measure_xi_s_direct.py
MP_S = np.arange(5.5, 130.0, 1.0)              # s at which the multipoles are evaluated (centres, 1 h^-1 Mpc steps)
MP_NR, MP_NY = 130, 165                        # model real-space cells (1 h^-1 Mpc): R in 0-130, y in 0-165 (y reaches s_par,max + v_max/conv)
MP_S_EDGES = np.arange(0.0, 131.0, 1.0)        # s_perp (rp) and s_par (pi) bins of the cartesian xi^s map
MP_N_MU = 400


def xi_s_map(pmf, w, y, conv, v, s_edges):
    """xi^s(s_perp, s_par>0) from per-cell velocity pmfs (rows = R cells, s_perp = R).
    pmf (n_R, n_y, n_v): probability per fine v-bin in each (R, r_los>0) cell; w (n_R, n_y) = 1 + xi(r).
    1 + xi^s(s) = int dy (1+xi) P(u = s - y | R, y), u = v/conv. A y<0 cell is the mirror image (y, v -> -y, -v) of a y>0 one,
    so each (cell, v-bin) lands once, at |y + v/conv|. Used for the MODELS; the simulation is measured directly."""
    n_R, n_y, n_v = pmf.shape
    n_s = len(s_edges) - 1
    ds, dy = s_edges[1] - s_edges[0], 1.0
    k = np.broadcast_to(np.floor(np.abs(y[None, :, None] + v[None, None, :] / conv) / ds).astype(int), pmf.shape)
    ok = k < n_s
    row = np.broadcast_to(np.arange(n_R)[:, None, None], pmf.shape)
    wt = w[:, :, None] * pmf * (dy / ds)
    g = np.bincount((row * n_s + k)[ok], weights=wt[ok], minlength=n_R * n_s).reshape(n_R, n_s)
    return g - 1.0


def multipoles_from_map(xi_map, s_out, ells=(0, 2, 4)):
    """xi_ell(s) = (2l+1) int_0^1 xi(s, mu) P_l(mu) dmu from a cartesian xi^s(s_perp, s_par) map (rows s_perp, columns s_par,
    1 h^-1 Mpc pixels, bilinear interpolation, coordinates clipped to the map's pixel centres)."""
    from scipy.interpolate import RegularGridInterpolator
    from scipy.special import eval_legendre
    c = 0.5 * (MP_S_EDGES[1:] + MP_S_EDGES[:-1])
    interp = RegularGridInterpolator((c, c), xi_map)
    mu = (np.arange(MP_N_MU) + 0.5) / MP_N_MU
    S, MU = np.meshgrid(s_out, mu, indexing='ij')
    pts = np.stack([np.clip(S * np.sqrt(1 - MU**2), c[0], c[-1]), np.clip(S * MU, c[0], c[-1])], axis=-1)
    xi = interp(pts)
    return np.array([(2 * l + 1) * np.mean(xi * eval_legendre(l, MU), axis=1) for l in ells])


def build_xi_s(f):
    """Simulation xi^s maps / multipoles from the direct pycorr measurements (measure_xi_s_direct.py), plus the model maps / multipoles."""
    paths = [XIS_DIRECT_TMPL.format(sim=s, k=K_XIS) for s in ALL_PLANNED_PHASES]
    data = [np.load(p) for p in paths if os.path.exists(p)]
    if len(data) < 2:
        raise RuntimeError('fewer than 2 xi_s_direct_mbin*.npz files found -- run measure_xi_s_direct.py (pycorr environment) first')
    sims = list(range(len(data)))
    xi_maps = np.array([d['xi_rppi'] for d in data])                       # (n_phases, 130, 130)
    r_cens, xi_r = data[0]['r_cens'], np.mean([d['xi_r'] for d in data], axis=0)
    xi_sim, xi_std = xi_maps.mean(axis=0), xi_maps.std(axis=0, ddof=1)
    mp_sim = np.array([multipoles_from_map(m, MP_S) for m in xi_maps])   # (n_phases, 3, n_s)

    # models: streaming integral over the FULL real-space grid (R from 0, no missing cells)
    Rc, yc = np.arange(MP_NR) + 0.5, np.arange(MP_NY) + 0.5
    r3 = np.hypot(Rc[:, None], yc[None, :])
    w = 1.0 + np.interp(r3, r_cens, xi_r, left=-1.0, right=xi_r[-1])
    b1, b2 = f[f'auto/fig1/bin_{K_XIS}/b1'][()], f[f'auto/fig1/bin_{K_XIS}/b2'][()]
    conv = lpt().conv
    r_tab = np.linspace(1.0, 200.0, 400)
    vr_spline = InterpolatedUnivariateSpline(r_tab, model_vr(r_tab, b1, b2), k=3, ext=3)
    v = PairHist.load(VR_AUTO_TMPL.format(sim=phases_for(K_XIS)[0], k=K_XIS)).v_centers      # same fine v-bins as the pair histograms
    dv = v[1] - v[0]
    pmf_t, pmf_g = np.zeros((MP_NR, MP_NY, len(v))), np.zeros((MP_NR, MP_NY, len(v)))
    for j in range(MP_NR):
        for i in range(MP_NY):
            pmf_t[j, i] = cm.Pv_compact(v, Rc[j], yc[i], vr_spline, PARS) * dv
            mu = vr_spline(r3[j, i]) * yc[i] / r3[j, i]
            sd = cm.sigma_mod(r3[j, i] - Rc[j], PARS[0], cm.A_rho_mod(Rc[j], PARS[1], PARS[2]), PARS[3])
            pmf_g[j, i] = np.exp(-0.5 * ((v - mu) / sd)**2) / (np.sqrt(2 * np.pi) * sd) * dv
    xi_t = xi_s_map(pmf_t, w, yc, conv, v, MP_S_EDGES)
    xi_g = xi_s_map(pmf_g, w, yc, conv, v, MP_S_EDGES)
    print(f'xi_s: {len(data)} phases, {int(np.mean([d["n_halos"] for d in data]))} halos per phase on average')

    put(f, 'auto/xi_s', dict(s_perp_edges=MP_S_EDGES, s_par_edges=MP_S_EDGES, xi_sim=xi_sim, xi_sim_std=xi_std,
                             n_phases=len(sims), xi_skewt=xi_t, xi_gauss=xi_g, conv=conv, mass_bin=K_XIS))
    put(f, 'auto/multipoles', dict(s=MP_S, ells=np.array([0, 2, 4]), xi_sim_phases=mp_sim, xi_sim=mp_sim.mean(axis=0),
                                   xi_sim_err=mp_sim.std(axis=0, ddof=1) / np.sqrt(len(sims)),
                                   xi_skewt=multipoles_from_map(xi_t, MP_S), xi_gauss=multipoles_from_map(xi_g, MP_S),
                                   n_phases=len(sims), mass_bin=K_XIS))


FIT_S_MIN, FIT_S_MAX, FIT_STRIDE = 20.0, 130.0, 3     # fit window [h^-1 Mpc] and thinning of the 1 h^-1 Mpc data bins
N_MU_FIT, N_Y_FIT = 40, 150                           # mu midpoints / integration points per half-line while fitting
N_MU_OUT, N_Y_OUT = 120, 300                          # ... for the final predictions
THETA_NAMES = ['b1', 'b2', 'sigma0', 'A_rho0', 'A_rho_p', 'sigma_rho', 'A_delta0', 'r_delta', 'sigma_delta', 'beta_delta',
               'alpha', 'nu_0', 'r_c']
THETA_LB = np.array([0.3, -6.0, 100.0, 0.0, 10.0, 5.0, 0.0, 5.0, 5.0, 0.2, 0.0, 2.5, 5.0])
THETA_UB = np.array([4.0, 6.0, 600.0, 0.99, 300.0, 100.0, 400.0, 300.0, 150.0, 6.0, 2.0, 200.0, 200.0])
_LPT_SPLINES = {}


def lpt_splines(b1, b2):
    """(xi_real, mean v_r) splines from LPT at (b1, b2): xi from xi_real, v_r from the consistent model_vr (as in Figs 1, 7, 8),
    NOT GetLPTInputs.pwv_mean. ext=3 (constant) beyond r = 220 instead of polynomial extrapolation."""
    key = (round(float(b1), 6), round(float(b2), 6))
    if key not in _LPT_SPLINES:
        if len(_LPT_SPLINES) > 8:
            _LPT_SPLINES.clear()
        r_lpt = np.arange(1.0, 220.0, 1.0)
        _LPT_SPLINES[key] = (InterpolatedUnivariateSpline(r_lpt, model_xihh(r_lpt, *key), k=3, ext=3),
                             InterpolatedUnivariateSpline(r_lpt, model_vr(r_lpt, *key), k=3, ext=3))
    return _LPT_SPLINES[key]


def model_xi_ell(theta, s, kind, n_mu, n_y, ells=(0, 2, 4)):
    """Streaming-model xi_ell(s) (production machinery: compact.streaming.simps_integrate) for theta = [b1, b2, 11 compact pars].
    kind 'skewt': Pv_compact. kind 'gauss': Gaussian with the skew-t's mean and standard deviation (same parameters)."""
    from scipy.special import eval_legendre
    xi_real, mean_vr = lpt_splines(theta[0], theta[1])
    pars, conv = np.asarray(theta[2:], dtype=float), lpt().conv
    if kind == 'skewt':
        pdf = lambda v, rp, rl: cm.Pv_compact(v, rp, rl, mean_vr, pars, unit_conversion=conv)
    else:
        def pdf(v, rp, rl):
            r = np.hypot(rp, rl)
            sd = cm.sigma_mod(r - rp, pars[0], cm.A_rho_mod(rp, pars[1], pars[2]), pars[3])
            return conv * np.exp(-0.5 * ((conv * v - mean_vr(r) * rl / r) / sd)**2) / (np.sqrt(2 * np.pi) * sd)
    mu_c = (np.arange(n_mu) + 0.5) / n_mu
    xi_s_mu = simps_integrate(np.asarray(s, dtype=float), mu_c, xi_real, pdf, limit=1.25 * np.max(s), epsilon=0.01, n=n_y)
    return np.array([(2 * l + 1) * np.mean(xi_s_mu * eval_legendre(l, mu_c), axis=1) for l in ells])


def build_fit_multipoles(f):
    """Least-squares fit of the 13 parameters to the simulation xi_0 and xi_2 (phase mean) on FIT_S_MIN-FIT_S_MAX, with the
    SINGLE-PHASE scatter (std over phases) as the per-bin error. xi_4 is then a prediction. Default curves use the Fig 1 (b1, b2)
    and Config._MEAN_PARAMS; the Gaussian is never fit."""
    s_all = f['auto/multipoles/s'][()]
    phases = f['auto/multipoles/xi_sim_phases'][()]                      # (n_phases, 3, n_s)
    data, sig = phases.mean(axis=0), phases.std(axis=0, ddof=1)
    sel = np.where((s_all >= FIT_S_MIN) & (s_all <= FIT_S_MAX))[0][::FIT_STRIDE]
    s_fit = s_all[sel]
    b1, b2 = f[f'auto/fig1/bin_{K_XIS}/b1'][()], f[f'auto/fig1/bin_{K_XIS}/b2'][()]
    theta0 = np.concatenate([[b1, b2], PARS])

    def resid(theta):
        m = model_xi_ell(theta, s_fit, 'skewt', N_MU_FIT, N_Y_FIT, ells=(0, 2))
        r = np.concatenate([(m[0] - data[0, sel]) / sig[0, sel], (m[1] - data[1, sel]) / sig[1, sel]])
        return np.nan_to_num(r, nan=1e3, posinf=1e3, neginf=-1e3)

    r0 = resid(theta0)
    print(f'fit_multipoles: {len(s_fit)} s points x 2 multipoles, start chi2 = {np.sum(r0**2):.1f}')
    res = least_squares(resid, np.clip(theta0, THETA_LB + 1e-6, THETA_UB - 1e-6), bounds=(THETA_LB, THETA_UB), x_scale='jac',
                        diff_step=1e-3, max_nfev=40, verbose=1)
    dof = 2 * len(s_fit) - len(theta0)
    print(f'fit_multipoles: chi2 {np.sum(r0**2):.1f} -> {np.sum(res.fun**2):.1f} (dof {dof}), success={res.success}')
    for name, a0, a1 in zip(THETA_NAMES, theta0, res.x):
        print(f'   {name:12s} {a0:10.4g} -> {a1:10.4g}')

    out = {k: model_xi_ell(th, s_all, kind, N_MU_OUT, N_Y_OUT) for k, th, kind in
           (('xi_fit', res.x, 'skewt'), ('xi_default', theta0, 'skewt'), ('xi_gauss', theta0, 'gauss'))}
    put(f, 'auto/multipoles_fit', dict(s=s_all, theta_names=THETA_NAMES, theta_fit=res.x, theta_default=theta0,
                                       chi2_default=float(np.sum(r0**2)), chi2_fit=float(np.sum(res.fun**2)), dof=dof,
                                       s_fit=s_fit, fit_s_min=FIT_S_MIN, success=bool(res.success), **out))


GRID_FIT_PATH = 'cosmo_grid_fit_results_numod.h5'     # 11-parameter fits to the AbacusSummit c130-c181 grid
Z6 = 0.2


def build_fig6(f):
    """Per grid sim and mass bin, the 11 parameters at z=0.2, plus each sim's [omega_b, omega_cdm, ln(1e10 A_s), n_s]
    (the emulator's predictor order). The fit file has no A_s, so it is read from each sim's own header via
    abacus_cosmo_core_auto.read_params, as cosmo_grid_vel_fit.py does; needs the repo root on sys.path."""
    import nb.calibration.runs.auto_runs.abacus_cosmo_core_auto as core
    with h5.File(GRID_FIT_PATH, 'r') as g:
        sim, z, bn, P = (g[k][()] for k in ('sim', 'z', 'bin', 'params'))
    at_z = np.isclose(z, Z6)
    sims = sorted(set(sim[at_z & (bn == 0)]) & set(sim[at_z & (bn == 1)]))
    params = np.full((len(sims), 2, 11), np.nan)
    cosmo = np.zeros((len(sims), 4))
    for n, s in enumerate(sims):
        for k in (0, 1):
            params[n, k] = P[at_z & (bn == k) & (sim == s)][0]
        par = core.read_params(s, Z6)
        cosmo[n] = [float(par['omega_b']), float(par['omega_cdm']), np.log(1e10 * float(par['A_s'])), float(par['n_s'])]
    print(f'fig6: {len(sims)} sims at z={Z6}; c130 (should be fiducial {Config.fiducial_cosmo}): {cosmo[0]}')
    put(f, 'auto/fig6', dict(sims=np.array(sims, dtype=np.int32), cosmo=cosmo, params=params))


SECTIONS = dict(fig0=build_fig0, fig1=build_fig1, fig23=build_fig23, fig5=build_fig5, fig6=build_fig6, lpt_sigma=build_lpt_sigma, xi_s=build_xi_s, fit_multipoles=build_fit_multipoles, fig7=build_fig7)

if __name__ == '__main__':
    todo = sys.argv[1:] or list(SECTIONS)
    with h5.File(CACHE, 'a') as f:
        for s in todo:
            print(f'=== {s}')
            SECTIONS[s](f)
            f.flush()