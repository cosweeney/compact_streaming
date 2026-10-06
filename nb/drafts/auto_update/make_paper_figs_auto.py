"""
Rebuilt make_paper_figures notebook, as a script with #%% cell breaks, using
the auto-statistics data (gather_auto_stats.py for v_r, compute_xihh_jk.py
for xi_hh) instead of the old cross-statistic pipeline. Figs 0 through 5
only, per your last message.

Changes from the earlier notebook version:
  - Fig 1: joint (b1, b2) fit now includes INTRINSIC SCATTER parameters
    (eps_xi, eps_v) again -- but with the properly normalized loss this
    time (chi2 + log-det of the retained eigenvalues), which is what makes
    eps well-behaved instead of running to infinity (the bug found and
    fixed earlier in this project). Uses phase-AVERAGED xi_hh and v_r
    (across however many phases both are available for), not a single
    phase, matching how xi_hh was always handled.
  - Figs 2-4: restricted to r_perp <= 30 h^-1 Mpc, and r (equivalently x,
    since x = r - r_perp) <= 70 h^-1 Mpc.
  - Fig 5: the 9-parameter mass-dependence check, reused directly from the
    separate vel_mod_mass_dep_auto.py work.

I have not run this myself -- no cluster access -- so treat this as a
structurally-complete first draft to check over, same as everything else
in this pipeline.
"""

#%% global setup
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, differential_evolution
from scipy.stats import t as student_t
from scipy.interpolate import InterpolatedUnivariateSpline
from joblib import Parallel, delayed
import psutil
psutil.Process().nice(10)

import matplotlib
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

FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

LOG_MASS_EDGES = np.array([[12, 12.01], [12.5, 12.6], [13, 13.1],
                           [13.5, 13.6], [13.9, 14.3]])
N_MASS = len(LOG_MASS_EDGES)   # derived, not hardcoded -- avoids the exact
                                # desync risk that made this edit non-trivial
M_CENS = (10**LOG_MASS_EDGES[:, 0] * 10**LOG_MASS_EDGES[:, 1])**0.5

VR_R_EDGES = np.arange(2.0, 142.0, 2.0)
R_MIN_FIT = 15.0

FID_COSMO = [0.6736, 0.02237, 0.1200, 2.083e-9, 0.9649]
FID_REDSHIFT_VAL = 0.2

# grid -- matches gather_auto_stats.py's R_edges/rlos_edges exactly
r_perp_edges = np.linspace(5, 140, 135 + 1)
r_los_edges = np.linspace(5, 140, 135 + 1)
r_p = 0.5 * (r_perp_edges[1:] + r_perp_edges[:-1])
r_l = 0.5 * (r_los_edges[1:] + r_los_edges[:-1])
r_p_2d, r_l_2d = np.meshgrid(r_p, r_l, indexing='ij')
r3_2d = np.sqrt(r_p_2d**2 + r_l_2d**2)
x_grid = r3_2d - r_p_2d

# Figs 2-4 restriction: r_perp <= 30, r (and x) <= 70
RP_MAX_PLOT = 30.0
R3_MAX_PLOT = 70.0
plot_mask_2d = (r_p_2d <= RP_MAX_PLOT) & (r3_2d <= R3_MAX_PLOT)
rp_row_mask = r_p <= RP_MAX_PLOT   # which rows of the grid to even loop over


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


NU_FIXED = 6
REBIN = 5
N_MIN = 500


def fit_skewt_grid(h, k=0, row_mask=None):
    """ST_pars[j,i,:] = [xi, omega, alpha] per (R, rlos) cell. row_mask
    restricts which R rows get fit at all (e.g. the r_perp<=30 cut) --
    skipping rows outside that saves real time, not just plotting effort."""
    n_k, sum_v_k, sum_v2_k, counts_k = h.n[k], h.sum_v[k], h.sum_v2[k], h.counts[k]
    cent_fine = h.v_centers
    dv = (cent_fine[1] - cent_fine[0]) * REBIN
    cent = cent_fine.reshape(-1, REBIN).mean(axis=1)

    n_R, n_l = n_k.shape
    ST_pars = np.full((n_R, n_l, 3), np.nan)
    rows = range(n_R) if row_mask is None else np.where(row_mask)[0]
    for j in rows:
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


#%% Fig 0 -- stacked P(v_los) in one mass bin, motivating nu
K_FIG0 = 0
fig0_sim = next((s for s in ALL_PLANNED_PHASES
                  if os.path.exists(VR_AUTO_TMPL.format(sim=s, k=K_FIG0))), None)

if fig0_sim is None:
    print('Fig 0: no phase has bin 0 auto-stats yet -- skipping')
else:
    h0 = PairHist.load(VR_AUTO_TMPL.format(sim=fig0_sim, k=K_FIG0))
    R_mask0 = r_p < 60
    l_mask0 = r_l < 60   # r_los > 0 only, true by construction of the accumulator's grid

    counts0 = h0.counts[K_FIG0][R_mask0][:, l_mask0].sum(axis=(0, 1))
    v_centers0 = h0.v_centers
    n_pairs0 = int(h0.n[K_FIG0][R_mask0][:, l_mask0].sum())

    hist0 = counts0 / (counts0.sum() * (v_centers0[1] - v_centers0[0]))
    mu0 = np.average(v_centers0, weights=counts0)
    width0 = np.sqrt(np.average((v_centers0 - mu0)**2, weights=counts0))

    p_free, _ = curve_fit(skewed_t_pdf, v_centers0, hist0, p0=[mu0, width0, -0.1, 6.0],
                          method='lm', maxfev=10_000)
    p_fixed, _ = curve_fit(
        lambda x, xi_, omega_, alpha_: skewed_t_pdf(x, xi_, omega_, alpha_, NU_FIXED),
        v_centers0, hist0, p0=[mu0, width0, -0.1], method='lm', maxfev=10_000)

    print(f'Fig 0 ({fig0_sim}, bin {K_FIG0}, {n_pairs0:,} pairs): '
          f'free nu={p_free[3]:.2f} (xi={p_free[0]:.1f}, omega={p_free[1]:.1f}, alpha={p_free[2]:.2f}) | '
          f'fixed nu=6 (xi={p_fixed[0]:.1f}, omega={p_fixed[1]:.1f}, alpha={p_fixed[2]:.2f})')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(v_centers0, hist0, 'o', ms=3, c='C0', alpha=0.5, label='data')
    vv = np.linspace(v_centers0.min(), v_centers0.max(), 500)
    ax.plot(vv, skewed_t_pdf(vv, *p_free), '-', c='C1', label=rf'free $\nu={p_free[3]:.1f}$')
    ax.plot(vv, skewed_t_pdf(vv, *p_fixed, NU_FIXED), '--', c='C2', label=r'fixed $\nu=6$')
    ax.set_xlabel(r'$v_{\rm los}$ [km/s]')
    ax.set_ylabel(r'$P(v_{\rm los})$')
    ax.legend()
    # fig.suptitle(f'Fig 0: stacked $P(v_{{\\rm los}})$, bin {K_FIG0}, {fig0_sim}')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig0_stacked_pvlos.png'), dpi=150)
    print(f'saved {os.path.join(FIGURES_DIR, "fig0_stacked_pvlos.png")}')


#%% Fig 1 -- models, intrinsic-scatter joint fit machinery
lpt = GetLPTInputs(redshift=FID_REDSHIFT_VAL, cosmo=FID_COSMO)
lpt.get_pklin_gsm()


def model_xihh(r, b1, b2, bs=0.0):
    pars = np.array([b1, b2, bs, 0, 0, 0, 0, 0, 0])
    return lpt.xi_real(r, pars)


def model_vr(r, b1, b2, bs=0.0):
    pars = np.array([b1, b2, bs, 0, 0, 0, 0, 0, 0])
    # NOT lpt.pwv_mean: that normalizes by a Tukey-windowed xi(r) that
    # doesn't match compute_xi_real's xi(r) (our actual xi_hh model) --
    # found and fixed earlier in this project; recomputed consistently here.
    lpt._ensure_cumulants(pars)
    v_int = np.interp(r, lpt.gsm.rint, lpt.gsm.veft)
    xi_consistent = lpt.gsm.compute_xi_real(r, *pars)
    return lpt.conv * lpt.f * v_int / (1 + xi_consistent)


def regularized_pinv_and_logdet(cov, max_rank):
    """Rank-truncated pseudo-inverse AND the log-determinant of the
    RETAINED eigenvalues -- the correct -2logL normalization for a
    covariance that's itself a function of fitted (intrinsic-scatter)
    parameters. Without this, eps has no reason not to run to infinity:
    inflating the covariance trivially shrinks chi2 with nothing to
    counteract it (the exact bug found and fixed earlier in this project).
    """
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    max_rank = min(max_rank, len(vals))
    vals = np.clip(vals[:max_rank], 1e-300, None)
    vecs = vecs[:, :max_rank]
    inv_cov = vecs @ np.diag(1.0 / vals) @ vecs.T
    logdet = np.sum(np.log(vals))
    return inv_cov, logdet


def fit_bias_intrinsic(r_min, r_xi, xihh, cov_xi, max_rank_xi,
                       r_vr, vr, cov_vr, max_rank_vr,
                       popsize=15, maxiter=200, tol=1e-6, seed=0):
    """Joint (b1, b2) fit to xi_hh and v_r, WITH intrinsic scatter
    parameters (eps_xi, eps_v): extra, FITTED fractional uncertainty
    (proportional to |data|, matching this project's original convention)
    added to each observable's covariance diagonal before inversion, meant
    to absorb genuine model inadequacy that the measured statistical
    covariance alone doesn't capture. Properly normalized this time -- see
    regularized_pinv_and_logdet's docstring.
    """
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
        chi2_xi = res_xi @ inv_xi @ res_xi + logdet_xi

        res_vr = model_vr(r_vr_fit, b1, b2) - vr_fit
        C_vr = cov_vr_fit + np.diag(eps_v**2 * vr_fit**2)
        inv_vr, logdet_vr = regularized_pinv_and_logdet(C_vr, max_rank_vr)
        chi2_vr = res_vr @ inv_vr @ res_vr + logdet_vr

        total = chi2_xi + chi2_vr
        return total if np.isfinite(total) else np.inf

    bounds = [(-0.3, 5.0), (-10.0, 10.0), (0.0, 2.0), (0.0, 2.0)]
    result = differential_evolution(loss, bounds, tol=tol, seed=seed,
                                    maxiter=maxiter, popsize=popsize, polish=True)
    result.dof = max(len(xihh_fit) + len(vr_fit) - 4, 1)
    return result


#%% Fig 1 -- load phase-averaged xi_hh and v_r (auto-statistics, common phases only)
xi_by_phase, xi_cov_jk_by_phase, xi_phases_used = {k: [] for k in range(N_MASS)}, {k: [] for k in range(N_MASS)}, {k: [] for k in range(N_MASS)}
vr_by_phase, vr_cov_by_phase, vr_r_ref = {k: [] for k in range(N_MASS)}, {k: [] for k in range(N_MASS)}, {k: None for k in range(N_MASS)}
vr_phases_used = {k: [] for k in range(N_MASS)}

for sim in ALL_PLANNED_PHASES:
    xi_path = XI_JK_TMPL.format(sim=sim)
    if os.path.exists(xi_path):
        d = np.load(xi_path)
        for k in range(N_MASS):
            xi_by_phase[k].append(d['xi_per_bin'][k])
            xi_cov_jk_by_phase[k].append(d['cov_jk_per_bin'][k])
            xi_phases_used[k].append(sim)
        r_xi = d['r_cens']

    for k in range(N_MASS):
        vr_path = VR_AUTO_TMPL.format(sim=sim, k=k)
        if not os.path.exists(vr_path):
            continue
        h = PairHist.load(vr_path)
        r_eff, mu, sd, npairs = vr_profile(h, 0, VR_R_EDGES)
        good = np.isfinite(mu) & (npairs > 1000)
        vr_by_phase[k].append(np.where(good, mu, np.nan))
        vr_phases_used[k].append(sim)
        if vr_r_ref[k] is None:
            vr_r_ref[k] = r_eff

n_xi_phases = {k: len(xi_phases_used[k]) for k in range(N_MASS)}
n_vr_phases = {k: len(vr_phases_used[k]) for k in range(N_MASS)}
for k in range(N_MASS):
    print(f'bin {k}: {n_xi_phases[k]} xi_hh phases, {n_vr_phases[k]} v_r phases')

xi_mean, xi_cov, xi_max_rank = {}, {}, {}
vr_mean, vr_var, vr_max_rank = {}, {}, {}
for k in range(N_MASS):
    if n_xi_phases[k] >= 2:
        xi_arr = np.array(xi_by_phase[k])
        xi_mean[k] = xi_arr.mean(axis=0)
        delta = xi_arr - xi_mean[k]
        xi_cov[k] = (delta.T @ delta) / (n_xi_phases[k] * (n_xi_phases[k] - 1))
        xi_max_rank[k] = max(1, n_xi_phases[k] - 2)
    if n_vr_phases[k] >= 2:
        vr_arr = np.array(vr_by_phase[k])
        vr_mean[k] = np.nanmean(vr_arr, axis=0)
        vr_var[k] = np.nanvar(vr_arr, axis=0, ddof=1) / n_vr_phases[k]
        vr_max_rank[k] = max(1, n_vr_phases[k] - 2)


#%% Fig 1 -- joint fit per mass bin (expensive: stores everything needed to
# re-plot quickly afterward, without re-fitting)
common_ks = sorted(set(xi_mean) & set(vr_mean))
fig1_results = {}   # k -> dict with fit results + arrays needed for plotting

if not common_ks:
    print('Fig 1: no bin has both xi_hh and v_r phase-averaged data yet -- skipping')
else:
    for k in common_ks:
        r_vr_k = vr_r_ref[k]
        vr_k, var_vr_k = vr_mean[k], vr_var[k]
        good_vr = np.isfinite(vr_k) & (var_vr_k > 0)
        # v_r only has a diagonal (phase-to-phase) estimate here, not a full
        # cross-r covariance -- build a diagonal matrix so it plugs into the
        # same intrinsic-scatter machinery as xi_hh's full covariance
        cov_vr_k = np.diag(np.where(good_vr, var_vr_k, 1e30))

        res = fit_bias_intrinsic(
            R_MIN_FIT, r_xi, xi_mean[k], xi_cov[k], xi_max_rank[k],
            r_vr_k, np.where(good_vr, vr_k, 0.0), cov_vr_k, vr_max_rank[k])
        b1, b2, eps_xi, eps_v = res.x
        print(f'bin {k}: success={res.success}, chi2={res.fun:.2f}, dof={res.dof}, '
              f'b1={b1:.3f}, b2={b2:.3f}, eps_xi={eps_xi:.3f}, eps_v={eps_v:.3f}')

        fig1_results[k] = dict(
            b1=b1, b2=b2, eps_xi=eps_xi, eps_v=eps_v,
            chi2=res.fun, dof=res.dof, success=res.success,
            r_xi=r_xi, xihh=xi_mean[k], xi_err=np.sqrt(np.diag(xi_cov[k])),
            r_vr=r_vr_k, vr=vr_k, vr_err=np.sqrt(var_vr_k),
            xi_model=model_xihh(r_xi, b1, b2),
            vr_model=model_vr(r_vr_k, b1, b2),
        )

#%% Fig 1 -- plot (fast to re-run: reads fig1_results, no re-fitting)
plot_ks = sorted(fig1_results.keys())   # no exclusion -- the widened bin 4 has
                                          # real statistics now, so all bins are shown

if not plot_ks:
    print('Fig 1: no bins left to plot -- skipping')
else:
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharex='col',
                            gridspec_kw={'height_ratios': [1.2, 1]})
    offset_step = 0.5

    def plot_ratio_bands(ax_, r, baseline, err, color):
        ax_.axhline(baseline, c=color, lw=0.7, alpha=0.5, zorder=0)
        ax_.fill_between(r, baseline - 2*err, baseline + 2*err, color=color, alpha=0.12, lw=0, zorder=0)
        ax_.fill_between(r, baseline - err, baseline + err, color=color, alpha=0.22, lw=0, zorder=0)

    for n, k in enumerate(plot_ks):
        d = fig1_results[k]
        lo, hi = LOG_MASS_EDGES[k]

        ax[0, 0].plot(d['r_vr'], d['vr'], 'o', ms=3, c=f'C{n}', zorder=2,
                      label=rf'$\log M \in [{lo:.2f}, {hi:.2f}]$')
        ax[0, 0].fill_between(d['r_vr'], d['vr'] - d['vr_err'], d['vr'] + d['vr_err'],
                              color=f'C{n}', alpha=0.25, lw=0)
        ax[0, 0].plot(d['r_vr'], d['vr_model'], '-', c=f'C{n}')

        xi_r2 = d['r_xi']**2 * d['xihh']
        ax[0, 1].plot(d['r_xi'], xi_r2, 'o', ms=3, c=f'C{n}', zorder=2)
        ax[0, 1].fill_between(d['r_xi'], xi_r2 - d['r_xi']**2*d['xi_err'], xi_r2 + d['r_xi']**2*d['xi_err'],
                              color=f'C{n}', alpha=0.25, lw=0)
        ax[0, 1].plot(d['r_xi'], d['r_xi']**2*d['xi_model'], '-', c=f'C{n}')

        baseline = 1 + n * offset_step

        vr_ratio = d['vr_model'] / d['vr']
        vr_ratio_err = np.abs(vr_ratio) * (d['vr_err'] / np.abs(d['vr']))
        plot_ratio_bands(ax[1, 0], d['r_vr'], baseline, vr_ratio_err, f'C{n}')
        ax[1, 0].plot(d['r_vr'], vr_ratio + n*offset_step, 'o-', ms=3, c=f'C{n}', zorder=2)

        xi_ratio = d['xi_model'] / d['xihh']
        xi_ratio_err = np.abs(xi_ratio) * (d['xi_err'] / np.abs(d['xihh']))
        xi_good = np.abs(d['xihh']) > d['xi_err']
        plot_ratio_bands(ax[1, 1], d['r_xi'][xi_good], baseline, xi_ratio_err[xi_good], f'C{n}')
        ax[1, 1].plot(d['r_xi'][xi_good], xi_ratio[xi_good] + n*offset_step, 'o-', ms=3, c=f'C{n}', zorder=2)

    for ax_ in ax.flat:
        ax_.axvline(R_MIN_FIT, color='gray', ls=':', lw=1)
        ax_.margins(x=0)
    ax[0, 0].set_ylabel(r'$\langle v_r|r\rangle$ [km/s]')
    ax[0, 1].set_ylabel(r'$r^2\,\xi_{\rm hh}(r|M)$ [$(h^{-1}\,{\rm Mpc})^2$]')
    ax[1, 0].set_ylabel('model / data (+ offset)')
    # ax[1, 1].set_ylabel('model / data (+ offset)')
    ax[1, 0].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
    ax[1, 1].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
    ax[0, 0].legend(fontsize=8, ncol=2)

    ax[1, 1].sharey(ax[1, 0])
    ax[1, 1].tick_params(labelleft=False)

    # fig.suptitle('Fig 1: joint (b1,b2) fit with intrinsic scatter, auto-statistics (highest mass bin excluded)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig1_joint_fit.png'), dpi=150)
    print(f'saved {os.path.join(FIGURES_DIR, "fig1_joint_fit.png")}')

#%% Fig 1 -- best-fit b1, b2 vs mass (reads fig1_results, no re-fitting)
plot_ks_b = sorted(fig1_results.keys())
if not plot_ks_b:
    print('No fitted bins available yet -- skipping b1/b2 vs mass plot')
else:
    b1_vals = [fig1_results[k]['b1'] for k in plot_ks_b]
    b2_vals = [fig1_results[k]['b2'] for k in plot_ks_b]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(M_CENS[plot_ks_b], b1_vals, 'o-', c='C0')
    ax[0].set_xlabel(r'$M\ [h^{-1} M_\odot]$')
    ax[0].set_ylabel(r'$b_1$')
    ax[0].set_xscale('log')

    ax[1].plot(M_CENS[plot_ks_b], b2_vals, 'o-', c='C1')
    ax[1].set_xlabel(r'$M\ [h^{-1} M_\odot]$')
    ax[1].set_ylabel(r'$b_2$')
    ax[1].set_xscale('log')

    # fig.suptitle('Fig 1: best-fit $b_1$, $b_2$ vs mass')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig1_b1_b2_vs_mass.png'), dpi=150)
    print(f'saved {os.path.join(FIGURES_DIR, "fig1_b1_b2_vs_mass.png")}')


#%% Figs 2 & 3 -- sigma_v^2(r|r_p), sigma_v^2(x), A_rho(r_p) -- r_perp<=30, r<=70
SIG_FLOOR = 0.1


def A_rho_mod(R, A_piv, A_rho_p, R_piv):
    return A_piv * np.exp(-(R - R_piv) / A_rho_p)


def sigma_mod(x, sigma0, A_rho, sigma_rho):
    arg = 2.0 * sigma0**2 * (1.0 - A_rho * np.exp(-x / sigma_rho))
    return np.sqrt(np.clip(arg, 1e-8, None))


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
    R_piv = np.median(RR)
    popt, _ = curve_fit(
        lambda c, s0, Ap, Arp, sr: sigma_mod_2d(c, s0, Ap, Arp, sr, R_piv),
        (X, RR), S, p0=_seed_sigma(X, RR, S, R_piv),
        bounds=([0.0, 1e-4, 1e-3, 1e-3], [np.inf, 1.0, np.inf, np.inf]), maxfev=20_000)
    return popt, R_piv


K_SINGLE = 2   # single mass bin used for Figs 2-4
fig23_sim = next((s for s in ALL_PLANNED_PHASES
                  if os.path.exists(VR_AUTO_TMPL.format(sim=s, k=K_SINGLE))), None)

if fig23_sim is None:
    print(f'Figs 2 & 3: no phase has bin {K_SINGLE} auto-stats yet -- skipping')
else:
    h23 = PairHist.load(VR_AUTO_TMPL.format(sim=fig23_sim, k=K_SINGLE))
    sig23 = meas_sigv_grid(h23, 0)

    # FIT: the full available grid, no r_perp or r3 restriction at all --
    # only SIG_FLOOR/finiteness gate what counts as "available"
    sig_params, R_piv = fit_sigma_joint(x_grid, r_p, sig23)
    print(f'Figs 2&3 ({fig23_sim}, bin {K_SINGLE}, full available data): sigma0={sig_params[0]:.1f}, '
          f'A_piv={sig_params[1]:.3f}, A_rho_p={sig_params[2]:.1f}, sigma_rho={sig_params[3]:.1f}')

    # PLOT (2-panel): DISPLAY-only restriction to r_perp<=30, r<=70; scatter
    # points (not connected lines), colorbar placed to the right of ax[:2]
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), layout='constrained')
    cmap = plt.cm.viridis(np.linspace(0, 1, rp_row_mask.sum()))
    skip = 2

    idx_plot = np.where(rp_row_mask)[0]
    for c, j in zip(cmap, idx_plot):
        m = (sig23[j] > SIG_FLOOR) & (r3_2d[j] <= R3_MAX_PLOT)
        ax[0].plot(r3_2d[j][m], sig23[j][m]**2, 'o', ms=3, c=c)
        ax[1].plot(x_grid[j][m], sig23[j][m]**2, 'o', ms=3, c=c)

    for c, j in zip(cmap, idx_plot):
        m = (sig23[j] > SIG_FLOOR) & (r3_2d[j] <= R3_MAX_PLOT)
        if m.sum() < 2:
            continue
        xs = np.linspace(x_grid[j][m].min(), x_grid[j][m].max(), 500)
        A_here = A_rho_mod(r_p[j], sig_params[1], sig_params[2], R_piv)
        ax[1].plot(xs, sigma_mod(xs, sig_params[0], A_here, sig_params[3])**2,
                   '--', c=c, lw=1)

    ax[0].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
    ax[0].set_ylabel(r'$\sigma_v^2(r|r_\perp)$ [(km/s)$^2$]')
    ax[1].set_xlabel(r'$x = r - r_\perp$ [$h^{-1}$ Mpc]')
    ax[1].set_ylabel(r'$\sigma_v^2(x|r_\perp)$ [(km/s)$^2$]')
    ax[0].margins(x=0)
    ax[1].margins(x=0)

    ax[0].set_yscale('log')
    ax[1].set_yscale('log')

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=5, vmax=RP_MAX_PLOT))
    fig.colorbar(sm, ax=ax, location='right', pad=0.02, label=r'$r_\perp$ [$h^{-1}$ Mpc]')

    fig.savefig(os.path.join(FIGURES_DIR, 'fig2_sigmav.png'), dpi=150)
    print(f'saved {os.path.join(FIGURES_DIR, "fig2_sigmav.png")}')

    # A_rho(r_p): fit + show over the FULL r_perp range (still thinned by skip)
    def fit_A_per_R(x, sig, sigma0, sigma_rho):
        n_R = x.shape[0]
        A_R = np.full(n_R, np.nan)
        A_R_err = np.full(n_R, np.nan)
        for j in range(n_R):
            m = np.isfinite(x[j]) & np.isfinite(sig[j]) & (sig[j] > SIG_FLOOR)
            if m.sum() < 5:
                continue
            try:
                popt, pcov = curve_fit(
                    lambda xx, A: sigma_mod(xx, sigma0, A, sigma_rho),
                    x[j][m], sig[j][m], p0=[0.5], bounds=(1e-4, 1.0), maxfev=5000)
                A_R[j], A_R_err[j] = popt[0], np.sqrt(pcov[0, 0])
            except RuntimeError:
                continue
        return A_R, A_R_err

    A_rho_R, A_rho_R_err = fit_A_per_R(x_grid, sig23, sig_params[0], sig_params[3])
    valid_A = np.isfinite(A_rho_R)

    fig3, ax3 = plt.subplots(figsize=(6, 5))
    ax3.errorbar(r_p[valid_A][::skip], A_rho_R[valid_A][::skip], A_rho_R_err[valid_A][::skip],
                fmt='o', c='k', ms=4, label=r'per-$r_\perp$ fit')
    ax3.plot(r_p, A_rho_mod(r_p, sig_params[1], sig_params[2], R_piv), c='C1',
             label=r'$A_{\rho,0}e^{-(r_\perp-R_0)/A_{\rho,p}}$')
    ax3.set_xlabel(r'$r_\perp$ [$h^{-1}$ Mpc]')
    ax3.set_ylabel(r'$A_\rho(r_\perp)$')
    ax3.legend(fontsize=8)
    ax3.margins(x=0)
    fig3.tight_layout()
    fig3.savefig(os.path.join(FIGURES_DIR, 'fig3_A_rho.png'), dpi=150)
    print(f'saved {os.path.join(FIGURES_DIR, "fig3_A_rho.png")}')



#%% Fig 4 -- Delta(r|r_p) and Delta(x), same mass bin as Fig 2
DEL_FLOOR = -1000.0
DEL_P0 = [125.0, 50.0, 10.0, 2.0, 1.0]


def A_delta_mod(R, A_delta0, r_delta):
    return A_delta0 * np.exp(-R / r_delta)   # exponential decay, matches model.py


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


if fig23_sim is None:
    print(f'Fig 4: no phase has bin {K_SINGLE} auto-stats yet -- skipping')
else:
    r_vr4, vr4, _, npairs4 = vr_profile(h23, 0, VR_R_EDGES)
    good_vr4 = np.isfinite(vr4) & (npairs4 > 1000)
    vr_interp4 = InterpolatedUnivariateSpline(r_vr4[good_vr4], vr4[good_vr4], k=3, ext=3)

    ST_pars4 = fit_skewt_grid(h23, 0, row_mask=rp_row_mask)   # display-only, restricted
    Delta4 = ST_pars4[:, :, 0] - vr_interp4(r3_2d) * r_l_2d / r3_2d

    ST_pars4_full = fit_skewt_grid(h23, 0, row_mask=None)     # fitting, unrestricted
    Delta4_full = ST_pars4_full[:, :, 0] - vr_interp4(r3_2d) * r_l_2d / r3_2d

    # FIT: full available data, no r_perp or r3 restriction
    del_params = fit_delta_joint(x_grid, r_p, Delta4_full)
    print(f'Fig 4 ({fig23_sim}, bin {K_SINGLE}, full available data): A_delta0={del_params[0]:.1f}, '
          f'r_delta={del_params[1]:.1f}, sigma_delta={del_params[2]:.2f}, '
          f'beta_delta={del_params[3]:.2f}, alpha={del_params[4]:.2f}')

    # PLOT (2-panel): DISPLAY-only restriction to r_perp<=30, r<=70; scatter
    # points, colorbar to the right
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    cmap = plt.cm.viridis(np.linspace(0, 1, rp_row_mask.sum()))
    skip = 2

    idx_plot = np.where(rp_row_mask)[0]
    for c, j in zip(cmap, idx_plot):
        m = np.isfinite(Delta4[j]) & (r3_2d[j] <= R3_MAX_PLOT)
        ax[0].plot(r3_2d[j][m], Delta4[j][m], 'o', ms=3, c=c)
        ax[1].plot(x_grid[j][m], Delta4[j][m], 'o', ms=3, c=c)

    for c, j in zip(cmap, idx_plot):
        m = np.isfinite(Delta4[j]) & (r3_2d[j] <= R3_MAX_PLOT)
        if m.sum() < 2:
            continue
        xs = np.linspace(x_grid[j][m].min(), x_grid[j][m].max(), 500)
        A_here = A_delta_mod(r_p[j], del_params[0], del_params[1])
        ax[1].plot(xs, delta_mod_alpha(xs, A_here, del_params[2], del_params[3], del_params[4]),
                   '--', c=c, lw=1)

    ax[0].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
    ax[0].set_ylabel(r'$\Delta(r|r_\perp)$ [km/s]')
    ax[1].set_xlabel(r'$x = r - r_\perp$ [$h^{-1}$ Mpc]')
    ax[1].set_ylabel(r'$\Delta(x)$ [km/s]')
    ax[0].margins(x=0)
    ax[1].margins(x=0)

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=5, vmax=RP_MAX_PLOT))
    fig.colorbar(sm, ax=ax, location='right', pad=0.02, label=r'$r_\perp$ [$h^{-1}$ Mpc]')

    fig.savefig(os.path.join(FIGURES_DIR, 'fig4_delta.png'), dpi=150)
    print(f'saved {os.path.join(FIGURES_DIR, "fig4_delta.png")}')

    # A_delta(r_p): fit + show over the FULL r_perp range
    def fit_Adelta_per_R(x, D, sigma_delta, beta_delta, alpha):
        n_R = x.shape[0]
        A_R = np.full(n_R, np.nan)
        A_R_err = np.full(n_R, np.nan)
        for j in range(n_R):
            m = np.isfinite(x[j]) & np.isfinite(D[j]) & (D[j] > DEL_FLOOR)
            if m.sum() < 5:
                continue
            try:
                popt, pcov = curve_fit(
                    lambda xx, A: delta_mod_alpha(xx, A, sigma_delta, beta_delta, alpha),
                    x[j][m], D[j][m], p0=[del_params[0]], maxfev=5000)
                A_R[j], A_R_err[j] = popt[0], np.sqrt(pcov[0, 0])
            except RuntimeError:
                continue
        return A_R, A_R_err

    A_delta_R, A_delta_R_err = fit_Adelta_per_R(x_grid, Delta4_full, del_params[2], del_params[3], del_params[4])
    valid_Ad = np.isfinite(A_delta_R)

    fig5, ax5 = plt.subplots(figsize=(6, 5))
    ax5.errorbar(r_p[valid_Ad][::skip], A_delta_R[valid_Ad][::skip], A_delta_R_err[valid_Ad][::skip],
                fmt='o', c='k', ms=4, label=r'per-$r_\perp$ fit')
    ax5.plot(r_p, A_delta_mod(r_p, del_params[0], del_params[1]), c='C1',
             label=r'$A_{\Delta,0}\,e^{-r_\perp/r_\Delta}$')
    ax5.set_xlabel(r'$r_\perp$ [$h^{-1}$ Mpc]')
    ax5.set_ylabel(r'$A_\Delta(r_\perp)$')
    ax5.legend(fontsize=8)
    ax5.margins(x=0)
    fig5.tight_layout()
    fig5.savefig(os.path.join(FIGURES_DIR, 'fig4b_A_delta.png'), dpi=150)
    print(f'saved {os.path.join(FIGURES_DIR, "fig4b_A_delta.png")}')

#%% Fig 5 -- fit (or load from cache) the 9 velocity-model parameters per (phase, bin)
FIG5_CACHE = os.path.join(FIGURES_DIR, 'fig5_fit_cache.h5')


def fit_one_phase_bin_fig5(sim, k):
    """Returns (sim, k, sig_params, del_params, err) -- err is None on full
    success, else a short string saying WHICH stage failed and why, so a
    failure is diagnosable instead of silently becoming None, None."""
    path = VR_AUTO_TMPL.format(sim=sim, k=k)
    if not os.path.exists(path):
        return sim, k, None, None, 'file missing'

    try:
        h = PairHist.load(path)
    except Exception as e:
        return sim, k, None, None, f'PairHist.load: {type(e).__name__}: {e}'

    try:
        sig = meas_sigv_grid(h, 0)
        ST_pars = fit_skewt_grid(h, 0)
    except Exception as e:
        return sim, k, None, None, f'meas_sigv_grid/fit_skewt_grid: {type(e).__name__}: {e}'

    try:
        r_vr, vr_mean_, _, npairs_vr = vr_profile(h, 0, VR_R_EDGES)
        good_vr = np.isfinite(vr_mean_) & (npairs_vr > 1000)
        if good_vr.sum() < 4:
            return sim, k, None, None, f'vr_profile: only {good_vr.sum()} usable radial bins'
        vr_interp = InterpolatedUnivariateSpline(r_vr[good_vr], vr_mean_[good_vr], k=3, ext=3)
        Delta = ST_pars[:, :, 0] - vr_interp(r3_2d) * r_l_2d / r3_2d
    except Exception as e:
        return sim, k, None, None, f'vr_profile/Delta: {type(e).__name__}: {e}'

    sig_params = None
    try:
        sig_params, _ = fit_sigma_joint(x_grid, r_p, sig)
    except Exception as e:
        err_sig = f'fit_sigma_joint: {type(e).__name__}: {e}'
    else:
        err_sig = None

    del_params = None
    try:
        del_params = fit_delta_joint(x_grid, r_p, Delta, p0=DEL_P0)
    except Exception as e:
        err_del = f'fit_delta_joint: {type(e).__name__}: {e}'
    else:
        err_del = None

    err = ' | '.join(e for e in (err_sig, err_del) if e) or None
    return sim, k, sig_params, del_params, err


def jackknife_mean_err(arr):
    n = arr.shape[0]
    total = arr.sum(axis=0)
    mean_vals = total / n
    loo_means = (total - arr) / (n - 1)
    err = np.sqrt((n - 1) / n * np.sum((loo_means - mean_vals)**2, axis=0))
    return mean_vals, err


labels5 = [r"$\sigma_0$ [km/s]", r"$A_{\rho,0}$", r"$A_{\rho,p}$ [$h^{-1}$ Mpc]", r"$\sigma_\rho$ [$h^{-1}$ Mpc]",
          r"$A_{\Delta,0}$", r"$r_\Delta$ [$h^{-1}$ Mpc]", r"$\sigma_\Delta$ [$h^{-1}$ Mpc]",
          r"$\beta_\Delta$", r"$\alpha_\Delta$"]

mean_vals5 = np.full((N_MASS, 9), np.nan)
jack_err5 = np.full((N_MASS, 9), np.nan)
per_bin_params5, per_bin_sims5 = {}, {}

if os.path.exists(FIG5_CACHE):
    print(f'Fig 5: loading cached fit results from {FIG5_CACHE} '
          f'(delete this file to force a re-fit)')
    with h5.File(FIG5_CACHE, 'r') as hdf:
        for k in range(N_MASS):
            gk = f'bin_{k}'
            if gk not in hdf:
                continue
            per_bin_params5[k] = hdf[gk]['params'][()]
            per_bin_sims5[k] = [s.decode() if isinstance(s, bytes) else s
                               for s in hdf[gk]['sims'][()]]
            mean_vals5[k], jack_err5[k] = jackknife_mean_err(per_bin_params5[k])
else:
    fig5_tasks = [(sim, k) for sim in ALL_PLANNED_PHASES for k in range(N_MASS)
                 if os.path.exists(VR_AUTO_TMPL.format(sim=sim, k=k))]
    print(f'Fig 5: {len(fig5_tasks)} (phase, bin) combinations available -- fitting '
          f'(will cache to {FIG5_CACHE} when done)')

    fig5_results = Parallel(n_jobs=-1, verbose=10)(
        delayed(fit_one_phase_bin_fig5)(sim, k) for sim, k in fig5_tasks)

    sig_by_bin5 = {k: {} for k in range(N_MASS)}
    del_by_bin5 = {k: {} for k in range(N_MASS)}
    err_counts = {}
    for sim, k, sig_params, del_params, err in fig5_results:
        if sig_params is not None:
            sig_by_bin5[k][sim] = sig_params
        if del_params is not None:
            del_by_bin5[k][sim] = del_params
        if err is not None:
            err_counts[err] = err_counts.get(err, 0) + 1

    for k in range(N_MASS):
        print(f'bin {k}: {len(sig_by_bin5[k])} sigma_v fits, '
              f'{len(del_by_bin5[k])} Delta fits, out of '
              f'{sum(1 for s, kk in fig5_tasks if kk == k)} attempted')

    if err_counts:
        print(f'\n{sum(err_counts.values())} task(s) failed; most common reasons:')
        for msg, n in sorted(err_counts.items(), key=lambda kv: -kv[1])[:10]:
            print(f'  {n:4d}x  {msg}')

    with h5.File(FIG5_CACHE, 'w') as hdf:
        for k in range(N_MASS):
            common_sims = sorted(set(sig_by_bin5[k]) & set(del_by_bin5[k]))
            if len(common_sims) >= 2:
                sig_arr = np.array([sig_by_bin5[k][s] for s in common_sims])
                del_arr = np.array([del_by_bin5[k][s] for s in common_sims])
                params = np.concatenate([sig_arr, del_arr], axis=-1)
                mean_vals5[k], jack_err5[k] = jackknife_mean_err(params)
                per_bin_params5[k] = params
                per_bin_sims5[k] = common_sims

                gk = hdf.create_group(f'bin_{k}')
                gk.create_dataset('params', data=params)
                gk.create_dataset('sims', data=np.array(common_sims, dtype=h5.string_dtype()))
            else:
                print(f'Fig 5, bin {k}: fewer than 2 phases with both fits succeeding -- skipping')

#%% Fig 5 -- plot (fast to re-run once cached: reads per_bin_params5, no re-fitting)
# no exclusion -- the widened bin 4 has real statistics now, so all bins are shown
available_ks5 = sorted(per_bin_params5.keys())

if available_ks5:
    fig, ax = plt.subplots(3, 3, figsize=(15, 13), layout='constrained', sharex='col')
    ax = ax.ravel()
    ax[0].set_xscale('log'); ax[1].set_xscale('log'); ax[2].set_xscale('log')  # propagates down each column
        
    all_sims_used5 = sorted(set(s for k in available_ks5 for s in per_bin_sims5[k]))

    # real (not jackknife) phase-to-phase std per bin, and the grand mean
    # over every phase AND every one of these 4 mass bins combined
    std_vals5 = np.full((N_MASS, 9), np.nan)
    for k in available_ks5:
        std_vals5[k] = per_bin_params5[k].std(axis=0)
    grand_mean5 = np.concatenate([per_bin_params5[k] for k in available_ks5], axis=0).mean(axis=0)

    for i, label in enumerate(labels5):
        for sim in all_sims_used5:
            ks_for_sim = [k for k in available_ks5 if sim in per_bin_sims5[k]]
            if not ks_for_sim:
                continue
            vals = [per_bin_params5[k][per_bin_sims5[k].index(sim), i] for k in ks_for_sim]
            ax[i].plot(M_CENS[ks_for_sim], vals, marker='o', alpha=0.3)

        # jackknife mean +/- jackknife error: markers only, capped, no
        # connecting line between mass bins
        ax[i].errorbar(M_CENS[available_ks5], mean_vals5[available_ks5, i],
                       yerr=jack_err5[available_ks5, i], marker='o', color='k',
                       ls='none', capsize=3, label='jackknife mean' if i == 0 else None,
                       zorder=5)

        # mu +/- sigma band: REAL std over phases (not the jackknife SEM),
        # dashed black
        mu, sd = mean_vals5[available_ks5, i], std_vals5[available_ks5, i]
        ax[i].plot(M_CENS[available_ks5], mu + sd, 'k--', lw=2,
                   label=r'$\pm$ standard dev over phases' if i == 0 else None)
        ax[i].plot(M_CENS[available_ks5], mu - sd, 'k--', lw=2)

        # grand mean over all phases and all 4 mass bins: dotted black
        ax[i].axhline(grand_mean5[i], c='k', ls=':', lw=3,
                      label='mean over all phases, bins' if i == 0 else None)

        ax[i].set_ylabel(label)
        if i >= 6:  # bottom row of the 3x3 grid
            ax[i].set_xlabel(r'$M\ [h^{-1} M_\odot]$')
        #ax[i].set_xscale('log')
        ax[i].set_xlim(M_CENS[available_ks5].min() / 1.5, M_CENS[available_ks5].max() * 1.5)

    handles, leg_labels = ax[0].get_legend_handles_labels()
    ax[6].legend(handles, leg_labels, fontsize=15, loc='upper left')
    # fig.suptitle(f'Fig 5: velocity-model parameters vs mass -- {len(available_ks5)}/{N_MASS} bins shown')
    fig.savefig(os.path.join(FIGURES_DIR, 'fig5_mass_dependence.png'), dpi=150)
    print(f'saved {os.path.join(FIGURES_DIR, "fig5_mass_dependence.png")}')
else:
    print('Fig 5: no bin has enough phases yet -- nothing to plot')

#%% Fig 5 -- velocity-model parameters vs best-fit b1, b2 (no bin exclusion)
common_ks_15 = sorted(set(available_ks5) & set(fig1_results.keys()))

if not common_ks_15:
    print('Not enough bins with both Fig 1 (b1,b2) and Fig 5 (velocity params) results -- skipping')
else:
    b1_vals_15 = np.array([fig1_results[k]['b1'] for k in common_ks_15])
    b2_vals_15 = np.array([fig1_results[k]['b2'] for k in common_ks_15])
    all_sims_used_15 = sorted(set(s for k in common_ks_15 for s in per_bin_sims5[k]))

    # --- each velocity-model parameter vs b1 ---
    fig, ax = plt.subplots(1, 9, figsize=(29, 5), layout='constrained')
    for i, label in enumerate(labels5):
        for sim in all_sims_used_15:
            # same convention as Fig 5's own plot: one line per phase,
            # connecting whatever bins that phase has, no explicit color so
            # matplotlib auto-cycles through the default sequence
            ks_for_sim = [k for k in common_ks_15 if sim in per_bin_sims5[k]]
            if not ks_for_sim:
                continue
            b1_for_sim = [fig1_results[k]['b1'] for k in ks_for_sim]
            vals_for_sim = [per_bin_params5[k][per_bin_sims5[k].index(sim), i] for k in ks_for_sim]
            ax[i].plot(b1_for_sim, vals_for_sim, marker='o', alpha=0.3)
        ax[i].errorbar(b1_vals_15, mean_vals5[common_ks_15, i], yerr=jack_err5[common_ks_15, i],
                       marker='o', color='k', ls='none', zorder=5, label='jackknife mean')
        ax[i].set_xlabel(r'$b_1$')
        ax[i].set_ylabel(label)
    ax[0].legend(fontsize=7)
    fig.savefig(os.path.join(FIGURES_DIR, 'fig5_params_vs_b1.png'), dpi=150)
    print(f'saved {os.path.join(FIGURES_DIR, "fig5_params_vs_b1.png")}')

    # --- each velocity-model parameter vs b2 ---
    fig, ax = plt.subplots(1, 9, figsize=(29, 5), layout='constrained')
    for i, label in enumerate(labels5):
        for sim in all_sims_used_15:
            ks_for_sim = [k for k in common_ks_15 if sim in per_bin_sims5[k]]
            if not ks_for_sim:
                continue
            b2_for_sim = [fig1_results[k]['b2'] for k in ks_for_sim]
            vals_for_sim = [per_bin_params5[k][per_bin_sims5[k].index(sim), i] for k in ks_for_sim]
            ax[i].plot(b2_for_sim, vals_for_sim, marker='o', alpha=0.3)
        ax[i].errorbar(b2_vals_15, mean_vals5[common_ks_15, i], yerr=jack_err5[common_ks_15, i],
                       marker='o', color='k', ls='none', zorder=5, label='jackknife mean')
        ax[i].set_xlabel(r'$b_2$')
        ax[i].set_ylabel(label)
    ax[0].legend(fontsize=7)
    fig.savefig(os.path.join(FIGURES_DIR, 'fig5_params_vs_b2.png'), dpi=150)
    print(f'saved {os.path.join(FIGURES_DIR, "fig5_params_vs_b2.png")}')

    # --- A_Delta,0 vs alpha_Delta, with individual-phase scatter ---
    A_delta0_idx, alpha_idx = 4, 8   # indices into labels5/mean_vals5
    fig, ax = plt.subplots(figsize=(6, 5))
    for sim in all_sims_used_15:
        ks_for_sim = [k for k in common_ks_15 if sim in per_bin_sims5[k]]
        if not ks_for_sim:
            continue
        alpha_for_sim = [per_bin_params5[k][per_bin_sims5[k].index(sim), alpha_idx] for k in ks_for_sim]
        Adelta0_for_sim = [per_bin_params5[k][per_bin_sims5[k].index(sim), A_delta0_idx] for k in ks_for_sim]
        ax.plot(alpha_for_sim, Adelta0_for_sim, marker='o', alpha=0.3)
    ax.errorbar(mean_vals5[common_ks_15, alpha_idx], mean_vals5[common_ks_15, A_delta0_idx],
               xerr=jack_err5[common_ks_15, alpha_idx], yerr=jack_err5[common_ks_15, A_delta0_idx],
               marker='o', color='k', ls='none', zorder=5, label='jackknife mean')
    ax.set_xlabel(labels5[alpha_idx])
    ax.set_ylabel(labels5[A_delta0_idx])
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig5_Adelta0_vs_alpha.png'), dpi=150)
    print(f'saved {os.path.join(FIGURES_DIR, "fig5_Adelta0_vs_alpha.png")}')


#%% Fig 7 -- P(v_los | r_perp, r_los): histogram vs GSM-form Gaussian vs compact skew-t, low-mass bin
from scipy.special import gamma as gamma_fn

K_LOW = 0
RP_SEL = [5.5, 50.5, 100.5]   # columns (r_perp)
RL_SEL = [5.5, 50.5, 100.5]   # rows (r_los)
V_LIM = 2500.0
R_INSET = 140.0               # quarter-disk radius in the inset (assumed r cut of the accumulator)
INSET_PAD = 0.12 * R_INSET    # left/bottom margin inside the inset


def skewt_from_moments(mu, xi, sigma, nu=NU_FIXED):
    """omega, alpha of a nu-dof skew-t with location xi whose mean is mu and
    std is sigma. Uses mean = xi + omega*delta*b, var = omega^2 nu/(nu-2) - (mean-xi)^2."""
    b = np.sqrt(nu / np.pi) * gamma_fn((nu - 1) / 2) / gamma_fn(nu / 2)
    d_off = mu - xi
    omega = np.sqrt((sigma**2 + d_off**2) * (nu - 2) / nu)
    delta = np.clip(d_off / (omega * b), -0.99, 0.99)
    return omega, delta / np.sqrt(1 - delta**2)


fig7_ready = K_LOW in per_bin_params5
if not fig7_ready:
    print(f'Fig 7: bin {K_LOW} has no cached Fig 5 fit -- skipping')
else:
    sim7 = per_bin_sims5[K_LOW][0]
    sig_p7, del_p7 = per_bin_params5[K_LOW][0][:4], per_bin_params5[K_LOW][0][4:]
    h7 = PairHist.load(VR_AUTO_TMPL.format(sim=sim7, k=K_LOW))

    # R_piv is not stored in the cache; recompute as fit_sigma_joint does
    sig7 = meas_sigv_grid(h7, 0)
    m7 = np.isfinite(sig7) & (sig7 > SIG_FLOOR)
    R_piv7 = np.median(np.broadcast_to(r_p[:, None], sig7.shape)[m7])

    r_vr7, vr7, sd_r7, npairs7 = vr_profile(h7, 0, VR_R_EDGES)
    good7 = np.isfinite(vr7) & (npairs7 > 1000)
    vr_interp7 = InterpolatedUnivariateSpline(r_vr7[good7], vr7[good7], k=3, ext=3)

    # --- GSM-form dispersions, measured from the sim, as functions of r only ---
    # sigma_los^2 (about mu*v12, the GSM centering) = mu^2 sig_par^2 + (1-mu^2) sig_perp^2
    mu2_7 = (r_l_2d / r3_2d)**2
    mu_grid7 = vr_interp7(r3_2d) * r_l_2d / r3_2d
    n7 = h7.n[0].astype(float)
    nz7 = np.where(n7 > 0, n7, 1.0)
    s2c7 = h7.sum_v2[0] / nz7 - 2 * mu_grid7 * h7.sum_v[0] / nz7 + mu_grid7**2
    sig_par2_7 = np.interp(r3_2d, r_vr7[good7], sd_r7[good7]**2)
    sig_perp2_cell7 = (s2c7 - mu2_7 * sig_par2_7) / (1 - mu2_7)
    nb7 = len(VR_R_EDGES) - 1
    idx7 = np.searchsorted(VR_R_EDGES, r3_2d, side='right') - 1
    sel7 = (np.isfinite(sig_perp2_cell7) & (n7 >= N_MIN) & (mu2_7 < 0.8)
            & (idx7 >= 0) & (idx7 < nb7))
    num7 = np.bincount(idx7[sel7], weights=(n7 * sig_perp2_cell7)[sel7], minlength=nb7)
    den7 = np.bincount(idx7[sel7], weights=n7[sel7], minlength=nb7)
    sig_perp2_r7 = num7 / np.where(den7 > 0, den7, np.nan)
    good_perp7 = np.isfinite(sig_perp2_r7) & (sig_perp2_r7 > 0) & good7

    cent7 = h7.v_centers.reshape(-1, REBIN).mean(axis=1)
    dv7 = (h7.v_centers[1] - h7.v_centers[0]) * REBIN
    vv7 = np.linspace(-V_LIM, V_LIM, 600)
    theta7 = np.linspace(0, np.pi / 2, 100)

    def plot_vlos_comparison(logy, fname):
        nr, nc = len(RL_SEL), len(RP_SEL)
        fig, axs = plt.subplots(nr, nc, figsize=(5 * nc, 4 * nr), layout='constrained',
                                sharex=True)
        axs = np.atleast_2d(axs)

        for a, rl_t in enumerate(RL_SEL):
            for b_, rp_t in enumerate(RP_SEL):
                ax_ = axs[a, b_]
                j, i = np.argmin(np.abs(r_p - rp_t)), np.argmin(np.abs(r_l - rl_t))
                n = h7.n[0][j, i]
                if n < N_MIN:
                    ax_.set_title('too few pairs')
                    continue

                counts = h7.counts[0][j, i].reshape(-1, REBIN).sum(axis=1)
                hist = counts / (counts.sum() * dv7)
                ax_.step(cent7, hist, where='mid', c='k', lw=1, label='simulation')

                # model quantities (mean is shared with the Gaussian)
                r3, x_c, R_c = r3_2d[j, i], x_grid[j, i], r_p[j]
                sigma_m = sigma_mod_2d((x_c, R_c), *sig_p7, R_piv7)
                mu_m = vr_interp7(r3) * r_l[i] / r3
                xi_m = mu_m + delta_mod_alpha_R((x_c, R_c), *del_p7)

                # GSM-form Gaussian: mean mu*v12, variance mu^2 sig_par^2(r) + (1-mu^2) sig_perp^2(r)
                mu2 = (r_l[i] / r3)**2
                sd_g = np.sqrt(mu2 * np.interp(r3, r_vr7[good7], sd_r7[good7]**2)
                               + (1 - mu2) * np.interp(r3, r_vr7[good_perp7], sig_perp2_r7[good_perp7]))
                ax_.plot(vv7, np.exp(-0.5 * ((vv7 - mu_m) / sd_g)**2) / (np.sqrt(2 * np.pi) * sd_g),
                         c='C1', lw=2, label='Gaussian (GSM form)')

                omega_m, alpha_m = skewt_from_moments(mu_m, xi_m, sigma_m)
                ax_.plot(vv7, skewed_t_pdf(vv7, xi_m, omega_m, alpha_m, NU_FIXED),
                         c='C0', lw=2, label='compact skew-t')

                ax_.axvline(0, c='gray', ls=':', lw=1)
                ax_.set_xlim(-V_LIM, V_LIM)
                if logy:
                    ax_.set_yscale('log')
                    ax_.set_ylim(hist.max() * 1e-4, hist.max() * 2)
                else:
                    ax_.set_yticks([])
                ax_.set_title(rf'$r_\perp\in[{r_perp_edges[j]:.2f}, {r_perp_edges[j+1]:.2f}]$, '
                              rf'$r_{{\rm los}}\in[{r_los_edges[i]:.2f}, {r_los_edges[i+1]:.2f}]$ '
                              r'$h^{-1}$Mpc', fontsize=12)
                if a == nr - 1:
                    ax_.set_xlabel(r'$v_{\rm los}$ [km/s]')

                # inset: where this bin sits on the (R, r_los) quarter-disk
                ins = ax_.inset_axes([0.78, 0.66, 0.20, 0.30])
                ins.fill(np.r_[-INSET_PAD, R_INSET, R_INSET * np.cos(theta7), -INSET_PAD],
                         np.r_[-INSET_PAD, -INSET_PAD, R_INSET * np.sin(theta7), R_INSET],
                         c='C1', alpha=0.5, lw=0)
                ins.axvspan(r_perp_edges[j], r_perp_edges[j + 1], color='C0', alpha=0.4, lw=0)
                ins.axhspan(r_los_edges[i], r_los_edges[i + 1], color='C0', alpha=0.4, lw=0)
                ins.axvline(r_p[j], c='gray', lw=1)
                ins.axhline(r_l[i], c='gray', lw=1)
                ins.set_xlim(-INSET_PAD, R_INSET)
                ins.set_ylim(-INSET_PAD, R_INSET)
                ins.set_xticks([])
                ins.set_yticks([])
                ins.set_xlabel(r'$R$', fontsize=8, labelpad=1)
                ins.set_ylabel(r'$r_{\rm LOS}$', fontsize=8, labelpad=1)

        fig.supylabel(r'$P(v_{\rm los}\,|\,r_\perp, r_{\rm los})$', fontsize=20)
        axs[0, 0].legend(fontsize=9, loc='upper left')
        fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150)
        print(f'saved {os.path.join(FIGURES_DIR, fname)} ({sim7}, bin {K_LOW}, R_piv={R_piv7:.1f})')

    plot_vlos_comparison(logy=False, fname='fig7_vlos_hist_vs_model.png')
#%% Fig 7 (log y-axis) -- requires the Fig 7 cell above to have been run
if fig7_ready:
    plot_vlos_comparison(logy=True, fname='fig7_vlos_hist_vs_model_logy.png')

#%%
import inspect
from compact.model import Pv_compact
from types import SimpleNamespace


#%% Fig 7 -- P(v_los | r_perp, r_los): histogram vs moment-matched Gaussian vs compact skew-t (nu = 6, bin-mean Fig 5 params), low-mass bin, all phases
K_LOW = 2
RP_SEL = [15.5, 55.5, 135.5]  # columns (r_perp)
RL_SEL = [15.5, 55.5, 135.5]  # rows (r_los)
ERR_MODE = 'none'          # histogram error bars: 'poisson', 'phase' (phase-to-phase), or 'none'
V_LIM = 2500.0                # range over which the model curves are computed
V_PLOT = 2000.0               # displayed x-range (all panels)
MIN_COUNT = 1                 # rebinned v-bins with fewer pairs are masked (1 = only empty bins)
Y_FLOOR = 1e-5                # log-y lower limit, as a fraction of each panel's peak
SUM_FIELDS = ('n', 'sum_v', 'sum_v2', 'counts', 'sum_vr', 'sum_vr2')


def load_stacked_pairhist(sims, k, cells=()):
    """Sum the accumulator arrays over phases. Returns an object exposing the same
    attributes (indexed [0]) that this cell uses. For each (j, i) in `cells`, also
    keeps the per-phase normalized, rebinned v_los densities
    (phase_dens[(j, i)] has shape (n_phases, n_vbins))."""
    acc, first = None, None
    phase_dens = {c: [] for c in cells}
    for s in sims:
        h_ = PairHist.load(VR_AUTO_TMPL.format(sim=s, k=k))
        if acc is None:
            first = h_
            acc = {f: np.array(getattr(h_, f)[0], dtype=np.float64) for f in SUM_FIELDS}
        else:
            for f in SUM_FIELDS:
                acc[f] += getattr(h_, f)[0]
        dv_ = (h_.v_centers[1] - h_.v_centers[0]) * REBIN
        for (j, i) in cells:
            c_ = h_.counts[0][j, i].reshape(-1, REBIN).sum(axis=1)
            tot = c_.sum()
            phase_dens[(j, i)].append(c_ / (tot * dv_) if tot > 0 else np.full(c_.shape, np.nan))
    return SimpleNamespace(R_edges=first.R_edges, rlos_edges=first.rlos_edges,
                           v_centers=first.v_centers,
                           phase_dens={c: np.array(v) for c, v in phase_dens.items()},
                           **{f: [acc[f]] for f in SUM_FIELDS})


fig7_ready = (K_LOW in per_bin_params5) and (K_LOW in fig1_results)
if not fig7_ready:
    print(f'Fig 7: bin {K_LOW} needs both a Fig 5 fit and a Fig 1 (b1, b2) fit -- skipping')
elif ERR_MODE not in ('poisson', 'phase', 'none'):
    raise ValueError(f"ERR_MODE must be 'poisson', 'phase' or 'none', got {ERR_MODE!r}")
elif ERR_MODE == 'phase' and len(per_bin_sims5[K_LOW]) < 2:
    raise ValueError('ERR_MODE = "phase" needs at least 2 phases')
else:
    sims7 = per_bin_sims5[K_LOW]
    cells7 = [(int(np.argmin(np.abs(r_p - rp))), int(np.argmin(np.abs(r_l - rl))))
              for rl in RL_SEL for rp in RP_SEL]
    h7 = load_stacked_pairhist(sims7, K_LOW, cells=cells7 if ERR_MODE == 'phase' else ())

    # --- model inputs for Pv_compact ---
    # mean radial velocity: Fig 1 LPT model at this bin's best-fit (b1, b2), tabulated to a spline
    b1_7, b2_7 = fig1_results[K_LOW]['b1'], fig1_results[K_LOW]['b2']
    r_tab7 = np.linspace(1.0, 200.0, 400)
    mean_vr7 = InterpolatedUnivariateSpline(r_tab7, model_vr(r_tab7, b1_7, b2_7), k=3, ext=3)
    # velocity parameters: this mass bin's mean over phases (the black points in Fig 5)
    pars_pv7 = mean_vals5[K_LOW]

    def sigma_v_model(rp, rl):
        """sigma_v (km/s) of the compact model at (r_perp, r_los); the skew-t's standard deviation by construction."""
        x = np.hypot(rp, rl) - rp
        return cm.sigma_mod(x, pars_pv7[0], cm.A_rho_mod(rp, pars_pv7[1], pars_pv7[2]), pars_pv7[3])

    cent7 = h7.v_centers.reshape(-1, REBIN).mean(axis=1)
    dv7 = (h7.v_centers[1] - h7.v_centers[0]) * REBIN
    edges7 = np.r_[cent7 - dv7 / 2, cent7[-1] + dv7 / 2]
    vv7 = np.linspace(-V_LIM, V_LIM, 600)

    # inset quarter-disk radius: just beyond the farthest selected cell, rounded up to 10
    r3_sel7 = [np.hypot(r_p[np.argmin(np.abs(r_p - rp))], r_l[np.argmin(np.abs(r_l - rl))])
               for rp in RP_SEL for rl in RL_SEL]
    R_INSET = 10.0 * np.ceil((max(r3_sel7) + 5.0) / 10.0)
    INSET_PAD = 0.12 * R_INSET    # left/bottom margin inside the inset
    theta7 = np.linspace(0, np.pi / 2, 100)

    def gauss_pdf(v, mu, sd):
        return np.exp(-0.5 * ((v - mu) / sd)**2) / (np.sqrt(2 * np.pi) * sd)

    fmt_var = lambda v: rf'${v / 1e5:.2f}\times10^5$'

    def plot_vlos_comparison(fname, logy):
        nr, nc = len(RL_SEL), len(RP_SEL)
        fig, axs = plt.subplots(nr, nc, figsize=(5 * nc, 4 * nr), layout='constrained',
                                sharex=True)
        axs = np.atleast_2d(axs)

        for a, rl_t in enumerate(RL_SEL):
            for b_, rp_t in enumerate(RP_SEL):
                ax_ = axs[a, b_]
                j, i = np.argmin(np.abs(r_p - rp_t)), np.argmin(np.abs(r_l - rl_t))
                n = h7.n[0][j, i]
                if n < N_MIN:
                    ax_.set_title('too few pairs')
                    continue

                counts = h7.counts[0][j, i].reshape(-1, REBIN).sum(axis=1)
                norm = counts.sum() * dv7
                hist = counts / norm
                ok = counts >= MIN_COUNT
                # stairs on the true bin edges; masked bins drop to a value far below the
                # axis floor (instead of NaN, which breaks the path and loses the vertical
                # edges of isolated bins)
                ax_.stairs(np.where(ok, hist, 1e-30), edges7, color='k', lw=1,
                           label='simulation')

                # histogram errors: Poisson on the stacked counts, or the standard error of
                # the mean of the per-phase densities
                if ERR_MODE != 'none':
                    if ERR_MODE == 'poisson':
                        err = np.sqrt(counts) / norm
                    else:
                        pd = h7.phase_dens[(int(j), int(i))]
                        err = np.nanstd(pd, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(pd), axis=0))
                    err_lo = np.minimum(err, 0.999 * hist) if logy else err   # keep log-axis lower bar positive
                    ax_.errorbar(cent7[ok], hist[ok], yerr=[err_lo[ok], err[ok]], fmt='none',
                                 ecolor='k', elinewidth=0.8, capsize=0)

                # Gaussian matched to the histogram's mean and variance (from the stacked pair sums)
                mean_h = h7.sum_v[0][j, i] / n
                var_h = h7.sum_v2[0][j, i] / n - mean_h**2
                pdf_g = gauss_pdf(vv7, mean_h, np.sqrt(var_h))

                # compact model (nu = 6) at the Fig 1 mean and bin-mean Fig 5 parameters
                var_c = sigma_v_model(r_p[j], r_l[i])**2
                pdf_c = Pv_compact(vv7, r_p[j], r_l[i], mean_vr7, pars_pv7)

                ax_.plot(vv7, pdf_g, c='C1', lw=2,
                         label=rf'Gaussian (moments): $\sigma^2=$ {fmt_var(var_h)} (km/s)$^2$')
                ax_.plot(vv7, pdf_c, c='C0', lw=2,
                         label=rf'compact skew-t: $\sigma^2=$ {fmt_var(var_c)} (km/s)$^2$')

                ax_.axvline(0, c='gray', ls=':', lw=1)
                ax_.set_xlim(-V_PLOT, V_PLOT)
                if logy:
                    ax_.set_yscale('log')
                    ax_.set_ylim(np.nanmax(hist) * Y_FLOOR, np.nanmax(hist) * 2)
                else:
                    ax_.set_ylim(0, 1.1 * max(np.nanmax(hist), pdf_g.max(), pdf_c.max()))
                ax_.legend(fontsize=7.5, framealpha=0.85, labelspacing=0.3,
                           loc='lower center' if logy else 'upper left')
                ax_.set_title(rf'$r_\perp\in[{r_perp_edges[j]:.2f}, {r_perp_edges[j+1]:.2f}]$, '
                              rf'$r_{{\rm los}}\in[{r_los_edges[i]:.2f}, {r_los_edges[i+1]:.2f}]$ '
                              r'$h^{-1}$Mpc', fontsize=12)
                if a == nr - 1:
                    ax_.set_xlabel(r'$v_{\rm los}$ [km/s]')

                # inset: where this bin sits on the (R, r_los) quarter-disk
                ins = ax_.inset_axes([0.82, 0.72, 0.16, 0.24])
                ins.fill(np.r_[-INSET_PAD, R_INSET, R_INSET * np.cos(theta7), -INSET_PAD],
                         np.r_[-INSET_PAD, -INSET_PAD, R_INSET * np.sin(theta7), R_INSET],
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
        fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150)
        print(f'saved {os.path.join(FIGURES_DIR, fname)} ({len(sims7)} phases stacked, '
              f'bin {K_LOW}, b1={b1_7:.3f}, b2={b2_7:.3f}, inset radius {R_INSET:.0f}, errors: {ERR_MODE})')

    plot_vlos_comparison('fig7_vlos_hist_vs_model_binmean.png', logy=True)
    plot_vlos_comparison('fig7_vlos_hist_vs_model_binmean_linear.png', logy=False)

#%% Fig 7 (nu scan) -- compact skew-t at the bin-mean Fig 5 parameters for several nu, vs histogram, low-mass bin, all phases
import compact.model as cm

K_LOW = 0
RP_SEL = [15.5, 55.5, 135.5]  # columns (r_perp)
RL_SEL = [15.5, 55.5, 135.5]  # rows (r_los)
NU_LIST = [4, 6, 9, 12, 18, 24]
NU_COLORS = ['C3', 'C0', 'C2', 'C1', 'C4', 'C5']   # red, blue, green, orange, purple, brown
ERR_MODE = 'poisson'          # histogram error bars: 'poisson', 'phase' (phase-to-phase), or 'none'
V_LIM = 2500.0                # range over which the model curves are computed
V_PLOT = 2000.0               # displayed x-range (all panels)
MIN_COUNT = 1                 # rebinned v-bins with fewer pairs are masked (1 = only empty bins)
Y_FLOOR = 1e-5                # log-y lower limit, as a fraction of each panel's peak
SUM_FIELDS = ('n', 'sum_v', 'sum_v2', 'counts', 'sum_vr', 'sum_vr2')


def load_stacked_pairhist(sims, k, cells=()):
    """Sum the accumulator arrays over phases. Returns an object exposing the same
    attributes (indexed [0]) that this cell uses. For each (j, i) in `cells`, also
    keeps the per-phase normalized, rebinned v_los densities
    (phase_dens[(j, i)] has shape (n_phases, n_vbins))."""
    acc, first = None, None
    phase_dens = {c: [] for c in cells}
    for s in sims:
        h_ = PairHist.load(VR_AUTO_TMPL.format(sim=s, k=k))
        if acc is None:
            first = h_
            acc = {f: np.array(getattr(h_, f)[0], dtype=np.float64) for f in SUM_FIELDS}
        else:
            for f in SUM_FIELDS:
                acc[f] += getattr(h_, f)[0]
        dv_ = (h_.v_centers[1] - h_.v_centers[0]) * REBIN
        for (j, i) in cells:
            c_ = h_.counts[0][j, i].reshape(-1, REBIN).sum(axis=1)
            tot = c_.sum()
            phase_dens[(j, i)].append(c_ / (tot * dv_) if tot > 0 else np.full(c_.shape, np.nan))
    return SimpleNamespace(R_edges=first.R_edges, rlos_edges=first.rlos_edges,
                           v_centers=first.v_centers,
                           phase_dens={c: np.array(v) for c, v in phase_dens.items()},
                           **{f: [acc[f]] for f in SUM_FIELDS})


fig7_ready = (K_LOW in per_bin_params5) and (K_LOW in fig1_results)
if not fig7_ready:
    print(f'Fig 7 (nu scan): bin {K_LOW} needs both a Fig 5 fit and a Fig 1 (b1, b2) fit -- skipping')
elif ERR_MODE not in ('poisson', 'phase', 'none'):
    raise ValueError(f"ERR_MODE must be 'poisson', 'phase' or 'none', got {ERR_MODE!r}")
elif ERR_MODE == 'phase' and len(per_bin_sims5[K_LOW]) < 2:
    raise ValueError('ERR_MODE = "phase" needs at least 2 phases')
else:
    sims7 = per_bin_sims5[K_LOW]
    cells7 = [(int(np.argmin(np.abs(r_p - rp))), int(np.argmin(np.abs(r_l - rl))))
              for rl in RL_SEL for rp in RP_SEL]
    h7 = load_stacked_pairhist(sims7, K_LOW, cells=cells7 if ERR_MODE == 'phase' else ())

    # --- model inputs for Pv_compact ---
    # mean radial velocity: Fig 1 LPT model at this bin's best-fit (b1, b2), tabulated to a spline
    b1_7, b2_7 = fig1_results[K_LOW]['b1'], fig1_results[K_LOW]['b2']
    r_tab7 = np.linspace(1.0, 200.0, 400)
    mean_vr7 = InterpolatedUnivariateSpline(r_tab7, model_vr(r_tab7, b1_7, b2_7), k=3, ext=3)
    # velocity parameters: this mass bin's mean over phases (the black points in Fig 5)
    pars_pv7 = mean_vals5[K_LOW]

    # one skew-t <-> (Delta, sigma_v) interpolator per nu
    skew_by_nu7 = {nu: cm.SkewTMeanInterpolator(nu=nu) for nu in NU_LIST}

    def pv_at_nu(nu, v, rp, rl):
        """Pv_compact with the module-level nu temporarily swapped; original restored after."""
        orig = cm.skew_interp
        cm.skew_interp = skew_by_nu7[nu]
        try:
            return Pv_compact(v, rp, rl, mean_vr7, pars_pv7)
        finally:
            cm.skew_interp = orig

    cent7 = h7.v_centers.reshape(-1, REBIN).mean(axis=1)
    dv7 = (h7.v_centers[1] - h7.v_centers[0]) * REBIN
    edges7 = np.r_[cent7 - dv7 / 2, cent7[-1] + dv7 / 2]
    vv7 = np.linspace(-V_LIM, V_LIM, 600)

    # inset quarter-disk radius: just beyond the farthest selected cell, rounded up to 10
    r3_sel7 = [np.hypot(r_p[np.argmin(np.abs(r_p - rp))], r_l[np.argmin(np.abs(r_l - rl))])
               for rp in RP_SEL for rl in RL_SEL]
    R_INSET = 10.0 * np.ceil((max(r3_sel7) + 5.0) / 10.0)
    INSET_PAD = 0.12 * R_INSET    # left/bottom margin inside the inset
    theta7 = np.linspace(0, np.pi / 2, 100)

    def plot_nu_scan(fname, logy):
        nr, nc = len(RL_SEL), len(RP_SEL)
        fig, axs = plt.subplots(nr, nc, figsize=(5 * nc, 4 * nr), layout='constrained',
                                sharex=True)
        axs = np.atleast_2d(axs)

        for a, rl_t in enumerate(RL_SEL):
            for b_, rp_t in enumerate(RP_SEL):
                ax_ = axs[a, b_]
                j, i = np.argmin(np.abs(r_p - rp_t)), np.argmin(np.abs(r_l - rl_t))
                n = h7.n[0][j, i]
                if n < N_MIN:
                    ax_.set_title('too few pairs')
                    continue

                counts = h7.counts[0][j, i].reshape(-1, REBIN).sum(axis=1)
                norm = counts.sum() * dv7
                hist = counts / norm
                ok = counts >= MIN_COUNT
                # stairs on the true bin edges; masked bins drop to a value far below the
                # axis floor (instead of NaN, which breaks the path and loses the vertical
                # edges of isolated bins)
                ax_.stairs(np.where(ok, hist, 1e-30), edges7, color='k', lw=1,
                           label='simulation')

                # histogram errors: Poisson on the stacked counts, or the standard error of
                # the mean of the per-phase densities
                if ERR_MODE != 'none':
                    if ERR_MODE == 'poisson':
                        err = np.sqrt(counts) / norm
                    else:
                        pd = h7.phase_dens[(int(j), int(i))]
                        err = np.nanstd(pd, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(pd), axis=0))
                    err_lo = np.minimum(err, 0.999 * hist) if logy else err   # keep log-axis lower bar positive
                    ax_.errorbar(cent7[ok], hist[ok], yerr=[err_lo[ok], err[ok]], fmt='none',
                                 ecolor='k', elinewidth=0.8, capsize=0)

                ymax = np.nanmax(hist)
                for n_, nu in enumerate(NU_LIST):
                    pdf_nu = pv_at_nu(nu, vv7, r_p[j], r_l[i])
                    ymax = max(ymax, pdf_nu.max())
                    ax_.plot(vv7, pdf_nu, c=NU_COLORS[n_ % len(NU_COLORS)],
                             lw=2.4 if nu == 6 else 1.6, label=rf'$\nu={nu}$')

                ax_.axvline(0, c='gray', ls=':', lw=1)
                ax_.set_xlim(-V_PLOT, V_PLOT)
                if logy:
                    ax_.set_yscale('log')
                    ax_.set_ylim(np.nanmax(hist) * Y_FLOOR, np.nanmax(hist) * 2)
                else:
                    ax_.set_ylim(0, 1.1 * ymax)
                ax_.set_title(rf'$r_\perp\in[{r_perp_edges[j]:.2f}, {r_perp_edges[j+1]:.2f}]$, '
                              rf'$r_{{\rm los}}\in[{r_los_edges[i]:.2f}, {r_los_edges[i+1]:.2f}]$ '
                              r'$h^{-1}$Mpc', fontsize=12)
                if a == nr - 1:
                    ax_.set_xlabel(r'$v_{\rm los}$ [km/s]')

                # inset: where this bin sits on the (R, r_los) quarter-disk
                ins = ax_.inset_axes([0.82, 0.72, 0.16, 0.24])
                ins.fill(np.r_[-INSET_PAD, R_INSET, R_INSET * np.cos(theta7), -INSET_PAD],
                         np.r_[-INSET_PAD, -INSET_PAD, R_INSET * np.sin(theta7), R_INSET],
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
        axs[0, 0].legend(fontsize=8, ncol=2, loc='lower center' if logy else 'upper left')
        fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150)
        print(f'saved {os.path.join(FIGURES_DIR, fname)} ({len(sims7)} phases stacked, '
              f'bin {K_LOW}, b1={b1_7:.3f}, b2={b2_7:.3f}, inset radius {R_INSET:.0f}, errors: {ERR_MODE})')

    plot_nu_scan('fig7_nu_scan.png', logy=True)
    plot_nu_scan('fig7_nu_scan_linear.png', logy=False)



from scipy.optimize import minimize_scalar

#%% Fig 7 (nu scan, Delta fit per panel) -- compact skew-t with Delta fitted to each panel for each nu, low-mass bin, all phases
import compact.model as cm

K_LOW = 0
RP_SEL = [15.5, 55.5, 135.5]  # columns (r_perp)
RL_SEL = [15.5, 55.5, 135.5]  # rows (r_los)
NU_LIST = [4, 6, 9, 12, 18, 24]
NU_COLORS = ['C3', 'C0', 'C2', 'C1', 'C4', 'C5']   # red, blue, green, orange, purple, brown
ERR_MODE = 'poisson'          # histogram error bars: 'poisson', 'phase' (phase-to-phase), or 'none'
V_LIM = 2500.0                # range over which the model curves are computed
V_PLOT = 2000.0               # displayed x-range (all panels)
V_FIT = 2500.0                # Delta fit uses bins within +/- V_FIT of the model mean (as in fit_skewt_grid)
MIN_COUNT = 1                 # rebinned v-bins with fewer pairs are masked (1 = only empty bins)
Y_FLOOR = 1e-5                # log-y lower limit, as a fraction of each panel's peak
SUM_FIELDS = ('n', 'sum_v', 'sum_v2', 'counts', 'sum_vr', 'sum_vr2')


def load_stacked_pairhist(sims, k, cells=()):
    """Sum the accumulator arrays over phases. Returns an object exposing the same
    attributes (indexed [0]) that this cell uses. For each (j, i) in `cells`, also
    keeps the per-phase normalized, rebinned v_los densities
    (phase_dens[(j, i)] has shape (n_phases, n_vbins))."""
    acc, first = None, None
    phase_dens = {c: [] for c in cells}
    for s in sims:
        h_ = PairHist.load(VR_AUTO_TMPL.format(sim=s, k=k))
        if acc is None:
            first = h_
            acc = {f: np.array(getattr(h_, f)[0], dtype=np.float64) for f in SUM_FIELDS}
        else:
            for f in SUM_FIELDS:
                acc[f] += getattr(h_, f)[0]
        dv_ = (h_.v_centers[1] - h_.v_centers[0]) * REBIN
        for (j, i) in cells:
            c_ = h_.counts[0][j, i].reshape(-1, REBIN).sum(axis=1)
            tot = c_.sum()
            phase_dens[(j, i)].append(c_ / (tot * dv_) if tot > 0 else np.full(c_.shape, np.nan))
    return SimpleNamespace(R_edges=first.R_edges, rlos_edges=first.rlos_edges,
                           v_centers=first.v_centers,
                           phase_dens={c: np.array(v) for c, v in phase_dens.items()},
                           **{f: [acc[f]] for f in SUM_FIELDS})


fig7_ready = (K_LOW in per_bin_params5) and (K_LOW in fig1_results)
if not fig7_ready:
    print(f'Fig 7 (nu scan): bin {K_LOW} needs both a Fig 5 fit and a Fig 1 (b1, b2) fit -- skipping')
elif ERR_MODE not in ('poisson', 'phase', 'none'):
    raise ValueError(f"ERR_MODE must be 'poisson', 'phase' or 'none', got {ERR_MODE!r}")
elif ERR_MODE == 'phase' and len(per_bin_sims5[K_LOW]) < 2:
    raise ValueError('ERR_MODE = "phase" needs at least 2 phases')
else:
    sims7 = per_bin_sims5[K_LOW]
    cells7 = [(int(np.argmin(np.abs(r_p - rp))), int(np.argmin(np.abs(r_l - rl))))
              for rl in RL_SEL for rp in RP_SEL]
    h7 = load_stacked_pairhist(sims7, K_LOW, cells=cells7 if ERR_MODE == 'phase' else ())

    # --- model inputs ---
    # mean radial velocity: Fig 1 LPT model at this bin's best-fit (b1, b2), tabulated to a spline
    b1_7, b2_7 = fig1_results[K_LOW]['b1'], fig1_results[K_LOW]['b2']
    r_tab7 = np.linspace(1.0, 200.0, 400)
    mean_vr7 = InterpolatedUnivariateSpline(r_tab7, model_vr(r_tab7, b1_7, b2_7), k=3, ext=3)
    # velocity parameters: this mass bin's mean over phases (the black points in Fig 5)
    pars_pv7 = mean_vals5[K_LOW]

    # one skew-t <-> (Delta, sigma_v) interpolator per nu
    skew_by_nu7 = {nu: cm.SkewTMeanInterpolator(nu=nu) for nu in NU_LIST}

    def mu_sig7(rp, rl):
        """Model mean (projected on the LOS) and sigma_v, exactly as Pv_compact computes them."""
        r = np.hypot(rp, rl)
        mu = mean_vr7(r) * rl / r
        sig = cm.sigma_mod(r - rp, pars_pv7[0], cm.A_rho_mod(rp, pars_pv7[1], pars_pv7[2]), pars_pv7[3])
        return mu, sig

    def pdf_at_delta(nu, delta, v, rp, rl):
        """Compact skew-t at the model mean and sigma_v, with Delta supplied instead of Delta(x, r_perp)."""
        mu, sig = mu_sig7(rp, rl)
        omega, alpha = skew_by_nu7[nu].get_skew_params(delta, sig)
        return cm.skewed_t_pdf(v, mu + delta, omega, alpha, nu)

    def delta_calibrated(rp, rl):
        """Delta from the 9-parameter model (what Pv_compact would use), for reference."""
        x = np.hypot(rp, rl) - rp
        A_d = cm.A_delta_mod(rp, pars_pv7[4], pars_pv7[5])
        return cm.delta_mod_alpha(x, A_d, pars_pv7[6], pars_pv7[7], pars_pv7[8])

    def fit_delta(nu, rp, rl, cent, hist):
        """Least squares on the density (as in the calibration), Delta the only free parameter."""
        mu, sig = mu_sig7(rp, rl)
        xr = skew_by_nu7[nu].R_to_alpha.x              # valid Delta/sigma range of the interpolator
        sel = np.abs(cent - mu) <= V_FIT
        norm2 = np.sum(hist[sel]**2)
        loss = lambda d: np.sum((pdf_at_delta(nu, d, cent[sel], rp, rl) - hist[sel])**2) / norm2
        res = minimize_scalar(loss, bounds=(0.98 * xr[0] * sig, 0.98 * xr[-1] * sig),
                              method='bounded', options={'xatol': 1e-3})
        return float(res.x)

    cent7 = h7.v_centers.reshape(-1, REBIN).mean(axis=1)
    dv7 = (h7.v_centers[1] - h7.v_centers[0]) * REBIN
    edges7 = np.r_[cent7 - dv7 / 2, cent7[-1] + dv7 / 2]
    vv7 = np.linspace(-V_LIM, V_LIM, 600)

    # --- fit Delta for every (panel, nu) once; reused by both figures ---
    fits7 = {}      # (j, i) -> {nu: (Delta_fit, pdf on vv7)}
    print(f'Fitted Delta [km/s] per panel (bin {K_LOW}); "cal" = Delta from the 9-parameter model')
    for (j, i) in dict.fromkeys(cells7):
        if h7.n[0][j, i] < N_MIN:
            continue
        counts_ = h7.counts[0][j, i].reshape(-1, REBIN).sum(axis=1)
        hist_ = counts_ / (counts_.sum() * dv7)
        fits7[(j, i)] = {}
        for nu in NU_LIST:
            d_fit = fit_delta(nu, r_p[j], r_l[i], cent7, hist_)
            fits7[(j, i)][nu] = (d_fit, pdf_at_delta(nu, d_fit, vv7, r_p[j], r_l[i]))
        print(f'  r_perp={r_p[j]:6.1f}, r_los={r_l[i]:6.1f}:  cal={delta_calibrated(r_p[j], r_l[i]):6.1f} | '
              + '  '.join(f'nu={nu}: {fits7[(j, i)][nu][0]:6.1f}' for nu in NU_LIST))

    # inset quarter-disk radius: just beyond the farthest selected cell, rounded up to 10
    r3_sel7 = [np.hypot(r_p[np.argmin(np.abs(r_p - rp))], r_l[np.argmin(np.abs(r_l - rl))])
               for rp in RP_SEL for rl in RL_SEL]
    R_INSET = 10.0 * np.ceil((max(r3_sel7) + 5.0) / 10.0)
    INSET_PAD = 0.12 * R_INSET    # left/bottom margin inside the inset
    theta7 = np.linspace(0, np.pi / 2, 100)

    def plot_nu_scan(fname, logy):
        nr, nc = len(RL_SEL), len(RP_SEL)
        fig, axs = plt.subplots(nr, nc, figsize=(5 * nc, 4 * nr), layout='constrained',
                                sharex=True)
        axs = np.atleast_2d(axs)

        for a, rl_t in enumerate(RL_SEL):
            for b_, rp_t in enumerate(RP_SEL):
                ax_ = axs[a, b_]
                j, i = np.argmin(np.abs(r_p - rp_t)), np.argmin(np.abs(r_l - rl_t))
                n = h7.n[0][j, i]
                if n < N_MIN:
                    ax_.set_title('too few pairs')
                    continue

                counts = h7.counts[0][j, i].reshape(-1, REBIN).sum(axis=1)
                norm = counts.sum() * dv7
                hist = counts / norm
                ok = counts >= MIN_COUNT
                # stairs on the true bin edges; masked bins drop to a value far below the
                # axis floor (instead of NaN, which breaks the path and loses the vertical
                # edges of isolated bins)
                ax_.stairs(np.where(ok, hist, 1e-30), edges7, color='k', lw=1,
                           label='simulation')

                # histogram errors: Poisson on the stacked counts, or the standard error of
                # the mean of the per-phase densities
                if ERR_MODE != 'none':
                    if ERR_MODE == 'poisson':
                        err = np.sqrt(counts) / norm
                    else:
                        pd = h7.phase_dens[(int(j), int(i))]
                        err = np.nanstd(pd, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(pd), axis=0))
                    err_lo = np.minimum(err, 0.999 * hist) if logy else err   # keep log-axis lower bar positive
                    ax_.errorbar(cent7[ok], hist[ok], yerr=[err_lo[ok], err[ok]], fmt='none',
                                 ecolor='k', elinewidth=0.8, capsize=0)

                # model Delta (blank legend entry, then one entry per nu with its best-fit Delta)
                ax_.plot([], [], ' ', label=rf'$\Delta_{{\rm model}}={delta_calibrated(r_p[j], r_l[i]):.1f}$ km/s')
                ymax = np.nanmax(hist)
                for n_, nu in enumerate(NU_LIST):
                    d_fit, pdf_nu = fits7[(int(j), int(i))][nu]
                    ymax = max(ymax, pdf_nu.max())
                    ax_.plot(vv7, pdf_nu, c=NU_COLORS[n_ % len(NU_COLORS)],
                             lw=2.4 if nu == 6 else 1.6,
                             label=rf'$\nu={nu}$: $\Delta_{{\rm fit}}={d_fit:.1f}$')

                ax_.axvline(0, c='gray', ls=':', lw=1)
                ax_.set_xlim(-V_PLOT, V_PLOT)
                if logy:
                    ax_.set_yscale('log')
                    ax_.set_ylim(np.nanmax(hist) * Y_FLOOR, np.nanmax(hist) * 2)
                else:
                    ax_.set_ylim(0, 1.1 * ymax)
                ax_.legend(fontsize=7, ncol=1, framealpha=0.85, labelspacing=0.25,
                           loc='lower center' if logy else 'upper left')
                ax_.set_title(rf'$r_\perp\in[{r_perp_edges[j]:.2f}, {r_perp_edges[j+1]:.2f}]$, '
                              rf'$r_{{\rm los}}\in[{r_los_edges[i]:.2f}, {r_los_edges[i+1]:.2f}]$ '
                              r'$h^{-1}$Mpc', fontsize=12)
                if a == nr - 1:
                    ax_.set_xlabel(r'$v_{\rm los}$ [km/s]')

                # inset: where this bin sits on the (R, r_los) quarter-disk
                ins = ax_.inset_axes([0.82, 0.72, 0.16, 0.24])
                ins.fill(np.r_[-INSET_PAD, R_INSET, R_INSET * np.cos(theta7), -INSET_PAD],
                         np.r_[-INSET_PAD, -INSET_PAD, R_INSET * np.sin(theta7), R_INSET],
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
        fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150)
        print(f'saved {os.path.join(FIGURES_DIR, fname)} ({len(sims7)} phases stacked, '
              f'bin {K_LOW}, b1={b1_7:.3f}, b2={b2_7:.3f}, inset radius {R_INSET:.0f}, errors: {ERR_MODE})')

    plot_nu_scan('fig7_nu_scan_fitdelta.png', logy=True)
    plot_nu_scan('fig7_nu_scan_fitdelta_linear.png', logy=False)


#%% Free-nu skew-t test, ALL radial bins -- Delta residual (free nu minus fixed nu), nu(r|r_perp) with power-law fit, and nu(x|r_perp), single phase, same bin as Fig 4
NU_BOUNDS = (2.1, 1000.0)    # free-nu bounds (variance needs nu > 2; ~1000 is effectively skew-normal)


def fit_skewt_row_free(n_row, sv_row, sv2_row, counts_row, cent_fine):
    """[xi, omega, alpha, nu] per cell in one R row: same recipe as fit_skewt_grid, but nu is free.
    nu is fitted as ln(nu) so the optimizer is well-conditioned over a wide range."""
    dv = (cent_fine[1] - cent_fine[0]) * REBIN
    cent = cent_fine.reshape(-1, REBIN).mean(axis=1)
    out = np.full((len(n_row), 4), np.nan)
    for i in range(len(n_row)):
        n = n_row[i]
        if n < N_MIN:
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
            p, _ = curve_fit(lambda x, xi_, om_, al_, lnnu_: skewed_t_pdf(x, xi_, om_, al_, np.exp(lnnu_)),
                             x_w, hist, p0=[mu, width, -0.1, np.log(float(NU_FIXED))],
                             bounds=([-np.inf, 1e-3, -50.0, np.log(NU_BOUNDS[0])],
                                     [np.inf, np.inf, 50.0, np.log(NU_BOUNDS[1])]), max_nfev=5000)
            out[i] = [p[0], p[1], p[2], np.exp(p[3])]
        except RuntimeError:
            continue
    return out


if fig23_sim is None:
    print(f'Free-nu test: no phase has bin {K_SINGLE} auto-stats yet -- skipping')
else:
    rows_free = np.arange(r3_2d.shape[0])          # every R row, no r_perp restriction
    res_free = Parallel(n_jobs=-1, verbose=5)(
        delayed(fit_skewt_row_free)(h23.n[0][j], h23.sum_v[0][j], h23.sum_v2[0][j],
                                    h23.counts[0][j], h23.v_centers) for j in rows_free)
    ST_free = np.full(r3_2d.shape + (4,), np.nan)
    for j, row in zip(rows_free, res_free):
        ST_free[j] = row

    # Delta_fixed uses the unrestricted Fig 4 calibration data (Delta4 is only filled for r_perp <= RP_MAX_PLOT)
    Delta_free = ST_free[:, :, 0] - vr_interp4(r3_2d) * r_l_2d / r3_2d
    nu_free = ST_free[:, :, 3]

    fin = np.isfinite(nu_free)
    print(f'Free-nu fits ({fig23_sim}, bin {K_SINGLE}): {fin.sum()} converged of {fin.size} cells; '
          f'median nu = {np.nanmedian(nu_free):.1f}; '
          f'{np.mean(nu_free[fin] > 0.99 * NU_BOUNDS[1]):.1%} at the upper bound, '
          f'{np.mean(nu_free[fin] < 1.01 * NU_BOUNDS[0]):.1%} at the lower bound')

    fig, ax = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
    cmap = plt.cm.viridis(np.linspace(0, 1, len(rows_free)))
    for c, j in zip(cmap, rows_free):
        m = np.isfinite(Delta_free[j])
        mr = m & np.isfinite(Delta4_full[j])                          # residual needs both fits
        ax[0].plot(x_grid[j][mr], Delta_free[j][mr] - Delta4_full[j][mr], 'o', ms=2, c=c, rasterized=True)
        ax[1].plot(r3_2d[j][m], nu_free[j][m], 'o', ms=2, c=c, rasterized=True)
        ax[2].plot(x_grid[j][m], nu_free[j][m], 'o', ms=2, c=c, rasterized=True)

    # single power-law fit nu = A r^p to all scatter points (least squares in log-log);
    # points pinned at the nu bounds are censored values, not measurements, so they are excluded
    r_all = r3_2d[fin]
    nu_all = nu_free[fin]
    free_pt = (nu_all > 1.01 * NU_BOUNDS[0]) & (nu_all < 0.99 * NU_BOUNDS[1])
    p_pl, lnA_pl = np.polyfit(np.log(r_all[free_pt]), np.log(nu_all[free_pt]), 1)
    rr = np.linspace(r_all.min(), r_all.max(), 200)
    ax[1].plot(rr, np.exp(lnA_pl) * rr**p_pl, c='r', lw=2.5, zorder=6,
               label=rf'power law: $\nu={np.exp(lnA_pl):.2g}\,r^{{{p_pl:.2f}}}$')
    print(f'nu(r) power-law fit: nu = {np.exp(lnA_pl):.3g} * r^{p_pl:.3f} '
          f'({free_pt.sum()} points used, {(~free_pt).sum()} at a bound excluded)')

    ax[0].axhline(0, c='k', ls='--', lw=1.5)
    for a_ in ax[1:]:
        a_.axhline(NU_FIXED, c='k', ls='--', lw=1.5)
        a_.set_yscale('log')
    ax[0].set_xlabel(r'$x = r - r_\perp$ [$h^{-1}$ Mpc]')
    ax[0].set_ylabel(rf'$\Delta_{{\rm free\ \nu}} - \Delta_{{\nu={NU_FIXED}}}$ [km/s]')
    ax[1].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
    ax[1].set_ylabel(r'$\nu(r|r_\perp)$')
    ax[2].set_xlabel(r'$x = r - r_\perp$ [$h^{-1}$ Mpc]')
    ax[2].set_ylabel(r'$\nu(x|r_\perp)$')
    for a_ in ax:
        a_.margins(x=0)
    ax[1].plot([], [], '--', c='k', label=rf'fixed $\nu={NU_FIXED}$')
    ax[1].legend(fontsize=9)

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=r_p.min(), vmax=r_p.max()))
    fig.colorbar(sm, ax=ax, location='right', pad=0.02, label=r'$r_\perp$ [$h^{-1}$ Mpc]')
    fig.savefig(os.path.join(FIGURES_DIR, 'fig_free_nu_delta_nu_all.png'), dpi=150)
    print(f'saved {os.path.join(FIGURES_DIR, "fig_free_nu_delta_nu_all.png")}')



#%% nu(r) functional forms -- fit to per-r-bin medians of all converged cells (uses r_all, nu_all, free_pt from the all-bins cell)
from scipy.optimize import least_squares


R_BIN = 6.0                  # r-bin width for the medians [h^-1 Mpc]
MIN_BIN_PTS = 30             # minimum cells per bin

edges_r = np.arange(r_all.min(), r_all.max() + R_BIN, R_BIN)
ib = np.digitize(r_all, edges_r) - 1
r_med, nu_med, f_pin = [], [], []
for b in range(len(edges_r) - 1):
    s = ib == b
    if s.sum() < MIN_BIN_PTS:
        continue
    r_med.append(r_all[s].mean())
    nu_med.append(np.median(nu_all[s]))
    f_pin.append(np.mean((nu_all[s] > 0.99 * NU_BOUNDS[1]) | (nu_all[s] < 1.01 * NU_BOUNDS[0])))
r_med, nu_med, f_pin = np.array(r_med), np.array(nu_med), np.array(f_pin)
if (f_pin > 0.3).any():
    print(f'WARNING: {np.sum(f_pin > 0.3)} bin(s) have >30% of cells pinned at a nu bound; their medians are unreliable '
          f'(r = {np.round(r_med[f_pin > 0.3], 0)})')

# (function, starting values, (lower bounds, upper bounds))
forms = {
    r'$A\,r^p$':                              (lambda r, A, p: A * r**p,                                      [4.0, 0.4],        ([1e-6, -5], [1e3, 5])),
    r'$2 + A\,r^p$':                          (lambda r, A, p: 2 + A * r**p,                                  [1.0, 0.7],        ([1e-6, -5], [1e3, 5])),
    r'$2 + (\nu_\infty-2)/(1+(r_c/r)^q)$':    (lambda r, ni, rc, q: 2 + (ni - 2) / (1 + (rc / r)**q),         [25.0, 40.0, 2.0], ([2.01, 1.0, 0.1], [1e3, 500.0, 20.0])),
    r'$\nu_\infty-(\nu_\infty-2)e^{-r/r_c}$': (lambda r, ni, rc: ni - (ni - 2) * np.exp(-r / rc),             [30.0, 80.0],      ([2.01, 1.0], [1e3, 1e4])),
}

fig, ax = plt.subplots(figsize=(7.5, 5.5), layout='constrained')
ax.plot(r_all[free_pt], nu_all[free_pt], 'o', ms=1.5, c='0.75', alpha=0.4, rasterized=True, label='cells')
ax.plot(r_med, nu_med, 'ko', ms=4, zorder=5, label='bin medians')
rr = np.linspace(r_all.min(), r_all.max(), 300)
print(f'{"form":42s} {"rms ln nu (medians)":>20s} {"rms ln nu (cells)":>18s}  parameters')
for n_, (name, (f, p0, bnds)) in enumerate(forms.items()):
    sol = least_squares(lambda th: np.log(np.clip(f(r_med, *th), 1e-6, None)) - np.log(nu_med),
                        p0, bounds=bnds, loss='soft_l1', f_scale=0.1)
    th = sol.x
    rms_med = np.sqrt(np.mean((np.log(f(r_med, *th)) - np.log(nu_med))**2))
    rms_cell = np.sqrt(np.mean((np.log(f(r_all[free_pt], *th)) - np.log(nu_all[free_pt]))**2))
    print(f'{name:42s} {rms_med:20.3f} {rms_cell:18.3f}  ' + ', '.join(f'{v:.3g}' for v in th))
    ax.plot(rr, f(rr, *th), lw=2, c=f'C{n_}', label=name)
ax.axhline(NU_FIXED, c='k', ls='--', lw=1.5)
ax.set_yscale('log')
ax.set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
ax.set_ylabel(r'$\nu(r|r_\perp)$')
ax.legend(fontsize=8, loc='lower right')
fig.savefig(os.path.join(FIGURES_DIR, 'fig_nu_r_forms.png'), dpi=150)
print(f'saved {os.path.join(FIGURES_DIR, "fig_nu_r_forms.png")}')
#%% Compact model with nu(r) from the single-phase free-nu power law, Delta refit to the nu(r)-based calibration data -- single phase, same bin as Fig 4
NU_A, NU_P = np.exp(lnA_pl), p_pl      # nu(r) = NU_A * r^NU_P, from the single-phase free-nu cell
NU_CLIP = (6.0, 200.0)                 # nu(r) is clipped to this range (it is only calibrated for r <~ 70)
RP_SEL = [5.5, 15.5, 29.5]             # columns (r_perp)
RL_SEL = [5.5, 30.5, 60.5]             # rows (r_los)
LOGY = False                            # log or linear y-axis
V_LIM = 2500.0                         # range over which the model curves are computed
V_PLOT = 2000.0                        # displayed x-range
MIN_COUNT = 1                          # rebinned v-bins with fewer pairs are masked
Y_FLOOR = 1e-5                         # log-y lower limit, as a fraction of each panel's peak


def nu_of_r(r):
    return np.clip(NU_A * np.asarray(r, dtype=float)**NU_P, *NU_CLIP)


def fit_skewt_row_nuvec(n_row, sv_row, sv2_row, counts_row, cent_fine, nu_row):
    """[xi, omega, alpha] per cell in one R row: same recipe as fit_skewt_grid, but with a
    per-cell nu (nu_row) instead of the constant NU_FIXED."""
    dv = (cent_fine[1] - cent_fine[0]) * REBIN
    cent = cent_fine.reshape(-1, REBIN).mean(axis=1)
    out = np.full((len(n_row), 3), np.nan)
    for i in range(len(n_row)):
        n = n_row[i]
        if n < N_MIN:
            continue
        mu = sv_row[i] / n
        width = np.sqrt(max(sv2_row[i] / n - mu**2, 0.0))
        counts = counts_row[i].reshape(-1, REBIN).sum(axis=1)
        sel = np.abs(cent - mu) <= 2500
        c_w, x_w = counts[sel], cent[sel]
        if c_w.sum() == 0:
            continue
        hist = c_w / (c_w.sum() * dv)
        nu_i = nu_row[i]
        try:
            p, _ = curve_fit(lambda xx, xi_, om_, al_: skewed_t_pdf(xx, xi_, om_, al_, nu_i),
                             x_w, hist, p0=[mu, width, -0.1], method='lm', maxfev=5000)
            out[i] = p
        except RuntimeError:
            continue
    return out


if fig23_sim is None:
    print(f'nu(r) refit: no phase has bin {K_SINGLE} auto-stats yet -- skipping')
else:
    # --- new Delta calibration data: skew-t fit per cell with nu = nu(r) ---
    nu_grid = nu_of_r(r3_2d)
    res_nu = Parallel(n_jobs=-1, verbose=5)(
        delayed(fit_skewt_row_nuvec)(h23.n[0][j], h23.sum_v[0][j], h23.sum_v2[0][j],
                                     h23.counts[0][j], h23.v_centers, nu_grid[j])
        for j in range(r3_2d.shape[0]))
    ST_nu = np.array(res_nu)
    Delta_nu = ST_nu[:, :, 0] - vr_interp4(r3_2d) * r_l_2d / r3_2d

    # --- refit the existing Delta function to it (full available grid, as in Fig 4) ---
    del_params_nu = fit_delta_joint(x_grid, r_p, Delta_nu, p0=del_params)
    print(f'nu(r) = {NU_A:.3g} * r^{NU_P:.3f}, clipped to {NU_CLIP}')
    print(f'Delta params (A_delta0, r_delta, sigma_delta, beta_delta, alpha):')
    print(f'  nu = 6 (Fig 4): ' + ', '.join(f'{v:.3g}' for v in del_params))
    print(f'  nu = nu(r)    : ' + ', '.join(f'{v:.3g}' for v in del_params_nu))

    # --- model curves: measured mean (as in the calibration), sigma_v from the Fig 2&3 fit ---
    skew6 = cm.SkewTMeanInterpolator(nu=6)

    def model_pdf(skew_obj, nu, dpars, rp, rl, v):
        r = np.hypot(rp, rl)
        x = r - rp
        mu = vr_interp4(r) * rl / r
        sig = sigma_mod_2d((x, rp), *sig_params, R_piv)
        delta = delta_mod_alpha_R((x, rp), *dpars)
        omega, alpha = skew_obj.get_skew_params(delta, sig)
        return skewed_t_pdf(v, mu + delta, omega, alpha, nu)

    cent_h = h23.v_centers.reshape(-1, REBIN).mean(axis=1)
    dv_h = (h23.v_centers[1] - h23.v_centers[0]) * REBIN
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
            if h23.n[0][j, i] < N_MIN:
                ax_.set_title('too few pairs')
                continue

            counts = h23.counts[0][j, i].reshape(-1, REBIN).sum(axis=1)
            norm = counts.sum() * dv_h
            hist = counts / norm
            ok = counts >= MIN_COUNT
            ax_.stairs(np.where(ok, hist, 1e-30), edges_h, color='k', lw=1, label='simulation')
            err = np.sqrt(counts) / norm
            err_lo = np.minimum(err, 0.999 * hist) if LOGY else err
            ax_.errorbar(cent_h[ok], hist[ok], yerr=[err_lo[ok], err[ok]], fmt='none',
                         ecolor='k', elinewidth=0.8, capsize=0)

            r3 = r3_2d[j, i]
            nu_c = float(nu_of_r(r3))
            pdf_ref = model_pdf(skew6, 6, del_params, r_p[j], r_l[i], vv)
            pdf_new = model_pdf(cm.SkewTMeanInterpolator(nu=nu_c), nu_c, del_params_nu, r_p[j], r_l[i], vv)
            ax_.plot(vv, pdf_ref, c='C1', lw=1.8, ls='--', label=r'$\nu=6$, Fig 4 $\Delta$')
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
    fname = 'fig_nu_function_refit_delta.png'
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150)
    print(f'saved {os.path.join(FIGURES_DIR, fname)} ({fig23_sim}, bin {K_SINGLE})')

#%% Free-nu power law over all phases and masses: nu(r) = nu_piv * (r / R_NU_PIV)^p per (phase, mass bin), Fig 5-style summary
NU_BOUNDS = (2.1, 1000.0)    # free-nu bounds (same as the single-phase test)
R_NU_PIV = 30.0              # pivot for the power law; set to 1.0 to get nu = A r^p
MIN_NU_PTS = 20              # minimum unbounded nu values needed to fit a (phase, bin)
NU_CACHE = os.path.join(FIGURES_DIR, 'fig_nu_powerlaw_cache.h5')


def fit_skewt_row_free(n_row, sv_row, sv2_row, counts_row, cent_fine, keep):
    """[xi, omega, alpha, nu] per cell in one R row: same recipe as fit_skewt_grid, but nu is free
    (fitted as ln nu). Only cells with keep=True are fit."""
    dv = (cent_fine[1] - cent_fine[0]) * REBIN
    cent = cent_fine.reshape(-1, REBIN).mean(axis=1)
    out = np.full((len(n_row), 4), np.nan)
    for i in range(len(n_row)):
        n = n_row[i]
        if (not keep[i]) or n < N_MIN:
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
            p, _ = curve_fit(lambda x, xi_, om_, al_, lnnu_: skewed_t_pdf(x, xi_, om_, al_, np.exp(lnnu_)),
                             x_w, hist, p0=[mu, width, -0.1, np.log(float(NU_FIXED))],
                             bounds=([-np.inf, 1e-3, -50.0, np.log(NU_BOUNDS[0])],
                                     [np.inf, np.inf, 50.0, np.log(NU_BOUNDS[1])]), max_nfev=5000)
            out[i] = [p[0], p[1], p[2], np.exp(p[3])]
        except RuntimeError:
            continue
    return out


def fit_nu_powerlaw_phase_bin(sim, k):
    """Returns (sim, k, [nu_piv, p] or None, fraction of fits at a bound, err string or None)."""
    path = VR_AUTO_TMPL.format(sim=sim, k=k)
    if not os.path.exists(path):
        return sim, k, None, np.nan, 'file missing'
    try:
        h = PairHist.load(path)
    except Exception as e:
        return sim, k, None, np.nan, f'PairHist.load: {type(e).__name__}: {e}'

    r_pts, nu_pts = [], []
    for j in np.where(rp_row_mask)[0]:
        row = fit_skewt_row_free(h.n[0][j], h.sum_v[0][j], h.sum_v2[0][j], h.counts[0][j],
                                 h.v_centers, r3_2d[j] <= R3_MAX_PLOT)
        ok = np.isfinite(row[:, 3])
        r_pts.append(r3_2d[j][ok])
        nu_pts.append(row[ok, 3])
    r_pts, nu_pts = np.concatenate(r_pts), np.concatenate(nu_pts)

    # points pinned at the nu bounds are censored values, not measurements: excluded from the fit
    free = (nu_pts > 1.01 * NU_BOUNDS[0]) & (nu_pts < 0.99 * NU_BOUNDS[1])
    if free.sum() < MIN_NU_PTS:
        return sim, k, None, 1 - free.mean() if len(free) else np.nan, f'only {free.sum()} unbounded nu values'
    p, ln_nu_piv = np.polyfit(np.log(r_pts[free] / R_NU_PIV), np.log(nu_pts[free]), 1)
    return sim, k, np.array([np.exp(ln_nu_piv), p]), 1 - free.mean(), None


def jackknife_mean_err(arr):
    n = arr.shape[0]
    total = arr.sum(axis=0)
    mean_vals = total / n
    loo_means = (total - arr) / (n - 1)
    err = np.sqrt((n - 1) / n * np.sum((loo_means - mean_vals)**2, axis=0))
    return mean_vals, err


labels_nu = [rf'$\nu(r={R_NU_PIV:g}\,h^{{-1}}\,{{\rm Mpc}})$', r'$p_\nu$']
mean_nu = np.full((N_MASS, 2), np.nan)
jack_nu = np.full((N_MASS, 2), np.nan)
per_bin_nu, per_bin_sims_nu = {}, {}

if os.path.exists(NU_CACHE):
    print(f'nu power law: loading cached fits from {NU_CACHE} (delete this file to force a re-fit)')
    with h5.File(NU_CACHE, 'r') as hdf:
        for k in range(N_MASS):
            gk = f'bin_{k}'
            if gk not in hdf:
                continue
            per_bin_nu[k] = hdf[gk]['params'][()]
            per_bin_sims_nu[k] = [s.decode() if isinstance(s, bytes) else s for s in hdf[gk]['sims'][()]]
            mean_nu[k], jack_nu[k] = jackknife_mean_err(per_bin_nu[k])
else:
    nu_tasks = [(sim, k) for sim in ALL_PLANNED_PHASES for k in range(N_MASS)
                if os.path.exists(VR_AUTO_TMPL.format(sim=sim, k=k))]
    print(f'nu power law: {len(nu_tasks)} (phase, bin) combinations -- fitting (will cache to {NU_CACHE})')
    nu_results = Parallel(n_jobs=-1, verbose=10)(
        delayed(fit_nu_powerlaw_phase_bin)(sim, k) for sim, k in nu_tasks)

    by_bin, err_counts = {k: {} for k in range(N_MASS)}, {}
    frac_bound = {k: [] for k in range(N_MASS)}
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

#%% nu power law -- plot (fast to re-run once cached)
labels_nu = [rf'$\nu(r={R_NU_PIV:g}\,h^{{-1}}\,{{\rm Mpc}})$', r'$\nu_s$']

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
            if not ks_sim:
                continue
            vals = [per_bin_nu[k][per_bin_sims_nu[k].index(sim), i] for k in ks_sim]
            ax[i].plot(M_CENS[ks_sim], vals, marker='o', alpha=0.3)

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

    handles, leg_labels = ax[0].get_legend_handles_labels()
    ax[0].legend(handles, leg_labels, fontsize=9, loc='upper left')
    fig.savefig(os.path.join(FIGURES_DIR, 'fig_nu_powerlaw_mass_dependence.png'), dpi=150)
    print(f'saved {os.path.join(FIGURES_DIR, "fig_nu_powerlaw_mass_dependence.png")}')
else:
    print('nu power law: no bin has enough phases yet -- nothing to plot')

#%% Fig 7 (two mass bins) -- P(v_los | r_perp, r_los): histograms vs compact skew-t vs Gaussian, all phases
from scipy.special import gamma as gamma_fn
from types import SimpleNamespace
from matplotlib.lines import Line2D

K_LO, K_HI = 0, N_MASS - 1
COL_LO, COL_HI = 'C0', 'C1'
RP_SEL = [5.5, 15.5, 29.5]    # columns (r_perp)
RL_SEL = [5.5, 15.5, 29.5]    # rows (r_los)
V_LIM = 2500.0                # range over which the model curves are computed
V_PLOT = 2000.0               # displayed x-range (all panels)
MIN_COUNT = 1                 # rebinned v-bins with fewer pairs are masked (1 = only empty bins)
Y_FLOOR = 1e-5                # log-y lower limit, as a fraction of the panel's highest peak
R_INSET = 140.0               # quarter-disk radius in the inset (assumed r cut of the accumulator)
INSET_PAD = 0.12 * R_INSET    # left/bottom margin inside the inset
SUM_FIELDS = ('n', 'sum_v', 'sum_v2', 'counts', 'sum_vr', 'sum_vr2')


def skewt_from_moments(mu, xi, sigma, nu=NU_FIXED):
    """omega, alpha of a nu-dof skew-t with location xi whose mean is mu and
    std is sigma. Uses mean = xi + omega*delta*b, var = omega^2 nu/(nu-2) - (mean-xi)^2."""
    b = np.sqrt(nu / np.pi) * gamma_fn((nu - 1) / 2) / gamma_fn(nu / 2)
    d_off = mu - xi
    omega = np.sqrt((sigma**2 + d_off**2) * (nu - 2) / nu)
    delta = np.clip(d_off / (omega * b), -0.99, 0.99)
    return omega, delta / np.sqrt(1 - delta**2)


def gauss_pdf(v, mu, sd):
    return np.exp(-0.5 * ((v - mu) / sd)**2) / (np.sqrt(2 * np.pi) * sd)


def load_stacked_pairhist(sims, k):
    """Sum the accumulator arrays over phases. Returns an object exposing the same
    attributes (indexed [0]) that vr_profile / meas_sigv_grid / this cell use."""
    acc, first = None, None
    for s in sims:
        h_ = PairHist.load(VR_AUTO_TMPL.format(sim=s, k=k))
        if acc is None:
            first = h_
            acc = {f: np.array(getattr(h_, f)[0], dtype=np.float64) for f in SUM_FIELDS}
        else:
            for f in SUM_FIELDS:
                acc[f] += getattr(h_, f)[0]
    return SimpleNamespace(R_edges=first.R_edges, rlos_edges=first.rlos_edges,
                           v_centers=first.v_centers, **{f: [acc[f]] for f in SUM_FIELDS})


def setup_mass_bin(k):
    """Stacked data + phase-averaged model parameters + sim-measured GSM dispersions for bin k."""
    sims = per_bin_sims5[k]
    p = per_bin_params5[k].mean(axis=0)
    h = load_stacked_pairhist(sims, k)

    # R_piv is not stored in the cache; recompute as fit_sigma_joint does (on the stacked data)
    sig = meas_sigv_grid(h, 0)
    m = np.isfinite(sig) & (sig > SIG_FLOOR)
    R_piv = np.median(np.broadcast_to(r_p[:, None], sig.shape)[m])

    r_vr, vr, sd_r, npairs = vr_profile(h, 0, VR_R_EDGES)
    good = np.isfinite(vr) & (npairs > 1000)
    vr_interp = InterpolatedUnivariateSpline(r_vr[good], vr[good], k=3, ext=3)

    # sigma_los^2 (about mu*v12) = mu^2 sig_par^2 + (1-mu^2) sig_perp^2
    mu2 = (r_l_2d / r3_2d)**2
    mu_grid = vr_interp(r3_2d) * r_l_2d / r3_2d
    n = h.n[0].astype(float)
    nz = np.where(n > 0, n, 1.0)
    s2c = h.sum_v2[0] / nz - 2 * mu_grid * h.sum_v[0] / nz + mu_grid**2
    sig_par2 = np.interp(r3_2d, r_vr[good], sd_r[good]**2)
    sig_perp2_cell = (s2c - mu2 * sig_par2) / (1 - mu2)
    nb = len(VR_R_EDGES) - 1
    idx = np.searchsorted(VR_R_EDGES, r3_2d, side='right') - 1
    sel = (np.isfinite(sig_perp2_cell) & (n >= N_MIN) & (mu2 < 0.8) & (idx >= 0) & (idx < nb))
    num = np.bincount(idx[sel], weights=(n * sig_perp2_cell)[sel], minlength=nb)
    den = np.bincount(idx[sel], weights=n[sel], minlength=nb)
    sig_perp2_r = num / np.where(den > 0, den, np.nan)
    good_perp = np.isfinite(sig_perp2_r) & (sig_perp2_r > 0) & good

    cent = h.v_centers.reshape(-1, REBIN).mean(axis=1)
    dv = (h.v_centers[1] - h.v_centers[0]) * REBIN
    return SimpleNamespace(k=k, h=h, n_phases=len(sims), sig_p=p[:4], del_p=p[4:], R_piv=R_piv,
                           r_vr=r_vr, sd_r=sd_r, good=good, vr_interp=vr_interp,
                           sig_perp2_r=sig_perp2_r, good_perp=good_perp,
                           cent=cent, dv=dv, edges=np.r_[cent - dv / 2, cent[-1] + dv / 2])


def bin_panel(B, j, i):
    """Histogram + the two model curves for bin-set B at grid cell (j, i); None if too few pairs."""
    if B.h.n[0][j, i] < N_MIN:
        return None
    counts = B.h.counts[0][j, i].reshape(-1, REBIN).sum(axis=1)
    hist = counts / (counts.sum() * B.dv)
    r3, x_c, R_c = r3_2d[j, i], x_grid[j, i], r_p[j]
    sigma_m = sigma_mod_2d((x_c, R_c), *B.sig_p, B.R_piv)
    mu_m = B.vr_interp(r3) * r_l[i] / r3
    xi_m = mu_m + delta_mod_alpha_R((x_c, R_c), *B.del_p)
    mu2 = (r_l[i] / r3)**2
    sd_g = np.sqrt(mu2 * np.interp(r3, B.r_vr[B.good], B.sd_r[B.good]**2)
                   + (1 - mu2) * np.interp(r3, B.r_vr[B.good_perp], B.sig_perp2_r[B.good_perp]))
    omega_m, alpha_m = skewt_from_moments(mu_m, xi_m, sigma_m)
    return dict(hist=np.where(counts >= MIN_COUNT, hist, 1e-30),
                gauss=gauss_pdf(vv7m, mu_m, sd_g),
                skewt=skewed_t_pdf(vv7m, xi_m, omega_m, alpha_m, NU_FIXED))


fig7m_ready = (K_LO in per_bin_params5) and (K_HI in per_bin_params5)
if not fig7m_ready:
    print(f'Fig 7 (two mass bins): need Fig 5 fits for bins {K_LO} and {K_HI} -- skipping')
else:
    B_lo, B_hi = setup_mass_bin(K_LO), setup_mass_bin(K_HI)
    vv7m = np.linspace(-V_LIM, V_LIM, 600)
    theta7m = np.linspace(0, np.pi / 2, 100)
    lab = lambda k: rf'$\log M\in[{LOG_MASS_EDGES[k][0]:.2f}, {LOG_MASS_EDGES[k][1]:.2f}]$'

    nr, nc = len(RL_SEL), len(RP_SEL)
    fig, axs = plt.subplots(nr, nc, figsize=(5 * nc, 4 * nr), layout='constrained', sharex=True)
    axs = np.atleast_2d(axs)

    for a, rl_t in enumerate(RL_SEL):
        for b_, rp_t in enumerate(RP_SEL):
            ax_ = axs[a, b_]
            j, i = np.argmin(np.abs(r_p - rp_t)), np.argmin(np.abs(r_l - rl_t))
            peaks = []
            for B, col in ((B_lo, COL_LO), (B_hi, COL_HI)):
                P = bin_panel(B, j, i)
                if P is None:
                    continue
                ax_.stairs(P['hist'], B.edges, color=col, lw=1.2)
                ax_.plot(vv7m, P['skewt'], c=col, ls='--', lw=2)
                ax_.plot(vv7m, P['gauss'], c=col, ls=':', lw=2)
                peaks.append(np.nanmax(np.where(P['hist'] > 1e-29, P['hist'], np.nan)))

            ax_.axvline(0, c='gray', ls=':', lw=1)
            ax_.set_xlim(-V_PLOT, V_PLOT)
            ax_.set_yscale('log')
            if peaks:
                ax_.set_ylim(max(peaks) * Y_FLOOR, max(peaks) * 2)
            ax_.set_title(rf'$r_\perp\in[{r_perp_edges[j]:.2f}, {r_perp_edges[j+1]:.2f}]$, '
                          rf'$r_{{\rm los}}\in[{r_los_edges[i]:.2f}, {r_los_edges[i+1]:.2f}]$ '
                          r'$h^{-1}$Mpc', fontsize=12)
            if a == nr - 1:
                ax_.set_xlabel(r'$v_{\rm los}$ [km/s]')

            # inset: where this bin sits on the (R, r_los) quarter-disk
            ins = ax_.inset_axes([0.82, 0.72, 0.16, 0.24])
            ins.fill(np.r_[-INSET_PAD, R_INSET, R_INSET * np.cos(theta7m), -INSET_PAD],
                     np.r_[-INSET_PAD, -INSET_PAD, R_INSET * np.sin(theta7m), R_INSET],
                     c='gray', alpha=0.4, lw=0)
            ins.axvspan(r_perp_edges[j], r_perp_edges[j + 1], color='k', alpha=0.4, lw=0)
            ins.axhspan(r_los_edges[i], r_los_edges[i + 1], color='k', alpha=0.4, lw=0)
            ins.axvline(r_p[j], c='k', lw=1)
            ins.axhline(r_l[i], c='k', lw=1)
            ins.set_xlim(-INSET_PAD, R_INSET)
            ins.set_ylim(-INSET_PAD, R_INSET)
            ins.set_xticks([])
            ins.set_yticks([])
            ins.set_xlabel(r'$R$', fontsize=7, labelpad=1)
            ins.set_ylabel(r'$r_{\rm LOS}$', fontsize=7, labelpad=1)

    handles = [Line2D([], [], c=COL_LO, lw=1.5, label=lab(K_LO)),
               Line2D([], [], c=COL_HI, lw=1.5, label=lab(K_HI)),
               Line2D([], [], c='k', ls='--', lw=2, label='compact skew-t'),
               Line2D([], [], c='k', ls=':', lw=2, label='Gaussian (GSM form)')]
    axs[0, 0].legend(handles=handles, fontsize=9, loc='upper left')
    fig.supylabel(r'$P(v_{\rm los}\,|\,r_\perp, r_{\rm los})$', fontsize=20)
    fname = 'fig7_two_mass_bins.png'
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150)
    print(f'saved {os.path.join(FIGURES_DIR, fname)} '
          f'(bin {K_LO}: {B_lo.n_phases} phases, bin {K_HI}: {B_hi.n_phases} phases)')

