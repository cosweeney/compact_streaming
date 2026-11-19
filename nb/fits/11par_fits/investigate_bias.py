"""
investigate_bias.py -- parameter shifts of the 11-parameter compact model measured on mocks, and the systematic error they add to the DR1 fits.

Cells (#%%), run in order from the folder containing fit_lib_11par.py, in the compact environment:
  1. setup and chain inventory (only the chains this file needs)
  2. validation plot of the mean-of-mocks fit with the averaged covariance (MAP model vs data, chi2)
  3. triangle plots of that fit: cosmology (against the single-mock-covariance fit, with the c000 truth) and bias parameters; offsets from the truth
  4. shift characterization from the 24 single-mock joint fits at s_min = 20: per-mock estimates, pooled bias with its uncertainty, Hotelling test,
     pull / error-calibration checks, rank (coverage) test, and a comparison with the averaged-covariance fit
  5. DR1 shift correction and systematic error: (1) the shift b is measured on the mocks; (2) it is applied to the DR1 baseline joint fit, theta -> theta - b;
     (3) the systematic error is half the shift, +- |b|/2 per parameter (Sigma_syst = b b^T / 4).
b = mean over mocks of (theta_hat_i - theta_true). The scatter of the individual theta_hat_i is statistical noise, already in each posterior, and is not
added to the systematic. The same estimator (posterior mean by default) should be used for the mocks and for quoting the DR1 result.
"""

#%% [markdown]
# ## 1. Setup

#%%
import sys, json, warnings
from pathlib import Path

import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import CubicSpline
from scipy.stats import chi2 as chi2_dist, f as f_dist, norm, kstest, t as t_dist
from getdist import MCSamples, plots
try:
    from IPython.display import display
except ImportError:
    display = print

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
sys.path.insert(0, str(HERE))
import fit_lib_11par as fl

warnings.filterwarnings('ignore', message='.*tight_layout.*')
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
                     'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False,
                     'axes.xmargin': 0, 'figure.dpi': 110})

# ---------------- configuration ----------------
CHAIN_ROOT = fl.CHAIN_ROOT
PLOT_DIR = CHAIN_ROOT / 'plots' / 'bias'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

AVGCOV = 'compact_mean_joint_avgcov_smin20'          # mean of mocks, averaged covariance
SINGLECOV = 'compact_mean_joint_smin20'              # mean of mocks, single-mock covariance
DR1 = 'compact_dr1_joint_smin20'                     # DR1 baseline joint fit
N_MOCKS = 24
MOCK_FITS = {i: f'compact_mock{i:02d}_joint_smin20' for i in range(N_MOCKS)}

PARAMS = ['omch2', 'ombh2', 'H0', 'logA', 'ns']
LABELS = [r'\Omega_c h^2', r'\Omega_b h^2', 'H_0', r'\ln(10^{10}A_s)', 'n_s']
TEST_PARAMS = ['omch2', 'H0', 'logA']                # constrained parameters used for the joint (Hotelling) test
TRUTH = dict(H0=67.36, ombh2=0.02237, omch2=0.12, logA=3.0363943, ns=0.9649)     # AbacusSummit c000
TRUTH_VEC = np.array([TRUTH[p] for p in PARAMS])
ESTIMATOR = 'mean'                                   # 'mean', 'median' or 'map': the point estimate taken from each chain
GETDIST_SETTINGS = {'smooth_scale_1D': 0.5, 'smooth_scale_2D': 0.5}
BURN_NSIGMA = 2.0
THIN_TAUS = fl.THIN_TAUS
N_BOOT = 5000

TRACER_NAMES = ['BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG2', 'QSO']
TRACER_COLORS = ['yellowgreen', 'orange', 'orangered', 'firebrick', 'steelblue', 'seagreen']
TRACER_SHORT = {fl.kname(k): n for k, n in zip(fl.FIT_KEYS, TRACER_NAMES)}
TRACER_COLOR = {k: c for k, c in zip(fl.FIT_KEYS, TRACER_COLORS)}

#%%
def find_burn_in(logp):
    # logp: (nsteps, nwalkers). Returns (burn_in, per-step median, smoothed median, late level, late scatter).
    lp = np.where(np.isfinite(logp), logp, np.nan)
    m = np.nanmedian(lp, axis=1)
    n = len(m)
    w = max(25, n // 200)
    ms = uniform_filter1d(np.nan_to_num(m, nan=np.nanmin(m)), size=w, mode='nearest')
    late = ms[n // 2:]
    level = np.median(late)
    scatter = max(1.4826 * np.median(np.abs(late - level)), 1e-12)
    above = ms >= level - BURN_NSIGMA * scatter
    ok = np.convolve(above, np.ones(w), 'valid') == w
    burn = int(np.argmax(ok)) if ok.any() else n // 2
    return burn, m, ms, level, scatter

def load_info(path):
    with h5py.File(path, 'r') as f:
        it = int(f['mcmc'].attrs['iteration'])
        logp = f['mcmc/log_prob'][:it]
    burn = find_burn_in(logp)[0]
    sp = Path(str(path)[:-3] + '.status.json')
    return dict(path=path, iterations=it, burn=burn, status=json.load(open(sp)) if sp.exists() else {})


INFO = {}
for name in [AVGCOV, SINGLECOV, DR1, *MOCK_FITS.values()]:
    p = CHAIN_ROOT / 'compact' / f'{name}.h5'
    if p.exists():
        INFO[name] = load_info(p)
missing = [n for n in [AVGCOV, SINGLECOV, DR1, *MOCK_FITS.values()] if n not in INFO]
print(f'{len(INFO)} chains found; missing: {missing if missing else "none"}')
unconverged = [n for n, i in INFO.items() if not i['status'].get('converged')]
if unconverged:
    print(f'WARNING: chains without a converged status: {unconverged}')


def post_samples(name):
    # equal-weight post-burn-in samples (thinned by THIN_TAUS * tau_max), all parameters
    info = INFO[name]
    thin = max(1, int(THIN_TAUS * info['status'].get('tau_max', 1.0)))
    with h5py.File(info['path'], 'r') as f:
        x = f['mcmc/chain'][info['burn']:info['iterations']:thin, :, :]
    return x.reshape(-1, x.shape[2])


def cosmo5(name):
    # shared cosmology (omch2, ombh2, H0, logA, ns) of a joint chain: parameter vector = [h, ombh2, omch2, logA, ns, biases...]
    x = post_samples(name)
    return np.column_stack([x[:, 2], x[:, 1], 100 * x[:, 0], x[:, 3], x[:, 4]])


def map_theta(info):
    # sample with the highest log-posterior after the burn-in
    with h5py.File(info['path'], 'r') as f:
        lp = f['mcmc/log_prob'][info['burn']:info['iterations']]
        t, w = np.unravel_index(np.nanargmax(np.where(np.isfinite(lp), lp, -np.inf)), lp.shape)
        theta = f['mcmc/chain'][info['burn'] + t, w, :]
    return theta, float(lp[t, w])

def map_cosmo5(name):
    th = map_theta(INFO[name])[0]
    return np.array([th[2], th[1], 100 * th[0], th[3], th[4]])

#%% [markdown]
# ## 2. Validation: mean-of-mocks fit with the averaged covariance
# MAP model against the data for each tracer-bin, with error bars from the averaged covariance (single-mock covariance divided by the number of mocks, per
# tracer). With this covariance a perfect model would give chi2/dof ~ 1; the excess measures the model systematic in units of the noise of the mean mock.

#%%
def validation_figure(name, save=True):
    info = INFO[name]
    spec = next(s for s in fl.build_specs() if s['name'] == name)
    problem = fl.FitProblem(spec)
    theta, lp_max = map_theta(info)
    h, ombh2, omch2, lnAs, ns, biases = problem.unpack(theta)
    aps = fl.ap_multi(h, ombh2, omch2, ns, problem.z_effs)

    n = len(problem.data)
    fig = plt.figure(figsize=(2.9 * n, 5.0))
    gs = fig.add_gridspec(2, n, height_ratios=[3, 1], hspace=0.06, wspace=0.32)
    chi2_rows = []
    for j, (key, d, (b1, b2), (qp, ql)) in enumerate(zip(spec['keys'], problem.data, biases, aps)):
        s = d['s_data']
        model = fl.model_one(d['emus'], [b1, b2, ombh2, omch2, h, lnAs, ns], s, d['W'], qp, ql).reshape(2, -1)
        data = d['data_vec'].reshape(2, -1)
        sig = np.sqrt(np.diag(np.linalg.inv(d['cov_inv']))).reshape(2, -1)
        resid = (data - model).ravel()
        chi2_rows.append((TRACER_SHORT[fl.kname(key)], resid.size, float(resid @ d['cov_inv'] @ resid)))
        color = TRACER_COLOR[key]
        ax, axr = fig.add_subplot(gs[0, j]), fig.add_subplot(gs[1, j])
        sf = np.linspace(s[0], s[-1], 300)
        for i, (marker, ls, filled) in enumerate((('o', '-', True), ('s', '--', False))):
            ax.errorbar(s, s ** 2 * data[i], yerr=s ** 2 * sig[i], fmt=marker, ms=3.3, color=color, ecolor=color,
                        mfc=color if filled else 'white', elinewidth=1.1, capsize=0)
            ax.plot(sf, CubicSpline(s, s ** 2 * model[i])(sf), ls=ls, color=color, lw=1.4)
            axr.plot(s, (data[i] - model[i]) / sig[i], marker=marker, ms=3, ls='-' if filled else '--', lw=0.8,
                     color=color, mfc=color if filled else 'white')
        axr.axhspan(-1, 1, color='0.8', zorder=0)
        axr.axhline(0, color='0.55', lw=0.7)
        axr.set_ylim(-5, 5)
        ax.set_title(TRACER_SHORT[fl.kname(key)], fontsize=9)
        ax.tick_params(labelbottom=False, labelsize=7)
        axr.tick_params(labelsize=7)
        axr.set_xlabel(r'$s\;[h^{-1}{\rm Mpc}]$')
        if j == 0:
            ax.set_ylabel(r'$s^2\xi_\ell(s)$')
            axr.set_ylabel(r'$\Delta/\sigma$')
            ax.legend(handles=[Line2D([0], [0], color='k', lw=1.4, label='MAP'),
                               Line2D([0], [0], color='k', marker='o', ms=3.3, ls='none', label='data')],
                      frameon=False, fontsize=7, loc='upper right')
    fig.text(0.5, 0.005, 'solid/filled: monopole   dashed/open: quadrupole', ha='center', fontsize=7)
    if save:
        fig.savefig(PLOT_DIR / f'validation_{name}.pdf', bbox_inches='tight')
        fig.savefig(PLOT_DIR / f'validation_{name}.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    tot_chi2, tot_n = sum(r[2] for r in chi2_rows), sum(r[1] for r in chi2_rows)
    print(f'{name}: MAP ln P = {lp_max:.2f}; chi2/n_data per tracer: ' +
          ', '.join(f'{t} {c:.1f}/{k}' for t, k, c in chi2_rows) + f'; total {tot_chi2:.1f}/{tot_n}')
    return chi2_rows, lp_max

if AVGCOV in INFO:
    rows, lp = validation_figure(AVGCOV)
    n_tot, chi2_tot = sum(r[1] for r in rows), sum(r[2] for r in rows)
    dof = n_tot - (5 + 2 * len(rows))
    print(f'total chi2 = {chi2_tot:.1f} for {dof} dof: chi2/dof = {chi2_tot / dof:.2f}, p = {chi2_dist.sf(chi2_tot, dof):.3g}; '
          f'expected spread of chi2/dof for a perfect model: +-{np.sqrt(2 / dof):.2f}')
else:
    print(f'{AVGCOV} not found')

#%% [markdown]
# ## 3. Triangle plots of the averaged-covariance fit
# Shared cosmology (against the single-mock-covariance fit, c000 truth marked) and the 12 bias parameters. The table gives the MAP and the posterior mean
# relative to the truth in units of each posterior's standard deviation; omega_b and n_s priors are centred on the truth, so their offsets are not informative.

#%%
BIAS_NAMES = [f'{b}_{t}' for t in TRACER_NAMES for b in ('b1', 'b2')]
BIAS_LABELS = [rf'b_{b[1]}^{{\rm {t}}}' for t in TRACER_NAMES for b in ('b1', 'b2')]
N_BIAS = len(BIAS_NAMES)
LAB_ONE, LAB_AVG = 'Mean mocks, single-mock covariance', 'Mean mocks, averaged covariance'


def cosmo_mc(name, label):
    return MCSamples(samples=cosmo5(name), names=PARAMS, labels=LABELS, label=label, settings=GETDIST_SETTINGS)


def bias_mc(name, label):
    x = post_samples(name)[:, 5:5 + N_BIAS]
    return MCSamples(samples=x, names=BIAS_NAMES, labels=BIAS_LABELS, label=label, settings=GETDIST_SETTINGS)


if AVGCOV in INFO and SINGLECOV in INFO:
    s_one, s_avg = cosmo_mc(SINGLECOV, LAB_ONE), cosmo_mc(AVGCOV, LAB_AVG)
    g = plots.get_subplot_plotter()
    g.triangle_plot([s_one, s_avg], params=PARAMS, filled=False, contour_colors=['tab:blue', 'k'], contour_lws=[1.3, 1.6],
                    legend_labels=[LAB_ONE, LAB_AVG], markers=TRUTH)
    g.export(str(PLOT_DIR / 'avgcov_cosmology.pdf')); g.export(str(PLOT_DIR / 'avgcov_cosmology.png'))
    plt.show()

    g = plots.get_subplot_plotter(width_inch=11)
    g.triangle_plot([bias_mc(SINGLECOV, LAB_ONE), bias_mc(AVGCOV, LAB_AVG)], params=BIAS_NAMES, filled=False,
                    contour_colors=['tab:blue', 'k'], contour_lws=[1.2, 1.5], legend_labels=[LAB_ONE, LAB_AVG])
    g.export(str(PLOT_DIR / 'avgcov_bias.pdf')); g.export(str(PLOT_DIR / 'avgcov_bias.png'))
    plt.show()

    rows = []
    for label, name, smp in (('single-mock cov', SINGLECOV, s_one), ('averaged cov', AVGCOV, s_avg)):
        mp = map_cosmo5(name)
        for k, p in enumerate(PARAMS):
            sd = smp.std(p)
            rows.append({'fit': label, 'parameter': p, 'truth': TRUTH[p], 'MAP': mp[k], 'mean': smp.mean(p), 'std': sd,
                         'MAP offset [sigma]': (mp[k] - TRUTH[p]) / sd, 'mean offset [sigma]': (smp.mean(p) - TRUTH[p]) / sd})
    display(pd.DataFrame(rows).set_index(['fit', 'parameter']).round(4))

#%% [markdown]
# ## 4. Shift characterization from the 24 single-mock fits (s_min = 20)
# Each mock fit gives a point estimate theta_hat_i (`ESTIMATOR`) and a posterior width sigma_i. Computed here: the pooled bias b with its standard error and
# bootstrap interval; per-parameter t statistics; the Hotelling T^2 test of the joint offset in `TEST_PARAMS` using the mocks' own scatter; the pull
# (theta_hat_i - truth)/sigma_i (mean should be 0, RMS 1) and the scatter of theta_hat_i relative to the typical sigma_i (error calibration); the rank of the truth within
# each posterior (uniform for a calibrated, unbiased fit; KS test); and a comparison with the offset of the averaged-covariance fit.

#%%
mock_names = [MOCK_FITS[i] for i in range(N_MOCKS) if MOCK_FITS[i] in INFO]
N = len(mock_names)
print(f'{N} single-mock chains available')

per_mock = {k: [] for k in ('mean', 'median', 'map')}
sig_i, rank_i = [], []
for n in mock_names:
    s5 = cosmo5(n)
    per_mock['mean'].append(s5.mean(0)); per_mock['median'].append(np.median(s5, axis=0)); per_mock['map'].append(map_cosmo5(n))
    sig_i.append(s5.std(0))
    rank_i.append((s5 < TRUTH_VEC).mean(0))
per_mock = {k: np.array(v) for k, v in per_mock.items()}
sig_i, rank_i = np.array(sig_i), np.array(rank_i)

theta_hat = per_mock[ESTIMATOR]
d = theta_hat - TRUTH_VEC                                   # (N, 5) offsets
b = d.mean(0)                                               # pooled bias
S = np.cov(d.T)                                             # scatter of the estimates (statistical)
Sigma_b = S / N                                             # covariance of the pooled bias estimate
se = np.sqrt(np.diag(Sigma_b))
z = b / se
rng = np.random.default_rng(0)
boot = np.array([d[rng.integers(N, size=N)].mean(0) for _ in range(N_BOOT)])
pulls = d / sig_i

ix = [PARAMS.index(p) for p in TEST_PARAMS]
T2 = N * b[ix] @ np.linalg.solve(S[np.ix_(ix, ix)], b[ix])
p_dim = len(ix)
F = (N - p_dim) / (p_dim * (N - 1)) * T2
p_hot = f_dist.sf(F, p_dim, N - p_dim)
print(f'estimator: {ESTIMATOR}. Hotelling T^2 on {TEST_PARAMS}: T^2 = {T2:.2f}, F({p_dim},{N - p_dim}) = {F:.2f}, p = {p_hot:.3g} '
      f'(= {norm.isf(p_hot / 2):.1f} sigma)')

tab = pd.DataFrame({
    'truth': TRUTH_VEC, 'bias b': b, 'std err (scatter/sqrt N)': se, 't = b/se': z,
    'p (t-test)': 2 * t_dist.sf(np.abs(z), N - 1),
    'boot 16%': np.percentile(boot, 16, axis=0), 'boot 84%': np.percentile(boot, 84, axis=0),
    'scatter of theta_hat': np.sqrt(np.diag(S)), 'median posterior sigma': np.median(sig_i, axis=0),
    'scatter / sigma': np.sqrt(np.diag(S)) / np.median(sig_i, axis=0),
    'pull mean': pulls.mean(0), 'pull RMS': np.sqrt((pulls ** 2).mean(0)),
    'rank KS p': [kstest(rank_i[:, k], 'uniform').pvalue for k in range(len(PARAMS))]}, index=PARAMS)
display(tab.round(4))

cmp_tab = pd.DataFrame({f'b ({k})': per_mock[k].mean(0) - TRUTH_VEC for k in ('mean', 'median', 'map')}, index=PARAMS)
if AVGCOV in INFO:
    a5 = cosmo5(AVGCOV)
    cmp_tab['avg-cov fit: mean - truth'] = a5.mean(0) - TRUTH_VEC
    cmp_tab['avg-cov fit: MAP - truth'] = map_cosmo5(AVGCOV) - TRUTH_VEC
display(cmp_tab.round(4))
print('(the pooled bias of the 24 single-mock fits and the offset of the fit to their average should agree if the shift is robust to the choice of statistic)')

# ---- figure A: per-mock estimates with the pooled bias
fig, axes = plt.subplots(len(PARAMS), 1, figsize=(7.0, 1.9 * len(PARAMS)), sharex=True)
xs = np.arange(N)
for k, ax in enumerate(axes):
    ax.errorbar(xs, theta_hat[:, k], yerr=sig_i[:, k], fmt='o', ms=4, color='0.25', capsize=2, elinewidth=1.0, zorder=3)
    ax.axhline(TRUTH_VEC[k], ls='--', color='gray', lw=1.0, zorder=1)
    ax.axhspan(TRUTH_VEC[k] + b[k] - se[k], TRUTH_VEC[k] + b[k] + se[k], color='tab:red', alpha=0.25, zorder=0)
    ax.axhline(TRUTH_VEC[k] + b[k], color='tab:red', lw=1.0, zorder=2)
    ax.set_ylabel(f'${LABELS[k]}$')
    ax.set_xlim(-1, N)
axes[-1].set_xlabel('mock')
axes[0].legend(handles=[Line2D([0], [0], marker='o', ls='none', color='0.25', label=f'per-mock {ESTIMATOR} and posterior $\\sigma$'),
                        Line2D([0], [0], color='gray', ls='--', label='c000 truth'),
                        Line2D([0], [0], color='tab:red', label='pooled bias $\\pm$ s.e.')], frameon=False, fontsize=7, ncol=3, loc='upper right')
fig.tight_layout()
fig.savefig(PLOT_DIR / 'shift_per_mock.pdf', bbox_inches='tight'); fig.savefig(PLOT_DIR / 'shift_per_mock.png', dpi=150, bbox_inches='tight')
plt.show()

# ---- figure B: pulls
fig, axes = plt.subplots(1, len(PARAMS), figsize=(2.6 * len(PARAMS), 2.6))
grid = np.linspace(-4, 4, 200)
for k, ax in enumerate(axes):
    ax.hist(pulls[:, k], bins=np.linspace(-4, 4, 17), density=True, color='0.6')
    ax.plot(grid, norm.pdf(grid), 'k-', lw=1.0)
    ax.set_xlabel(f'pull of ${LABELS[k]}$')
    ax.text(0.03, 0.95, f'mean {pulls[:, k].mean():.2f}\nRMS {np.sqrt((pulls[:, k] ** 2).mean()):.2f}', transform=ax.transAxes, va='top', fontsize=7)
    ax.set_yticks([])
fig.tight_layout()
fig.savefig(PLOT_DIR / 'shift_pulls.pdf', bbox_inches='tight'); fig.savefig(PLOT_DIR / 'shift_pulls.png', dpi=150, bbox_inches='tight')
plt.show()

# ---- figure C: offsets in the constrained directions, with the pooled bias and its error ellipse
pairs = [('omch2', 'logA'), ('H0', 'logA'), ('omch2', 'H0')]
fig, axes = plt.subplots(1, len(pairs), figsize=(3.4 * len(pairs), 3.3))
for ax, (p1, p2) in zip(axes, pairs):
    i1, i2 = PARAMS.index(p1), PARAMS.index(p2)
    ax.plot(d[:, i1], d[:, i2], 'o', ms=4, color='0.4')
    ax.plot(0, 0, 'k+', ms=10)
    cov2 = Sigma_b[np.ix_([i1, i2], [i1, i2])]
    w_, v_ = np.linalg.eigh(cov2)
    ang = np.degrees(np.arctan2(v_[1, 1], v_[0, 1]))
    ax.add_patch(Ellipse((b[i1], b[i2]), 2 * np.sqrt(w_[1]) * 1.52, 2 * np.sqrt(w_[0]) * 1.52, angle=ang, fill=False, ec='tab:red', lw=1.4))
    ax.plot(b[i1], b[i2], 'x', color='tab:red', ms=8)
    ax.set_xlabel(rf'$\Delta\,{LABELS[i1]}$'); ax.set_ylabel(rf'$\Delta\,{LABELS[i2]}$')
fig.tight_layout()
fig.savefig(PLOT_DIR / 'shift_offset_planes.pdf', bbox_inches='tight'); fig.savefig(PLOT_DIR / 'shift_offset_planes.png', dpi=150, bbox_inches='tight')
plt.show()

#%% [markdown]
# ## 5. Shift correction and systematic error for the DR1 fit
# (1) The shift b is measured on the 24 mocks (section 4). (2) It is applied to the DR1 baseline joint fit: every posterior sample is shifted, theta -> theta - b,
# all five parameters together, so correlated shifts move jointly. (3) The systematic error is half the shift, +- |b_i|/2 per parameter: the corrected posterior is
# convolved with N(0, Sigma_syst), Sigma_syst = b b^T / 4 (`SYST_CORRELATED = True`: the uncertainty acts along the shift direction) or diag(b_i^2 / 4) (False). The per-parameter
# systematic is |b_i|/2 either way; the choice only changes the joint contours. `INCLUDE_SHIFT_UNCERTAINTY = True` also adds the uncertainty of the shift estimate,
# Sigma_b = Cov(theta_hat_i)/N. omega_b and n_s are prior-dominated, so their rows are not informative.

#%%
SYST_CORRELATED = True
INCLUDE_SHIFT_UNCERTAINTY = False

if DR1 not in INFO:
    print(f'{DR1} not found')
else:
    dr1 = cosmo5(DR1)
    Sigma_stat = np.cov(dr1.T)
    Sigma_syst = 0.25 * (np.outer(b, b) if SYST_CORRELATED else np.diag(b ** 2))
    if INCLUDE_SHIFT_UNCERTAINTY:
        Sigma_syst = Sigma_syst + Sigma_b
    Sigma_tot = Sigma_stat + Sigma_syst
    sig_stat, sig_syst, sig_tot = np.sqrt(np.diag(Sigma_stat)), np.sqrt(np.diag(Sigma_syst)), np.sqrt(np.diag(Sigma_tot))
    dr1_corr = dr1 - b                                                    # step 2: apply the shift to every posterior sample
    out = pd.DataFrame({'DR1 mean (uncorrected)': dr1.mean(0), 'shift b': b, 't = b/se': z, 'DR1 mean (corrected)': dr1_corr.mean(0),
                        'sigma_stat': sig_stat, 'sigma_syst': sig_syst, 'sigma_tot': sig_tot, 'sigma_tot / sigma_stat': sig_tot / sig_stat,
                        'b / sigma_stat': b / sig_stat}, index=PARAMS)
    display(out.round(4))
    out.to_csv(PLOT_DIR / 'dr1_shift_and_systematic.csv')
    print('shift applied to the DR1 fit, in units of its statistical error (b / sigma_stat): ' + ', '.join(f'{p} {v:+.2f}' for p, v in zip(PARAMS, b / sig_stat)))

    # step 3: convolve the corrected posterior with the systematic
    eps = np.random.default_rng(1).multivariate_normal(np.zeros(len(PARAMS)), Sigma_syst, size=len(dr1), check_valid='ignore')
    dr1_tot = dr1_corr + eps
    mk = lambda x, lab: MCSamples(samples=x, names=PARAMS, labels=LABELS, label=lab, settings=GETDIST_SETTINGS)
    s_orig, s_corr, s_tot = mk(dr1, 'DR1, uncorrected'), mk(dr1_corr, 'DR1, shift-corrected (statistical error)'), \
        mk(dr1_tot, r'DR1, shift-corrected, statistical + systematic ($\pm\frac{1}{2}$ shift)')
    g = plots.get_subplot_plotter()
    g.triangle_plot([s_orig, s_corr, s_tot], params=PARAMS, filled=False, contour_colors=['tab:blue', '0.5', 'k'], contour_lws=[1.3, 1.3, 1.7],
                    legend_labels=[s.getLabel() for s in (s_orig, s_corr, s_tot)])
    g.export(str(PLOT_DIR / 'dr1_shift_corrected_triangle.pdf')); g.export(str(PLOT_DIR / 'dr1_shift_corrected_triangle.png'))
    plt.show()

    # per-parameter summary: uncorrected vs corrected, with statistical and total errors
    fig, axes = plt.subplots(len(PARAMS), 1, figsize=(4.6, 1.7 * len(PARAMS)), sharex=True)
    for k, ax in enumerate(axes):
        ax.errorbar(0, dr1[:, k].mean(), yerr=sig_stat[k], fmt='o', color='tab:blue', capsize=3, elinewidth=1.6, ms=6)
        ax.errorbar(1, dr1_corr[:, k].mean(), yerr=sig_tot[k], fmt='none', ecolor='k', capsize=5, elinewidth=1.0)
        ax.errorbar(1, dr1_corr[:, k].mean(), yerr=sig_stat[k], fmt='o', color='k', capsize=3, elinewidth=2.0, ms=6)
        ax.set_ylabel(f'${LABELS[k]}$')
        ax.set_xlim(-0.6, 1.6)
    axes[-1].set_xticks([0, 1])
    axes[-1].set_xticklabels(['DR1\nuncorrected', 'shift-corrected\n(thick: stat, thin: total)'], fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'dr1_shift_summary.pdf', bbox_inches='tight'); fig.savefig(PLOT_DIR / 'dr1_shift_summary.png', dpi=150, bbox_inches='tight')
    plt.show()

    # bar chart of the error budget (constrained parameters)
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    cols = [PARAMS.index(p) for p in TEST_PARAMS]
    w = 0.27
    for j, (lab, vals, colr) in enumerate((('statistical', sig_stat, 'tab:blue'), ('systematic ($\\frac{1}{2}$ shift)', sig_syst, 'tab:red'), ('total', sig_tot, 'k'))):
        ax.bar(np.arange(len(cols)) + (j - 1) * w, [vals[c] / sig_stat[c] for c in cols], w * 0.92, color=colr, label=lab)
    ax.axhline(1.0, color='0.6', lw=0.8, ls=':')
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f'${LABELS[c]}$' for c in cols])
    ax.set_ylabel(r'error / $\sigma_{\rm stat}$')
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'dr1_error_budget.pdf', bbox_inches='tight'); fig.savefig(PLOT_DIR / 'dr1_error_budget.png', dpi=150, bbox_inches='tight')
    plt.show()
# %%
