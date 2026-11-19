"""
Fit-validation figures for the v2 baseline fits (as in the v1 plotting script): one panel per tracer bin with s^2 xi_ell of the data (monopole filled, quadrupole open, error bars from the
covariance diagonal) and the model at the best-fit point, plus a residual strip (data - model)/sigma with a +-1 band, and a chi^2 table. Three fits are shown. Reduced chi^2 is NOT
reported by default (only chi^2, N_data, N_par and the p-value; set SHOW_REDUCED_CHI2 = True to add it to the table):
  dr1          compact_dr1_joint_smin20            DR1 data (windowed model)
  mock         compact_mean_joint_smin20           mean of the mocks, single-mock covariance (error bars are those of ONE mock)
  mock_avgcov  compact_mean_joint_avgcov_smin20    mean of the mocks, covariance / n_mocks: chi^2 ~ N_data - N_par for a perfect model, so this is the real model test
Run cell by cell (#%% markers) or as a script in the compact environment, from anywhere (FITS_DIR points at the folder with fits_apfix.py). The model is evaluated with fits_apfix's own
FitProblem / AP code, so it is exactly the likelihood that was sampled. The best-fit point is the highest-posterior post-burn-in sample ('map', noisy: a few lnP units below the true maximum)
or the posterior mean ('mean'); chi^2 is evaluated there. Figures go to CHAIN_ROOT/plots_validation (PDF + PNG at DPI).
"""
#%%
import sys
from pathlib import Path
FITS_DIR = Path('/home/cosweeney/code/compact_streaming/nb/fits/11par_fits')
sys.path.insert(0, str(FITS_DIR))
import fits_apfix as fa                     # first: it pins the BLAS/OpenMP/TF thread counts before numpy is imported
import json
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
from scipy.stats import chi2 as chi2_dist

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False})

CHAIN_ROOT = fa.CHAIN_ROOT
PLOT_DIR = CHAIN_ROOT / 'plots_validation'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
FITS = {'dr1': ('compact_dr1_joint_smin20', 'DR1 data'),
        'mock': ('compact_mean_joint_smin20', 'mean of mocks (single-mock covariance)'),
        'mock_avgcov': ('compact_mean_joint_avgcov_smin20', 'mean of mocks (averaged covariance)')}
POINT = 'map'            # 'map' = highest-posterior post-burn sample, 'mean' = posterior mean
SHOW_REDUCED_CHI2 = False  # never reported by default; True adds a chi2/dof column to the table (the figures never show it)
RESID_YLIM = None        # residual strip limits: None = automatic (at least +-1.5, common to all panels of a figure), or a number
NCOLS = 6                # panels per row: 6 = one row of six (as in v1), 3 = two rows of three (narrower, for a paper column)
DPI = 600
TRACER_NAMES = ['BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG2', 'QSO']             # fa.FIT_KEYS order
TRACER_COLORS = ['yellowgreen', 'orange', 'orangered', 'firebrick', 'steelblue', 'seagreen']

#%% [markdown]
# ## Best-fit points and model evaluation

#%%
SPECS = {s['name']: s for s in fa.build_specs()}


def chain_point(name):
    """Best-fit parameter vector of a chain (post-burn-in, as saved in status.json) and its log-posterior (None for the mean)."""
    root = CHAIN_ROOT / SPECS[name]['model'] / name
    st = json.load(open(f'{root}.status.json'))
    with h5py.File(f'{root}.h5', 'r') as f:
        it = int(f['mcmc'].attrs['iteration'])
        lp = f['mcmc/log_prob'][st['discard']:it]
        ch = f['mcmc/chain'][st['discard']:it]
    if POINT == 'mean':
        return ch.reshape(-1, ch.shape[-1]).mean(0), None, st
    t, w = np.unravel_index(np.nanargmax(np.where(np.isfinite(lp), lp, -np.inf)), lp.shape)
    return ch[t, w], float(lp[t, w]), st


def evaluate(problem, theta):
    """Model multipoles (2, n_s) of every tracer at theta, windowed for DR1, exactly as in the likelihood."""
    h, ombh2, omch2, lnAs, ns, biases = problem.unpack(theta)
    aps = fa.ap_multi(h, ombh2, omch2, ns, problem.z_effs)
    return [fa.model_one(d['emus'], [b1, b2, ombh2, omch2, h, lnAs, ns], d['s_data'], d['W'], qp, ql).reshape(2, -1)
            for d, (b1, b2), (qp, ql) in zip(problem.data, biases, aps)]


RESULTS = {}
for key, (name, label) in FITS.items():
    root = CHAIN_ROOT / SPECS[name]['model'] / f'{name}.status.json'
    if not root.exists():
        print(f'{name}: no status.json yet, skipped')
        continue
    theta, lp, st = chain_point(name)
    problem = fa.FitProblem(SPECS[name])
    models = evaluate(problem, theta)
    rows = []
    for j, (d, tn) in enumerate(zip(problem.data, TRACER_NAMES)):
        r = d['data_vec'] - models[j].ravel()
        n = r.size
        rows.append({'tracer': tn, 'n_data': n, 'chi2': float(r @ d['cov_inv'] @ r)})
    tab = pd.DataFrame(rows).set_index('tracer')
    tab.loc['TOTAL'] = [tab.n_data.sum(), tab.chi2.sum()]
    tab['n_par'] = np.where(tab.index == 'TOTAL', problem.ndim, np.nan)                  # the parameters are shared, so only the total has a meaningful dof
    tab['dof'] = np.where(tab.index == 'TOTAL', tab.n_data - problem.ndim, np.nan)
    tab['p-value'] = np.where(tab.index == 'TOTAL', chi2_dist.sf(tab.chi2, tab.dof), np.nan)
    if SHOW_REDUCED_CHI2:
        tab['chi2/dof'] = tab.chi2 / tab.dof
    tab.to_csv(PLOT_DIR / f'v2_validation_chi2_{key}.csv')
    RESULTS[key] = dict(problem=problem, models=models, table=tab, lp=lp, converged=st['converged'])
    print(f"\n{name} ({label}), {POINT} point{'' if lp is None else f', lnP = {lp:.2f}'}, chain converged = {st['converged']}:")
    print(tab.round(3).to_string())

#%% [markdown]
# ## Figures
# Top: s^2 xi_ell with data and the model (solid line through the model values at the data bins); bottom: (data - model)/sigma. chi^2 and N_data of each tracer are in the panel titles, the totals below the figure.

#%%
def validation_figure(key):
    R = RESULTS[key]
    problem, models, tab = R['problem'], R['models'], R['table']
    n = len(problem.data)
    nblocks = int(np.ceil(n / NCOLS))
    heights = []
    for b in range(nblocks):
        heights += [3, 1.6] + ([0.9] if b < nblocks - 1 else [])             # data panel, residual strip, spacer between rows of panels
    fig = plt.figure(figsize=(2.6 * NCOLS, 3.5 * nblocks + 0.4 * (nblocks - 1)))
    gs = fig.add_gridspec(len(heights), NCOLS, height_ratios=heights, hspace=0.08, wspace=0.34)
    sigs = [np.sqrt(np.diag(np.linalg.inv(d['cov_inv']))).reshape(2, -1) for d in problem.data]
    resid = [(d['data_vec'].reshape(2, -1) - m) / sg for d, m, sg in zip(problem.data, models, sigs)]
    lim = RESID_YLIM or max(1.5, np.ceil(2 * 1.15 * max(np.abs(r).max() for r in resid)) / 2)
    for j, (d, name, color) in enumerate(zip(problem.data, TRACER_NAMES, TRACER_COLORS)):
        blk, col = divmod(j, NCOLS)
        ax, axr = fig.add_subplot(gs[3 * blk, col]), fig.add_subplot(gs[3 * blk + 1, col])
        s = d['s_data']
        data, model = d['data_vec'].reshape(2, -1), models[j]
        sig = sigs[j]
        sf = np.linspace(s[0], s[-1], 300)
        for i, (marker, ls, filled) in enumerate((('o', '-', True), ('s', '--', False))):
            ax.errorbar(s, s ** 2 * data[i], yerr=s ** 2 * sig[i], fmt=marker, ms=3.3, color=color, ecolor=color, mfc=color if filled else 'white', elinewidth=1.1, capsize=0)
            ax.plot(sf, CubicSpline(s, s ** 2 * model[i])(sf), ls=ls, color=color, lw=1.4)
            axr.plot(s, resid[j][i], marker=marker, ms=3.8, ls=ls, lw=1.0, color=color, mfc=color if filled else 'white')
        axr.axhspan(-1, 1, color='0.8', zorder=0)
        axr.axhline(0, color='0.55', lw=0.7)
        axr.set_ylim(-lim, lim)
        axr.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_title(rf"{name}  ($\chi^2={tab.loc[name, 'chi2']:.1f}$, $N={int(tab.loc[name, 'n_data'])}$)", fontsize=8.5)
        ax.tick_params(labelbottom=False, labelsize=7)
        axr.tick_params(labelsize=7)
        axr.set_xlabel(r'$s\;[h^{-1}{\rm Mpc}]$')
        if col == 0:
            ax.set_ylabel(r'$s^2\xi_\ell(s)$')
            axr.set_ylabel(r'$\Delta/\sigma$')
        if j == 0:                                                       # legend in the BGS data panel (markers = data, lines = model), with headroom so it cannot cover the data
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.3 * (hi - lo))
            ax.legend(handles=[Line2D([0], [0], color='0.3', marker='o', ms=3.5, ls='-', lw=1.4, label=r'$\ell=0$: data, model'),
                               Line2D([0], [0], color='0.3', marker='s', ms=3.5, mfc='white', ls='--', lw=1.4, label=r'$\ell=2$: data, model')], frameon=False, fontsize=6.5, loc='upper right', handlelength=2.4)
    fig.text(0.99, -0.07 / nblocks, rf"total $\chi^2={tab.loc['TOTAL', 'chi2']:.1f}$, $N_{{\rm data}}={int(tab.loc['TOTAL', 'n_data'])}$, $N_{{\rm par}}={int(tab.loc['TOTAL', 'n_par'])}$", ha='right', fontsize=7.5)
    fig.savefig(PLOT_DIR / f'v2_validation_{key}.pdf', bbox_inches='tight')
    fig.savefig(PLOT_DIR / f'v2_validation_{key}.png', dpi=DPI, bbox_inches='tight')
    plt.show()
    plt.close(fig)


for key in RESULTS:
    validation_figure(key)
# %%
