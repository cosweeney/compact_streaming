"""
getdist plots and tables for the b1-ONLY fits of fits_apfix_11par_v3.py (group 'b1only': compact_v3_noslope_b1only_<dr1|mean>_{joint,indiv}_..._smin25; b2 is not sampled), analysed on their own, without the b1+b2 fits.
Style and helpers (loading, getdist plotter, DESI chains) are those of plot_v3_chains_gd.py: one chain per walker, KDE bandwidth 1.25 x the automatic one, unitless sampled parameters, vector PDF + PNG in PLOT_DIR,
error bars = the 1-D marginalized 68% limits (the widths of the triangle plots; ERR = 'std' switches to the posterior std). Run cell by cell (#%% markers) or as a script, compact environment (getdist, h5py);
cells 5-6 also import fits_apfix_11par_v3 (and through it your fits_apfix.py and common.py) to evaluate the model at the MAP.

  1. joint b1-only fit vs DESI FS-only and BAO (DR1; mean of mocks with the c000 truth marked)
  2. individual tracers vs the joint fit
  3. the b1 of each tracer: triangle of the six b1 (DR1 and mean of mocks), and joint vs individual b1 per tracer
  4. the five sampled parameters for the joint and each individual fit (DR1 and mean of mocks; truth marked for the mocks)
  5. goodness of fit at the MAP: chi2, N_data, N_par and p-value (no reduced chi2) for the joint and individual fits
  6. data vs the MAP model multipoles of each tracer-bin: joint-fit MAP (solid) and the tracer's own individual-fit MAP (dashed)
  7. chain health (N/tau, split-half test) and the fraction of each posterior outside the NN training box
  8. summary table of every fit: posterior mean, std and 1-D 68% limits of the sampled parameters and of b1, and offsets from the c000 truth for the mocks
Not tested on chains: only checked for syntax, with the helper functions copied from the tested plotting script.
"""
#%%
import glob
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from getdist import MCSamples, loadMCSamples, plots

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False})
NB_DPI = 200            # resolution of figures shown in the notebook (and so of copy / paste into slides); with the 'retina' format below the inline PNG has twice this density. Saved files use DPI.
plt.rcParams['figure.dpi'] = NB_DPI
try:
    from matplotlib_inline.backend_inline import set_matplotlib_formats
    set_matplotlib_formats('retina')
except ImportError:         # not running in a Jupyter / VS Code notebook: nothing to set
    pass


DESI_FS = '/spiffball/cosweeney/DESI/DR1/full-shape-cosmo-params/v1.0/cobaya/base/desi-reptvelocileptors-fs-all_schoneberg2024-bbn_planck2018-ns10/'
DESI_BAO = '/spiffball/cosweeney/DESI/DR1/bao-cosmo-params/v1.0/cobaya/base/desi-bao-all_schoneberg2024-bbn/'      # adjust to your local directory name
DPI = 600               # PNG resolution (the PDFs are vector)
DPI_MATRIX = 1200       # the constraining-power matrix has small text, so it gets a higher resolution
JOINT_COLOR = 'k'       # joint (all-tracer) fits are drawn filled black in the joint-vs-tracers plots
WIDTH = 7.2             # figure width in inches for the triangle plots (two-column page width); the 12-parameter bias triangle uses WIDTH_BIAS
WIDTH_BIAS = 10.0
SMOOTH = {'smooth_scale_1D': -1.25, 'smooth_scale_2D': -1.25}      # getdist KDE bandwidth = automatic x 1.25 (-1 = default)
UNITLESS = True         # True (default): plot the 5 sampled dimensionless parameters (h, omega_b, omega_cdm, ln A_s, n_s); False: Omega_m, H0, ... instead
SMINS = (20, 25, 30, 35, 40)
W_NU = 0.0006442        # omega_nu of one massive 0.06 eV species, so Omega_m includes neutrinos like CAMB's
DATASETS = {'dr1': 'DR1', 'mean': 'mean of mocks'}

SAMPLED = ['h', 'ombh2', 'omch2', 'logA', 'ns']
PARAMS = ['omegam', 'H0', 'ombh2', 'omch2', 'logA', 'ns']
if UNITLESS:
    PARAMS = SAMPLED
LABELS = {'h': r'h', 'omegam': r'\Omega_m', 'H0': r'H_0', 'ombh2': r'\omega_b', 'omch2': r'\omega_{\rm cdm}', 'logA': r'\ln(10^{10}A_s)', 'ns': r'n_s'}
TRUTH_S = {'h': 0.6736, 'ombh2': 0.02237, 'omch2': 0.12, 'logA': float(np.log(1e10 * 2.0830e-9)), 'ns': 0.9649}              # AbacusSummit c000, sampled parameters
TRUTH = {**TRUTH_S, 'omegam': (0.02237 + 0.12 + W_NU) / 0.6736 ** 2, 'H0': 67.36}
TRACERS = [('BGS', 'BGS', 'yellowgreen'), ('LRG1', 'LRG_0.4', 'orange'), ('LRG2', 'LRG_0.6', 'orangered'), ('LRG3', 'LRG_0.8', 'firebrick'),
           ('ELG2', 'ELG', 'steelblue'), ('QSO', 'QSO', 'seagreen')]      # (short name, substring of the chain name, DESI colour)
RED, ORANGE, PURPLE = 'red', '#FF8C1A', '#D58BCB'
DESI_POINT = '#9A4FA5'
for s, key, _c in TRACERS:
    LABELS.update({f'b1_{s}': rf'b_1^{{\rm {s}}}', f'b2_{s}': rf'b_2^{{\rm {s}}}'})

import sys
from scipy.stats import chi2 as chi2_dist

CHAIN_ROOT_V3 = Path('/spiffball/cosweeney/data/chains/fits_11par_apfix_v2_v3')
PLOT_DIR = CHAIN_ROOT_V3 / 'plots_gd_b1only'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
B1_SMIN = 25            # s_min of the b1-only fits
ERR = '68'              # error bars in cells 3-4: '68' (1-D marginalized 68% limits) or 'std' (posterior standard deviation, symmetric about the mean)
PLOTS_TO_RUN = set(range(1, 9))
MODELS = {'b1': dict(folder='compact_v3_noslope', prefix='compact_v3_noslope_b1only', root=CHAIN_ROOT_V3, label=r'v3, slopes zero, $b_1$ only', color='tab:blue', filled=True, ls='-')}
M = 'b1'                # the only model key (the copied helpers below take a model argument)
LABELS['b1'] = r'b_1'
TRUTH_FOR = {'dr1': None, 'mean': TRUTH}
NN_BOX = {'h': (0.55, 0.90), 'ombh2': (0.0207, 0.0243), 'omch2': (0.104, 0.140), 'logA': (2.0, 3.91), 'ns': (0.901, 1.025), 'b1': (0.0, 2.0)}     # as BOX in fits_apfix.py (no b2: not sampled)
TAU_FACTOR = 100       # the fits' convergence rule: chain length > TAU_FACTOR x tau for every parameter

#%% [markdown]
# ## Loading and figure helpers (copied from plot_v3_chains_gd.py)

#%%
_cache = {}


def full3(model, rest):
    """(array (steps, walkers, n_params), parameter names, status dict): post-burn-in and thinned as saved in status.json."""
    if (model, rest) not in _cache:
        root = fit_path(model, rest)
        st = json.load(open(f'{root}.status.json'))
        with h5py.File(f'{root}.h5', 'r') as f:
            it = int(f['mcmc'].attrs['iteration'])
            arr = f['mcmc/chain'][st['discard']:it:st['thin'], :, :]
        _cache[(model, rest)] = (arr, st['param_names'], st)
    return _cache[(model, rest)]


def columns(model, rest):
    """dict column -> array (steps, walkers), with derived Omega_m and H0."""
    arr, names, _ = full3(model, rest)
    c = {n: arr[..., i].copy() for i, n in enumerate(names)}
    c['H0'] = 100 * c['h']
    c['omegam'] = (c['ombh2'] + c['omch2'] + W_NU) / c['h'] ** 2
    return c


def mcs(model, rest, params, label=None):
    """getdist MCSamples with one chain per walker for the parameters of `params` that the chain has."""
    c = columns(model, rest)
    use = [p for p in params if p in c]
    stack = np.stack([c[p] for p in use], axis=-1)
    return MCSamples(samples=[stack[:, w] for w in range(stack.shape[1])], names=use, labels=[LABELS[p] for p in use], label=label or MODELS[model]['label'], settings=SMOOTH)


def flat(model, rest, params=PARAMS + ['h']):
    c = columns(model, rest)
    return pd.DataFrame({p: c[p].ravel() for p in params if p in c})


def desi(path, label):
    try:
        s = loadMCSamples(path + 'chain', settings={'ignore_rows': 0.3})
    except Exception as e:
        print(f'DESI chain not found at {path}: {e}')
        return None
    n = s.getParamNames().list()
    has = lambda *a: any(x in n for x in a)
    col = lambda *a: s.samples[:, n.index(next(x for x in a if x in n))]
    h, ob = col('H0') / 100, col('ombh2', 'omegabh2')
    c = {'h': h, 'H0': 100 * h, 'ombh2': ob}
    c['omch2'] = col('omch2') if has('omch2') else col('omegam') * h ** 2 - ob - W_NU
    c['omegam'] = col('omegam') if has('omegam') else (ob + c['omch2'] + W_NU) / h ** 2
    c.update({k: col(k) for k in ('logA', 'ns') if has(k)})                                       # the BAO chains have no logA / ns
    names = [p for p in ['h', 'H0', 'omegam', 'ombh2', 'omch2', 'logA', 'ns'] if p in c]
    return MCSamples(samples=np.column_stack([c[p] for p in names]), weights=s.weights, names=names, labels=[LABELS[p] for p in names], label=label, settings=SMOOTH)


def map_point(model, rest):
    """Highest-posterior post-burn-in sample of a chain as {column: value}, with Omega_m and H0 added."""
    root = fit_path(model, rest)
    st = json.load(open(f'{root}.status.json'))
    with h5py.File(f'{root}.h5', 'r') as f:
        it = int(f['mcmc'].attrs['iteration'])
        lp = f['mcmc/log_prob'][st['discard']:it]
        t, w = np.unravel_index(np.nanargmax(np.where(np.isfinite(lp), lp, -np.inf)), lp.shape)
        theta = f['mcmc/chain'][st['discard'] + t, w, :]
    c = dict(zip(st['param_names'], theta))
    c['H0'] = 100 * c['h']
    c['omegam'] = (c['ombh2'] + c['omch2'] + W_NU) / c['h'] ** 2
    return c

def plotter(width):
    """Paper-style getdist plotter: fixed font sizes (not rescaled with the width), no legend frame."""
    st = plots.GetDistPlotSettings()
    st.axes_fontsize, st.lab_fontsize, st.legend_fontsize = 8, 10, 9
    st.legend_frame = False
    st.axis_tick_max_labels = 4
    return plots.get_subplot_plotter(width_inch=width, scaling=False, settings=st)


def item(root, color, filled=False, ls='-'):
    return None if root is None else (root, color, filled, ls)


def figure(fname, items, params, truth=None, width=None):
    """Triangle plot of the (non-None) items = (MCSamples, colour, filled, linestyle), drawn in order (the last is on top); saves PDF + PNG (DPI) to PLOT_DIR."""
    items = [i for i in items if i is not None]
    if len(items) < 2:
        print(f'{fname}: nothing to compare, skipped')
        return
    roots = [i[0] for i in items]
    present = {p for r in roots for p in r.getParamNames().list()}
    params = [p for p in params if p in present]
    g = plotter(width or WIDTH)
    g.triangle_plot(roots, params, filled=[i[2] for i in items], contour_colors=[i[1] for i in items], contour_ls=[i[3] for i in items],
                    contour_lws=[1.2] * len(items), legend_labels=[r.getLabel() for r in roots], markers={k: v for k, v in (truth or {}).items() if k in params})
    g.export(str(PLOT_DIR / f'{fname}.pdf'))
    g.fig.savefig(PLOT_DIR / f'{fname}.png', dpi=DPI, bbox_inches='tight')
    plt.show()
    plt.close(g.fig)
    print('saved', fname, flush=True)

#%% [markdown]
# ## b1-only chain helpers (the rest of the loading and figure helpers above are copied from plot_v3_chains_gd.py)

#%%
def fit_path(model, rest):
    """Path stem of the b1-only chain compact_v3_noslope_b1only_<rest>, e.g. rest = 'dr1_joint_smin25'."""
    m = MODELS[model]
    return m['root'] / m['folder'] / f"{m['prefix']}_{rest}"


def have(model, rest):
    return Path(f'{fit_path(model, rest)}.status.json').exists()


def indiv_rest(tag, key):
    """`rest` of the individual-tracer b1-only chain whose name contains `_indiv_<key>`, or None."""
    m = MODELS[M]
    files = glob.glob(str(m['root'] / m['folder'] / f"{m['prefix']}_{tag}_indiv_*_smin{B1_SMIN}.status.json"))
    match = [Path(f).name[:-len('.status.json')][len(m['prefix']) + 1:] for f in files if f'_indiv_{key}' in Path(f).name]
    return match[0] if match else None


def all_rests():
    m = MODELS[M]
    return sorted(Path(p).name[:-len('.status.json')][len(m['prefix']) + 1:] for p in glob.glob(str(m['root'] / m['folder'] / f"{m['prefix']}_*.status.json")))


def short_fit_name(rest):
    """Shorter label for a fit name, e.g. dr1_joint_smin25 -> DR1 joint, mean_indiv_LRG_0.4-0.6_smin25 -> mean LRG_0.4-0.6."""
    tag, _, tail = rest.partition('_')
    tail = tail.replace('joint_smin25', 'joint').replace('indiv_', '').replace('_smin25', '')
    return f"{'DR1' if tag == 'dr1' else 'mean'} {tail}"


def joint_b1_columns(rest):
    """dict b1_<tracer> -> samples (steps, walkers) of a joint b1-only chain (the chain's b1_<tracer-bin> columns are in TRACERS order)."""
    arr, names, _ = full3(M, rest)
    idx = [i for i, n in enumerate(names) if n.startswith('b1_')]
    return {f'b1_{t[0]}': arr[..., i] for t, i in zip(TRACERS, idx)}


def margstats(by_name):
    """getdist 1-D marginalized statistics of a dict name -> samples (steps, walkers): {name: dict(mean, std, lower, upper)}, lower / upper = the 68% limits."""
    names = list(by_name)
    stack = np.stack([by_name[n] for n in names], axis=-1)
    smp = MCSamples(samples=[stack[:, w] for w in range(stack.shape[1])], names=names, labels=[LABELS.get(n, n) for n in names], settings=SMOOTH)
    ms = smp.getMargeStats()
    out = {}
    for n in names:
        par = ms.parWithName(n)
        lim = par.limits[0]
        out[n] = dict(mean=par.mean, std=par.err, lower=lim.lower, upper=lim.upper)
    return out


def bar(ax, x, st, color, **kw):
    """Point at the posterior mean and the error bar: the 1-D 68% limits (ERR = '68') or +- the posterior std (ERR = 'std')."""
    lo, hi = (st['lower'], st['upper']) if ERR == '68' else (st['mean'] - st['std'], st['mean'] + st['std'])
    ax.errorbar(x, 0.5 * (lo + hi), yerr=0.5 * (hi - lo), fmt='none', ecolor=color, capsize=2, elinewidth=1.0)
    ax.plot(x, st['mean'], 'o', ms=4, color=color, **kw)


def report_chains():
    rests = all_rests()
    bad = [r for r in rests if not json.load(open(f'{fit_path(M, r)}.status.json')).get('converged')]
    print(f"{MODELS[M]['label']}: {len(rests)} finished chains in {MODELS[M]['root'] / MODELS[M]['folder']}" + (f'; NOT CONVERGED: {bad}' if bad else ''))


report_chains()

#%% [markdown]
# ## 1. Joint b1-only fit vs DESI FS-only and BAO
# DR1, and the mean of the mocks with the c000 truth marked. Filled blue = the b1-only fit.

#%%
fs = bao = None
if 1 in PLOTS_TO_RUN:
    fs, bao = desi(DESI_FS, 'DESI DR1 FS-only'), desi(DESI_BAO, 'DESI DR1 BAO')
    for tag in DATASETS:
        rest = f'{tag}_joint_smin{B1_SMIN}'
        mine = item(mcs(M, rest, PARAMS), MODELS[M]['color'], True) if have(M, rest) else None
        figure(f'b1_1_joint_vs_desi_{tag}', [item(bao, ORANGE, True), item(fs, PURPLE, True), mine], PARAMS, TRUTH_FOR[tag])

#%% [markdown]
# ## 2. Individual tracers vs the joint fit
# Individual fits in the DESI tracer colours, the joint fit filled black and on top.

#%%
if 2 in PLOTS_TO_RUN:
    for tag in DATASETS:
        items = []
        for short, key, color in TRACERS:
            rest = indiv_rest(tag, key)
            items.append(item(mcs(M, rest, PARAMS, short), color, True) if rest else None)
        joint = f'{tag}_joint_smin{B1_SMIN}'
        figure(f'b1_2_individual_{tag}', items + [item(mcs(M, joint, PARAMS, 'Joint'), JOINT_COLOR, True) if have(M, joint) else None], PARAMS, TRUTH_FOR[tag])

#%% [markdown]
# ## 3. The b1 of each tracer
# (a) triangle of the six b1 of the joint fit: DR1 (filled) and mean of the mocks (outline). (b) b1 of each tracer from the joint fit and from the tracer's own individual fit, with the 1-D errors (ERR); no truth is marked
# (the mock b1 is not known). Table: b1_3_b1.csv.

#%%
def b1_mcs(rest, label):
    c = joint_b1_columns(rest)
    names = list(c)
    stack = np.stack([c[n] for n in names], axis=-1)
    return MCSamples(samples=[stack[:, w] for w in range(stack.shape[1])], names=names, labels=[LABELS[n] for n in names], label=label, settings=SMOOTH)


if 3 in PLOTS_TO_RUN:
    cols = [f'b1_{t[0]}' for t in TRACERS]
    rj = {tag: f'{tag}_joint_smin{B1_SMIN}' for tag in DATASETS}
    figure('b1_3_b1_triangle', [item(b1_mcs(rj['mean'], 'mean of mocks'), 'tab:blue', False) if have(M, rj['mean']) else None,
                                item(b1_mcs(rj['dr1'], 'DR1'), 'tab:blue', True) if have(M, rj['dr1']) else None], cols, width=WIDTH_BIAS)
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(WIDTH, 2.8), sharey=True, squeeze=False)
    rows = []
    for ax, (tag, title) in zip(axes[0], DATASETS.items()):
        jstats = margstats(joint_b1_columns(rj[tag])) if have(M, rj[tag]) else {}
        for i, (short, key, color) in enumerate(TRACERS):
            if f'b1_{short}' in jstats:
                st = jstats[f'b1_{short}']
                bar(ax, i - 0.12, st, JOINT_COLOR)
                rows.append(dict(dataset=title, tracer=short, fit='joint', **st))
            ri = indiv_rest(tag, key)
            if ri:
                st = margstats({'b1': columns(M, ri)['b1']})['b1']
                bar(ax, i + 0.12, st, color)
                rows.append(dict(dataset=title, tracer=short, fit='individual', **st))
        ax.set_xticks(range(len(TRACERS)))
        ax.set_xticklabels([t[0] for t in TRACERS], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0, 0].set_ylabel(r'$b_1$', fontsize=9)
    fig.legend(handles=[Line2D([], [], marker='o', color=JOINT_COLOR, ls='none', label='joint fit'), Line2D([], [], marker='o', color='0.5', ls='none', label='individual fit (tracer colour)')],
               frameon=False, fontsize=7.5, loc='lower center', ncol=2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(PLOT_DIR / 'b1_3_b1_per_tracer.pdf', bbox_inches='tight')
    fig.savefig(PLOT_DIR / 'b1_3_b1_per_tracer.png', dpi=DPI, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    if rows:
        pd.DataFrame(rows).to_csv(PLOT_DIR / 'b1_3_b1.csv', index=False)

#%% [markdown]
# ## 4. The sampled parameters for the joint and each individual fit
# Posterior mean with the 1-D error bars (ERR) for h, omega_b, omega_cdm, ln A_s and n_s: left DR1, right the mean of the mocks (dashed = c000 truth). Joint fit black, individual tracers in their DESI colours.

#%%
if 4 in PLOTS_TO_RUN:
    fig, axes = plt.subplots(len(SAMPLED), len(DATASETS), figsize=(WIDTH, 1.25 * len(SAMPLED) + 0.7), sharex=True, squeeze=False)
    for col, (tag, title) in enumerate(DATASETS.items()):
        fits = [('Joint', f'{tag}_joint_smin{B1_SMIN}', JOINT_COLOR)] + [(short, indiv_rest(tag, key), color) for short, key, color in TRACERS]
        for i, (label, rest, color) in enumerate(fits):
            if not rest or not have(M, rest):
                continue
            c = columns(M, rest)
            stats = margstats({p: c[p] for p in SAMPLED})
            for row, p in enumerate(SAMPLED):
                bar(axes[row, col], i, stats[p], color)
        for row, p in enumerate(SAMPLED):
            if tag == 'mean':
                axes[row, col].axhline(TRUTH_S[p], ls='--', color='0.5', lw=0.9)
            axes[row, col].tick_params(labelsize=7)
            if col == 0:
                axes[row, col].set_ylabel(f'${LABELS[p]}$', fontsize=9)
        axes[0, col].set_title(title, fontsize=9)
        axes[-1, col].set_xticks(range(len(fits)))
        axes[-1, col].set_xticklabels([f[0] for f in fits], fontsize=7, rotation=45)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'b1_4_parameters_per_fit.pdf', bbox_inches='tight')
    fig.savefig(PLOT_DIR / 'b1_4_parameters_per_fit.png', dpi=DPI, bbox_inches='tight')
    plt.show()
    plt.close(fig)

#%% [markdown]
# ## 5. Goodness of fit at the MAP
# chi2 of the data given the MAP (highest-posterior) sample, with N_data, N_par (the sampled parameters of the fit: b2 is not one of them) and the p-value of chi2 for N_data - N_par degrees of freedom; no reduced chi2.
# The mean-of-mocks fits use the single-mock covariance, so their chi2 is multiplied by the number of mocks before the p-value is taken. For the joint fit the per-tracer chi2 is listed as well (its N_par is the joint
# fit's, shared by the tracers, so no per-tracer p-value is given). Tables: b1_5_goodness_of_fit.csv. Needs fits_apfix_11par_v3 (data files and emulators).

#%%
HERE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
fv = None
if 5 in PLOTS_TO_RUN or 6 in PLOTS_TO_RUN:
    try:
        sys.path.insert(0, str(HERE))
        import fits_apfix_11par_v3 as fv                          # imports your fits_apfix.py and common.py
        SPECS = {s['name']: s for s in fv.build_specs_v3()}
    except Exception as e:
        print(f'cells 5 and 6 skipped: cannot import fits_apfix_11par_v3 ({type(e).__name__}: {e})')
        fv = None


def evaluate_at_map(rest):
    """(FitProblem, per-tracer list of dict(s, data, model, sigma, chi2, n)) at the MAP of the b1-only chain `rest`."""
    problem = fv.fa.FitProblem(SPECS[f"{MODELS[M]['prefix']}_{rest}"])        # FitProblemB1: the parameter vector has no b2
    names = full3(M, rest)[1]
    mp = map_point(M, rest)
    full = problem._expand(np.array([mp[n] for n in names]))                  # b2 = B2_OF_B1(b1) put back
    biases = problem.unpack(full)[5]
    h, ombh2, omch2, lnAs, ns = problem.model_params(full)
    aps = fv.fa.ap_multi(h, ombh2, omch2, ns, problem.z_effs)
    out = []
    for d, (b1, b2), (q_perp, q_par) in zip(problem.data, biases, aps):
        mod = fv.fa.model_one(d['emus'], [b1, b2, ombh2, omch2, h, lnAs, ns], d['s_data'], d['W'], q_perp, q_par)
        r = mod - d['data_vec']
        out.append(dict(s=d['s_data'], data=d['data_vec'], model=mod, sigma=np.sqrt(np.diag(np.linalg.inv(d['cov_inv']))), chi2=float(r @ d['cov_inv'] @ r), n=len(r)))
    return problem, out


if 5 in PLOTS_TO_RUN and fv is not None:
    n_mocks = min(fv.fa.mock_counts().values())
    rows = []
    for tag, title in DATASETS.items():
        scale = n_mocks if tag == 'mean' else 1
        jr = f'{tag}_joint_smin{B1_SMIN}'
        if have(M, jr):
            problem, ev = evaluate_at_map(jr)
            chi2, n_data = scale * sum(e['chi2'] for e in ev), sum(e['n'] for e in ev)
            rows.append({'dataset': title, 'fit': 'joint', 'chi2': chi2, 'N_data': n_data, 'N_par': problem.ndim, 'dof': n_data - problem.ndim, 'p-value': float(chi2_dist.sf(chi2, n_data - problem.ndim)), 'chi2 scaled by N_mocks': scale != 1})
            for t, e in zip(TRACERS, ev):
                rows.append({'dataset': title, 'fit': f'joint, tracer {t[0]}', 'chi2': scale * e['chi2'], 'N_data': e['n'], 'chi2 scaled by N_mocks': scale != 1})
        for short, key, color in TRACERS:
            ir = indiv_rest(tag, key)
            if ir:
                problem, ev = evaluate_at_map(ir)
                chi2, n_data = scale * ev[0]['chi2'], ev[0]['n']
                rows.append({'dataset': title, 'fit': f'individual {short}', 'chi2': chi2, 'N_data': n_data, 'N_par': problem.ndim, 'dof': n_data - problem.ndim, 'p-value': float(chi2_dist.sf(chi2, n_data - problem.ndim)),
                             'chi2 scaled by N_mocks': scale != 1})
    if rows:
        GOF = pd.DataFrame(rows)
        GOF.to_csv(PLOT_DIR / 'b1_5_goodness_of_fit.csv', index=False)
        pd.set_option('display.width', 250)
        print(GOF.to_string(float_format=lambda v: f'{v:.4g}'))
        fig, axes = plt.subplots(1, len(DATASETS), figsize=(WIDTH, 2.8), squeeze=False)
        for ax, (tag, title) in zip(axes[0], DATASETS.items()):
            g = GOF[(GOF['dataset'] == title) & GOF['dof'].notna()]
            colors = {'joint': JOINT_COLOR, **{f'individual {t[0]}': t[2] for t in TRACERS}}
            for i, (_, r) in enumerate(g.iterrows()):
                ax.plot(i, r['chi2'], 'o', color=colors.get(r['fit'], 'k'))
                ax.plot([i - 0.3, i + 0.3], [r['dof']] * 2, 'k-', lw=1.0)
            ax.set_yscale('log')
            ax.set_xticks(range(len(g)))
            ax.set_xticklabels([f.replace('individual ', '') for f in g['fit']], fontsize=7, rotation=45)
            ax.set_ylabel(r'$\chi^2$ at the MAP' + (r' ($\times N_{\rm mocks}$)' if tag == 'mean' else ''), fontsize=9)
            ax.set_title(title, fontsize=9)
            ax.tick_params(labelsize=7)
        axes[0, 0].legend(handles=[Line2D([], [], marker='o', color='0.3', ls='none', label=r'$\chi^2$'), Line2D([], [], color='k', lw=1.0, label=r'$N_{\rm data}-N_{\rm par}$')], frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / 'b1_5_goodness_of_fit.pdf', bbox_inches='tight')
        fig.savefig(PLOT_DIR / 'b1_5_goodness_of_fit.png', dpi=DPI, bbox_inches='tight')
        plt.show()
        plt.close(fig)

#%% [markdown]
# ## 6. Data vs the MAP model: multipoles of each tracer-bin
# Rows: s^2 xi_0 and s^2 xi_2 with the diagonal data errors, each followed by the residual (model - data) / sigma. Solid blue: the MAP model of the joint fit; dashed (tracer colour): the MAP model of the tracer's own individual fit.
# For the mean of the mocks the errors are those of a single mock. Figures: b1_6_data_vs_map_{dr1,mean}.

#%%
if 6 in PLOTS_TO_RUN and fv is not None:
    for tag, title in DATASETS.items():
        jr = f'{tag}_joint_smin{B1_SMIN}'
        if not have(M, jr):
            continue
        ev_joint = evaluate_at_map(jr)[1]
        fig, axes = plt.subplots(4, len(TRACERS), figsize=(WIDTH * 1.9, 6.6), sharex='col', gridspec_kw={'height_ratios': [3, 1.3, 3, 1.3]}, squeeze=False)
        for j, (short, key, color) in enumerate(TRACERS):
            ref = ev_joint[j]
            s, n = ref['s'], len(ref['s'])
            ir = indiv_rest(tag, key)
            ev_ind = evaluate_at_map(ir)[1][0] if ir else None
            for ell_i, (row, label) in enumerate(((0, r'$s^2\xi_0$'), (2, r'$s^2\xi_2$'))):
                sl = slice(ell_i * n, (ell_i + 1) * n)
                ax, ar = axes[row, j], axes[row + 1, j]
                ax.errorbar(s, s ** 2 * ref['data'][sl], yerr=s ** 2 * ref['sigma'][sl], fmt='o', ms=2.5, color='k', elinewidth=0.7, capsize=0, label='data' if j == 0 and ell_i == 0 else None)
                ax.plot(s, s ** 2 * ref['model'][sl], color=MODELS[M]['color'], lw=1.0, label='joint MAP' if j == 0 and ell_i == 0 else None)
                ar.plot(s, (ref['model'][sl] - ref['data'][sl]) / ref['sigma'][sl], color=MODELS[M]['color'], lw=1.0)
                if ev_ind is not None:
                    ax.plot(s, s ** 2 * ev_ind['model'][sl], color=color, ls='--', lw=1.0, label='individual MAP' if j == 0 and ell_i == 0 else None)
                    ar.plot(s, (ev_ind['model'][sl] - ev_ind['data'][sl]) / ev_ind['sigma'][sl], color=color, ls='--', lw=1.0)
                ar.axhline(0, color='0.5', lw=0.7)
                ar.axhspan(-1, 1, color='0.85', lw=0, zorder=0)
                if j == 0:
                    ax.set_ylabel(label, fontsize=9)
                    ar.set_ylabel(r'$\Delta/\sigma$', fontsize=8)
                ax.tick_params(labelsize=6)
                ar.tick_params(labelsize=6)
            axes[0, j].set_title(short, fontsize=9)
            axes[3, j].set_xlabel(r'$s\;[h^{-1}{\rm Mpc}]$', fontsize=8)
        axes[0, 0].legend(frameon=False, fontsize=6)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f'b1_6_data_vs_map_{tag}.pdf', bbox_inches='tight')
        fig.savefig(PLOT_DIR / f'b1_6_data_vs_map_{tag}.png', dpi=DPI, bbox_inches='tight')
        plt.show()
        plt.close(fig)

#%% [markdown]
# ## 7. Chain health and coverage of the NN training box
# For every finished b1-only chain: N / tau_max (the fits stop when it exceeds TAU_FACTOR = 100 and tau is stable), the split-half test (largest shift of a sampled cosmological parameter's mean between the first and second half
# of the post-burn-in chain, in posterior stds; dotted line = 0.1 sigma as a guide, not a criterion), and the fraction of samples outside the NN training box in any sampled parameter (b1 included). Table: b1_7_chain_health.csv.

#%%
if 7 in PLOTS_TO_RUN:
    rows = []
    for rest in all_rests():
        arr, names, st = full3(M, rest)
        half = arr.shape[0] // 2
        a, b = arr[:half].reshape(-1, arr.shape[-1]), arr[half:2 * half].reshape(-1, arr.shape[-1])
        shifts = {n: abs(a[:, i].mean() - b[:, i].mean()) / np.concatenate([a[:, i], b[:, i]]).std() for i, n in enumerate(names) if n in SAMPLED}
        worst = max(shifts, key=shifts.get) if shifts else None
        out = None
        for i, n in enumerate(names):
            key = 'b1' if n.startswith('b1') else n
            if key in NN_BOX:
                o = (arr[..., i] < NN_BOX[key][0]) | (arr[..., i] > NN_BOX[key][1])
                out = o if out is None else out | o
        rows.append({'fit': rest, 'converged': st.get('converged'), 'iterations': st.get('iterations'), 'tau_max': st.get('tau_max'), 'N/tau_max': st.get('n_over_tau_max'), 'max split-half shift [std]': shifts[worst] if worst else np.nan,
                     'worst parameter': worst, 'fraction outside NN box': float(out.mean()) if out is not None else np.nan})
    if rows:
        HEALTH = pd.DataFrame(rows)
        HEALTH.to_csv(PLOT_DIR / 'b1_7_chain_health.csv', index=False)
        print(HEALTH.to_string(float_format=lambda v: f'{v:.4g}'))
        fig, axes = plt.subplots(3, 1, figsize=(WIDTH * 1.2, 5.6), sharex=True)
        x = np.arange(len(HEALTH))
        for ax, col, ylabel in zip(axes, ['N/tau_max', 'max split-half shift [std]', 'fraction outside NN box'], [r'$N/\tau_{\rm max}$', 'max split-half shift [std]', 'fraction outside NN box']):
            ax.plot(x, HEALTH[col], 'o', color=MODELS[M]['color'], ms=4)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0].axhline(TAU_FACTOR, color='0.4', ls='--', lw=0.9)
        axes[0].set_yscale('log')
        axes[1].axhline(0.1, color='0.4', ls=':', lw=0.9)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([short_fit_name(f) for f in HEALTH['fit']], rotation=90, fontsize=5.5)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / 'b1_7_chain_health.pdf', bbox_inches='tight')
        fig.savefig(PLOT_DIR / 'b1_7_chain_health.png', dpi=DPI, bbox_inches='tight')
        plt.show()
        plt.close(fig)

#%% [markdown]
# ## 8. Summary table
# Every b1-only fit: posterior mean, std and 1-D 68% limits of h, omega_b, omega_cdm, ln A_s, n_s and b1 (the joint fit has one b1 per tracer-bin), and for the mean of the mocks (mean - truth) / std. Table: b1_8_summary.csv.

#%%
if 8 in PLOTS_TO_RUN:
    rows = []
    for rest in all_rests():
        c = columns(M, rest)
        by_name = {p: c[p] for p in SAMPLED}
        if '_joint_' in rest:
            by_name.update(joint_b1_columns(rest))
        elif 'b1' in c:
            by_name['b1'] = c['b1']
        for p, st in margstats(by_name).items():
            row = {'fit': rest, 'parameter': p, **st}
            if rest.startswith('mean') and p in TRUTH_S:
                row['(mean - truth)/std'] = (st['mean'] - TRUTH_S[p]) / st['std']
            rows.append(row)
    if rows:
        SUMMARY = pd.DataFrame(rows)
        SUMMARY.to_csv(PLOT_DIR / 'b1_8_summary.csv', index=False)
        pd.set_option('display.width', 250)
        print(SUMMARY[SUMMARY['fit'].str.contains('joint')].to_string(float_format=lambda v: f'{v:.4g}'))
# %%
