"""
getdist plots for the 11par_v3 fits (fits_apfix_11par_v3.py): the two v3 emulators, compact_v3 (cosmology slopes on) and compact_v3_noslope (all slopes zero), against the baseline v2
fits (fits_11par_apfix_v2) and DESI DR1. Style and helpers follow plot_v2_chains_gd.py: our chains are given to getdist one chain per walker, KDE bandwidth SMOOTH x the automatic one,
unitless sampled parameters by default, figures saved as vector PDF + PNG (DPI) in PLOT_DIR. Run cell by cell (#%% markers) or as a script, compact environment (needs getdist, h5py).

Colours: v3 slopes on = red, filled (the reference); v3 slopes zero = blue outline; baseline v2 = black dashed outline; DESI in the DESI-paper colours; in the joint-vs-individual-tracer
plots the joint fit is filled black and the tracers have their DESI colours. Chains with no .status.json (not finished) are skipped with a message.

  1. v3 vs baseline vs DESI: joint fits at s_min = 20 (DR1 and mean of mocks; c000 truth marked for mocks)
  2. s_min = 20-40 for each model
  3. individual tracers vs the joint fit, for each model
  4. bias parameters: the three models
  5. constraining power relative to DESI FS-only (volume ratios, 1-D and pairwise ratios) for the three models
  6. posterior mean vs s_min with the 1-D marginalized 68% limits, for the three models (DR1 and mean of mocks)
  7. summary tables and figure: how far the models move each other (DR1) and how far each is from the c000 truth (mocks), and a one-line-per-fit table of every chain found
Not made (no v3 chains for them yet): single-mock offsets and the shift (tier 3), leave-one-out tensions, flat-prior fits, the mass-scheme, Gaussian and w0wa comparisons.
"""
#%%
import glob
import json
import textwrap
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

CHAIN_ROOT_V2 = Path('/spiffball/cosweeney/data/chains/fits_11par_apfix_v2')
CHAIN_ROOT_V3 = Path('/spiffball/cosweeney/data/chains/fits_11par_apfix_v2_v3')
PLOT_DIR = CHAIN_ROOT_V3 / 'plots_gd'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
DESI_FS = '/spiffball/cosweeney/DESI/DR1/full-shape-cosmo-params/v1.0/cobaya/base/desi-reptvelocileptors-fs-all_schoneberg2024-bbn_planck2018-ns10/'
DESI_BAO = '/spiffball/cosweeney/DESI/DR1/bao-cosmo-params/v1.0/cobaya/base/desi-bao-all_schoneberg2024-bbn/'      # adjust to your local directory name

PLOTS_TO_RUN = set(range(1, 8))
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

MODELS = {   # key -> chain-name prefix (= folder name), chain root, legend label and the way it is drawn
    'v2':   dict(prefix='compact',            root=CHAIN_ROOT_V2, label='Baseline (v2)',  color='k',        filled=False, ls='--'),
    'v3':   dict(prefix='compact_v3',         root=CHAIN_ROOT_V3, label='v3, slopes on',   color=RED,        filled=True,  ls='-'),
    'v3ns': dict(prefix='compact_v3_noslope', root=CHAIN_ROOT_V3, label='v3, slopes zero', color='tab:blue', filled=False, ls='-'),
}

#%% [markdown]
# ## Loading helpers

#%%
def fit_path(model, rest):
    """Path stem of the chain of `model` whose name is <prefix>_<rest>, e.g. rest = 'dr1_joint_smin20'."""
    m = MODELS[model]
    return m['root'] / m['prefix'] / f"{m['prefix']}_{rest}"


def have(model, rest):
    return Path(f'{fit_path(model, rest)}.status.json').exists()


def indiv_rest(model, tag, key):
    """`rest` of the individual-tracer chain whose name contains `_indiv_<key>`, or None."""
    m = MODELS[model]
    files = glob.glob(str(m['root'] / m['prefix'] / f"{m['prefix']}_{tag}_indiv_*_smin20.status.json"))
    match = [Path(p).name[:-len('.status.json')][len(m['prefix']) + 1:] for p in files if f'_indiv_{key}' in Path(p).name]
    return match[0] if match else None


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


def report_chains():
    """Print which chains exist and whether they converged."""
    for model, m in MODELS.items():
        files = sorted(glob.glob(str(m['root'] / m['prefix'] / f"{m['prefix']}_*.status.json")))
        bad = [Path(f).name for f in files if not json.load(open(f)).get('converged')]
        print(f"{m['label']:18s}: {len(files)} finished chains in {m['root'] / m['prefix']}" + (f"; NOT CONVERGED: {bad}" if bad else ''))


report_chains()

#%% [markdown]
# ## Figure helpers

#%%
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


def model_item(model, rest, params, label=None, filled=None):
    """The chain of `model` named <prefix>_<rest> drawn in the model's own style (or None if it is not finished)."""
    if not have(model, rest):
        return None
    m = MODELS[model]
    return item(mcs(model, rest, params, label), m['color'], m['filled'] if filled is None else filled, m['ls'])


TRUTH_FOR = {'dr1': None, 'mean': TRUTH}

#%% [markdown]
# ## 1. v3 vs baseline vs DESI: joint fits at s_min = 20
# The v3 slopes-on fit is filled red; the baseline (black dashed) and the zero-slope fit (blue) are outlines drawn on top of it so they stay visible.

#%%
fs = bao = None
if 1 in PLOTS_TO_RUN:
    fs, bao = desi(DESI_FS, 'DESI DR1 FS-only'), desi(DESI_BAO, 'DESI DR1 BAO')
    for tag in DATASETS:
        rest = f'{tag}_joint_smin20'
        figure(f'v3_1_models_vs_desi_{tag}', [item(bao, ORANGE, True), item(fs, PURPLE, True), model_item('v3', rest, PARAMS),
                                              model_item('v2', rest, PARAMS), model_item('v3ns', rest, PARAMS)], PARAMS, TRUTH_FOR[tag])

#%% [markdown]
# ## 2. s_min for each model
# s_min = 25-40 as outlines, s_min = 20 filled in the model's colour.

#%%
if 2 in PLOTS_TO_RUN:
    colors = {25: 'tab:orange', 30: 'tab:green', 35: 'tab:blue', 40: 'tab:purple'}
    for model, m in MODELS.items():
        for tag in DATASETS:
            items = [item(mcs(model, f'{tag}_joint_smin{s}', PARAMS, rf'$s_{{\rm min}}={s}$'), colors[s], False, '-') if have(model, f'{tag}_joint_smin{s}') else None for s in SMINS[1:]]
            figure(f'v3_2_smin_{model}_{tag}', items + [model_item(model, f'{tag}_joint_smin20', PARAMS, r'$s_{\rm min}=20$ (' + m['label'] + ')', filled=True)], PARAMS, TRUTH_FOR[tag])

#%% [markdown]
# ## 3. Individual-tracer fits vs the joint fit, for each model
# Individual fits are filled in the DESI tracer colours; the joint fit (s_min = 20) is filled black and on top.

#%%
if 3 in PLOTS_TO_RUN:
    for model, m in MODELS.items():
        for tag in DATASETS:
            items = []
            for short, key, color in TRACERS:
                rest = indiv_rest(model, tag, key)
                items.append(item(mcs(model, rest, PARAMS, short), color, True) if rest else None)
            joint = f'{tag}_joint_smin20'
            figure(f'v3_3_individual_{model}_{tag}', items + [item(mcs(model, joint, PARAMS, 'Joint'), JOINT_COLOR, True) if have(model, joint) else None], PARAMS, TRUTH_FOR[tag])

#%% [markdown]
# ## 4. Bias parameters
# b1, b2 for the six tracer bins from the joint s_min = 20 fits of the three models.

#%%
def bias_mcs(model, rest, label):
    arr, names, _ = full3(model, rest)
    b1, b2 = [i for i, n in enumerate(names) if n.startswith('b1_')], [i for i, n in enumerate(names) if n.startswith('b2_')]
    idx, cols = [], []
    for (short, _k, _c), i1, i2 in zip(TRACERS, b1, b2):
        idx += [i1, i2]
        cols += [f'b1_{short}', f'b2_{short}']
    stack = arr[..., idx]
    return MCSamples(samples=[stack[:, w] for w in range(stack.shape[1])], names=cols, labels=[LABELS[c] for c in cols], label=label, settings=SMOOTH)


if 4 in PLOTS_TO_RUN:
    cols = [f'{b}_{t[0]}' for t in TRACERS for b in ('b1', 'b2')]
    for tag in DATASETS:
        rest = f'{tag}_joint_smin20'
        items = {m: item(bias_mcs(m, rest, MODELS[m]['label']), MODELS[m]['color'], MODELS[m]['filled'], MODELS[m]['ls']) if have(m, rest) else None for m in MODELS}
        figure(f'v3_4_bias_{tag}', [items['v3'], items['v2'], items['v3ns']], cols, width=WIDTH_BIAS)

#%% [markdown]
# ## 5. Constraining power relative to DESI FS-only
# sqrt(det C) of the posterior covariance C of a set of parameters is proportional to the volume of the posterior's 1-sigma ellipsoid, so the ratio ours / DESI FS-only is the relative
# volume: below 1 means our constraints are tighter there. The table (v3_5_constraining_power_<tag>.csv) has it for several parameter spaces (`per dim` = the d-th root); the figure shows
# the 1-D (diagonal) and pairwise (off-diagonal) ratios of the five sampled parameters for the three models side by side. Needs the DESI FS chain.

#%%
def sqdet(smp, names):
    """sqrt(det of the covariance) of the named parameters; NaN if the chain lacks one of them."""
    if smp is None or any(n not in smp.getParamNames().list() for n in names):
        return np.nan
    return float(np.exp(0.5 * np.linalg.slogdet(np.atleast_2d(smp.cov(pars=names)))[1]))


if 5 in PLOTS_TO_RUN:
    if fs is None:
        fs = desi(DESI_FS, 'DESI DR1 FS-only')
    ALLP = ['h', 'H0', 'omegam', 'ombh2', 'omch2', 'logA', 'ns']
    HP = 'h' if UNITLESS else 'H0'
    SPACES = {'sampled (h, wb, wcdm, lnAs, ns)': ['h', 'ombh2', 'omch2', 'logA', 'ns'], 'sampled minus wb': ['h', 'omch2', 'logA', 'ns'],
              f'physical (Om, {HP}, wb, lnAs, ns)': ['omegam', HP, 'ombh2', 'logA', 'ns'], f'(Om, {HP})': ['omegam', HP], '(h, wcdm)': ['h', 'omch2'], '(lnAs, ns)': ['logA', 'ns']}
    MATRIX = ['h', 'ombh2', 'omch2', 'logA', 'ns']
    for tag in DATASETS:
        rest = f'{tag}_joint_smin20'
        chains = {m: mcs(m, rest, ALLP) for m in MODELS if have(m, rest)}
        if fs is None or not chains:
            print(f'plot 5 ({tag}): DESI FS chain or our chains missing, skipped')
            continue
        rows = []
        for label, names in SPACES.items():
            d, ref = len(names), sqdet(fs, names)
            row = {'space': label, 'dim': d, 'sqrt det DESI FS': ref}
            for m, smp in chains.items():
                v = sqdet(smp, names)
                row.update({f'{MODELS[m]["label"]} / DESI': v / ref, f'{MODELS[m]["label"]} / DESI per dim': (v / ref) ** (1 / d)})
            rows.append(row)
        table = pd.DataFrame(rows).set_index('space')
        table.to_csv(PLOT_DIR / f'v3_5_constraining_power_{tag}.csv')
        print(f'--- {DATASETS[tag]}: volume ratio to DESI FS-only')
        print(table.to_string(float_format=lambda x: f'{x:.4g}'))

        n = len(MATRIX)
        fig, axes = plt.subplots(1, len(chains), figsize=(WIDTH * len(chains) / 2.0, 0.55 * WIDTH), squeeze=False)
        for ax, (m, smp) in zip(axes[0], chains.items()):
            R = np.full((n, n), np.nan)
            for i, a in enumerate(MATRIX):
                for j, b in enumerate(MATRIX):
                    R[i, j] = (sqdet(smp, [a]) / sqdet(fs, [a])) if i == j else sqdet(smp, [a, b]) / sqdet(fs, [a, b])
            if np.allclose(R, R.T, rtol=1e-6, equal_nan=True):
                R = np.where(np.tri(n, dtype=bool), R, np.nan)
            cmap = plt.get_cmap('RdBu_r').copy()
            cmap.set_bad('white')
            im = ax.imshow(np.ma.masked_invalid(np.log10(R)), cmap=cmap, vmin=-0.7, vmax=0.7)
            for i in range(n):
                for j in range(n):
                    if np.isfinite(R[i, j]):
                        ax.text(j, i, f'{R[i, j]:.2f}', ha='center', va='center', fontsize=8.5)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels([f'${LABELS[p]}$' for p in MATRIX], fontsize=9, rotation=45)
            ax.set_yticklabels([f'${LABELS[p]}$' for p in MATRIX] if ax is axes[0][0] else [], fontsize=9)
            ax.set_xlabel(f'{MODELS[m]["label"]} / DESI FS-only', fontsize=10)
        cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, pad=0.02)
        cb.set_ticks([-0.7, -0.35, 0, 0.35, 0.7])
        cb.set_ticklabels(['0.2', '0.45', '1', '2.2', '5'])
        cb.set_label('ratio (blue: ours tighter)', fontsize=9)
        cb.ax.tick_params(labelsize=8)
        fig.savefig(PLOT_DIR / f'v3_5_constraining_power_matrix_{tag}.pdf', bbox_inches='tight')
        fig.savefig(PLOT_DIR / f'v3_5_constraining_power_matrix_{tag}.png', dpi=DPI_MATRIX, bbox_inches='tight')
        plt.show()
        plt.close(fig)

#%% [markdown]
# ## 6. Parameters vs s_min
# The five sampled parameters (rows) of each joint fit at s_min = 20-40, DR1 on the left and the mean of the mocks on the right (dashed line = c000 truth), for the three models side by side. The point is the 1-D marginalized posterior
# mean (S6_POINT = 'map' plots the joint maximum-posterior sample instead, which is a noisy statistic: it scatters by roughly 0.5 posterior std per parameter in a 17-parameter fit and can sit away from the marginal peak), and the error
# bars are by default (S6_ERR = '68') the 1-D marginalized 68% limits computed by getdist from the same MCSamples (one chain per walker, the same smoothing) as the triangle plots, so the widths here are the ones those plots show;
# S6_ERR = 'std' draws the posterior standard deviation instead (symmetric about the plotted point).

#%%
S6_POINT = 'mean'        # 'mean' (1-D marginalized posterior mean) or 'map' (joint highest-posterior sample)
S6_ERR = '68'            # error bars: '68' (1-D marginalized 68% limits, the widths of the triangle plots; asymmetric) or 'std' (posterior standard deviation, symmetric about the plotted point)
if 6 in PLOTS_TO_RUN:
    fig, axes = plt.subplots(len(SAMPLED), len(DATASETS), figsize=(WIDTH, 1.2 * len(SAMPLED) + 0.7), sharex=True, squeeze=False)
    offsets = {'v2': -0.9, 'v3': 0.0, 'v3ns': 0.9}
    for col, (tag, title) in enumerate(DATASETS.items()):
        for model, m in MODELS.items():
            for sm in SMINS:
                rest = f'{tag}_joint_smin{sm}'
                if not have(model, rest):
                    continue
                stats = mcs(model, rest, SAMPLED).getMargeStats()
                pt = map_point(model, rest) if S6_POINT == 'map' else None
                for row, p in enumerate(SAMPLED):
                    par = stats.parWithName(p)
                    lim = par.limits[0]                                   # 68% limits of the 1-D marginalized posterior (getdist's default first contour level)
                    x, y = sm + offsets[model], (par.mean if S6_POINT == 'mean' else pt[p])
                    lo, hi = (lim.lower, lim.upper) if S6_ERR == '68' else (y - par.err, y + par.err)       # par.err = posterior std
                    axes[row, col].errorbar(x, 0.5 * (lo + hi), yerr=0.5 * (hi - lo), fmt='none', ecolor=m['color'], capsize=2, elinewidth=1.0)
                    axes[row, col].plot(x, y, 'o', ms=4, color=m['color'], mfc=m['color'] if m['filled'] else 'white')
        for row, p in enumerate(SAMPLED):
            if tag == 'mean':
                axes[row, col].axhline(TRUTH_S[p], ls='--', color='0.5', lw=0.9)
            axes[row, col].tick_params(labelsize=7)
            if col == 0:
                axes[row, col].set_ylabel(f'${LABELS[p]}$', fontsize=9)
        axes[0, col].set_title(title, fontsize=9)
        axes[-1, col].set_xlabel(r'$s_{\rm min}\;[h^{-1}{\rm Mpc}]$', fontsize=9)
        axes[-1, col].set_xticks(SMINS)
        axes[-1, col].set_xlim(min(SMINS) - 3, max(SMINS) + 3)
    fig.legend(handles=[Line2D([], [], marker='o', color=m['color'], mfc=m['color'] if m['filled'] else 'white', ls='none', label=m['label']) for m in MODELS.values()] +
                       [Line2D([], [], color='0.5', ls='--', label='c000 truth (mocks)'),
                        Line2D([], [], color='0.3', marker='o', ms=4, lw=1.0, label=('posterior mean' if S6_POINT == 'mean' else 'MAP') + (', 1-D 68% limits' if S6_ERR == '68' else r', $\pm$ posterior std'))], frameon=False, fontsize=7.5, loc='lower center', ncol=3)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(PLOT_DIR / 'v3_6_params_vs_smin.pdf', bbox_inches='tight')
    fig.savefig(PLOT_DIR / 'v3_6_params_vs_smin.png', dpi=DPI, bbox_inches='tight')
    plt.show()
    plt.close(fig)

#%% [markdown]
# ## 7. Summary
# Joint fits at s_min = 20. DR1: the shift of the posterior mean of each model relative to another in units of the reference model's posterior std (v3 - v2, zero-slope - v2, zero-slope - v3). Mocks: the offset of
# each model's posterior mean from the c000 truth in units of its own posterior std. Plus the ratio of posterior stds. Tables: v3_7_model_comparison.csv and v3_7_all_fits.csv (every chain found: convergence and posterior means and stds).

#%%
if 7 in PLOTS_TO_RUN:
    rows = []
    shifts = {}
    for tag in DATASETS:
        rest = f'{tag}_joint_smin20'
        d = {m: flat(m, rest) for m in MODELS if have(m, rest)}
        for p in PARAMS:
            row = {'dataset': DATASETS[tag], 'parameter': p}
            for m, df in d.items():
                row[f'mean {m}'], row[f'std {m}'] = df[p].mean(), df[p].std()
                if tag == 'mean':
                    row[f'(mean - truth)/std {m}'] = (df[p].mean() - TRUTH[p]) / df[p].std()
            for a, b, ref in (('v3', 'v2', 'v2'), ('v3ns', 'v2', 'v2'), ('v3ns', 'v3', 'v3')):
                if a in d and b in d:
                    row[f'({a} - {b})/std {ref}'] = (d[a][p].mean() - d[b][p].mean()) / d[ref][p].std()
                    row[f'std {a} / std {b}'] = d[a][p].std() / d[b][p].std()
                    shifts[(tag, p, a, b)] = row[f'({a} - {b})/std {ref}']
            rows.append(row)
    if rows:
        COMPARE = pd.DataFrame(rows).set_index(['dataset', 'parameter'])
        COMPARE.to_csv(PLOT_DIR / 'v3_7_model_comparison.csv')
        pd.set_option('display.width', 250)
        print(COMPARE.to_string(float_format=lambda x: f'{x:.3g}'))
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8), sharey=False)
        x = np.arange(len(PARAMS))
        for k, (a, b, color, label) in enumerate((('v3', 'v2', RED, r'v3 $-$ v2'), ('v3ns', 'v2', 'tab:blue', r'zero-slope $-$ v2'), ('v3ns', 'v3', '0.5', r'zero-slope $-$ v3'))):
            axes[0].plot(x + (k - 1) * 0.2, [shifts.get(('dr1', p, a, b), np.nan) for p in PARAMS], 'o', color=color, label=label)
        axes[0].axhline(0, color='k', lw=0.8)
        axes[0].set_ylabel(r'shift of the posterior mean [std of the reference]', fontsize=8)
        axes[0].set_title('DR1', fontsize=9)
        axes[0].legend(frameon=False, fontsize=7)
        for model, m in MODELS.items():
            axes[1].plot(x + (list(MODELS).index(model) - 1) * 0.2, [COMPARE[f'(mean - truth)/std {model}'].get(('mean of mocks', p), np.nan) if f'(mean - truth)/std {model}' in COMPARE else np.nan for p in PARAMS],
                         'o', color=m['color'], mfc=m['color'] if m['filled'] else 'white', label=m['label'])
        axes[1].axhline(0, color='k', lw=0.8)
        axes[1].set_ylabel(r'(mean $-$ truth) / std', fontsize=8)
        axes[1].set_title('mean of mocks', fontsize=9)
        axes[1].legend(frameon=False, fontsize=7)
        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels([f'${LABELS[p]}$' for p in PARAMS], fontsize=9)
            ax.tick_params(labelsize=7)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / 'v3_7_model_comparison.pdf', bbox_inches='tight')
        fig.savefig(PLOT_DIR / 'v3_7_model_comparison.png', dpi=DPI, bbox_inches='tight')
        plt.show()
        plt.close(fig)

    allrows = []
    for model, m in MODELS.items():
        for sp in sorted(glob.glob(str(m['root'] / m['prefix'] / f"{m['prefix']}_*.status.json"))):
            rest = Path(sp).name[:-len('.status.json')][len(m['prefix']) + 1:]
            st = json.load(open(sp))
            row = {'model': m['label'], 'fit': rest, 'converged': st.get('converged'), 'iterations': st.get('iterations'), 'N/tau_max': st.get('n_over_tau_max')}
            if Path(sp[:-len('.status.json')] + '.h5').exists():
                df = flat(model, rest)
                for p in SAMPLED:
                    row[f'{p} mean'], row[f'{p} std'] = df[p].mean(), df[p].std()
            allrows.append(row)
    if allrows:
        pd.DataFrame(allrows).to_csv(PLOT_DIR / 'v3_7_all_fits.csv', index=False)
        print(f'v3_7_all_fits.csv: {len(allrows)} chains')


#%% [markdown]
# # Validation of the v3 fits
# Cells 8-12 (appended; they only use the helpers defined above). 8: chain health (N/tau and a split-half test); 9: goodness of fit at the MAP (chi2, N_data, N_par, p-value) for the three models; 10: data vs the MAP
# model multipoles of each tracer-bin with the model-model differences; 11: truth recovery on the mocks vs s_min; 12: how much of each posterior lies outside the NN training box and outside the calibration
# grid of the cosmology emulator. Cells 9 and 10 evaluate the models, so they import fits_apfix_11par_v3 (and through it your fits_apfix.py and common.py): compact environment, data files and emulators needed.

#%%
import sys
from scipy.stats import chi2 as chi2_dist

VALIDATION_PLOTS = set(range(8, 13))
TAU_FACTOR = 100             # the fits' convergence rule: chain length > TAU_FACTOR x tau for every parameter
OFFSET = {'v2': -0.2, 'v3': 0.0, 'v3ns': 0.2}      # horizontal offsets so the three models do not overlap (fit-index axis)
SMIN_OFFSET = {k: 4 * v for k, v in OFFSET.items()}      # the same on an s_min axis


def all_rests(model):
    """Names (without the model prefix) of every finished chain of `model`, sorted."""
    m = MODELS[model]
    return sorted(Path(p).name[:-len('.status.json')][len(m['prefix']) + 1:] for p in glob.glob(str(m['root'] / m['prefix'] / f"{m['prefix']}_*.status.json")))


def short_fit_name(rest):
    """Shorter label for a fit name, e.g. dr1_joint_smin20 -> DR1 joint 20, mean_indiv_LRG_0.4-0.6_smin20 -> mean LRG_0.4-0.6."""
    tag, _, tail = rest.partition('_')
    tail = tail.replace('joint_smin', 'joint ').replace('indiv_', '').replace('_smin20', '')
    return f"{'DR1' if tag == 'dr1' else 'mean'} {tail}"


#%% [markdown]
# ## 8. Chain health
# For every finished chain of the three models: N / tau_max (post-run chain length over the largest autocorrelation time; the fits stop when this exceeds TAU_FACTOR = 100 and tau is stable) and a split-half test: the largest shift of
# the mean of a sampled cosmological parameter between the first and second half of the post-burn-in chain, in units of its posterior std (dotted line at 0.1 sigma as a guide, not a criterion). Table: v3_8_chain_health.csv.

#%%
if 8 in VALIDATION_PLOTS:
    rows = []
    for model in MODELS:
        for rest in all_rests(model):
            arr, names, st = full3(model, rest)
            half = arr.shape[0] // 2
            a, b = arr[:half].reshape(-1, arr.shape[-1]), arr[half:2 * half].reshape(-1, arr.shape[-1])
            shifts = {n: abs(a[:, i].mean() - b[:, i].mean()) / np.concatenate([a[:, i], b[:, i]]).std() for i, n in enumerate(names) if n in SAMPLED}
            worst = max(shifts, key=shifts.get) if shifts else None
            rows.append({'model': model, 'fit': rest, 'converged': st.get('converged'), 'iterations': st.get('iterations'), 'tau_max': st.get('tau_max'), 'N/tau_max': st.get('n_over_tau_max'),
                         'max split-half shift [std]': shifts[worst] if worst else np.nan, 'worst parameter': worst})
    HEALTH = pd.DataFrame(rows)
    if len(HEALTH):
        HEALTH.to_csv(PLOT_DIR / 'v3_8_chain_health.csv', index=False)
        print(HEALTH[HEALTH['converged'] != True].to_string() if (HEALTH['converged'] != True).any() else 'all chains report converged = True')
        fits_all = sorted(set(HEALTH['fit']))
        x = {f: i for i, f in enumerate(fits_all)}
        fig, axes = plt.subplots(2, 1, figsize=(WIDTH * 1.5, 4.8), sharex=True)
        for model, m in MODELS.items():
            h = HEALTH[HEALTH['model'] == model]
            xs = [x[f] + OFFSET[model] for f in h['fit']]
            kw = dict(color=m['color'], mfc=m['color'] if m['filled'] else 'white', ms=4, ls='none', marker='o', label=m['label'])
            axes[0].plot(xs, h['N/tau_max'], **kw)
            axes[1].plot(xs, h['max split-half shift [std]'], **kw)
        axes[0].axhline(TAU_FACTOR, color='0.4', ls='--', lw=0.9)
        axes[0].set_yscale('log')
        axes[0].set_ylabel(r'$N/\tau_{\rm max}$', fontsize=9)
        axes[0].legend(frameon=False, fontsize=7, ncol=3)
        axes[1].axhline(0.1, color='0.4', ls=':', lw=0.9)
        axes[1].set_ylabel('max split-half shift [std]', fontsize=9)
        axes[1].set_xticks(range(len(fits_all)))
        axes[1].set_xticklabels([short_fit_name(f) for f in fits_all], rotation=90, fontsize=5)
        for ax in axes:
            ax.tick_params(labelsize=7)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / 'v3_8_chain_health.pdf', bbox_inches='tight')
        fig.savefig(PLOT_DIR / 'v3_8_chain_health.png', dpi=DPI, bbox_inches='tight')
        plt.show()
        plt.close(fig)

#%% [markdown]
# ## 9. Goodness of fit at the MAP
# chi2 of the data given the MAP (highest-posterior) sample, with N_data, N_par (all sampled parameters) and the p-value of chi2 for N_data - N_par degrees of freedom; no reduced chi2. The mean-of-mocks fits use the single-mock
# covariance, so their chi2 is multiplied by the number of mocks (equivalent to the covariance of the mean) before the p-value is taken: it is the chi2 of the mean-of-mocks data vector against the model. Per-tracer chi2 of the joint s_min = 20
# fits is also listed. Tables: v3_9_goodness_of_fit.csv, v3_9_chi2_per_tracer.csv.

#%%
HERE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
fv = None
if 9 in VALIDATION_PLOTS or 10 in VALIDATION_PLOTS:
    try:
        sys.path.insert(0, str(HERE))
        import fits_apfix_11par_v3 as fv                         # imports your fits_apfix.py and common.py (data, emulators)
        SPECS = {s['name']: s for s in fv._baseline_build_specs()}
        SPECS.update({s['name']: s for s in fv.build_specs_v3()})
    except Exception as e:
        print(f'cells 9 and 10 skipped: cannot import fits_apfix_11par_v3 ({type(e).__name__}: {e})')
        fv = None


def evaluate_at_map(model, rest):
    """Per-tracer model and data at the MAP of a chain: (FitProblem, theta, list of dict(s, data, model, sigma, chi2, n))."""
    spec = SPECS[f"{MODELS[model]['prefix']}_{rest}"]
    problem = fv.fa.FitProblem(spec)
    names = full3(model, rest)[1]
    mp = map_point(model, rest)
    theta = np.array([mp[n] for n in names])
    biases = problem.unpack(theta)[5]
    h, ombh2, omch2, lnAs, ns = problem.model_params(theta)
    aps = fv.fa.ap_multi(h, ombh2, omch2, ns, problem.z_effs)
    out = []
    for d, (b1, b2), (q_perp, q_par) in zip(problem.data, biases, aps):
        mod = fv.fa.model_one(d['emus'], [b1, b2, ombh2, omch2, h, lnAs, ns], d['s_data'], d['W'], q_perp, q_par)
        r = mod - d['data_vec']
        out.append(dict(s=d['s_data'], data=d['data_vec'], model=mod, sigma=np.sqrt(np.diag(np.linalg.inv(d['cov_inv']))), chi2=float(r @ d['cov_inv'] @ r), n=len(r)))
    return problem, theta, out


if 9 in VALIDATION_PLOTS and fv is not None:
    n_mocks = min(fv.fa.mock_counts().values())
    rows, per_tracer = [], []
    for model in MODELS:
        for tag in DATASETS:
            for sm in SMINS:
                rest = f'{tag}_joint_smin{sm}'
                if not have(model, rest):
                    continue
                problem, _, ev = evaluate_at_map(model, rest)
                scale = n_mocks if tag == 'mean' else 1
                chi2 = scale * sum(e['chi2'] for e in ev)
                n_data, n_par = sum(e['n'] for e in ev), problem.ndim
                rows.append({'model': model, 'dataset': DATASETS[tag], 's_min': sm, 'chi2': chi2, 'N_data': n_data, 'N_par': n_par, 'dof': n_data - n_par,
                             'p-value': float(chi2_dist.sf(chi2, n_data - n_par)), 'chi2 scaled by N_mocks': scale != 1})
                if sm == 20:
                    per_tracer += [{'model': model, 'dataset': DATASETS[tag], 'tracer': t[0], 'chi2': scale * e['chi2'], 'N_data': e['n']} for t, e in zip(TRACERS, ev)]
    if rows:
        GOF = pd.DataFrame(rows)
        GOF.to_csv(PLOT_DIR / 'v3_9_goodness_of_fit.csv', index=False)
        pd.DataFrame(per_tracer).to_csv(PLOT_DIR / 'v3_9_chi2_per_tracer.csv', index=False)
        print(GOF.to_string(float_format=lambda v: f'{v:.4g}'))
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8), sharey=False)
        for ax, (tag, title) in zip(axes, DATASETS.items()):
            g = GOF[GOF['dataset'] == title]
            for model, m in MODELS.items():
                gm = g[g['model'] == model]
                ax.plot(gm['s_min'] + SMIN_OFFSET[model], gm['chi2'], 'o', color=m['color'], mfc=m['color'] if m['filled'] else 'white', label=m['label'])
            if len(g):
                dof = g.groupby('s_min')['dof'].first()
                ax.plot(dof.index, dof.values, 'k--', lw=0.9, label=r'$N_{\rm data}-N_{\rm par}$')
            ax.set_xlabel(r'$s_{\rm min}\;[h^{-1}{\rm Mpc}]$', fontsize=9)
            ax.set_ylabel(r'$\chi^2$ at the MAP' + (r' ($\times N_{\rm mocks}$)' if tag == 'mean' else ''), fontsize=9)
            ax.set_title(title, fontsize=9)
            ax.tick_params(labelsize=7)
        axes[0].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / 'v3_9_goodness_of_fit.pdf', bbox_inches='tight')
        fig.savefig(PLOT_DIR / 'v3_9_goodness_of_fit.png', dpi=DPI, bbox_inches='tight')
        plt.show()
        plt.close(fig)

#%% [markdown]
# ## 10. Data vs the MAP model: multipoles of each tracer-bin
# Joint s_min = 20 fits. Rows: s^2 xi_0 and s^2 xi_2 (data with the diagonal errors of its covariance, and the MAP model of each of the three fits), each followed by the residual (model - data) / sigma. The three models are on
# the same data and covariance, so differences between the lines are the model-to-model differences. For the mean-of-mocks the errors are those of a single mock. Figures: v3_10_data_vs_map_{dr1,mean}.

#%%
if 10 in VALIDATION_PLOTS and fv is not None:
    for tag, title in DATASETS.items():
        rest = f'{tag}_joint_smin20'
        evs = {model: evaluate_at_map(model, rest)[2] for model in MODELS if have(model, rest)}
        if not evs:
            continue
        nt = len(TRACERS)
        fig, axes = plt.subplots(4, nt, figsize=(WIDTH * 1.9, 6.6), sharex='col', gridspec_kw={'height_ratios': [3, 1.3, 3, 1.3]}, squeeze=False)
        for j, (short, _key, _color) in enumerate(TRACERS):
            ref = next(iter(evs.values()))[j]
            s, n = ref['s'], len(ref['s'])
            for ell_i, (row, label) in enumerate(((0, r'$s^2\xi_0$'), (2, r'$s^2\xi_2$'))):
                sl = slice(ell_i * n, (ell_i + 1) * n)
                ax, ar = axes[row, j], axes[row + 1, j]
                ax.errorbar(s, s ** 2 * ref['data'][sl], yerr=s ** 2 * ref['sigma'][sl], fmt='o', ms=2.5, color='k', elinewidth=0.7, capsize=0, label='data' if j == 0 and ell_i == 0 else None)
                for model, ev in evs.items():
                    m = MODELS[model]
                    ax.plot(s, s ** 2 * ev[j]['model'][sl], color=m['color'], ls=m['ls'] if model != 'v3ns' else '-', lw=1.0, label=m['label'] if j == 0 and ell_i == 0 else None)
                    ar.plot(s, (ev[j]['model'][sl] - ev[j]['data'][sl]) / ev[j]['sigma'][sl], color=m['color'], ls=m['ls'] if model != 'v3ns' else '-', lw=1.0)
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
        fig.savefig(PLOT_DIR / f'v3_10_data_vs_map_{tag}.pdf', bbox_inches='tight')
        fig.savefig(PLOT_DIR / f'v3_10_data_vs_map_{tag}.png', dpi=DPI, bbox_inches='tight')
        plt.show()
        plt.close(fig)

#%% [markdown]
# ## 11. Truth recovery on the mocks vs s_min
# Mean-of-mocks joint fits: the offset of the posterior mean from the c000 truth in posterior stds for each sampled parameter (the band is +-1), and the p-value of the 5-parameter Mahalanobis distance of the truth,
# d^T C^-1 d with C the posterior covariance, for 5 degrees of freedom. These fits use the single-mock covariance on the mean of 25 mocks, so offsets well below 1 std are expected. Table: v3_11_truth_recovery.csv.

#%%
if 11 in VALIDATION_PLOTS:
    rows = []
    for model in MODELS:
        for sm in SMINS:
            rest = f'mean_joint_smin{sm}'
            if not have(model, rest):
                continue
            df = flat(model, rest, SAMPLED)[SAMPLED]
            d = df.mean().to_numpy() - np.array([TRUTH_S[p] for p in SAMPLED])
            d2 = float(d @ np.linalg.solve(np.cov(df.to_numpy().T), d))
            row = {'model': model, 's_min': sm, 'Mahalanobis d2': d2, 'p-value (5 dof)': float(chi2_dist.sf(d2, len(SAMPLED)))}
            row.update({f'{p} (mean - truth)/std': float(d[i] / df[p].std()) for i, p in enumerate(SAMPLED)})
            rows.append(row)
    if rows:
        TRUTHREC = pd.DataFrame(rows)
        TRUTHREC.to_csv(PLOT_DIR / 'v3_11_truth_recovery.csv', index=False)
        print(TRUTHREC.to_string(float_format=lambda v: f'{v:.3g}'))
        fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.2), sharex=True)
        for ax, p in zip(axes.ravel(), SAMPLED + ['p']):
            for model, m in MODELS.items():
                g = TRUTHREC[TRUTHREC['model'] == model]
                y = g['p-value (5 dof)'] if p == 'p' else g[f'{p} (mean - truth)/std']
                ax.plot(g['s_min'] + SMIN_OFFSET[model], y, 'o', color=m['color'], mfc=m['color'] if m['filled'] else 'white', ms=4, label=m['label'])
            if p == 'p':
                ax.set_ylabel('p-value of the truth (5 dof)', fontsize=8)
                ax.set_ylim(0, 1.02)
            else:
                ax.axhline(0, color='k', lw=0.8)
                ax.axhspan(-1, 1, color='0.88', lw=0, zorder=0)
                ax.set_ylabel(rf'$({LABELS[p]}-{LABELS[p]}^{{\rm true}})/\sigma$', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_xticks(SMINS)
        for ax in axes[-1]:
            ax.set_xlabel(r'$s_{\rm min}\;[h^{-1}{\rm Mpc}]$', fontsize=8)
        axes[0, 0].legend(frameon=False, fontsize=6.5)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / 'v3_11_truth_recovery.pdf', bbox_inches='tight')
        fig.savefig(PLOT_DIR / 'v3_11_truth_recovery.png', dpi=DPI, bbox_inches='tight')
        plt.show()
        plt.close(fig)

#%% [markdown]
# ## 12. Posterior coverage of the emulator: NN training box and calibration grid
# Fraction of each posterior's samples outside (a) the range the NN emulators were trained on (BOX of fits_apfix.py: all sampled cosmological and bias parameters) and (b) the range of the AbacusSummit cosmology grid the cosmology
# emulator was calibrated on (omega_b, omega_cdm, ln A_s, n_s and h of the sims c130-c181 in cosmologies.csv; skipped if the file is not found). Outside the box the NNs extrapolate; outside the calibration grid the ridge slopes extrapolate.
# Table with the per-parameter fractions: v3_12_coverage.csv; figure: the joint fits.

#%%
COSMO_CSV = Path('/home/cosweeney/code/compact_streaming/nb/calibration/runs/auto_runs/cosmologies.csv')
NN_BOX = {'h': (0.55, 0.90), 'ombh2': (0.0207, 0.0243), 'omch2': (0.104, 0.140), 'logA': (2.0, 3.91), 'ns': (0.901, 1.025), 'b1': (0.0, 2.0), 'b2': (-5.0, 10.0)}     # as BOX in fits_apfix.py
GRID_BOX = None
if 12 in VALIDATION_PLOTS and COSMO_CSV.exists():
    t = pd.read_csv(COSMO_CSV, skipinitialspace=True)
    t.columns = [c.strip() for c in t.columns]
    t['root'] = t['root'].str.strip()
    t = t[t['root'].isin([f'abacus_cosm{n:03d}' for n in range(130, 182)])]
    v = {'h': t['h'], 'ombh2': t['omega_b'], 'omch2': t['omega_cdm'], 'logA': np.log(1e10 * t['A_s']), 'ns': t['n_s']}
    GRID_BOX = {k: (float(x.min()), float(x.max())) for k, x in v.items()}
    print('calibration grid ranges (sims c130-c181):', {k: tuple(round(a, 4) for a in b) for k, b in GRID_BOX.items()})

if 12 in VALIDATION_PLOTS:
    rows = []
    for model in MODELS:
        for rest in all_rests(model):
            c = columns(model, rest)
            out_any_box, out_any_grid = None, None
            row = {'model': model, 'fit': rest}
            for name, a in c.items():
                key = 'b1' if name.startswith('b1') else 'b2' if name.startswith('b2') else name
                if key in NN_BOX:
                    o = (a < NN_BOX[key][0]) | (a > NN_BOX[key][1])
                    row[f'{name} outside NN box'] = float(o.mean())
                    out_any_box = o if out_any_box is None else out_any_box | o
                if GRID_BOX and name in GRID_BOX:
                    o = (a < GRID_BOX[name][0]) | (a > GRID_BOX[name][1])
                    row[f'{name} outside calibration grid'] = float(o.mean())
                    out_any_grid = o if out_any_grid is None else out_any_grid | o
            row['any parameter outside NN box'] = float(out_any_box.mean()) if out_any_box is not None else np.nan
            row['any parameter outside calibration grid'] = float(out_any_grid.mean()) if out_any_grid is not None else np.nan
            rows.append(row)
    if rows:
        COVER = pd.DataFrame(rows)
        COVER.to_csv(PLOT_DIR / 'v3_12_coverage.csv', index=False)
        summ = COVER[COVER['fit'].str.contains('joint')][['model', 'fit', 'any parameter outside NN box', 'any parameter outside calibration grid']]
        print(summ.to_string(float_format=lambda f: f'{f:.3g}'))
        quantities = [('any parameter outside NN box', 'outside the NN box')] + ([('any parameter outside calibration grid', 'outside the calibration grid')] if GRID_BOX else [])
        fig, axes = plt.subplots(len(quantities), len(DATASETS), figsize=(WIDTH, 1.9 * len(quantities) + 0.4), sharex=True, squeeze=False)
        for r, (col, ylabel) in enumerate(quantities):
            for c_i, (tag, title) in enumerate(DATASETS.items()):
                ax = axes[r, c_i]
                for model, m in MODELS.items():
                    g = COVER[(COVER['model'] == model) & COVER['fit'].str.match(rf'^{tag}_joint_smin\d+$')]
                    ax.plot(g['fit'].str.extract(r'smin(\d+)')[0].astype(int) + SMIN_OFFSET[model], g[col], 'o', color=m['color'], mfc=m['color'] if m['filled'] else 'white', ms=4,
                            label=m['label'] if (r, c_i) == (0, 0) else None)
                ax.set_ylabel(f'fraction of samples\n{ylabel}', fontsize=7.5)
                ax.tick_params(labelsize=7)
                if r == 0:
                    ax.set_title(title, fontsize=9)
        for ax in axes[-1]:
            ax.set_xlabel(r'$s_{\rm min}\;[h^{-1}{\rm Mpc}]$', fontsize=8)
            ax.set_xticks(SMINS)
        axes[0, 0].legend(frameon=False, fontsize=6.5)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / 'v3_12_coverage.pdf', bbox_inches='tight')
        fig.savefig(PLOT_DIR / 'v3_12_coverage.png', dpi=DPI, bbox_inches='tight')
        plt.show()
        plt.close(fig)


#%% [markdown]
# # De-biased fits
# Cells 13-14 (appended): the de-biased DR1 fits of fits_apfix_11par_v3.py. Baseline model = slopes zero (v3ns), s_min = 25; the shift is calibrated on that model's mean-of-mocks fit at s_min = 25 and put in the likelihood
# (the model is evaluated at p + shift), so the chains are already corrected: the joint fit of all six tracer-bins (shift25_dr1_joint_smin25), the six tracers alone, and the joint fit without the QSO. The systematic
# error of the shift is added here, to the samples (it is not in the chains).

#%%
DEBIAS_PLOTS = {13, 14}
DEB_MODEL, DEB_SMIN = 'v3ns', 25
SYS_FRACTION = 0.5       # systematic error of the shift = SYS_FRACTION x |shift| (1 sigma)
SYS_MODE = 'independent'  # 'independent': each parameter gets its own N(0, (SYS_FRACTION shift_i)^2); 'correlated': the whole shift vector is scaled by (1 + SYS_FRACTION z), z ~ N(0, 1)
N_MOCKS = 25             # statistical error of a shift computed here from a mean-of-mocks fit = posterior std / sqrt(N_MOCKS)
DEB_COLOR, DEB_DARK = MODELS[DEB_MODEL]['color'], '#00008B'
DEB_JOINT = f'shift{DEB_SMIN}_dr1_joint_smin{DEB_SMIN}'
DEB_NOQSO = f'shift{DEB_SMIN}_dr1_joint_noQSO_smin{DEB_SMIN}'


def deb_indiv(key):
    """`rest` of the de-biased individual-tracer chain whose name contains `_indiv_<key>`, or None."""
    m = MODELS[DEB_MODEL]
    files = glob.glob(str(m['root'] / m['prefix'] / f"{m['prefix']}_shift{DEB_SMIN}_dr1_indiv_*_smin{DEB_SMIN}.status.json"))
    match = [Path(f).name[:-len('.status.json')][len(m['prefix']) + 1:] for f in files if f'_indiv_{key}' in Path(f).name]
    return match[0] if match else None


def shift_block(rest):
    """(shift dict {parameter: value}, status 'shift' block) of a de-biased chain of DEB_MODEL, or (None, None)."""
    if not have(DEB_MODEL, rest):
        return None, None
    blk = json.load(open(f'{fit_path(DEB_MODEL, rest)}.status.json')).get('shift')
    return (dict(zip(blk['names'], blk['values'])), blk) if blk else (None, None)


def with_systematic(c, shift, frac=None, mode=None, seed=0):
    """Copy of the column dict `c` with the systematic error of the shift added to the sampled parameters (Omega_m and H0 re-derived)."""
    frac, mode = SYS_FRACTION if frac is None else frac, SYS_MODE if mode is None else mode
    c = {k: v.copy() for k, v in c.items()}
    names = [p for p in SAMPLED if p in shift and p in c]
    rng = np.random.default_rng(seed)
    if mode == 'independent':
        for p in names:
            c[p] = c[p] + rng.normal(0.0, frac * abs(shift[p]), size=c[p].shape)
    else:
        z = rng.standard_normal(c[names[0]].shape)
        for p in names:
            c[p] = c[p] + frac * shift[p] * z
    c['H0'] = 100 * c['h']
    c['omegam'] = (c['ombh2'] + c['omch2'] + W_NU) / c['h'] ** 2
    return c


def mcs_cols(c, params, label):
    use = [p for p in params if p in c]
    stack = np.stack([c[p] for p in use], axis=-1)
    return MCSamples(samples=[stack[:, w] for w in range(stack.shape[1])], names=use, labels=[LABELS[p] for p in use], label=label, settings=SMOOTH)


#%% [markdown]
# ## 13. De-biased posteriors
# (0) v3_13_debiased_vs_desi: only the de-biased + systematic chain with DESI FS-only (purple) and BAO (orange). (a) joint fit: DESI FS-only (purple), de-biased + systematic (filled), de-biased only (dark outline), uncorrected s_min = 25 fit (dashed). (b) the six individual tracers (de-biased, DESI colours) with the joint de-biased fit (filled black).
# (c) joint without the QSO (orange outline) vs the all-tracer joint fit (filled black). (d) the shift in units of the posterior std with its statistical error (posterior std / sqrt(N_MOCKS)) at s_min = 20 and
# at s_min = 25 for the baseline model, from its mean-of-mocks fits (only the s_min = 25 shift is used by the fits). Tables: v3_13_debiased_joint.csv (joint fit: shift and posterior means / stds before and after),
# v3_13_debiased_tracers.csv (the individual tracers and the no-QSO fit relative to the all-tracer joint fit). Panels b and c have no systematic added (statistical comparison).

#%%
if 13 in DEBIAS_PLOTS:
    if fs is None:
        fs = desi(DESI_FS, 'DESI DR1 FS-only')
    if bao is None:
        bao = desi(DESI_BAO, 'DESI DR1 BAO')
    raw_rest = f'dr1_joint_smin{DEB_SMIN}'
    shift, blk = shift_block(DEB_JOINT)
    if shift is None or not have(DEB_MODEL, raw_rest):
        print(f'cell 13: the de-biased joint fit ({DEB_JOINT}) or the uncorrected {raw_rest} chain of {DEB_MODEL} is not finished; skipped')
    else:
        c_raw, c_deb = columns(DEB_MODEL, raw_rest), columns(DEB_MODEL, DEB_JOINT)
        c_sys = with_systematic(c_deb, shift)
        lab = MODELS[DEB_MODEL]['label']
        figure('v3_13_debiased_vs_desi', [item(bao, ORANGE, True), item(fs, PURPLE, True),
                                          item(mcs_cols(c_sys, PARAMS, rf'{lab}, de-biased $\pm$ {SYS_FRACTION:g} shift ({SYS_MODE})'), DEB_COLOR, True)], PARAMS)
        figure('v3_13_debiased_joint', [item(fs, PURPLE, True), item(mcs_cols(c_sys, PARAMS, rf'{lab}, de-biased $\pm$ {SYS_FRACTION:g} shift ({SYS_MODE})'), DEB_COLOR, True),
                                        item(mcs_cols(c_deb, PARAMS, f'{lab}, de-biased only'), DEB_DARK, False), item(mcs_cols(c_raw, PARAMS, f'{lab}, uncorrected'), DEB_COLOR, False, '--')], PARAMS)
        rows = []
        for p in SAMPLED:
            a, b, sy = c_raw[p].ravel(), c_deb[p].ravel(), c_sys[p].ravel()
            err = blk['std_error'][blk['names'].index(p)]
            rows.append({'parameter': p, 'shift': shift[p], 'stat. error of shift': err, 'shift / stat. error': shift[p] / err if err else np.nan, 'mean uncorrected': a.mean(), 'std uncorrected': a.std(),
                         'mean de-biased': b.mean(), 'std de-biased': b.std(), 'std de-biased + systematic': sy.std(), 'std ratio (with syst / de-biased)': sy.std() / b.std(),
                         '(de-biased - uncorrected)/std uncorrected': (b.mean() - a.mean()) / a.std()})
        JOINTTAB = pd.DataFrame(rows).set_index('parameter')
        JOINTTAB.to_csv(PLOT_DIR / 'v3_13_debiased_joint.csv')
        pd.set_option('display.width', 250)
        print(JOINTTAB.to_string(float_format=lambda v: f'{v:.4g}'))

        indiv = [(short, deb_indiv(key), color) for short, key, color in TRACERS]
        items = [item(mcs_cols(columns(DEB_MODEL, r), PARAMS, short), color, True) if r else None for short, r, color in indiv]
        figure('v3_13_debiased_individual', items + [item(mcs_cols(c_deb, PARAMS, 'Joint'), JOINT_COLOR, True)], PARAMS)

        if have(DEB_MODEL, DEB_NOQSO):
            figure('v3_13_debiased_noQSO', [item(mcs_cols(c_deb, PARAMS, 'all tracers'), JOINT_COLOR, True), item(mcs_cols(columns(DEB_MODEL, DEB_NOQSO), PARAMS, 'without QSO'), 'tab:orange', False)], PARAMS)
        rows = []
        for label, r in [(short, r) for short, r, _c in indiv if r] + [('joint without QSO', DEB_NOQSO if have(DEB_MODEL, DEB_NOQSO) else None)]:
            if r is None:
                continue
            cc = columns(DEB_MODEL, r)
            for p in SAMPLED:
                a = cc[p].ravel()
                rows.append({'fit': label, 'parameter': p, 'mean': a.mean(), 'std': a.std(), '(mean - all-tracer joint mean)/joint std': (a.mean() - c_deb[p].mean()) / c_deb[p].std(), 'std / joint std': a.std() / c_deb[p].std()})
        if rows:
            pd.DataFrame(rows).to_csv(PLOT_DIR / 'v3_13_debiased_tracers.csv', index=False)

        fig, ax = plt.subplots(figsize=(WIDTH * 0.8, 2.8))
        x = np.arange(len(SAMPLED))
        for k, smin in enumerate((20, DEB_SMIN)):
            rest = f'mean_joint_smin{smin}'
            if not have(DEB_MODEL, rest):
                continue
            cm = columns(DEB_MODEL, rest)
            sd = np.array([cm[p].std() for p in SAMPLED])
            d = np.array([cm[p].mean() - TRUTH_S[p] for p in SAMPLED])
            ax.errorbar(x + (k - 0.5) * 0.2, d / sd, yerr=1 / np.sqrt(N_MOCKS) * np.ones(len(SAMPLED)), fmt='o', color=DEB_COLOR, mfc=DEB_COLOR if smin == DEB_SMIN else 'white', capsize=2,
                        label=rf'$s_{{\rm min}}={smin}$' + (' (used)' if smin == DEB_SMIN else ''))
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'${LABELS[p]}$' for p in SAMPLED], fontsize=9)
        ax.set_ylabel(r'shift $\pm$ stat. error [posterior std]', fontsize=8)
        ax.set_title(f"{MODELS[DEB_MODEL]['label']}, mean of mocks", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / 'v3_13_shifts_smin.pdf', bbox_inches='tight')
        fig.savefig(PLOT_DIR / 'v3_13_shifts_smin.png', dpi=DPI, bbox_inches='tight')
        plt.show()
        plt.close(fig)

#%% [markdown]
# ## 14. Constraining power of the de-biased fit, with the systematic error of the shift
# As plot 5 (ratio of sqrt(det C) to DESI FS-only, below 1 = ours tighter) for the baseline model's joint s_min = 25 fit in three versions: uncorrected, de-biased, and de-biased with the systematic error of the shift added to the samples
# (SYS_FRACTION x |shift|, SYS_MODE; the sample covariance is then C_posterior + C_systematic). The systematic widens the de-biased posterior, so its ratio is the one to quote. A fourth panel shows the joint fit without the QSO
# (de-biased, same shift and systematic) against DESI FS-only; the table also gives its volume ratio to the all-tracer fit. Table: v3_14_debiased_constraining_power.csv;
# matrices: v3_14_debiased_constraining_power_matrix.

#%%
if 14 in DEBIAS_PLOTS:
    if fs is None:
        fs = desi(DESI_FS, 'DESI DR1 FS-only')
    raw_rest = f'dr1_joint_smin{DEB_SMIN}'
    shift, _ = shift_block(DEB_JOINT)
    if fs is None or shift is None or not have(DEB_MODEL, raw_rest):
        print('cell 14: the DESI FS chain, the de-biased joint fit or the uncorrected chain is missing; skipped')
    else:
        ALL14 = ['h', 'H0', 'omegam', 'ombh2', 'omch2', 'logA', 'ns']
        HP14 = 'h' if UNITLESS else 'H0'
        SPACES14 = {'sampled (h, wb, wcdm, lnAs, ns)': ['h', 'ombh2', 'omch2', 'logA', 'ns'], 'sampled minus wb': ['h', 'omch2', 'logA', 'ns'], f'physical (Om, {HP14}, wb, lnAs, ns)': ['omegam', HP14, 'ombh2', 'logA', 'ns'],
                    f'(Om, {HP14})': ['omegam', HP14], '(h, wcdm)': ['h', 'omch2'], '(lnAs, ns)': ['logA', 'ns']}
        MATRIX14 = ['h', 'ombh2', 'omch2', 'logA', 'ns']
        c_deb = columns(DEB_MODEL, DEB_JOINT)
        versions = {'uncorrected': columns(DEB_MODEL, raw_rest), 'de-biased': c_deb, 'de-biased + systematic': with_systematic(c_deb, shift)}
        NOQSO_KEY = 'de-biased + systematic, no QSO'
        shift_q, _ = shift_block(DEB_NOQSO)
        if shift_q is not None:                                  # the joint fit without the QSO, with the same shift and the same systematic
            versions[NOQSO_KEY] = with_systematic(columns(DEB_MODEL, DEB_NOQSO), shift_q)
        chains = {k: mcs_cols(c, ALL14, k) for k, c in versions.items()}
        rows = []
        for label, names in SPACES14.items():
            d, ref = len(names), sqdet(fs, names)
            row = {'space': label, 'dim': d, 'sqrt det DESI FS': ref}
            for k, smp in chains.items():
                v = sqdet(smp, names)
                row.update({f'{k} / DESI': v / ref, f'{k} / DESI per dim': (v / ref) ** (1 / d)})
            if NOQSO_KEY in chains:
                row['no QSO / all tracers (both de-biased + systematic)'] = sqdet(chains[NOQSO_KEY], names) / sqdet(chains['de-biased + systematic'], names)
            rows.append(row)
        CPTAB = pd.DataFrame(rows).set_index('space')
        CPTAB.to_csv(PLOT_DIR / 'v3_14_debiased_constraining_power.csv')
        print(CPTAB.to_string(float_format=lambda v: f'{v:.4g}'))
        n = len(MATRIX14)
        fig, axes = plt.subplots(1, len(chains), figsize=(WIDTH * len(chains) / 2.0, 0.55 * WIDTH), squeeze=False)
        for ax, (k, smp) in zip(axes[0], chains.items()):
            R = np.full((n, n), np.nan)
            for i, a in enumerate(MATRIX14):
                for j, b in enumerate(MATRIX14):
                    R[i, j] = (sqdet(smp, [a]) / sqdet(fs, [a])) if i == j else sqdet(smp, [a, b]) / sqdet(fs, [a, b])
            R = np.where(np.tri(n, dtype=bool), R, np.nan) if np.allclose(R, R.T, rtol=1e-3, equal_nan=True) else R
            cmap = plt.get_cmap('RdBu_r').copy()
            cmap.set_bad('white')
            im = ax.imshow(np.ma.masked_invalid(np.log10(R)), cmap=cmap, vmin=-0.7, vmax=0.7)
            for i in range(n):
                for j in range(n):
                    if np.isfinite(R[i, j]):
                        ax.text(j, i, f'{R[i, j]:.2f}', ha='center', va='center', fontsize=8.5)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels([f'${LABELS[p]}$' for p in MATRIX14], fontsize=9, rotation=45)
            ax.set_yticklabels([f'${LABELS[p]}$' for p in MATRIX14] if ax is axes[0][0] else [], fontsize=9)
            ax.set_xlabel(textwrap.fill(f'{k} / DESI FS-only', 22), fontsize=9)
        cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, pad=0.02)
        cb.set_ticks([-0.7, -0.35, 0, 0.35, 0.7])
        cb.set_ticklabels(['0.2', '0.45', '1', '2.2', '5'])
        cb.set_label('ratio (blue: ours tighter)', fontsize=9)
        cb.ax.tick_params(labelsize=8)
        fig.savefig(PLOT_DIR / 'v3_14_debiased_constraining_power_matrix.pdf', bbox_inches='tight')
        fig.savefig(PLOT_DIR / 'v3_14_debiased_constraining_power_matrix.png', dpi=DPI_MATRIX, bbox_inches='tight')
        plt.show()
        plt.close(fig)