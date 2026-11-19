"""
getdist version of plot_v2_chains.py: the same ten plot sets for the AP-fixed refits in fits_11par_apfix_v2, with paper-quality output (vector PDF + PNG at DPI = 600). Run cell by
cell (#%% markers) or as a script, compact environment (needs getdist, h5py). Ours are red (filled where it is the reference chain, drawn last so it is on top); DESI chains are
filled in the DESI-paper colours; variant fits are outlines. Our chains are given to getdist as one chain per walker, so its bandwidth uses the correct correlation information,
and all chains use a KDE bandwidth of SMOOTH x the automatic one (getdist's default plus a little: slightly smoother contours for a few percent of broadening).
Figures go to CHAIN_ROOT/plots_gd as PDF + PNG; chains that have not finished (no .status.json) are skipped with a message. Parameter columns are identified by the `param_names` stored
in each chain's status.json.

  1. baseline vs DESI FS and BAO        6. single-mock fits per parameter + mean offset     (DR1 and mock-mean baselines, separately)
  2. baseline vs s_min                  7. DR1 baseline shifted by minus the offset in 6, vs the uncorrected one and DESI
  3. baseline vs Gaussian model         8. bias parameters of the DR1 and mock-mean baselines
  4. baseline vs mass schemes           9. w0wa baseline vs LCDM baseline
  5. baseline vs individual tracers    10. w0wa fits at s_min = 20 and 30
Plots 1-5 and 9-10 are made for DR1 and for the mean-of-mocks fits (the latter with the c000 truth marked). Joint-vs-individual-tracer plots (5, 11, 13, 14) draw the joint fit filled black.
 11. mean (+- std) vs joint MAP and 1-D marginal mode: ours and DESI's joint   12. MAP vs s_min (DR1 and mocks, five sampled parameters)
 13. each tracer vs all the others (omega_cdm, h, ln A_s) + the observed N_sigma of each tracer   14. joint vs individual tracers in the Omega_m-sigma8 and Omega_m-H0 planes (DESI V Fig. 14 analogue)
 15. baseline with flat omega_b, n_s priors (compact_flatprior_*) vs the Gaussian-prior baseline and DESI FS-only
 16. 1-D marginalized n_s posterior of the DR1 joint fit with its mean and MAP marked (diagnostic for the MAP-vs-mean offset in plot 11)
"""
#%%
import glob
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import chi2 as chi2_dist, gaussian_kde, norm
from matplotlib.patches import Patch
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

CHAIN_ROOT = Path('/spiffball/cosweeney/data/chains/fits_11par_apfix_v2')
PLOT_DIR = CHAIN_ROOT / 'plots_gd'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
DESI_FS = '/spiffball/cosweeney/DESI/DR1/full-shape-cosmo-params/v1.0/cobaya/base/desi-reptvelocileptors-fs-all_schoneberg2024-bbn_planck2018-ns10/'
DESI_BAO = '/spiffball/cosweeney/DESI/DR1/bao-cosmo-params/v1.0/cobaya/base/desi-bao-all_schoneberg2024-bbn/'      # adjust to your local directory name
DESI_BURN = 0.3                  # plot 11: fraction of each DESI cobaya chain file dropped as burn-in (as in desi())
DESI_POINT = '#9A4FA5'           # plot 11: colour of the DESI joint-FS points (ours take the tracer colours)

PLOTS_TO_RUN = set(range(1, 17))
DPI = 600               # PNG resolution (the PDFs are vector)
DPI_MATRIX = 1200       # the constraining-power matrix has small text, so it gets a higher resolution
JOINT_COLOR = 'k'       # joint (all-tracer) fits are drawn filled black in the joint-vs-tracers plots (5, 11, 13, 14)
TENSION_PARAMS = ['omch2', 'h', 'logA']          # parameters of the tracer-vs-rest tension test (plot 13)
N_PAIRS = 20000         # sample pairs of the parameter-difference (Monte Carlo) tension test
S8_TRAIN, S8_VALID, S8_JOBS = 160, 40, 8         # plot 14: CAMB runs to build / validate the sigma8 emulator, and parallel jobs
WIDTH = 7.2             # figure width in inches for the triangle plots (two-column page width); the 12-parameter bias triangle uses WIDTH_BIAS
WIDTH_BIAS = 10.0
SMOOTH = {'smooth_scale_1D': -1.25, 'smooth_scale_2D': -1.25}      # getdist KDE bandwidth = automatic x 1.25 (-1 = default); larger is smoother but broadens the contours
FILL_VARIANTS = False   # True: fill the variant chains in plots 2-4 and 9-10 too (the baseline is always filled)
UNITLESS = True         # True (default): plot only the 5 sampled dimensionless parameters (h, omega_b, omega_cdm, ln A_s, n_s; plus w0, wa in plots 9-10); False: Omega_m, H0, ... instead
SMINS = (20, 25, 30, 35, 40)
W_NU = 0.0006442        # omega_nu of one massive 0.06 eV species, so Omega_m includes neutrinos like CAMB's
DATASETS = {'dr1': 'DR1', 'mean': 'mean of mocks'}

SAMPLED = ['h', 'ombh2', 'omch2', 'logA', 'ns']
PARAMS = ['omegam', 'H0', 'ombh2', 'omch2', 'logA', 'ns']
COMPARE_PARAMS = ['omegam', 'H0', 'omch2', 'logA', 'ns']            # omega_b is prior-dominated: dropped from the multi-chain comparisons
W0WA_PARAMS = ['omegam', 'H0', 'omch2', 'logA', 'ns', 'w0', 'wa']
if UNITLESS:
    PARAMS = ['h', 'ombh2', 'omch2', 'logA', 'ns']
    COMPARE_PARAMS = PARAMS
    W0WA_PARAMS = PARAMS + ['w0', 'wa']
DE_BOX = {'w0': (-1.271, -0.726), 'wa': (-0.628, 0.621)}      # w0wa NN training box = the prior limits of the w0wa fits; plots 9-10 show w0, wa only inside it
FLAT_BOX = {'ombh2': (0.0207, 0.0243), 'ns': (0.901, 1.025)}      # the NN box = the flat prior ranges of the compact_flatprior_* fits (plot 15)
LABELS = {'sigma8': r'\sigma_8', 'h': r'h', 'omegam': r'\Omega_m', 'H0': r'H_0', 'ombh2': r'\omega_b', 'omch2': r'\omega_{\rm cdm}', 'logA': r'\ln(10^{10}A_s)', 'ns': r'n_s', 'w0': r'w_0', 'wa': r'w_a'}
TRUTH_S = {'h': 0.6736, 'ombh2': 0.02237, 'omch2': 0.12, 'logA': float(np.log(1e10 * 2.0830e-9)), 'ns': 0.9649}              # AbacusSummit c000, sampled parameters
TRUTH = {**TRUTH_S, 'omegam': (0.02237 + 0.12 + W_NU) / 0.6736 ** 2, 'H0': 67.36, 'w0': -1.0, 'wa': 0.0, 'sigma8': 0.8077}      # sigma8 as computed by CAMB (m_nu = 0.06 eV) for c000
TRACERS = [('BGS', 'BGS', 'yellowgreen'), ('LRG1', 'LRG_0.4', 'orange'), ('LRG2', 'LRG_0.6', 'orangered'), ('LRG3', 'LRG_0.8', 'firebrick'),
           ('ELG2', 'ELG', 'steelblue'), ('QSO', 'QSO', 'seagreen')]      # (short name, substring of the chain name, DESI colour)
RED, ORANGE, PURPLE = 'red', '#FF8C1A', '#D58BCB'
for s, key, _c in TRACERS:
    LABELS.update({f'b1_{s}': rf'b_1^{{\rm {s}}}', f'b2_{s}': rf'b_2^{{\rm {s}}}'})

#%% [markdown]
# ## Loading helpers

#%%
def model_dir(name):
    for prefix in ('compact_lowM', 'compact_highM', 'compact_w0wa', 'gauss'):
        if name.startswith(prefix):
            return prefix
    return 'compact'


def have(name):
    return (CHAIN_ROOT / model_dir(name) / f'{name}.status.json').exists()


_cache = {}
S8_EMU = None


def full3(name):
    """(array (steps, walkers, n_params), parameter names, status dict): post-burn-in and thinned as saved in status.json."""
    if name not in _cache:
        root = CHAIN_ROOT / model_dir(name) / name
        st = json.load(open(f'{root}.status.json'))
        with h5py.File(f'{root}.h5', 'r') as f:
            it = int(f['mcmc'].attrs['iteration'])
            arr = f['mcmc/chain'][st['discard']:it:st['thin'], :, :]
        _cache[name] = (arr, st['param_names'], st)
    return _cache[name]


def columns(name, shift=None, systematic=None, seed=0):
    """dict column -> array (steps, walkers), with derived Omega_m and H0; `shift` (dict) is subtracted from the sampled parameters first, then `systematic` = (parameter names, covariance) adds
    to every sample an independent draw of that Gaussian (the uncertainty of the shift itself)."""
    arr, names, _ = full3(name)
    c = {n: arr[..., i].copy() for i, n in enumerate(names)}
    for p, v in (shift or {}).items():
        c[p] = c[p] - v
    if systematic is not None:
        names_s, cov_s = systematic
        eps = np.random.default_rng(seed).multivariate_normal(np.zeros(len(names_s)), cov_s, size=c[names_s[0]].shape)
        for i, p in enumerate(names_s):
            c[p] = c[p] + eps[..., i]
    c['H0'] = 100 * c['h']
    c['omegam'] = (c['ombh2'] + c['omch2'] + W_NU) / c['h'] ** 2
    if S8_EMU is not None and all(k in c for k in ('ombh2', 'omch2', 'h', 'ns', 'logA')):
        c['sigma8'] = S8_EMU(c)                                                       # only once plot 14 has built the emulator
    return c


def mcs(name, params, label=None, shift=None, ranges=None, systematic=None):
    """getdist MCSamples with one chain per walker for the parameters of `params` that the chain has."""
    c = columns(name, shift, systematic)
    use = [p for p in params if p in c]
    stack = np.stack([c[p] for p in use], axis=-1)
    return MCSamples(samples=[stack[:, w] for w in range(stack.shape[1])], names=use, labels=[LABELS[p] for p in use], label=label, settings=SMOOTH,
                     ranges={p: v for p, v in {**DE_BOX, **(ranges or {})}.items() if p in use})          # hard prior limits (w0, wa, or `ranges`): getdist applies its boundary correction there


def flat(name, params=PARAMS + ['h']):
    c = columns(name)
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
    names = [p for p in ['h', 'H0', 'omegam', 'ombh2', 'omch2', 'logA', 'ns'] if p in c]          # every available column; figure() picks the ones plotted
    return MCSamples(samples=np.column_stack([c[p] for p in names]), weights=s.weights, names=names, labels=[LABELS[p] for p in names], label=label, settings=SMOOTH)


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


def figure(fname, items, params, truth=None, width=None, limits=None, edges=None):
    """Triangle plot of the (non-None) items = (MCSamples, colour, filled, linestyle), drawn in order (the last is on top); saves PDF + PNG (DPI) to PLOT_DIR.
    `limits` = {parameter: (lo, hi)} sets the plotted range of those parameters; `edges` = {parameter: (lo, hi)} draws dotted lines at those values (hard prior edges) in every panel of that parameter."""
    items = [i for i in items if i is not None]
    if len(items) < 2 and not fname.startswith('v2_8'):
        print(f'{fname}: nothing to compare, skipped')
        return
    roots = [i[0] for i in items]
    present = {p for r in roots for p in r.getParamNames().list()}
    params = [p for p in params if p in present]
    g = plotter(width or WIDTH)
    g.triangle_plot(roots, params, filled=[i[2] for i in items], contour_colors=[i[1] for i in items], contour_ls=[i[3] for i in items],
                    contour_lws=[1.2] * len(items), legend_labels=[r.getLabel() for r in roots], markers={k: v for k, v in (truth or {}).items() if k in params},
                    param_limits={k: v for k, v in (limits or {}).items() if k in params})
    for p, (lo, hi) in (edges or {}).items():
        if p not in params:
            continue
        k = params.index(p)
        for i in range(len(params)):
            for j in range(len(params)):
                ax = g.subplots[i, j]
                if ax is None:
                    continue
                if j == k and i >= k:                                   # x axis is p: its 1D panel and the 2D panels below it
                    for v in (lo, hi):
                        ax.axvline(v, color='0.35', ls=':', lw=0.9)
                elif i == k and j < k:                                  # y axis is p
                    for v in (lo, hi):
                        ax.axhline(v, color='0.35', ls=':', lw=0.9)
    g.export(str(PLOT_DIR / f'{fname}.pdf'))
    g.fig.savefig(PLOT_DIR / f'{fname}.png', dpi=DPI, bbox_inches='tight')
    plt.show()
    plt.close(g.fig)
    print('saved', fname, flush=True)


def variant(name, params, label, color, ls='-'):
    return item(mcs(name, params, label), color, FILL_VARIANTS, ls) if have(name) else None


def baseline(tag, params, label=None, color=RED):
    name = f'compact_{tag}_joint_smin20'
    return item(mcs(name, params, label or f'Baseline, {DATASETS[tag]}'), color, True) if have(name) else None


TRUTH_FOR = {'dr1': None, 'mean': TRUTH}

#%% [markdown]
# ## 1. Baseline vs DESI FS-only and BAO

#%%
if 1 in PLOTS_TO_RUN:
    fs, bao = desi(DESI_FS, 'DESI DR1 FS-only'), desi(DESI_BAO, 'DESI DR1 BAO')
    for tag in DATASETS:
        figure(f'v2_1_baseline_vs_desi_{tag}', [item(bao, ORANGE, True), item(fs, PURPLE, True), baseline(tag, PARAMS)], PARAMS, TRUTH_FOR[tag])

#%% [markdown]
# ## 1b. Constraining power: baseline vs DESI DR1 FS-only
# sqrt(det C) of the posterior covariance C is proportional to the volume of the posterior's 1-sigma ellipsoid in a parameter space, so the ratio ours / DESI FS-only is the
# relative volume: below 1 means our constraints are tighter there. The ratio is unchanged by linear reparametrizations (so H0 vs h makes no difference); `per dim` is its d-th
# root, the geometric-mean sigma ratio, which makes spaces of different dimension comparable. The DR1 baseline and DESI FS-only are fits to the same data; the mean-of-mocks
# baseline is shown for reference (its noise is that of one mock, comparable to DR1). omega_b and n_s are prior-dominated and have the same priors as DESI, so they pull the ratios
# towards 1. Covariances are weighted (DESI) or flat (ours), so they ignore non-Gaussianity such as truncation by the priors. Edit SPACES to change the spaces compared.

#%%
if 1 in PLOTS_TO_RUN:
    if globals().get('fs') is None:
        fs = desi(DESI_FS, 'DESI DR1 FS-only')
    ALLP = ['h', 'H0', 'omegam', 'ombh2', 'omch2', 'logA', 'ns']
    HP = 'h' if UNITLESS else 'H0'                                   # h and H0 give identical ratios (linear rescaling); h is dimensionless
    SPACES = {'sampled (h, wb, wcdm, lnAs, ns)': ['h', 'ombh2', 'omch2', 'logA', 'ns'], 'sampled minus wb': ['h', 'omch2', 'logA', 'ns'],
              f'physical (Om, {HP}, wb, lnAs, ns)': ['omegam', HP, 'ombh2', 'logA', 'ns'], f'physical minus wb (Om, {HP}, lnAs, ns)': ['omegam', HP, 'logA', 'ns'],
              f'(Om, {HP})': ['omegam', HP], '(h, wcdm)': ['h', 'omch2'], '(lnAs, ns)': ['logA', 'ns']}
    base_chains = {tag: mcs(f'compact_{tag}_joint_smin20', ALLP, tag) for tag in DATASETS if have(f'compact_{tag}_joint_smin20')}

    def sqdet(smp, names):
        """sqrt(det of the covariance) of the named parameters; NaN if the chain lacks one of them."""
        if smp is None or any(n not in smp.getParamNames().list() for n in names):
            return np.nan
        return float(np.exp(0.5 * np.linalg.slogdet(np.atleast_2d(smp.cov(pars=names)))[1]))

    if fs is not None and base_chains:
        rows = []
        for label, names in SPACES.items():
            d, ref = len(names), sqdet(fs, names)
            row = {'space': label, 'dim': d, 'sqrt det DESI FS': ref}
            for tag, smp in base_chains.items():
                v = sqdet(smp, names)
                row.update({f'sqrt det {DATASETS[tag]}': v, f'{DATASETS[tag]} / DESI': v / ref, f'{DATASETS[tag]} / DESI per dim': (v / ref) ** (1 / d)})
            rows.append(row)
        SPACE_TABLE = pd.DataFrame(rows).set_index('space')
        SPACE_TABLE.to_csv(PLOT_DIR / 'v2_1b_constraining_power_spaces.csv')
        pd.set_option('display.width', 220)
        print(SPACE_TABLE.to_string(float_format=lambda x: f'{x:.4g}'))
        for tag in base_chains:
            r = SPACE_TABLE[f'{DATASETS[tag]} / DESI']
            print(f'{DATASETS[tag]} baseline is tighter than DESI FS-only in {(r < 1).sum()} of {len(r)} spaces (volume ratio {r.min():.2f} to {r.max():.2f})')

        # one-dimensional sigma ratios (diagonal) and the pairwise 2D area ratios (sqrt det of each 2x2 covariance; symmetric, so only the lower triangle is drawn)
        MATRIX = ['h', 'ombh2', 'omch2', 'logA', 'ns']                                   # the five sampled parameters (no derived Omega_m)
        sig = pd.DataFrame({'DESI FS': {p: sqdet(fs, [p]) for p in MATRIX}, **{DATASETS[t]: {p: sqdet(smp, [p]) for p in MATRIX} for t, smp in base_chains.items()}})
        for t in base_chains:
            sig[f'{DATASETS[t]} / DESI'] = sig[DATASETS[t]] / sig['DESI FS']
        sig.to_csv(PLOT_DIR / 'v2_1b_constraining_power_1d.csv')
        print(sig.to_string(float_format=lambda x: f'{x:.4g}'))

        n = len(MATRIX)
        fig, axes = plt.subplots(1, len(base_chains), figsize=(WIDTH, 0.55 * WIDTH), squeeze=False)
        for ax, (tag, smp) in zip(axes[0], base_chains.items()):
            R = np.full((n, n), np.nan)
            for i, a in enumerate(MATRIX):
                for j, b in enumerate(MATRIX):
                    R[i, j] = (sqdet(smp, [a]) / sqdet(fs, [a])) if i == j else sqdet(smp, [a, b]) / sqdet(fs, [a, b])
            if np.allclose(R, R.T, rtol=1e-6, equal_nan=True):                    # the 2D ratio of (a, b) equals that of (b, a): show the lower triangle (with the diagonal) only
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
            ax.set_xlabel(f'{DATASETS[tag]} baseline / DESI FS-only', fontsize=10)
        cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, pad=0.02)
        cb.set_ticks([-0.7, -0.35, 0, 0.35, 0.7])
        cb.set_ticklabels(['0.2', '0.45', '1', '2.2', '5'])
        cb.set_label('ratio (blue: ours tighter)', fontsize=9)
        cb.ax.tick_params(labelsize=8)
        fig.savefig(PLOT_DIR / 'v2_1b_constraining_power_matrix.pdf', bbox_inches='tight')
        fig.savefig(PLOT_DIR / 'v2_1b_constraining_power_matrix.png', dpi=DPI_MATRIX, bbox_inches='tight')
        plt.show()
        plt.close(fig)

#%% [markdown]
# ## 2. Baseline vs different s_min

#%%
if 2 in PLOTS_TO_RUN:
    colors = {25: 'tab:orange', 30: 'tab:green', 35: 'tab:blue', 40: 'tab:purple'}
    for tag in DATASETS:
        items = [variant(f'compact_{tag}_joint_smin{s}', COMPARE_PARAMS, rf'$s_{{\rm min}}={s}$', colors[s]) for s in SMINS[1:]] + [baseline(tag, COMPARE_PARAMS, r'$s_{\rm min}=20$ (baseline)')]
        figure(f'v2_2_smin_{tag}', items, COMPARE_PARAMS, TRUTH_FOR[tag])

#%% [markdown]
# ## 3. Baseline vs Gaussian model

#%%
if 3 in PLOTS_TO_RUN:
    for tag in DATASETS:
        figure(f'v2_3_gaussian_{tag}', [variant(f'gauss_{tag}_joint_smin20', COMPARE_PARAMS, 'Gaussian model', 'tab:blue'), baseline(tag, COMPARE_PARAMS, 'Compact (11-par), baseline')],
               COMPARE_PARAMS, TRUTH_FOR[tag])

#%% [markdown]
# ## 4. Baseline vs mass schemes

#%%
if 4 in PLOTS_TO_RUN:
    for tag in DATASETS:
        figure(f'v2_4_mass_{tag}', [variant(f'compact_lowM_{tag}_joint_smin20', COMPARE_PARAMS, r'low $M$', 'tab:blue'),
                                    variant(f'compact_highM_{tag}_joint_smin20', COMPARE_PARAMS, r'high $M$', 'tab:green'),
                                    baseline(tag, COMPARE_PARAMS, r'avg $M$ (baseline)')], COMPARE_PARAMS, TRUTH_FOR[tag])

#%% [markdown]
# ## 5. Baseline vs individual-tracer fits
# Individual fits are filled in the DESI tracer colours (as in the DESI paper); the joint baseline is filled black (JOINT_COLOR) and on top.

#%%
if 5 in PLOTS_TO_RUN:
    for tag in DATASETS:
        files = sorted(Path(p).name[:-len('.status.json')] for p in glob.glob(str(CHAIN_ROOT / 'compact' / f'compact_{tag}_indiv_*_smin20.status.json')))
        items = []
        for short, key, color in TRACERS:
            match = [f for f in files if f'_indiv_{key}' in f]
            items.append(item(mcs(match[0], COMPARE_PARAMS, short), color, True) if match else None)
        figure(f'v2_5_individual_{tag}', items + [baseline(tag, COMPARE_PARAMS, 'Joint baseline', color=JOINT_COLOR)], COMPARE_PARAMS, TRUTH_FOR[tag])

#%% [markdown]
# ## 6. Single-mock fits: per-parameter scatter and the mean offset
# Posterior mean +- std of each single-mock baseline fit (s_min = 20); the red line and band are the mean of the means +- its standard error, the dashed line is the c000
# truth. `shift` is mean(means) - truth for the sampled parameters (h, omega_b, omega_cdm, ln A_s, n_s); Omega_m and H0 are shown for reference.

#%%
SHIFT, SHIFT_COV, MOCK_TABLE = None, None, None
if 6 in PLOTS_TO_RUN or 7 in PLOTS_TO_RUN:
    mock_names = sorted(Path(p).name[:-len('.status.json')] for p in glob.glob(str(CHAIN_ROOT / 'compact' / 'compact_mock*_joint_smin20.status.json')))
    means, stds = [], []
    for n in mock_names:
        d = flat(n)
        means.append(d.mean())
        stds.append(d.std())
    M, S = pd.DataFrame(means), pd.DataFrame(stds)
    N = len(M)
    print(f'{N} single-mock fits found')
    if N:
        rows = []
        for p in PARAMS:
            off, se = M[p].mean() - TRUTH[p], M[p].std(ddof=1) / np.sqrt(N)
            rows.append({'parameter': p, 'truth': TRUTH[p], 'mean of posterior means': M[p].mean(), 'offset': off, 'offset s.e.': se,
                         'offset [mean posterior std]': off / S[p].mean(), 'mean pull': np.mean((M[p] - TRUTH[p]) / S[p]),
                         'scatter / mean std': M[p].std(ddof=1) / S[p].mean()})
        MOCK_TABLE = pd.DataFrame(rows).set_index('parameter')
        MOCK_TABLE.to_csv(PLOT_DIR / 'v2_6_mock_offsets.csv')
        print(MOCK_TABLE.round(4).to_string())
        SHIFT = {p: float(M[p].mean() - TRUTH_S[p]) for p in SAMPLED}                 # subtracted from the baseline samples in plot 7
        print('shift (mean - truth) of the sampled parameters:', {k: round(v, 5) for k, v in SHIFT.items()})
        SHIFT_COV = M[SAMPLED].cov().to_numpy() / N                                    # covariance of the estimated mean offset = the systematic error of the shift (plot 7)
        SHIFT_TABLE = pd.DataFrame({'shift': pd.Series(SHIFT), 'std. error': np.sqrt(np.diag(SHIFT_COV)), 'shift / s.e.': pd.Series(SHIFT) / np.sqrt(np.diag(SHIFT_COV))})
        SHIFT_TABLE.to_csv(PLOT_DIR / 'v2_6_shift_with_error.csv')
        print(SHIFT_TABLE.to_string(float_format=lambda x: f'{x:.4g}'))

#%%
if 6 in PLOTS_TO_RUN and MOCK_TABLE is not None:
    mean_fit = 'compact_mean_joint_smin20'
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 0.5 * WIDTH), sharex=True)
    for ax in axes.ravel()[len(PARAMS):]:
        ax.axis('off')
    for ax, p in zip(axes.ravel(), PARAMS):
        ax.errorbar(np.arange(N), M[p], yerr=S[p], fmt='o', ms=3, color='k', elinewidth=0.8, capsize=0)
        ax.axhline(TRUTH[p], ls='--', color='0.4', lw=0.9)
        r = MOCK_TABLE.loc[p]
        ax.axhline(r['mean of posterior means'], color=RED, lw=1.0)
        ax.axhspan(r['mean of posterior means'] - r['offset s.e.'], r['mean of posterior means'] + r['offset s.e.'], color=RED, alpha=0.25, lw=0)
        if have(mean_fit):
            d = flat(mean_fit)
            ax.errorbar(N + 0.5, d[p].mean(), yerr=d[p].std(), fmt='D', ms=4, color='tab:blue', elinewidth=0.8, capsize=0)
        ax.set_ylabel(f'${LABELS[p]}$', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.text(0.03, 0.95, rf"offset = ${r['offset']:+.3g}\pm{r['offset s.e.']:.2g}$ ({r['offset [mean posterior std]']:+.2f}$\sigma$)", transform=ax.transAxes, va='top', fontsize=6.5,
                bbox=dict(facecolor='w', alpha=0.8, lw=0))
    for ax in axes[1]:
        ax.set_xlabel('mock index', fontsize=9)
    fig.legend(handles=[plt.Line2D([], [], marker='o', color='k', ls='none', label='single-mock fit'), plt.Line2D([], [], color=RED, label=r'mean offset $\pm$ s.e.'),
                        plt.Line2D([], [], color='0.4', ls='--', label='c000 truth'), plt.Line2D([], [], marker='D', color='tab:blue', ls='none', label='mean-of-mocks fit')],
               frameon=False, fontsize=7, loc='lower center', ncol=4)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(PLOT_DIR / 'v2_6_mock_scatter.pdf', bbox_inches='tight')
    fig.savefig(PLOT_DIR / 'v2_6_mock_scatter.png', dpi=DPI, bbox_inches='tight')
    plt.show()
    plt.close(fig)

#%% [markdown]
# ## 7. Shift-corrected DR1 baseline
# The DR1 baseline samples with the offset from plot 6 subtracted from each sampled parameter (Omega_m and H0 re-derived). Three versions: red dashed outline = uncorrected, dark-red outline = shifted,
# red fill = shifted AND with a systematic error on the shift: every sample gets an independent draw eps with standard deviation SYS_FRACTION x |shift| (default 0.5, i.e. the shift +- shift/2), either
# independently for each parameter (SYS_MODE = 'independent') or, 'correlated', as one common scaling of the whole shift vector (shift x (1 - 0.5 z), z ~ N(0, 1)). This widens the posterior by the assumed
# uncertainty of the correction. (The purely statistical standard error of the mean offset, from the scatter of the mocks, is in the table printed by plot 6.) `SHIFT_APPLY` selects which parameters are shifted.

#%%
SHIFT_APPLY = SAMPLED
SYS_FRACTION = 0.5       # systematic error of the shift = SYS_FRACTION x |shift| (1 sigma)
SYS_MODE = 'independent'  # 'independent': each parameter gets its own N(0, (SYS_FRACTION shift_i)^2); 'correlated': the whole shift vector is scaled by (1 - SYS_FRACTION z), z ~ N(0, 1)


def systematic_cov(apply):
    """Covariance of the systematic error on the shift, for the parameters in `apply`."""
    d = SYS_FRACTION * np.array([SHIFT[p] for p in apply])
    return np.diag(d ** 2) if SYS_MODE == 'independent' else np.outer(d, d)


if 7 in PLOTS_TO_RUN and SHIFT is not None and have('compact_dr1_joint_smin20'):
    fs, bao = desi(DESI_FS, 'DESI DR1 FS-only'), desi(DESI_BAO, 'DESI DR1 BAO')
    name = 'compact_dr1_joint_smin20'
    shift_d = {p: SHIFT[p] for p in SHIFT_APPLY}
    sys_d = (SHIFT_APPLY, systematic_cov(SHIFT_APPLY))
    figure('v2_7_shift_corrected', [item(bao, ORANGE, True), item(fs, PURPLE, True), item(mcs(name, PARAMS, rf'Baseline, DR1 (shift $\pm$ {SYS_FRACTION:g} shift)', shift=shift_d, systematic=sys_d), RED, True),
                                    item(mcs(name, PARAMS, 'Baseline, DR1 (shift only)', shift=shift_d), '#8B0000', False), item(mcs(name, PARAMS, 'Baseline, DR1 (uncorrected)'), RED, False, '--')], PARAMS)

# Same correction applied to the mean-of-mocks baseline, with the c000 truth marked (same systematic error as above). The offset comes from the single-mock fits of the same mocks, so this is a closure test of the shift (does subtracting
# the mean offset of the single-mock fits move the fit to the mean data to the truth, i.e. is the offset additive?), not an independent validation.
if 7 in PLOTS_TO_RUN and SHIFT is not None and have('compact_mean_joint_smin20'):
    name_m = 'compact_mean_joint_smin20'
    shift_m = {p: SHIFT[p] for p in SHIFT_APPLY}
    sys_m = (SHIFT_APPLY, systematic_cov(SHIFT_APPLY))
    figure('v2_7_shift_corrected_mock', [item(mcs(name_m, PARAMS, rf'Baseline, mean of mocks (shift $\pm$ {SYS_FRACTION:g} shift)', shift=shift_m, systematic=sys_m), RED, True),
                                         item(mcs(name_m, PARAMS, 'Baseline, mean of mocks (shift only)', shift=shift_m), '#8B0000', False),
                                         item(mcs(name_m, PARAMS, 'Baseline, mean of mocks (uncorrected)'), RED, False, '--')], PARAMS, TRUTH)
    d_raw, cc, cs = flat(name_m), columns(name_m, shift=shift_m), columns(name_m, shift=shift_m, systematic=sys_m)
    d_cor, d_sys = pd.DataFrame({q: cc[q].ravel() for q in d_raw.columns}), pd.DataFrame({q: cs[q].ravel() for q in d_raw.columns})
    tr = pd.Series(TRUTH)[d_raw.columns]
    CLOSURE = pd.DataFrame({'truth': tr, 'uncorrected offset [std]': (d_raw.mean() - tr) / d_raw.std(), 'corrected offset [std]': (d_cor.mean() - tr) / d_cor.std(),
                            'corrected + systematic offset [std]': (d_sys.mean() - tr) / d_sys.std(), 'std ratio (with systematic / without)': d_sys.std() / d_cor.std()}).loc[[q for q in PARAMS if q in d_raw.columns]]
    CLOSURE.to_csv(PLOT_DIR / 'v2_7_shift_closure_mock.csv')
    print(CLOSURE.to_string(float_format=lambda x: f'{x:.3g}'))

#%% [markdown]
# ## 8. Bias parameters of the baseline fits
# b1, b2 for the six tracer bins: DR1 baseline (red, filled, on top) and mean-of-mocks baseline (blue outline).

#%%
def bias_mcs(name, label):
    arr, names, _ = full3(name)
    b1, b2 = [i for i, n in enumerate(names) if n.startswith('b1_')], [i for i, n in enumerate(names) if n.startswith('b2_')]
    idx, cols = [], []
    for (short, _k, _c), i1, i2 in zip(TRACERS, b1, b2):
        idx += [i1, i2]
        cols += [f'b1_{short}', f'b2_{short}']
    stack = arr[..., idx]
    return MCSamples(samples=[stack[:, w] for w in range(stack.shape[1])], names=cols, labels=[LABELS[c] for c in cols], label=label, settings=SMOOTH)


if 8 in PLOTS_TO_RUN:
    cols = [f'{b}_{t[0]}' for t in TRACERS for b in ('b1', 'b2')]
    items = [item(bias_mcs('compact_mean_joint_smin20', 'Baseline, mean of mocks'), 'tab:blue', FILL_VARIANTS) if have('compact_mean_joint_smin20') else None,
             item(bias_mcs('compact_dr1_joint_smin20', 'Baseline, DR1'), RED, True) if have('compact_dr1_joint_smin20') else None]
    figure('v2_8_bias', items, cols, width=WIDTH_BIAS)

#%% [markdown]
# ## 9. w0wa baseline vs LCDM baseline
# w0wa baseline (s_min = 20, red, filled) against the LCDM baseline (outline) in the shared parameters plus w0, wa; the c000 truth (w0, wa) = (-1, 0) is marked for mocks.
# w0 and wa are limited to the NN training box by the prior (w0 in [-1.271, -0.726], wa in [-0.628, 0.621]).

#%%
if 9 in PLOTS_TO_RUN:
    for tag in DATASETS:
        name = f'compact_w0wa_{tag}_joint_smin20'
        figure(f'v2_9_w0wa_{tag}', [variant(f'compact_{tag}_joint_smin20', W0WA_PARAMS, 'LCDM baseline', 'tab:blue'),
                                    item(mcs(name, W0WA_PARAMS, r'$w_0w_a$CDM baseline'), RED, True) if have(name) else None], W0WA_PARAMS, TRUTH_FOR[tag], limits=DE_BOX)

#%% [markdown]
# ## 10. w0wa fits at s_min = 20 and 30

#%%
if 10 in PLOTS_TO_RUN:
    for tag in DATASETS:
        a, b = f'compact_w0wa_{tag}_joint_smin20', f'compact_w0wa_{tag}_joint_smin30'
        figure(f'v2_10_w0wa_smin_{tag}', [variant(b, W0WA_PARAMS, r'$w_0w_a$CDM, $s_{\rm min}=30$', 'tab:blue'),
                                          item(mcs(a, W0WA_PARAMS, r'$w_0w_a$CDM, $s_{\rm min}=20$ (baseline)'), RED, True) if have(a) else None], W0WA_PARAMS, TRUTH_FOR[tag], limits=DE_BOX)

#%% [markdown]
# ## 11. Posterior mean vs joint MAP and 1-D marginal mode: our DR1 fits and DESI's joint FS chain
# One row per parameter (PARAMS); filled circle = posterior mean with the posterior std as its error bar, open diamond = joint MAP (the highest-posterior post-burn-in sample in the full parameter space), open triangle = 1-D marginal mode (the peak of the
# Gaussian KDE of that parameter's samples, scipy's default bandwidth, so it depends somewhat on the bandwidth). Columns: our individual-tracer fits (tracer colours), our joint fit (black), and, at the far right behind a dashed line, DESI's public joint FS chain
# (purple; MAP = lowest `minuslogpost`, including all nuisance parameters; mean, std and mode weighted by the chain weights). The black number above each point set is (MAP - mean) / posterior std, the green one (mode - mean) / posterior std. A joint MAP far from
# the mean signals a skewed or prior-volume-affected posterior; the MAP of an MCMC chain is itself noisy (a few lnP units below the true maximum), so offsets of a few tenths of a sigma are not meaningful on their own. Table: v2_11_map_vs_mean_individual.csv.

#%%
def map_point(name):
    """Highest-posterior post-burn-in sample of a chain as {column: value}, with Omega_m and H0 added."""
    root = CHAIN_ROOT / model_dir(name) / name
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


def desi_frame(path):
    """A DESI cobaya chain (all chain.N.txt files in `path`) as a DataFrame with h, H0, ombh2, omch2, omegam, [logA, ns], weight, minuslogpost; the first DESI_BURN of each file is dropped. None if not found."""
    files = sorted(glob.glob(path + 'chain.[0-9]*.txt')) or glob.glob(path + 'chain.txt') if path else []
    if not files:
        return None
    frames = []
    for fn in files:
        names = open(fn).readline().lstrip('#').split()
        d = pd.read_csv(fn, sep=r'\s+', comment='#', header=None, names=names)
        frames.append(d.iloc[int(DESI_BURN * len(d)):])
    d = pd.concat(frames, ignore_index=True)
    h = d['H0'] / 100
    ob = d['ombh2'] if 'ombh2' in d else d['omegabh2']
    oc = d['omch2'] if 'omch2' in d else d['omegam'] * h ** 2 - ob - W_NU
    out = pd.DataFrame({'h': h, 'H0': d['H0'], 'ombh2': ob, 'omch2': oc, 'omegam': d['omegam'] if 'omegam' in d else (ob + oc + W_NU) / h ** 2, 'weight': d['weight'], 'minuslogpost': d['minuslogpost']})
    for k in ('logA', 'ns'):
        if k in d:
            out[k] = d[k]
    return out


def marg_mode(x, w=None, n_max=50000, seed=0):
    """Peak of the Gaussian KDE (scipy's default bandwidth) of the 1-D samples x, optionally weighted; at most n_max samples are used (drawn at random, by weight if weighted)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    if len(x) > n_max:
        idx = rng.choice(len(x), n_max, replace=w is not None, p=None if w is None else np.asarray(w, dtype=float) / np.sum(w))
        x, w = x[idx], None
    grid = np.linspace(*np.percentile(x, [0.05, 99.95]), 500)
    return float(grid[np.argmax(gaussian_kde(x, weights=w)(grid))])


def indiv_chain(tag, key):
    """Name of the individual-tracer chain whose name contains `_indiv_<key>`, or None."""
    files = [Path(p).name[:-len('.status.json')] for p in glob.glob(str(CHAIN_ROOT / 'compact' / f'compact_{tag}_indiv_*_smin20.status.json'))]
    match = [f for f in files if f'_indiv_{key}' in f]
    return match[0] if match else None


if 11 in PLOTS_TO_RUN:
    fits = [(short, indiv_chain('dr1', key), color) for short, key, color in TRACERS] + [('Joint', 'compact_dr1_joint_smin20', JOINT_COLOR)]
    fits = [(lab, n, c) for lab, n, c in fits if n and have(n)]
    dd = desi_frame(DESI_FS)
    if dd is None:
        print(f'plot 11: DESI joint FS chain not found at {DESI_FS}; only ours is drawn')
    if fits or dd is not None:
        table = []
        fig, axes = plt.subplots(len(PARAMS), 1, figsize=(WIDTH, 1.2 * len(PARAMS) + 0.7), sharex=True)

        def draw(ax, k, mp, mean, mode, sd, color):
            ax.errorbar(k - 0.24, mean, yerr=sd, fmt='o', ms=4.5, color=color, capsize=2.5, elinewidth=1.1)
            ax.plot(k, mp, 'D', ms=4.5, mfc='white', mec=color, mew=1.2)
            ax.plot(k + 0.24, mode, '^', ms=5, mfc='white', mec=color, mew=1.2)
            top = max(mean + sd, mp, mode)
            ax.annotate(rf'${(mp - mean) / sd:+.2f}\,\sigma$', (k, top), xytext=(0, 2.5), textcoords='offset points', ha='center', va='bottom', fontsize=6.5)
            ax.annotate(rf'${(mode - mean) / sd:+.2f}\,\sigma$', (k, top), xytext=(0, 10), textcoords='offset points', ha='center', va='bottom', fontsize=6.5, color='tab:green')

        def record(lab, source, p, mp, mean, mode, sd):
            table.append({'chain': lab, 'source': source, 'parameter': p, 'MAP': mp, 'mean': mean, '1-D marginal mode': mode, 'std': sd, '(MAP-mean)/std': (mp - mean) / sd, '(mode-mean)/std': (mode - mean) / sd})

        for k, (lab, name, color) in enumerate(fits):
            m, d = map_point(name), flat(name)
            for ax, p in zip(axes, PARAMS):
                sd, mode = d[p].std(), marg_mode(d[p].to_numpy())
                draw(ax, k, m[p], d[p].mean(), mode, sd, color)
                record(lab, 'ours', p, m[p], d[p].mean(), mode, sd)
        labels = [f[0] for f in fits]
        if dd is not None:
            k, imap = len(fits), dd['minuslogpost'].idxmin()
            for ax, p in zip(axes, PARAMS):
                mean = float(np.average(dd[p], weights=dd['weight']))
                sd = float(np.sqrt(np.average((dd[p] - mean) ** 2, weights=dd['weight'])))
                mode = marg_mode(dd[p].to_numpy(), dd['weight'].to_numpy())
                draw(ax, k, float(dd[p][imap]), mean, mode, sd, DESI_POINT)
                ax.axvline(k - 0.5, color='0.6', lw=0.8, ls='--', zorder=0)
                record('DESI joint FS', 'DESI', p, float(dd[p][imap]), mean, mode, sd)
            labels.append('DESI FS')
        for ax, p in zip(axes, PARAMS):
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.36 * (hi - lo))                                  # headroom for the two discrepancy labels
            ax.set_ylabel(f'${LABELS[p]}$', fontsize=9)
            ax.tick_params(labelsize=7)
        axes[-1].set_xticks(range(len(labels)))
        axes[-1].set_xticklabels(labels, fontsize=8)
        axes[-1].set_xlim(-0.6, len(labels) - 0.4)
        fig.legend(handles=[Line2D([], [], marker='o', color='0.3', ls='none', label=r'posterior mean $\pm$ posterior std'), Line2D([], [], marker='D', mfc='white', color='0.3', ls='none', label=r'joint MAP (black label: (MAP $-$ mean)/std)'),
                            Line2D([], [], marker='^', mfc='white', color='0.3', ls='none', label=r'1-D marginal mode (green label: (mode $-$ mean)/std)')],
                   frameon=False, fontsize=7.5, loc='lower center', ncol=2)
        fig.tight_layout(rect=(0, 0.075, 1, 1))
        fig.savefig(PLOT_DIR / 'v2_11_map_vs_mean_individual.pdf', bbox_inches='tight')
        fig.savefig(PLOT_DIR / 'v2_11_map_vs_mean_individual.png', dpi=DPI, bbox_inches='tight')
        pd.DataFrame(table).to_csv(PLOT_DIR / 'v2_11_map_vs_mean_individual.csv', index=False)
        plt.show()
        plt.close(fig)


#%% [markdown]
# ## 12. MAP vs s_min (DR1 and mean of mocks)
# MAP of each baseline-model joint fit at s_min = 20-40 for the five sampled parameters (rows), DR1 on the left and the mean of the mocks on the right (dashed line = c000 truth). Error bars are the
# posterior std; the s_min = 20 baseline is red.

#%%
if 12 in PLOTS_TO_RUN:
    fig, axes = plt.subplots(len(SAMPLED), len(DATASETS), figsize=(WIDTH, 1.2 * len(SAMPLED) + 0.7), sharex=True, squeeze=False)
    for col, (tag, title) in enumerate(DATASETS.items()):
        for sm in SMINS:
            name = f'compact_{tag}_joint_smin{sm}'
            if not have(name):
                continue
            m, d = map_point(name), flat(name)
            for row, p in enumerate(SAMPLED):
                axes[row, col].errorbar(sm, m[p], yerr=d[p].std(), fmt='o', ms=4.5, color=RED if sm == 20 else 'k', capsize=2.5, elinewidth=1.1)
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
    fig.legend(handles=[Line2D([], [], marker='o', color='k', ls='none', label=r'MAP $\pm$ posterior std'), Line2D([], [], marker='o', color=RED, ls='none', label=r'baseline ($s_{\rm min}=20$)'),
                        Line2D([], [], color='0.5', ls='--', label='c000 truth (mocks)')], frameon=False, fontsize=7.5, loc='lower center', ncol=3)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(PLOT_DIR / 'v2_12_map_vs_smin.pdf', bbox_inches='tight')
    fig.savefig(PLOT_DIR / 'v2_12_map_vs_smin.png', dpi=DPI, bbox_inches='tight')
    plt.show()
    plt.close(fig)

#%% [markdown]
# ## 13. Each tracer against all the others (DR1)
# For every tracer: the posterior of that tracer alone (individual fit, tracer colour) and of the joint fit to all the OTHER tracers (leave-one-out fit, grey) in TENSION_PARAMS, with the joint fit to
# all tracers in filled black. The follow-up figure gives the observed N_sigma of each tracer from the rest, from the parameter-difference test of Lemos et al. (2021): draw pairs of samples,
# form d = theta_tracer - theta_rest, estimate the density of d with a KDE, and take p = the mass of d where the density exceeds that at d = 0, N_sigma = norm.isf((1 - p)/2). The Gaussian N_sigma uses
# chi^2 = dmu^T (C_A + C_B)^-1 dmu with dof = number of parameters. The Monte Carlo version is reliable to roughly 3 sigma (it saturates beyond; then use the Gaussian one), and both assume the two
# fits are independent (the QSO overlaps LRG3 and ELG2, and the cross-covariance is neglected); prior-truncated posteriors dilute tension.

#%%
def gaussian_tension(a, b):
    dmu = a.mean(0) - b.mean(0)
    C = np.atleast_2d(np.cov(a.T)) + np.atleast_2d(np.cov(b.T))
    q = float(dmu @ np.linalg.solve(C, dmu))
    return q, float(norm.isf(chi2_dist.sf(q, a.shape[1]) / 2))


def mc_tension(a, b, n=None, seed=0):
    n = n or N_PAIRS
    rng = np.random.default_rng(seed)
    d = a[rng.integers(len(a), size=n)] - b[rng.integers(len(b), size=n)]
    kde = gaussian_kde(d.T, bw_method='silverman')
    p = min(float(np.mean(kde(d.T) > kde(np.zeros((d.shape[1], 1))))), 1 - 1 / n)
    return p, float(norm.isf((1 - p) / 2))


if 13 in PLOTS_TO_RUN:
    joint = 'compact_dr1_joint_smin20'
    rows = []
    for short, key, color in TRACERS:
        ind, loo = indiv_chain('dr1', key), f'compact_dr1_joint_no{short}_smin20'
        if not (ind and have(ind) and have(loo) and have(joint)):
            print(f'{short}: individual / leave-one-out / joint chain missing, skipped')
            continue
        A, B = flat(ind, TENSION_PARAMS).to_numpy(), flat(loo, TENSION_PARAMS).to_numpy()
        p_in, n_mc = mc_tension(A, B)
        q, n_g = gaussian_tension(A, B)
        row = {'tracer': short, 'N_sigma (MC)': n_mc, 'N_sigma (Gaussian)': n_g, 'chi2 (Gaussian)': q, 'MC saturated': p_in >= 1 - 1.5 / N_PAIRS}
        for k, p in enumerate(TENSION_PARAMS):
            row[f'{p}: MC'], row[f'{p}: Gaussian'] = mc_tension(A[:, [k]], B[:, [k]], seed=k + 1)[1], gaussian_tension(A[:, [k]], B[:, [k]])[1]
        rows.append(row)
        figure(f'v2_13_tension_{short}', [item(mcs(loo, TENSION_PARAMS, 'all other tracers'), '0.55', True), item(mcs(ind, TENSION_PARAMS, f'{short} alone'), color, True),
                                          item(mcs(joint, TENSION_PARAMS, 'all tracers'), JOINT_COLOR, True)], TENSION_PARAMS, width=5.0)
    if rows:
        TENSION = pd.DataFrame(rows).set_index('tracer')
        TENSION.to_csv(PLOT_DIR / 'v2_13_tension.csv')
        print(TENSION.round(2).to_string())
        fig, ax = plt.subplots(figsize=(0.65 * WIDTH, 3.0))
        x = np.arange(len(TENSION))
        col_of = {short: c for short, _k, c in TRACERS}
        ax.bar(x, TENSION['N_sigma (MC)'], color=[col_of[t] for t in TENSION.index], width=0.6)
        ax.plot(x, TENSION['N_sigma (Gaussian)'], 'ko', ms=4)
        for v in (1, 2, 3):
            ax.axhline(v, color='0.6', ls=':', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(TENSION.index, fontsize=8)
        ax.set_ylabel(r'$N_\sigma$ of the tracer from all the others', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(handles=[Patch(color='0.7', label='parameter-difference (Monte Carlo)'), Line2D([], [], marker='o', color='k', ls='none', label='Gaussian')], frameon=False, fontsize=7.5, loc='upper left')
        ax.text(0.99, 0.97, ', '.join(f'${LABELS[p]}$' for p in TENSION_PARAMS), transform=ax.transAxes, ha='right', va='top', fontsize=7.5)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / 'v2_13_tension_summary.pdf', bbox_inches='tight')
        fig.savefig(PLOT_DIR / 'v2_13_tension_summary.png', dpi=DPI, bbox_inches='tight')
        plt.show()
        plt.close(fig)

#%% [markdown]
# ## 14. Joint vs individual-tracer fits in the Omega_m-sigma8 and Omega_m-H0 planes (analogue of DESI 2024 V Fig. 14)
# Individual tracers in their DESI colours, the joint fit in filled black, for DR1 and for the mean of the mocks (c000 truth dashed). sigma8 is not sampled, so it is derived for every sample from
# (omega_b, omega_cdm, h, n_s, ln A_s) with CAMB (total-matter sigma8, m_nu = 0.06 eV) through an emulator: sigma8 = exp(quadratic polynomial in ln omega_b, ln omega_cdm, ln h, n_s) x sqrt(A_s / A_ref),
# exact in A_s and fitted to S8_TRAIN CAMB runs at random posterior samples; its accuracy is printed from S8_VALID held-out runs (CAMB is called S8_TRAIN + S8_VALID times, S8_JOBS in parallel).

#%%
def camb_sigma8(ob, oc, h, ns, As=2.1e-9):
    import camb
    pars = camb.set_params(H0=100 * h, ombh2=ob, omch2=oc, ns=ns, As=As, mnu=0.06)
    pars.set_matter_power(redshifts=[0.0], kmax=2.0)
    pars.NonLinear = camb.model.NonLinear_none
    return camb.get_results(pars).get_sigma8_0()


class Sigma8Emulator:
    """sigma8 = exp(quadratic in (ln ombh2, ln omch2, ln h, ns)) x sqrt(A_s / A_REF), trained on CAMB at random samples of `pool` (columns ombh2, omch2, h, ns)."""
    AS_REF = 2.1e-9

    def __init__(self, pool, n_train, n_valid, jobs=1, seed=0):
        feats = self._raw(pool[:, 0], pool[:, 1], pool[:, 2], pool[:, 3])
        self.mu, self.sd = feats.mean(0), feats.std(0)
        idx = np.random.default_rng(seed).choice(len(pool), n_train + n_valid, replace=False)
        X = pool[idx]
        y = np.log(np.array(Parallel(n_jobs=jobs)(delayed(camb_sigma8)(*x, self.AS_REF) for x in X)))
        F = self._design(X[:, 0], X[:, 1], X[:, 2], X[:, 3])
        self.coef = np.linalg.lstsq(F[:n_train], y[:n_train], rcond=None)[0]
        err = np.exp(F[n_train:] @ self.coef - y[n_train:]) - 1
        print(f'sigma8 emulator: {n_train} training runs, held-out relative error: max {np.abs(err).max():.2e}, rms {np.sqrt(np.mean(err ** 2)):.2e}')
        if np.abs(err).max() > 5e-3:
            print('WARNING: the sigma8 emulator error exceeds 0.5%; raise S8_TRAIN')

    @staticmethod
    def _raw(ob, oc, h, ns):
        return np.column_stack([np.log(ob), np.log(oc), np.log(h), ns])

    def _design(self, ob, oc, h, ns):
        u = (self._raw(ob, oc, h, ns) - self.mu) / self.sd
        quad = [u[:, i] * u[:, j] for i in range(4) for j in range(i, 4)]
        return np.column_stack([np.ones(len(u)), u] + quad)

    def __call__(self, c):
        shape = np.shape(c['h'])
        flat_ = {k: np.ravel(c[k]) for k in ('ombh2', 'omch2', 'h', 'ns', 'logA')}
        s8 = np.exp(self._design(flat_['ombh2'], flat_['omch2'], flat_['h'], flat_['ns']) @ self.coef) * np.sqrt(np.exp(flat_['logA']) * 1e-10 / self.AS_REF)
        return s8.reshape(shape)


if 14 in PLOTS_TO_RUN:
    sets = {}
    for tag in DATASETS:
        fits = [(short, indiv_chain(tag, key), color) for short, key, color in TRACERS] + [('All', f'compact_{tag}_joint_smin20', JOINT_COLOR)]
        sets[tag] = [(lab, n, c) for lab, n, c in fits if n and have(n)]
    pool_names = [n for fits in sets.values() for _l, n, _c in fits]
    if pool_names:
        rng14 = np.random.default_rng(1)
        pool = np.concatenate([(lambda a: a[rng14.choice(len(a), min(len(a), 4000), replace=False)])(flat(n, ['ombh2', 'omch2', 'h', 'ns']).to_numpy()) for n in pool_names])
        S8_EMU = Sigma8Emulator(pool, S8_TRAIN, S8_VALID, jobs=S8_JOBS)
        for tag, title in DATASETS.items():
            fits = sets[tag]
            if len(fits) < 2:
                continue
            roots = [mcs(n, ['omegam', 'sigma8', 'H0'], lab) for lab, n, _c in fits]
            cols = [c for _l, _n, c in fits]
            g = plotter(WIDTH)
            g.plots_2d(roots, param_pairs=[['omegam', 'sigma8'], ['omegam', 'H0']], nx=2, filled=True, colors=cols, lws=[1.0] * len(roots))
            for lg in list(g.fig.legends):
                lg.remove()
            g.fig.axes[0].legend(handles=[Patch(color=c, label=lab) for (lab, _n, _c2), c in zip(fits, cols)], frameon=False, fontsize=7, loc='upper right')
            if tag == 'mean':
                for ax, yk in zip(g.fig.axes[:2], ('sigma8', 'H0')):
                    ax.axvline(TRUTH['omegam'], ls='--', color='0.4', lw=0.9)
                    ax.axhline(TRUTH[yk], ls='--', color='0.4', lw=0.9)
            g.export(str(PLOT_DIR / f'v2_14_joint_vs_tracers_{tag}.pdf'))
            g.fig.savefig(PLOT_DIR / f'v2_14_joint_vs_tracers_{tag}.png', dpi=DPI, bbox_inches='tight')
            plt.show()
            plt.close(g.fig)
            print('saved', f'v2_14_joint_vs_tracers_{tag}', flush=True)

#%% [markdown]
# ## 15. Flat priors on omega_b and n_s
# The `compact_flatprior_{dr1,mean}_joint_smin20` fits replace the DESI Gaussians on omega_b and n_s by flat priors over the NN training box (dotted lines: omega_b in [0.0207, 0.0243], n_s in
# [0.901, 1.025]). Triangle per dataset: DESI FS-only (purple, Gaussian priors), our baseline with Gaussian priors (red, filled) and the flat-prior fit (blue outline, drawn on top). getdist applies its
# boundary correction at the prior edges. The summary table gives the shift of each posterior mean in units of the baseline std, the ratio of the stds, and the fraction of the flat-prior samples within
# 10% of the box width of an edge (pile-up against a hard edge means the data do not close the posterior there, so the result depends on the prior range).

#%%
if 15 in PLOTS_TO_RUN:
    if globals().get('fs') is None:
        fs = desi(DESI_FS, 'DESI DR1 FS-only')
    summary = []
    for tag in DATASETS:
        flat_name, base_name = f'compact_flatprior_{tag}_joint_smin20', f'compact_{tag}_joint_smin20'
        if not (have(flat_name) and have(base_name)):
            print(f'{flat_name} or {base_name}: chain not finished, skipped')
            continue
        figure(f'v2_15_flatprior_{tag}', [item(fs, PURPLE, True), baseline(tag, PARAMS, 'Gaussian priors (baseline)'),
                                          item(mcs(flat_name, PARAMS, r'flat $\omega_b$, $n_s$ priors', ranges=FLAT_BOX), 'tab:blue', False)], PARAMS, TRUTH_FOR[tag], edges=FLAT_BOX)
        db, df_ = flat(base_name), flat(flat_name)
        for p in PARAMS:
            row = {'dataset': DATASETS[tag], 'parameter': p, 'mean (Gaussian priors)': db[p].mean(), 'std (Gaussian priors)': db[p].std(), 'mean (flat priors)': df_[p].mean(), 'std (flat priors)': df_[p].std(),
                   'mean shift [std]': (df_[p].mean() - db[p].mean()) / db[p].std(), 'std ratio (flat / Gaussian)': df_[p].std() / db[p].std()}
            if p in FLAT_BOX:
                lo, hi = FLAT_BOX[p]
                row['fraction within 10% of lower edge'] = float(np.mean(df_[p] < lo + 0.1 * (hi - lo)))
                row['fraction within 10% of upper edge'] = float(np.mean(df_[p] > hi - 0.1 * (hi - lo)))
            summary.append(row)
    if summary:
        FLAT_SUMMARY = pd.DataFrame(summary).set_index(['dataset', 'parameter'])
        FLAT_SUMMARY.to_csv(PLOT_DIR / 'v2_15_flatprior_summary.csv')
        print(FLAT_SUMMARY.to_string(float_format=lambda x: f'{x:.4g}'))


#%% [markdown]
# ## 16. n_s marginal posterior of the DR1 joint fit: mean and MAP
# Diagnostic for the MAP-vs-mean offset in plot 11: the 1-D marginalized n_s posterior (Gaussian KDE of the post-burn-in samples) with the posterior mean (dashed), the joint MAP (solid: the n_s coordinate of the highest-posterior sample in the
# full parameter space, as in plot 11) and the 1-D marginal mode (dotted: the peak of the KDE, which depends somewhat on its bandwidth). The legend gives each value, and for the joint MAP its percentile within the marginal and its offset from the mean in posterior stds.

#%%
if 16 in PLOTS_TO_RUN:
    name = 'compact_dr1_joint_smin20'
    if have(name):
        ns_s = flat(name)['ns'].to_numpy()
        ns_map, ns_mean, ns_std = map_point(name)['ns'], ns_s.mean(), ns_s.std()
        pct = 100 * float(np.mean(ns_s < ns_map))
        lo, hi = np.percentile(ns_s, [0.05, 99.95])
        grid = np.linspace(min(lo, ns_map - 0.2 * ns_std), max(hi, ns_map + 0.2 * ns_std), 500)
        dens = gaussian_kde(np.random.default_rng(0).choice(ns_s, min(len(ns_s), 50000), replace=False))(grid)
        fig, ax = plt.subplots(figsize=(3.6, 2.7))
        ax.fill_between(grid, dens, color=RED, alpha=0.25, lw=0)
        ax.plot(grid, dens, color=RED, lw=1.2)
        ax.axvline(ns_mean, color='k', ls='--', lw=1.1, label=f'mean = {ns_mean:.3f}')
        ax.axvline(ns_map, color='tab:blue', lw=1.1, label=rf'joint MAP = {ns_map:.3f} ({pct:.0f}th percentile, {(ns_map - ns_mean) / ns_std:+.2f}$\sigma$)')
        ns_mode = marg_mode(ns_s)                                                  # the same helper as plot 11, so the numbers agree
        ax.axvline(ns_mode, color='tab:green', ls=':', lw=1.4, label=rf'1-D marginal mode = {ns_mode:.3f} ({(ns_mode - ns_mean) / ns_std:+.2f}$\sigma$)')
        ax.set_xlabel(f'${LABELS["ns"]}$', fontsize=9)
        ax.set_ylabel('marginalized posterior', fontsize=9)
        ax.set_yticks([])
        ax.set_ylim(0, 1.05 * dens.max())
        ax.tick_params(labelsize=7)
        ax.legend(frameon=False, fontsize=7, loc='lower left', bbox_to_anchor=(0.0, 1.0))      # above the axes, so it never overlaps the mean / MAP lines
        fig.tight_layout()
        fig.savefig(PLOT_DIR / 'v2_16_ns_joint_map_vs_mean.pdf', bbox_inches='tight')
        fig.savefig(PLOT_DIR / 'v2_16_ns_joint_map_vs_mean.png', dpi=DPI, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print(f'n_s (DR1 joint): mean {ns_mean:.4f}, std {ns_std:.4f}, joint MAP {ns_map:.4f} = {pct:.1f}th percentile, {(ns_map - ns_mean) / ns_std:+.2f} sigma from the mean; 1-D marginal mode {ns_mode:.4f}')