"""
getdist plots for the final fits: the b1-only fits (compact_v3_noslope_b1only_*) and the hexadecapole fits (compact_v3_hex_*, compact_v3_noslope_hex_*), all at s_min = 25.
Helpers, style and chain conventions are those of plot_v3_chains_gd.py / plot_b1_only_gd.py: chains read from status.json + .h5 (post-burn-in, thinned), one chain per walker, KDE bandwidth 1.25 x automatic,
unitless sampled parameters, 1-D 68% limits for error bars, joint fits filled black against individual tracers, vector PDF + PNG in PLOT_DIR. Nothing is fitted; chains that do not exist yet are skipped.
Every figure prints one line before it is shown: the file name and the chain(s) in it. Run cell by cell (#%% markers) or as a script, compact environment (getdist, h5py; cells 5, 10, 11 import fits_apfix_11par_v3).

The mean-of-mocks fits use the SINGLE-mock covariance (fits_apfix.load_mock_fit_data; only the compact_mean_joint_avgcov fit divides it by the number of mocks). Error bars and residuals of the fit-quality
plots therefore use the single-mock sigma for DR1 and the mocks alike; the tables give chi2 with the fit covariance and chi2 x N_mocks (the chi2 of the mean-of-mocks data vector, whose covariance is C / N_mocks), the p-value is from the latter.

  b1-only
  1. joint b1-only vs b1+b2 (xi0, xi2) vs DESI FS-only; DR1 and mean of mocks (c000 truth marked)
  2. individual b1-only fits vs the joint b1-only fit
  3. b1 of each tracer-bin: b1-only joint, b1+b2 joint, b1-only individual
  4. constraining power relative to DESI FS-only
  5. MAP vs posterior mean for the b1-only and b1+b2 joint fits: parameter offsets and chi2, N_data, N_par, p-value at both points
  hexadecapole
  6. xi0, xi2, xi4 vs xi0, xi2 vs DESI FS-only, for both models
  7. bias parameters, xi0, xi2, xi4 vs xi0, xi2
  8. constraining power relative to DESI FS-only
  9. chain summary table of all fits used
  fit quality (data vs MAP model multipoles with residuals; chi2 tables printed, not plotted)
  10. b1-only joint fits, DR1 and mean of mocks
  11. hexadecapole joint fits, both models, DR1 and mean of mocks
"""
#%%
import glob
import json
import sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import chi2 as chi2_dist
from getdist import MCSamples, loadMCSamples, plots

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False})
plt.rcParams['figure.dpi'] = 200            # on-screen / notebook resolution; with the 'retina' format below the inline PNG has twice this density
try:
    from matplotlib_inline.backend_inline import set_matplotlib_formats
    set_matplotlib_formats('retina')
except ImportError:
    pass

CHAIN_ROOT = Path('/spiffball/cosweeney/data/chains/fits_11par_apfix_v2_v3')
PLOT_DIR = CHAIN_ROOT / 'plots_gd_final'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
FITS_DIR = '/home/cosweeney/code/compact_streaming/nb/fits/11par_v3'
DESI_FS = '/spiffball/cosweeney/DESI/DR1/full-shape-cosmo-params/v1.0/cobaya/base/desi-reptvelocileptors-fs-all_schoneberg2024-bbn_planck2018-ns10/'

PLOTS_TO_RUN = set(range(1, 12))
SMIN = 25
POINT = 'map'           # point of the fit-quality plots (cells 10, 11): 'map' (highest-posterior sample) or 'mean' (posterior mean)
DPI = 600               # PNG resolution (the PDFs are vector)
DPI_MATRIX = 1200
WIDTH = 7.2             # two-column page width in inches; WIDTH_BIAS for the 12-parameter bias triangle
WIDTH_BIAS = 10.0
SMOOTH = {'smooth_scale_1D': -1.25, 'smooth_scale_2D': -1.25}
W_NU = 0.0006442
DATASETS = {'dr1': 'DR1', 'mean': 'mean of mocks'}

SAMPLED = ['h', 'ombh2', 'omch2', 'logA', 'ns']
LABELS = {'h': r'h', 'omegam': r'\Omega_m', 'H0': r'H_0', 'ombh2': r'\omega_b', 'omch2': r'\omega_{\rm cdm}', 'logA': r'\ln(10^{10}A_s)', 'ns': r'n_s', 'b1': r'b_1'}
TRUTH = {'h': 0.6736, 'ombh2': 0.02237, 'omch2': 0.12, 'logA': float(np.log(1e10 * 2.0830e-9)), 'ns': 0.9649}      # AbacusSummit c000
TRUTH_FOR = {'dr1': None, 'mean': TRUTH}
TRACERS = [('BGS', 'BGS', 'yellowgreen'), ('LRG1', 'LRG_0.4', 'orange'), ('LRG2', 'LRG_0.6', 'orangered'), ('LRG3', 'LRG_0.8', 'firebrick'),
           ('ELG2', 'ELG', 'steelblue'), ('QSO', 'QSO', 'seagreen')]      # (short name, substring of the chain name, DESI colour)
PURPLE, RED, BLUE = '#D58BCB', 'red', 'tab:blue'
for s, _k, _c in TRACERS:
    LABELS.update({f'b1_{s}': rf'b_1^{{\rm {s}}}', f'b2_{s}': rf'b_2^{{\rm {s}}}'})

FITS = {   # key -> chain folder, chain-name prefix, legend label, colour, filled, linestyle
    'b12':   dict(folder='compact_v3_noslope', prefix='compact_v3_noslope',       label=r'$b_1+b_2$',       color='k',  filled=False, ls='--'),
    'b1':    dict(folder='compact_v3_noslope', prefix='compact_v3_noslope_b1only', label=r'$b_1$ only',      color=BLUE, filled=True,  ls='-'),
    'nshex': dict(folder='compact_v3_noslope', prefix='compact_v3_noslope_hex',   label=r'$\ell=0,2,4$',    color=BLUE, filled=True,  ls='-'),
    'sl':    dict(folder='compact_v3',         prefix='compact_v3',               label=r'$\ell=0,2$',      color='k',  filled=False, ls='--'),
    'slhex': dict(folder='compact_v3',         prefix='compact_v3_hex',           label=r'$\ell=0,2,4$',    color=RED,  filled=True,  ls='-'),
}
FITS['ns'] = dict(FITS['b12'], label=r'$\ell=0,2$')       # the baseline (slopes zero) xi0, xi2 fit under its multipole label
HEX_PAIRS = {'ns': ('nshex', 'ns'), 'sl': ('slhex', 'sl')}     # model (ns = slopes zero, sl = slopes on) -> (hexadecapole key, xi0 + xi2 key)

#%% [markdown]
# ## Loading helpers

#%%
def fit_path(key, rest):
    m = FITS[key]
    return CHAIN_ROOT / m['folder'] / f"{m['prefix']}_{rest}"


def cname(key, rest):
    """Chain name, e.g. compact_v3_noslope_b1only_dr1_joint_smin25."""
    return f"{FITS[key]['prefix']}_{rest}"


def have(key, rest):
    return Path(f'{fit_path(key, rest)}.status.json').exists()


def joint(tag):
    return f'{tag}_joint_smin{SMIN}'


def indiv_rest(key, tag, kfrag):
    """`rest` of the individual-tracer chain of `key` whose name contains `_indiv_<kfrag>`, or None."""
    m = FITS[key]
    files = glob.glob(str(CHAIN_ROOT / m['folder'] / f"{m['prefix']}_{tag}_indiv_*_smin{SMIN}.status.json"))
    match = [Path(f).name[:-len('.status.json')][len(m['prefix']) + 1:] for f in files if f'_indiv_{kfrag}' in Path(f).name]
    return match[0] if match else None


_cache = {}


def full3(key, rest):
    """(array (steps, walkers, n_params), parameter names, status dict): post-burn-in and thinned as saved in status.json."""
    if (key, rest) not in _cache:
        root = fit_path(key, rest)
        st = json.load(open(f'{root}.status.json'))
        with h5py.File(f'{root}.h5', 'r') as f:
            it = int(f['mcmc'].attrs['iteration'])
            arr = f['mcmc/chain'][st['discard']:it:st['thin'], :, :]
        _cache[(key, rest)] = (arr, st['param_names'], st)
    return _cache[(key, rest)]


def columns(key, rest):
    """dict column -> array (steps, walkers), with derived Omega_m and H0."""
    arr, names, _ = full3(key, rest)
    c = {n: arr[..., i].copy() for i, n in enumerate(names)}
    c['H0'] = 100 * c['h']
    c['omegam'] = (c['ombh2'] + c['omch2'] + W_NU) / c['h'] ** 2
    return c


def mcs(key, rest, params, label=None):
    """getdist MCSamples with one chain per walker for the `params` that the chain has."""
    c = columns(key, rest)
    use = [p for p in params if p in c]
    stack = np.stack([c[p] for p in use], axis=-1)
    return MCSamples(samples=[stack[:, w] for w in range(stack.shape[1])], names=use, labels=[LABELS[p] for p in use], label=label or FITS[key]['label'], settings=SMOOTH)


def bias_mcs(key, rest, label=None):
    """b1, b2 of the six tracer-bins of a joint chain, named b1_<short>, b2_<short> (the chain's bias columns are in TRACERS order)."""
    arr, names, _ = full3(key, rest)
    b1, b2 = [i for i, n in enumerate(names) if n.startswith('b1_')], [i for i, n in enumerate(names) if n.startswith('b2_')]
    idx, cols = [], []
    for (short, _k, _c), i1, i2 in zip(TRACERS, b1, b2):
        idx += [i1, i2]
        cols += [f'b1_{short}', f'b2_{short}']
    stack = arr[..., idx]
    return MCSamples(samples=[stack[:, w] for w in range(stack.shape[1])], names=cols, labels=[LABELS[c] for c in cols], label=label or FITS[key]['label'], settings=SMOOTH)


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
    c.update({k: col(k) for k in ('logA', 'ns') if has(k)})
    names = [p for p in ['h', 'H0', 'omegam', 'ombh2', 'omch2', 'logA', 'ns'] if p in c]
    return MCSamples(samples=np.column_stack([c[p] for p in names]), weights=s.weights, names=names, labels=[LABELS[p] for p in names], label=label, settings=SMOOTH)


def margstats(by_name):
    """getdist 1-D marginalized statistics of a dict name -> samples (steps, walkers): {name: dict(mean, std, lower, upper)}, lower / upper = the 68% limits."""
    names = list(by_name)
    stack = np.stack([by_name[n] for n in names], axis=-1)
    ms = MCSamples(samples=[stack[:, w] for w in range(stack.shape[1])], names=names, settings=SMOOTH).getMargeStats()
    out = {}
    for n in names:
        par = ms.parWithName(n)
        lim = par.limits[0]
        out[n] = dict(mean=par.mean, std=par.err, lower=lim.lower, upper=lim.upper)
    return out


def errbar(ax, x, st, color, **kw):
    """Point at the posterior mean with the 1-D 68% limits as the error bar."""
    ax.errorbar(x, 0.5 * (st['lower'] + st['upper']), yerr=0.5 * (st['upper'] - st['lower']), fmt='none', ecolor=color, capsize=2, elinewidth=1.0)
    ax.plot(x, st['mean'], 'o', ms=4, color=color, **kw)


def used_chains():
    """(key, rest) of every chain the plots use: the joint fits of each key and the six individual b1-only fits, for DR1 and the mean of mocks."""
    out = [(k, joint(tag)) for k in FITS if k != 'ns' for tag in DATASETS]
    out += [('b1', indiv_rest('b1', tag, kfrag)) for tag in DATASETS for _s, kfrag, _c in TRACERS]
    return [(k, r) for k, r in out if r and have(k, r)]


def report_chains():
    bad = [cname(k, r) for k, r in used_chains() if not json.load(open(f'{fit_path(k, r)}.status.json')).get('converged')]
    print(f'{len(used_chains())} chains found' + (f'; NOT CONVERGED: {bad}' if bad else ''))


report_chains()

#%% [markdown]
# ## Fit-layer helpers (cells 5, 10, 11)
# The fit layer is imported on first use; evaluation follows plot_b1_only_gd.py (evaluate_at_map), with the point given explicitly.

#%%
fv = SPECS = None


def load_fv():
    """Import fits_apfix_11par_v3 (and through it fits_apfix.py, common.py) once; False if that fails."""
    global fv, SPECS
    if fv is None:
        try:
            sys.path.insert(0, FITS_DIR)
            import fits_apfix_11par_v3 as _fv
            fv, SPECS = _fv, {s['name']: s for s in _fv.build_specs_v3()}
        except Exception as e:
            print(f'fit layer not available ({type(e).__name__}: {e}); chi2 tables and fit-quality plots skipped')
            return False
    return True


def theta_map(key, rest):
    """Highest-posterior post-burn-in sample of the chain."""
    root = fit_path(key, rest)
    st = json.load(open(f'{root}.status.json'))
    with h5py.File(f'{root}.h5', 'r') as f:
        it = int(f['mcmc'].attrs['iteration'])
        lp = f['mcmc/log_prob'][st['discard']:it]
        t, w = np.unravel_index(np.nanargmax(np.where(np.isfinite(lp), lp, -np.inf)), lp.shape)
        return f['mcmc/chain'][st['discard'] + t, w, :]


def theta_mean(key, rest):
    arr = full3(key, rest)[0]
    return arr.reshape(-1, arr.shape[-1]).mean(0)


def evaluate(problem, theta):
    """Per tracer-bin dict(s, data, model, sigma, chi2): all multipoles concatenated, sigma from the fit covariance (single-mock for the mocks), chi2 with the fit covariance."""
    full = problem._expand(theta) if hasattr(problem, '_expand') else theta          # b1-only problem: b2 put back
    biases = problem.unpack(full)[5]
    h, ombh2, omch2, lnAs, ns = problem.model_params(full)
    aps = fv.fa.ap_multi(h, ombh2, omch2, ns, problem.z_effs)
    out = []
    for d, (b1, b2), (q_perp, q_par) in zip(problem.data, biases, aps):
        mod = fv.fa.model_one(d['emus'], [b1, b2, ombh2, omch2, h, lnAs, ns], d['s_data'], d['W'], q_perp, q_par)
        r = mod - d['data_vec']
        out.append(dict(s=d['s_data'], data=d['data_vec'], model=mod, sigma=np.sqrt(np.diag(np.linalg.inv(d['cov_inv']))), chi2=float(r @ d['cov_inv'] @ r)))
    return out


def problem_for(key, rest):
    return fv.fa.FitProblem(SPECS[cname(key, rest)])


def n_scale(tag):
    """chi2 factor for the mean of mocks (fit covariance = single-mock covariance, the mean's is C / N_mocks); 1 for DR1."""
    return min(fv.fa.mock_counts().values()) if tag == 'mean' else 1


def gof_row(base, problem, ev, tag):
    """Table row of the joint chi2: chi2 with the fit covariance, chi2 x N_mocks, N_data, N_par and the p-value of the latter for N_data - N_par degrees of freedom."""
    chi2 = sum(e['chi2'] for e in ev)
    n_data = sum(len(e['data']) for e in ev)
    scaled = n_scale(tag) * chi2
    return {**base, 'chi2 (fit cov.)': chi2, 'chi2 x N_mocks': scaled, 'N_data': n_data, 'N_par': problem.ndim, 'p-value': float(chi2_dist.sf(scaled, n_data - problem.ndim))}


#%% [markdown]
# ## Figure helpers

#%%
def announce(fname, names):
    """One line before each figure: its file name and the chain(s) it shows."""
    print(f'{fname}: ' + ' | '.join(n for n in names if n), flush=True)


def plotter(width):
    st = plots.GetDistPlotSettings()
    st.axes_fontsize, st.lab_fontsize, st.legend_fontsize = 8, 10, 9
    st.legend_frame = False
    st.axis_tick_max_labels = 4
    return plots.get_subplot_plotter(width_inch=width, scaling=False, settings=st)


def item(smp, color, filled=False, ls='-', name=None):
    return None if smp is None else (smp, color, filled, ls, name)


def fit_item(key, rest, params, label=None, bias=False):
    """The chain `key`/`rest` drawn in its own style, or None if it does not exist."""
    if not have(key, rest):
        return None
    m = FITS[key]
    smp = bias_mcs(key, rest, label) if bias else mcs(key, rest, params, label)
    return item(smp, m['color'], m['filled'], m['ls'], cname(key, rest))


def figure(fname, items, params, truth=None, width=None):
    """Triangle plot of the (non-None) items = (MCSamples, colour, filled, linestyle, chain name), the last drawn on top; saves PDF + PNG."""
    items = [i for i in items if i is not None]
    roots = [i[0] for i in items]
    present = {p for r in roots for p in r.getParamNames().list()}
    params = [p for p in params if p in present]
    g = plotter(width or WIDTH)
    g.triangle_plot(roots, params, filled=[i[2] for i in items], contour_colors=[i[1] for i in items], contour_ls=[i[3] for i in items],
                    contour_lws=[1.2] * len(items), legend_labels=[r.getLabel() for r in roots], markers={k: v for k, v in (truth or {}).items() if k in params})
    g.export(str(PLOT_DIR / f'{fname}.pdf'))
    g.fig.savefig(PLOT_DIR / f'{fname}.png', dpi=DPI, bbox_inches='tight')
    announce(fname, [i[4] for i in items])
    plt.show()
    plt.close(g.fig)


def sqdet(smp, names):
    """sqrt(det of the posterior covariance) of the named parameters, proportional to the 1-sigma volume."""
    if smp is None or any(n not in smp.getParamNames().list() for n in names):
        return np.nan
    return float(np.exp(0.5 * np.linalg.slogdet(np.atleast_2d(smp.cov(pars=names)))[1]))


def matrix_figure(fname, chains, fs):
    """Constraining power relative to DESI FS-only: sqrt det ratio of the posterior covariances, 1-D on the diagonal and pairwise off it, one panel per (label, MCSamples, chain name) in `chains`.
    Also writes <fname>.csv with the 5-D ratio of the sampled parameters."""
    n = len(SAMPLED)
    fig, axes = plt.subplots(1, len(chains), figsize=(WIDTH * len(chains) / 2.0, 0.55 * WIDTH), squeeze=False)
    rows = []
    for ax, (label, smp, name) in zip(axes[0], chains):
        R = np.full((n, n), np.nan)
        for i, a in enumerate(SAMPLED):
            for j, b in enumerate(SAMPLED):
                R[i, j] = sqdet(smp, [a]) / sqdet(fs, [a]) if i == j else sqdet(smp, [a, b]) / sqdet(fs, [a, b])
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
        ax.set_xticklabels([f'${LABELS[p]}$' for p in SAMPLED], fontsize=9, rotation=45)
        ax.set_yticklabels([f'${LABELS[p]}$' for p in SAMPLED] if ax is axes[0][0] else [], fontsize=9)
        ax.set_xlabel(label, fontsize=10)
        d = sqdet(smp, SAMPLED) / sqdet(fs, SAMPLED)
        rows.append({'fit': name, 'sqrt det ratio (5-D)': d, 'per dimension': d ** (1 / n)})
    cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, pad=0.02)
    cb.set_ticks([-0.7, -0.35, 0, 0.35, 0.7])
    cb.set_ticklabels(['0.2', '0.45', '1', '2.2', '5'])
    cb.set_label(r'ratio to DESI FS-only', fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.savefig(PLOT_DIR / f'{fname}.pdf', bbox_inches='tight')
    fig.savefig(PLOT_DIR / f'{fname}.png', dpi=DPI_MATRIX, bbox_inches='tight')
    announce(fname, [c[2] for c in chains])
    plt.show()
    plt.close(fig)
    pd.DataFrame(rows).to_csv(PLOT_DIR / f'{fname}.csv', index=False)
    print(pd.DataFrame(rows).to_string(float_format=lambda v: f'{v:.4g}'))


fs = desi(DESI_FS, 'DESI DR1 FS-only')
fs_item = item(fs, PURPLE, True, name='DESI DR1 FS-only')

#%% [markdown]
# ## 1. Joint b1-only vs b1+b2 vs DESI FS-only

#%%
if 1 in PLOTS_TO_RUN:
    for tag in DATASETS:
        if have('b1', joint(tag)):
            figure(f'final_1_b1only_vs_b1b2_{tag}', [fs_item, fit_item('b1', joint(tag), SAMPLED), fit_item('b12', joint(tag), SAMPLED)], SAMPLED, TRUTH_FOR[tag])

#%% [markdown]
# ## 2. Individual b1-only fits vs the joint b1-only fit

#%%
if 2 in PLOTS_TO_RUN:
    for tag in DATASETS:
        if not have('b1', joint(tag)):
            continue
        items = []
        for short, kfrag, color in TRACERS:
            rest = indiv_rest('b1', tag, kfrag)
            items.append(item(mcs('b1', rest, SAMPLED, short), color, True, name=cname('b1', rest)) if rest else None)
        figure(f'final_2_b1only_individual_{tag}', items + [item(mcs('b1', joint(tag), SAMPLED, 'Joint'), 'k', True, name=cname('b1', joint(tag)))], SAMPLED, TRUTH_FOR[tag])

#%% [markdown]
# ## 3. b1 of each tracer-bin
# Posterior mean and 1-D 68% limits: b1-only joint, b1+b2 joint, b1-only individual.

#%%
if 3 in PLOTS_TO_RUN:
    fig, axes = plt.subplots(len(DATASETS), 1, figsize=(WIDTH, 2.0 * len(DATASETS) + 0.5), sharex=True, squeeze=False)
    series = [('b1', 'joint', r'$b_1$ only, joint', BLUE, -0.2, {}), ('b12', 'joint', r'$b_1+b_2$, joint', 'k', 0.0, dict(mfc='white')), ('b1', 'indiv', r'$b_1$ only, individual', 'tab:orange', 0.2, {})]
    used = []
    for ax, (tag, title) in zip(axes[:, 0], DATASETS.items()):
        for key, mode, _lab, color, dx, kw in series:
            for i, (short, kfrag, _c) in enumerate(TRACERS):
                rest = joint(tag) if mode == 'joint' else indiv_rest(key, tag, kfrag)
                if not rest or not have(key, rest):
                    continue
                c = columns(key, rest)
                name = [n for n in full3(key, rest)[1] if n.startswith('b1_')][i] if mode == 'joint' else 'b1'
                errbar(ax, i + dx, margstats({'b1': c[name]})['b1'], color, **kw)
                if mode == 'joint' or i == 0:
                    used.append(cname(key, rest) if mode == 'joint' else f"{cname(key, rest.replace(kfrag, '<tracer>'))} (six fits)")
        ax.set_ylabel(r'$b_1$', fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-0.5, len(TRACERS) - 0.5)
        ax.tick_params(labelsize=7)
    axes[-1, 0].set_xticks(range(len(TRACERS)))
    axes[-1, 0].set_xticklabels([t[0] for t in TRACERS], fontsize=8)
    fig.legend(handles=[Line2D([], [], marker='o', color=c, mfc=kw.get('mfc', c), ls='none', label=lab) for _k, _m, lab, c, _d, kw in series], frameon=False, fontsize=8, loc='lower center', ncol=3)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(PLOT_DIR / 'final_3_b1_per_tracer.pdf', bbox_inches='tight')
    fig.savefig(PLOT_DIR / 'final_3_b1_per_tracer.png', dpi=DPI, bbox_inches='tight')
    announce('final_3_b1_per_tracer', list(dict.fromkeys(used)))
    plt.show()
    plt.close(fig)

#%% [markdown]
# ## 4. Constraining power of the b1-only and b1+b2 fits

#%%
if 4 in PLOTS_TO_RUN and fs is not None:
    for tag in DATASETS:
        chains = [(FITS[k]['label'], mcs(k, joint(tag), SAMPLED), cname(k, joint(tag))) for k in ('b12', 'b1') if have(k, joint(tag))]
        if len(chains) == 2:
            matrix_figure(f'final_4_constraining_power_b1only_{tag}', chains, fs)

#%% [markdown]
# ## 5. MAP vs posterior mean, b1-only and b1+b2 joint fits
# Figure: (MAP - posterior mean) / posterior std for the sampled parameters (the MAP is the highest-posterior sample of the chain). Table (final_5_map_vs_mean.csv): chi2 of the data at the MAP and at the posterior mean,
# with N_data, N_par and the p-value; chi2 columns as in gof_row (single-mock covariance for the mocks, and x N_mocks).

#%%
CHECK_FITS = [('b12', r'$b_1+b_2$', 'k', -0.12), ('b1', r'$b_1$ only', BLUE, 0.12)]


def display_columns(names):
    """(display name, index in the chain) of the sampled cosmological parameters and the b1 of each tracer-bin (b2 left out)."""
    b1 = [i for i, n in enumerate(names) if n.startswith('b1_')]
    return [(n, names.index(n)) for n in SAMPLED] + [(f'b1_{t[0]}', i) for t, i in zip(TRACERS, b1)]


if 5 in PLOTS_TO_RUN:
    cols, used = None, []
    fig, axes = plt.subplots(len(DATASETS), 1, figsize=(WIDTH, 2.2 * len(DATASETS) + 0.5), sharex=True, sharey=True, squeeze=False)
    for ax, (tag, title) in zip(axes[:, 0], DATASETS.items()):
        for key, label, color, dx in CHECK_FITS:
            if not have(key, joint(tag)):
                continue
            arr, names, _ = full3(key, joint(tag))
            flat = arr.reshape(-1, arr.shape[-1])
            off = (theta_map(key, joint(tag)) - flat.mean(0)) / flat.std(0)
            cols = display_columns(names)
            ax.plot([i + dx for i in range(len(cols))], [off[j] for _n, j in cols], 'o', ms=4, color=color, label=label)
            used.append(cname(key, joint(tag)))
        ax.axhline(0, color='0.5', lw=0.9)
        ax.set_ylabel(r'$({\rm MAP}-\langle\theta\rangle)/\sigma$', fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
    if cols:
        axes[-1, 0].set_xticks(range(len(cols)))
        axes[-1, 0].set_xticklabels([f'${LABELS[n]}$' for n, _j in cols], fontsize=8, rotation=45)
        axes[-1, 0].set_xlim(-0.5, len(cols) - 0.5)
        axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / 'final_5_map_vs_mean.pdf', bbox_inches='tight')
        fig.savefig(PLOT_DIR / 'final_5_map_vs_mean.png', dpi=DPI, bbox_inches='tight')
        announce('final_5_map_vs_mean', used)
        plt.show()
    plt.close(fig)

    if load_fv():
        rows = []
        for tag, title in DATASETS.items():
            for key, label, _c, _dx in CHECK_FITS:
                if not have(key, joint(tag)):
                    continue
                problem = problem_for(key, joint(tag))
                for point, theta in (('MAP', theta_map(key, joint(tag))), ('mean', theta_mean(key, joint(tag)))):
                    rows.append(gof_row({'chain': cname(key, joint(tag)), 'dataset': title, 'point': point}, problem, evaluate(problem, theta), tag))
        if rows:
            table = pd.DataFrame(rows)
            table.to_csv(PLOT_DIR / 'final_5_map_vs_mean.csv', index=False)
            pd.set_option('display.width', 250)
            print(table.to_string(float_format=lambda v: f'{v:.4g}', index=False))

#%% [markdown]
# ## 6. Hexadecapole fits vs the xi0, xi2 fits and DESI FS-only
# One figure per model (ns = slopes zero = baseline, sl = slopes on) and dataset.

#%%
if 6 in PLOTS_TO_RUN:
    for model, (hk, bk) in HEX_PAIRS.items():
        for tag in DATASETS:
            if have(hk, joint(tag)):
                figure(f'final_6_hexadecapole_{model}_{tag}', [fs_item, fit_item(hk, joint(tag), SAMPLED), fit_item(bk, joint(tag), SAMPLED)], SAMPLED, TRUTH_FOR[tag])

#%% [markdown]
# ## 7. Bias parameters, hexadecapole vs xi0, xi2

#%%
if 7 in PLOTS_TO_RUN:
    cols = [f'{b}_{t[0]}' for t in TRACERS for b in ('b1', 'b2')]
    for model, (hk, bk) in HEX_PAIRS.items():
        for tag in DATASETS:
            if have(hk, joint(tag)):
                figure(f'final_7_hexadecapole_bias_{model}_{tag}', [fit_item(hk, joint(tag), cols, bias=True), fit_item(bk, joint(tag), cols, bias=True)], cols, width=WIDTH_BIAS)

#%% [markdown]
# ## 8. Constraining power of the hexadecapole fits

#%%
if 8 in PLOTS_TO_RUN and fs is not None:
    for model, (hk, bk) in HEX_PAIRS.items():
        for tag in DATASETS:
            chains = [(FITS[k]['label'], mcs(k, joint(tag), SAMPLED), cname(k, joint(tag))) for k in (bk, hk) if have(k, joint(tag))]
            if len(chains) == 2:
                matrix_figure(f'final_8_constraining_power_hexadecapole_{model}_{tag}', chains, fs)

#%% [markdown]
# ## 9. Chain summary
# Every chain used above: convergence, iterations, tau_max, N / tau_max.

#%%
if 9 in PLOTS_TO_RUN:
    rows = []
    for k, r in used_chains():
        st = json.load(open(f'{fit_path(k, r)}.status.json'))
        rows.append({'fit': cname(k, r), 'converged': st.get('converged'), 'iterations': st.get('iterations'), 'tau_max': st.get('tau_max'), 'N/tau_max': st.get('n_over_tau_max')})
    table = pd.DataFrame(rows)
    table.to_csv(PLOT_DIR / 'final_9_chain_summary.csv', index=False)
    pd.set_option('display.width', 250)
    print(table.to_string(float_format=lambda v: f'{v:.4g}'))

#%% [markdown]
# ## 10-11. Fit quality
# One 2 x 3 figure per fit: the six tracer-bins, each with s^2 xi_ell (points = data with the fit covariance's error bars, lines = model at POINT) over one residual panel Delta_ell / sigma per multipole
# (xi0 filled circles, solid; xi2 open squares, dashed; xi4 open triangles, dotted). Errors are the single-mock sigma for DR1 and the mocks alike (the covariance the fits use). chi2 tables are printed and saved, not plotted.

#%%
MARK, LS = ['o', 's', '^'], ['-', '--', ':']
COLOR = {short: color for short, _k, color in TRACERS}


def plot_fit(ev, fname):
    nl = len(ev[0]['data']) // len(ev[0]['s'])
    rc = {'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(7.2, 2 * (1.9 + 0.55 * nl)), layout='constrained')
        subs = fig.subfigures(2, 3, wspace=0.03, hspace=0.03)
        for k, ((short, _kf, color), e) in enumerate(zip(TRACERS, ev)):
            axs = subs[k // 3, k % 3].subplots(1 + nl, 1, sharex=True, gridspec_kw=dict(height_ratios=[2.2] + [1] * nl))
            ax, res = axs[0], axs[1:]
            s, n = e['s'], len(e['s'])
            for l in range(nl):
                sl = slice(l * n, (l + 1) * n)
                sig = e['sigma'][sl]
                mfc = color if l == 0 else 'white'
                ax.errorbar(s, s ** 2 * e['data'][sl], s ** 2 * sig, fmt=MARK[l], ms=2.8, mew=0.8, color=color, mfc=mfc, elinewidth=0.7, capsize=0, zorder=2)
                ax.plot(s, s ** 2 * e['model'][sl], ls=LS[l], color=color, lw=1.1, zorder=3)
                d = (e['data'][sl] - e['model'][sl]) / sig
                lim = max(3, int(np.ceil(np.abs(d).max())))
                t = 2 if lim <= 4 else 4
                r = res[l]
                r.axhspan(-1, 1, color='0.85', lw=0)
                r.axhline(0, color='0.5', lw=0.6)
                r.plot(s, d, MARK[l], ls='none', ms=2.8, mew=0.8, color=color, mfc=mfc)
                r.set_ylim(-lim, lim)
                r.set_yticks([-t, 0, t])
                r.set_ylabel(rf'$\Delta_{2 * l}/\sigma$', fontsize=8)
                r.tick_params(labelsize=7)
            ax.set_title(short, color=color, fontsize=10, loc='right', pad=3)
            ax.locator_params(axis='y', nbins=4)
            ax.tick_params(labelsize=7)
            if k % 3 == 0:
                ax.set_ylabel(r'$s^2\xi_\ell$', fontsize=9)
            ax.set_xlim(20, 130)
            ax.set_xticks([40, 80, 120])
            res[-1].set_xlabel(r'$s\ [h^{-1}{\rm Mpc}]$', fontsize=8)
            subs[k // 3, k % 3].align_ylabels(axs)
        fig.savefig(PLOT_DIR / f'{fname}.pdf', bbox_inches='tight')
        fig.savefig(PLOT_DIR / f'{fname}.png', dpi=DPI, bbox_inches='tight')
        return fig


def run_fit_quality(key, file_tag):
    """Fit-quality figure and chi2 table of the joint chain `key` for DR1 and the mean of mocks; file_tag names the figure files."""
    if not load_fv():
        return
    rows = []
    for tag, title in DATASETS.items():
        rest = joint(tag)
        if not have(key, rest):
            print(f'{cname(key, rest)}: no chain, skipped')
            continue
        try:
            problem = problem_for(key, rest)
        except Exception as e:
            print(f'{cname(key, rest)}: cannot build the problem ({type(e).__name__}: {e}), skipped')
            continue
        ev = evaluate(problem, theta_map(key, rest) if POINT == 'map' else theta_mean(key, rest))
        fname = f'final_fit_quality_{file_tag}_{tag}'
        fig = plot_fit(ev, fname)
        announce(fname, [cname(key, rest), f'{title}, model at the {"MAP" if POINT == "map" else "posterior mean"}'])
        plt.show()
        plt.close(fig)
        base = {'chain': cname(key, rest), 'dataset': title, 'point': POINT}
        rows.append(gof_row({**base, 'tracer': 'joint'}, problem, ev, tag))
        rows += [{**base, 'tracer': t[0], 'chi2 (fit cov.)': e['chi2'], 'chi2 x N_mocks': n_scale(tag) * e['chi2'], 'N_data': len(e['data'])} for t, e in zip(TRACERS, ev)]
    if rows:
        table = pd.DataFrame(rows)
        table.to_csv(PLOT_DIR / f'final_fit_quality_{file_tag}_chi2.csv', index=False)
        pd.set_option('display.width', 250)
        print(table.to_string(float_format=lambda v: f'{v:.4g}', index=False))


if 10 in PLOTS_TO_RUN:
    run_fit_quality('b1', 'b1only')

#%%
if 11 in PLOTS_TO_RUN:
    for model, (hk, _bk) in HEX_PAIRS.items():
        run_fit_quality(hk, f'hexadecapole_{model}')