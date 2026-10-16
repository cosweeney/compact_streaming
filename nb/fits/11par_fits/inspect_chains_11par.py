"""
Inspect the 11-parameter / Gaussian-model fit chains: burn-in detection, trace plots, MAP validation plots and
cosmology posteriors of the baseline s_min = 20 joint fits (with DESI DR1 2024 full-shape contours).
Converted from inspect_chains_11par.ipynb; run cell by cell (#%% markers) or as a script, from the folder containing
fit_lib_11par.py, in the compact environment.
"""

#%% [markdown]
# # Inspecting the 11-parameter / Gaussian-model fit chains
# Run from the folder containing `fit_lib_11par.py` (compact environment). Sections: (1) chain inventory and burn-in,
# (2) trace plots for every chain, (3) MAP validation plots of the baseline $s_{\rm min}=20\,h^{-1}$Mpc joint fits,
# (4) marginalized cosmology posteriors of the baseline fits with the DESI DR1 2024 full-shape contours.
#
# **Burn-in rule** (`find_burn_in`): the walker-median log-posterior per step is smoothed with a running mean
# (window = max(25, N/200) steps); the late-time level and scatter come from the second half of the chain; the burn-in is the
# first step from which the smoothed trace stays above `level - BURN_NSIGMA * scatter` for a full window.
# The dashed grey line marks $5\tau_{\max}$, the burn-in used for the saved getdist `.txt` chains, for comparison.

#%%
import sys, json, warnings
from pathlib import Path
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import CubicSpline
from getdist import MCSamples, loadMCSamples, plots
try:
    from IPython.display import display
except ImportError:
    display = print

try:
    HERE = Path(__file__).resolve().parent
except NameError:          # e.g. running a cell in an interactive console
    HERE = Path.cwd()
sys.path.insert(0, str(HERE))
import fit_lib_11par as fl

warnings.filterwarnings('ignore', message='.*tight_layout.*')
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
                     'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False,
                     'axes.xmargin': 0, 'figure.dpi': 110})

# ---------------- configuration ----------------
CHAIN_ROOT = fl.CHAIN_ROOT
PLOT_DIR = CHAIN_ROOT / 'plots'
(PLOT_DIR / 'traces').mkdir(parents=True, exist_ok=True)
DESI_BASE = '/spiffball/cosweeney/DESI/DR1/full-shape-cosmo-params/v1.0/cobaya/base/desi-reptvelocileptors-fs-all_schoneberg2024-bbn_planck2018-ns10/'

BASELINE = {   # baseline s_min = 20 joint fits: {case: {model: chain name}}
    'mock': {'compact': 'compact_mean_joint_smin20', 'gauss': 'gauss_mean_joint_smin20'},
    'dr1':  {'compact': 'compact_dr1_joint_smin20',  'gauss': 'gauss_dr1_joint_smin20'},
}
MODEL_LABEL = {'compact': r'Compact (11-par), $s_{\rm min}=20$', 'gauss': r'Gaussian, $s_{\rm min}=20$'}
MODEL_COLOR = {'compact': 'tab:blue', 'gauss': 'tab:orange'}

TRUTH = dict(H0=67.36, ombh2=0.02237, omch2=0.12, logA=3.0363943, ns=0.9649)   # AbacusSummit c000 (mock truth)
GETDIST_SETTINGS = {'smooth_scale_1D': 0.5, 'smooth_scale_2D': 0.5}

N_TRACE_WALKERS, TRACE_POINTS = 12, 2000         # walkers drawn per trace panel / max points per line
BURN_NSIGMA = 2.0
THIN_TAUS = fl.THIN_TAUS                          # posterior thinning = THIN_TAUS * tau_max, as in the saved chains
SHOW_TRACES_INLINE = {n for case in BASELINE.values() for n in case.values()}   # all traces are saved; these are also shown

# tracer display names / DESI per-tracer colours (DESI 2024 II, Appendix D), in fl.FIT_KEYS order
TRACER_NAMES = ['BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG2', 'QSO']
TRACER_COLORS = ['yellowgreen', 'orange', 'orangered', 'firebrick', 'steelblue', 'seagreen']
assert len(fl.FIT_KEYS) == 6
TRACER_SHORT = {fl.kname(k): n for k, n in zip(fl.FIT_KEYS, TRACER_NAMES)}
TRACER_COLOR = {k: c for k, c in zip(fl.FIT_KEYS, TRACER_COLORS)}

#%% [markdown]
# ## 1. Chain inventory and burn-in

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


def stem(h5path):
    return str(h5path)[:-3]


def read_status(h5path):
    p = Path(stem(h5path) + '.status.json')
    return json.load(open(p)) if p.exists() else {}


def pretty_param(name):
    base = {'h': 'h', 'ombh2': r'\omega_b', 'omch2': r'\omega_{\rm cdm}', 'logA': r'\ln(10^{10}A_s)', 'ns': 'n_s',
            'b1': 'b_1', 'b2': 'b_2'}
    if name in base:
        return f'${base[name]}$'
    for pre, sym in (('b1_', 'b_1'), ('b2_', 'b_2')):
        if name.startswith(pre):
            return f'${sym}^{{\\rm {TRACER_SHORT.get(name[len(pre):], name[len(pre):])}}}$'
    return name


def read_param_names(h5path, ndim):
    p = Path(stem(h5path) + '.paramnames')
    if p.exists():
        names = [ln.split()[0].rstrip('*') for ln in open(p) if ln.strip()]
        if len(names) == ndim:
            return names
    return [f'p{i}' for i in range(ndim)]


def h5_info(h5path):
    with h5py.File(h5path, 'r') as f:
        it = int(f['mcmc'].attrs['iteration'])
        logp = f['mcmc/log_prob'][:it]
    burn, m, ms, level, scatter = find_burn_in(logp)
    st = read_status(h5path)
    return dict(path=h5path, name=Path(stem(h5path)).name, model=h5path.parent.name, iterations=it, burn=burn,
                m=m, ms=ms, level=level, scatter=scatter, status=st, nwalkers=logp.shape[1])


files = sorted(CHAIN_ROOT.glob('*/*.h5'))
INFO = {}
for p in files:
    INFO[Path(stem(p)).name] = h5_info(p)
print(f'{len(INFO)} chains found under {CHAIN_ROOT}')

rows = []
for name, i in INFO.items():
    st = i['status']
    tau = st.get('tau_max', np.nan)
    rows.append(dict(chain=name, steps=i['iterations'], walkers=i['nwalkers'], tau_max=tau,
                     N_over_tau=i['iterations'] / tau if tau == tau else np.nan, converged=st.get('converged'),
                     burn_auto=i['burn'], burn_frac=i['burn'] / i['iterations'], five_tau=fl.BURN_TAUS * tau if tau == tau else np.nan))
SUMMARY = pd.DataFrame(rows).set_index('chain')
SUMMARY.to_csv(PLOT_DIR / 'chain_summary.csv')
display(SUMMARY.round({'tau_max': 0, 'N_over_tau': 0, 'burn_frac': 3, 'five_tau': 0}))

#%% [markdown]
# ## 2. Trace plots (all chains saved to `plots/traces/`; baseline fits also shown here)

#%%
def plot_traces(info, show=False):
    h5path, name = info['path'], info['name']
    with h5py.File(h5path, 'r') as f:
        it = info['iterations']
        nd = f['mcmc/chain'].shape[2]
        step = max(1, it // TRACE_POINTS)
        sub = f['mcmc/chain'][:it:step, :N_TRACE_WALKERS, :]
    steps = np.arange(0, it, step)
    names = read_param_names(h5path, nd)
    panels = ['logP'] + names
    ncols = 4
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 1.9 * nrows), sharex=True, squeeze=False)
    tau = info['status'].get('tau_max', np.nan)
    for ax, pname in zip(axes.ravel(), panels):
        if pname == 'logP':
            ax.plot(steps, info['m'][::step][:len(steps)], color='k', lw=0.6, label='walker median')
            ax.plot(steps, info['ms'][::step][:len(steps)], color='tab:red', lw=0.9)
            sc = max(info['scatter'], 1.0)
            ax.set_ylim(max(np.nanpercentile(info['m'], 1), info['level'] - 50 * sc), info['level'] + 6 * sc)
            ax.set_ylabel(r'$\ln P$')
            ax.text(0.02, 0.06, r'$\mathrm{' + name.replace('_', r'\_') + '}$', transform=ax.transAxes, fontsize=6.5)   # cmr10 has no underscore glyph
        else:
            j = names.index(pname)
            ax.plot(steps, sub[:, :, j], lw=0.4, alpha=0.35)
            ax.set_ylabel(pretty_param(pname))
        ax.axvline(info['burn'], color='tab:red', lw=1.2)
        if tau == tau:
            ax.axvline(fl.BURN_TAUS * tau, color='0.5', lw=1.0, ls='--')
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(panels):]:
        ax.axis('off')
    for ax in axes[-1]:
        ax.set_xlabel('step', fontsize=8)
    fig.tight_layout(h_pad=0.3, w_pad=0.5)
    fig.savefig(PLOT_DIR / 'traces' / f'{name}.png', dpi=110)
    if show:
        plt.show()
    plt.close(fig)


for name, info in INFO.items():
    plot_traces(info, show=name in SHOW_TRACES_INLINE)
print(f'trace plots saved in {PLOT_DIR / "traces"}')

#%% [markdown]
# ## 3. Validation plots: MAP model vs data for the baseline joint fits
# Top rows: $s^2\xi_\ell$ with the data (error bars from the fit covariance diagonal) and the maximum-posterior (MAP) model;
# bottom rows: $\Delta/\sigma=(\mathrm{data}-\mathrm{model})/\sigma$ with a $\pm1$ band. For the DR1 fits the model is the windowed prediction, as in the fit.

#%%
def map_theta(info):
    # sample with the highest log-posterior after the burn-in
    with h5py.File(info['path'], 'r') as f:
        lp = f['mcmc/log_prob'][info['burn']:info['iterations']]
        t, w = np.unravel_index(np.nanargmax(np.where(np.isfinite(lp), lp, -np.inf)), lp.shape)
        theta = f['mcmc/chain'][info['burn'] + t, w, :]
    return theta, float(lp[t, w])


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


for case in BASELINE.values():
    for name in case.values():
        validation_figure(name)

#%% [markdown]
# ## 4. Marginalized cosmology posteriors of the baseline fits, with DESI DR1 2024 full-shape contours (black, unfilled)
# Samples are taken after the burn-in found above and thinned by $0.5\tau_{\max}$. The mock case marks the AbacusSummit c000 truth (`TRUTH`).
# Three variants per case: compact + Gaussian + DESI (`_baseline`), compact only (`_baseline_compact_only`), and compact + Gaussian
# without the DESI contours (`_baseline_compact_gauss`).

#%%
PARAMS = ['omch2', 'ombh2', 'H0', 'logA', 'ns']
LABELS = [r'\Omega_c h^2', r'\Omega_b h^2', 'H_0', r'\ln(10^{10}A_s)', 'n_s']

def cosmo_samples(name, label):
    info = INFO[name]
    tau = info['status'].get('tau_max', 1.0)
    thin = max(1, int(THIN_TAUS * tau))
    with h5py.File(info['path'], 'r') as f:
        flat = f['mcmc/chain'][info['burn']:info['iterations']:thin, :, :5].reshape(-1, 5)   # shared cosmology = first 5 joint parameters
    h, ombh2, omch2, logA, ns = flat.T
    return MCSamples(samples=np.column_stack([omch2, ombh2, 100 * h, logA, ns]), names=PARAMS, labels=LABELS,
                     label=label, settings=GETDIST_SETTINGS)


desi = loadMCSamples(DESI_BASE + 'chain', settings=GETDIST_SETTINGS)
missing = [p for p in PARAMS if p not in desi.getParamNames().list()]
assert not missing, f'DESI chain lacks {missing}; available: {desi.getParamNames().list()[:30]}'

BASELINE_VARIANTS = {   # variant: (models shown, include the DESI DR1 2024 contours?)
    'full':          (('compact', 'gauss'), True),
    'compact_only':  (('compact',), False),
    'compact_gauss': (('compact', 'gauss'), False),
}
for case, models in BASELINE.items():
    for variant, (shown, with_desi) in BASELINE_VARIANTS.items():
        samples = [cosmo_samples(models[m], MODEL_LABEL[m]) for m in shown] + ([desi] if with_desi else [])
        colors = [MODEL_COLOR[m] for m in shown] + (['k'] if with_desi else [])
        legend = [MODEL_LABEL[m] for m in shown] + ([r'DESI DR1 2024 full-shape'] if with_desi else [])
        g = plots.get_subplot_plotter()
        g.triangle_plot(samples, params=PARAMS, filled=False, contour_colors=colors,
                        contour_lws=[1.2] * len(shown) + ([1.4] if with_desi else []), legend_labels=legend,
                        markers=TRUTH if case == 'mock' else None)
        suffix = '' if variant == 'full' else f'_{variant}'
        g.export(str(PLOT_DIR / f'posteriors_{case}_baseline{suffix}.pdf'))
        g.export(str(PLOT_DIR / f'posteriors_{case}_baseline{suffix}.png'))
        plt.show()


#%% [markdown]
# ### 4b. Compact DR1 fits and the DESI DR1 2024 chains only
# Unfilled contours: the compact joint fits to DR1 at each s_min (20, 40) with the DESI DR1 2024 full-shape contours in black.
 

#%%
dr1_names = {s: f'compact_dr1_joint_smin{s}' for s in (20,) if f'compact_dr1_joint_smin{s}' in INFO}
if not dr1_names:
    print('no compact DR1 joint chains found')
else:
    dr1_color = {20: 'tab:blue'}
    samples = [cosmo_samples(n, rf'Compact, DR1, $s_{{\rm min}}={s}$') for s, n in dr1_names.items()] + [desi]
    g = plots.get_subplot_plotter()
    g.triangle_plot(samples, params=PARAMS, filled=False,
                    contour_colors=[dr1_color[s] for s in dr1_names] + ['k'], contour_lws=[1.3] * len(dr1_names) + [1.4],
                    legend_labels=[sm.getLabel() for sm in samples[:-1]] + [r'DESI DR1 2024 full-shape'])
    g.export(str(PLOT_DIR / 'posteriors_dr1_compact_vs_desi.pdf'))
    g.export(str(PLOT_DIR / 'posteriors_dr1_compact_vs_desi.png'))
    plt.show()

#%% [markdown]
# ## 5. Compact joint fits at different s_min, and the single-mock MAPs
# Contours (unfilled) compare the compact joint fits at s_min = 20 and 40 for the mean mocks and for DR1, coloured by s_min
# (no DESI contours in this section). On the mock plot, each single-mock joint fit (5 mocks per s_min) is shown as one
# dot at its MAP, coloured by its s_min, with the AbacusSummit c000 truth marked. Tables of posterior means/widths and of the
# single-mock MAP scatter are printed below.

#%%
import re
from matplotlib.colors import to_hex

S_MINS = (20, 25, 30, 35, 40)
SMIN_COLOR = dict(zip(S_MINS, [to_hex(c) for c in ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple')]))   # distinct colours; hex strings for getdist


def cosmo_vector(theta):
    # (omch2, ombh2, H0, logA, ns) from a joint-fit parameter vector
    h, ombh2, omch2, logA, ns = theta[:5]
    return np.array([omch2, ombh2, 100 * h, logA, ns])


# MAP of every single-mock joint fit, grouped by s_min
MOCK_MAPS = {s: [] for s in S_MINS}
for name, info in INFO.items():
    m = re.fullmatch(r'compact_mock(\d+)_joint_smin(\d+)', name)
    if m and int(m.group(2)) in MOCK_MAPS:
        MOCK_MAPS[int(m.group(2))].append(cosmo_vector(map_theta(info)[0]))
print('single-mock MAPs found per s_min: ' + ', '.join(f'{s}: {len(v)}' for s, v in MOCK_MAPS.items()) +
      f'  (total {sum(len(v) for v in MOCK_MAPS.values())})')

COMPARE = {   # case: (chain-name pattern, legend prefix, overlay single-mock MAP dots?)
    'mock': ('compact_mean_joint_smin{}', 'Mean mocks', True),
    'dr1':  ('compact_dr1_joint_smin{}', 'DR1', False),
}
table_rows = []
for case, (pattern, prefix, with_dots) in COMPARE.items():
    chains = {s: pattern.format(s) for s in (20, 40) if pattern.format(s) in INFO}
    if not chains:
        print(f'{case}: no chains found for {pattern}')
        continue
    samples = [cosmo_samples(n, rf'{prefix}, $s_{{\rm min}}={s}$') for s, n in chains.items()]
    for s, smp in zip(chains, samples):
        table_rows.append({'case': case, 's_min': s, **{f'{p} mean': smp.mean(p) for p in PARAMS}, **{f'{p} std': smp.std(p) for p in PARAMS}})
    g = plots.get_subplot_plotter()
    g.triangle_plot(samples, params=PARAMS, filled=False,
                    contour_colors=[SMIN_COLOR[s] for s in chains], contour_lws=[1.4] * len(samples),
                    legend_labels=[sm.getLabel() for sm in samples],
                    markers=TRUTH if case == 'mock' else None)
    if with_dots:
        for i in range(len(PARAMS)):
            for j in range(i):
                ax = g.subplots[i, j]          # x = PARAMS[j], y = PARAMS[i]
                for s in S_MINS:
                    if MOCK_MAPS[s]:
                        v = np.array(MOCK_MAPS[s])
                        ax.scatter(v[:, j], v[:, i], s=16, color=SMIN_COLOR[s], edgecolor='k', linewidths=0.4, zorder=5)
        g.fig.legend(handles=[Line2D([0], [0], marker='o', ls='none', color=SMIN_COLOR[s], markeredgecolor='k', markersize=5,
                                     label=f'{s}') for s in S_MINS],
                     title=r'single-mock MAP, $s_{\rm min}$', loc='upper right', bbox_to_anchor=(0.97, 0.60), fontsize=8,
                     title_fontsize=8, frameon=False)
    g.export(str(PLOT_DIR / f'posteriors_{case}_compact_smin_comparison.pdf'))
    g.export(str(PLOT_DIR / f'posteriors_{case}_compact_smin_comparison.png'))
    plt.show()

display(pd.DataFrame(table_rows).set_index(['case', 's_min']).round(4))
rows = []
for s, v in MOCK_MAPS.items():
    if v:
        v = np.array(v)
        rows.append({'s_min': s, 'n_mocks': len(v), **{f'{p} MAP-truth mean': (v[:, k] - TRUTH[p]).mean() for k, p in enumerate(PARAMS)},
                     **{f'{p} MAP std': v[:, k].std() for k, p in enumerate(PARAMS)}})
if rows:
    display(pd.DataFrame(rows).set_index('s_min').round(4))

#%% [markdown]
# ### 5b. MAP and posterior width of each compact joint fit vs s_min
# One row per cosmology parameter; each point is a chain's MAP with its marginal posterior standard deviation as the error bar.
# Mock figure: squares = mean-of-mocks fits (s_min = 20, 40), circles = single-mock fits (5 mocks per s_min), dashed line = AbacusSummit
# c000 truth. DR1 figure: s_min = 20, 40.

#%%
def joint_map_and_std(name):
    # MAP (highest-posterior sample after burn-in) and marginal std of the shared cosmology (omch2, ombh2, H0, logA, ns)
    info = INFO[name]
    theta, _ = map_theta(info)
    thin = max(1, int(THIN_TAUS * info['status'].get('tau_max', 1.0)))
    with h5py.File(info['path'], 'r') as f:
        flat = f['mcmc/chain'][info['burn']:info['iterations']:thin, :, :5].reshape(-1, 5)
    samples = np.column_stack([flat[:, 2], flat[:, 1], 100 * flat[:, 0], flat[:, 3], flat[:, 4]])
    return cosmo_vector(theta), samples.std(axis=0)


def new_rows_figure(width=6.4, row_height=1.9):
    fig, axes = plt.subplots(len(PARAMS), 1, figsize=(width, row_height * len(PARAMS)), sharex=True)
    for ax, lab in zip(axes, LABELS):
        ax.set_ylabel(f'${lab}$')
    return fig, axes


# ---- mocks
fig, axes = new_rows_figure()
for s in (20, 40):
    n = f'compact_mean_joint_smin{s}'
    if n in INFO:
        mp, sd = joint_map_and_std(n)
        for k, ax in enumerate(axes):
            ax.errorbar(s - 2.4, mp[k], yerr=sd[k], fmt='s', ms=7, mfc=SMIN_COLOR[s], mec='k', ecolor='k', capsize=3, elinewidth=1.2, zorder=4)
for s in S_MINS:
    names = sorted(n for n in INFO if re.fullmatch(rf'compact_mock\d+_joint_smin{s}', n))
    for j, n in enumerate(names):
        mp, sd = joint_map_and_std(n)
        x = s + (j - (len(names) - 1) / 2) * 0.55
        for k, ax in enumerate(axes):
            ax.errorbar(x, mp[k], yerr=sd[k], fmt='o', ms=4.5, color=SMIN_COLOR[s], capsize=2, elinewidth=1.0)
for p, ax in zip(PARAMS, axes):
    ax.axhline(TRUTH[p], ls='--', color='gray', lw=1.0, zorder=1)
    ax.set_xlim(16.5, 42.5)
    ax.set_xticks(S_MINS)
axes[-1].set_xlabel(r'$s_{\rm min}\;[h^{-1}{\rm Mpc}]$')
axes[0].legend(handles=[Line2D([0], [0], marker='s', ls='none', mfc='w', mec='k', ms=7, label='mean of mocks'),
                        Line2D([0], [0], marker='o', ls='none', color='k', ms=4.5, label='single mock')],
               frameon=False, fontsize=8, loc='upper right', ncol=2)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'map_vs_smin_mock.pdf', bbox_inches='tight')
fig.savefig(PLOT_DIR / 'map_vs_smin_mock.png', dpi=150, bbox_inches='tight')
plt.show()

# ---- DR1
fig, axes = new_rows_figure(width=4.2)
for s in (20, 40):
    n = f'compact_dr1_joint_smin{s}'
    if n in INFO:
        mp, sd = joint_map_and_std(n)
        for k, ax in enumerate(axes):
            ax.errorbar(s, mp[k], yerr=sd[k], fmt='o', ms=6, color=SMIN_COLOR[s], capsize=3, elinewidth=1.2)
for ax in axes:
    ax.set_xlim(15, 45)
    ax.set_xticks([20, 40])
axes[-1].set_xlabel(r'$s_{\rm min}\;[h^{-1}{\rm Mpc}]$')
fig.tight_layout()
fig.savefig(PLOT_DIR / 'map_vs_smin_dr1.pdf', bbox_inches='tight')
fig.savefig(PLOT_DIR / 'map_vs_smin_dr1.png', dpi=150, bbox_inches='tight')
plt.show()

#%% [markdown]
# ## 6. Individual-tracer fits with the joint fit on top
# Filled cosmology contours from the six individual-tracer compact fits (s_min = 20, DESI per-tracer colours), with the joint fit
# on top in black. The mock plot marks the AbacusSummit c000 truth. Individual-tracer fits have 7 parameters
# ($h,\omega_b,\omega_{\rm cdm},\ln A_s,b_1,b_2,n_s$); only the cosmology is shown.

#%%
def indiv_samples(name, label):
    info = INFO[name]
    thin = max(1, int(THIN_TAUS * info['status'].get('tau_max', 1.0)))
    with h5py.File(info['path'], 'r') as f:
        flat = f['mcmc/chain'][info['burn']:info['iterations']:thin, :, :].reshape(-1, 7)    # [h, ombh2, omch2, logA, b1, b2, ns]
    h, ombh2, omch2, logA, b1, b2, ns = flat.T
    return MCSamples(samples=np.column_stack([omch2, ombh2, 100 * h, logA, ns]), names=PARAMS, labels=LABELS,
                     label=label, settings=GETDIST_SETTINGS)


INDIV = {   # case: (individual-fit chain-name pattern, joint chain name)
    'mock': ('compact_mean_indiv_{}_smin20', 'compact_mean_joint_smin20'),
    'dr1':  ('compact_dr1_indiv_{}_smin20',  'compact_dr1_joint_smin20'),
}
for case, (pattern, joint_name) in INDIV.items():
    keys = [k for k in fl.FIT_KEYS if pattern.format(fl.kname(k)) in INFO]
    if not keys or joint_name not in INFO:
        print(f'{case}: missing chains (found {len(keys)} individual fits, joint {"found" if joint_name in INFO else "missing"})')
        continue
    samples = [indiv_samples(pattern.format(fl.kname(k)), TRACER_SHORT[fl.kname(k)]) for k in keys] + [cosmo_samples(joint_name, 'Joint')]
    g = plots.get_subplot_plotter()
    g.triangle_plot(samples, params=PARAMS, filled=True,
                    contour_colors=[TRACER_COLOR[k] for k in keys] + ['k'],
                    legend_labels=[sm.getLabel() for sm in samples],
                    markers=TRUTH if case == 'mock' else None)
    g.export(str(PLOT_DIR / f'posteriors_{case}_individual_vs_joint.pdf'))
    g.export(str(PLOT_DIR / f'posteriors_{case}_individual_vs_joint.png'))
    plt.show()

#%% [markdown]
# ### 6b. MAP and posterior width of the individual-tracer fits vs the joint fit
# Same layout as 5b: one row per cosmology parameter, MAP with the marginal standard deviation as the error bar; tracers in their DESI
# colours, the joint fit in black, the c000 truth as a dashed line on the mock figure. (Uses `new_rows_figure` and `joint_map_and_std` from 5b.)

#%%
def indiv_map_and_std(name):
    info = INFO[name]
    theta, _ = map_theta(info)                       # [h, ombh2, omch2, logA, b1, b2, ns]
    thin = max(1, int(THIN_TAUS * info['status'].get('tau_max', 1.0)))
    with h5py.File(info['path'], 'r') as f:
        flat = f['mcmc/chain'][info['burn']:info['iterations']:thin, :, :].reshape(-1, 7)
    samples = np.column_stack([flat[:, 2], flat[:, 1], 100 * flat[:, 0], flat[:, 3], flat[:, 6]])
    mp = np.array([theta[2], theta[1], 100 * theta[0], theta[3], theta[6]])
    return mp, samples.std(axis=0)


for case, (pattern, joint_name) in INDIV.items():
    keys = [k for k in fl.FIT_KEYS if pattern.format(fl.kname(k)) in INFO]
    if not keys or joint_name not in INFO:
        print(f'{case}: missing chains (found {len(keys)} individual fits, joint {"found" if joint_name in INFO else "missing"})')
        continue
    fig, axes = new_rows_figure(width=6.0)
    for i, k in enumerate(keys):
        mp, sd = indiv_map_and_std(pattern.format(fl.kname(k)))
        for j, ax in enumerate(axes):
            ax.errorbar(i, mp[j], yerr=sd[j], fmt='o', ms=6, color=TRACER_COLOR[k], capsize=3, elinewidth=1.2)
    mp, sd = joint_map_and_std(joint_name)
    for j, ax in enumerate(axes):
        ax.errorbar(len(keys), mp[j], yerr=sd[j], fmt='o', ms=6, color='k', capsize=3, elinewidth=1.2)
        if case == 'mock':
            ax.axhline(TRUTH[PARAMS[j]], ls='--', color='gray', lw=1.0, zorder=1)
        ax.set_xlim(-0.6, len(keys) + 0.6)
    axes[-1].set_xticks(range(len(keys) + 1))
    axes[-1].set_xticklabels([TRACER_SHORT[fl.kname(k)] for k in keys] + ['Joint'])
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f'map_individual_vs_joint_{case}.pdf', bbox_inches='tight')
    fig.savefig(PLOT_DIR / f'map_individual_vs_joint_{case}.png', dpi=150, bbox_inches='tight')
    plt.show()

#%% [markdown]
# ## 7. Bias parameters of the baseline joint compact fits
# One triangle plot per baseline joint compact fit (mean mocks, DR1) of the $b_1$, $b_2$ of the six tracer-bins, grouped by tracer.
# The joint-fit parameter vector is [h, ω_b, ω_cdm, ln A_s, n_s, then (b_1, b_2) for each tracer-bin in `fl.FIT_KEYS` order].

#%%
BIAS_FILLED = False         # set True for filled contours
N_BIAS = 2 * len(TRACER_NAMES)
BIAS_NAMES = [f'{b}_{t}' for t in TRACER_NAMES for b in ('b1', 'b2')]
BIAS_LABELS = [rf'b_{b[1]}^{{\rm {t}}}' for t in TRACER_NAMES for b in ('b1', 'b2')]
BIAS_CASE_LABEL = {'mock': r'Mean mocks, compact, $s_{\rm min}=20$', 'dr1': r'DR1, compact, $s_{\rm min}=20$'}


def bias_samples(name, label):
    info = INFO[name]
    thin = max(1, int(THIN_TAUS * info['status'].get('tau_max', 1.0)))
    with h5py.File(info['path'], 'r') as f:
        flat = f['mcmc/chain'][info['burn']:info['iterations']:thin, :, 5:5 + N_BIAS].reshape(-1, N_BIAS)
    return MCSamples(samples=flat, names=BIAS_NAMES, labels=BIAS_LABELS, label=label, settings=GETDIST_SETTINGS)


for case, models in BASELINE.items():
    name = models['compact']
    if name not in INFO:
        print(f'{case}: chain {name} not found')
        continue
    smp = bias_samples(name, BIAS_CASE_LABEL[case])
    g = plots.get_subplot_plotter(width_inch=11)
    g.triangle_plot([smp], params=BIAS_NAMES, filled=BIAS_FILLED, contour_colors=[MODEL_COLOR['compact']],
                    contour_lws=[1.3], legend_labels=[smp.getLabel()])
    g.export(str(PLOT_DIR / f'bias_{case}_compact_baseline.pdf'))
    g.export(str(PLOT_DIR / f'bias_{case}_compact_baseline.png'))
    plt.show()

#%% [markdown]
# ## 8. Compact fits with each mass scheme (avg_M, low_M, high_M)
# Baseline joint fits (s_min = 20) with the three compact emulator schemes, for the mean mocks and for DR1: shared-cosmology
# contours (unfilled) plus a table of posterior means and widths. `MASS_EXTRA_PARAMS` adds bias parameters to the triangle,
# e.g. `['b2_QSO']` (names are those in `BIAS_NAMES`, section 7).

#%%
MASS_EXTRA_PARAMS = []          # e.g. ['b2_QSO']
MASS_SCHEMES = {                # scheme: (chain-name pattern per case, colour, linestyle, label)
    'avg_M':  ({'mock': 'compact_mean_joint_smin20',       'dr1': 'compact_dr1_joint_smin20'},       'tab:purple', '-',  r'avg $M$'),
    'low_M':  ({'mock': 'compact_lowM_mean_joint_smin20',  'dr1': 'compact_lowM_dr1_joint_smin20'},  'tab:blue',   '--', r'low $M$'),
    'high_M': ({'mock': 'compact_highM_mean_joint_smin20', 'dr1': 'compact_highM_dr1_joint_smin20'}, 'tab:red',   ':',  r'high $M$'),
}


def mass_samples(name, label):
    # shared cosmology (omch2, ombh2, H0, logA, ns) + optional bias parameters from a joint chain
    info = INFO[name]
    thin = max(1, int(THIN_TAUS * info['status'].get('tau_max', 1.0)))
    extra_idx = [5 + BIAS_NAMES.index(p) for p in MASS_EXTRA_PARAMS]
    with h5py.File(info['path'], 'r') as f:
        flat = f['mcmc/chain'][info['burn']:info['iterations']:thin, :, :5 + N_BIAS].reshape(-1, 5 + N_BIAS)
    h, ombh2, omch2, logA, ns = flat[:, :5].T
    cols = [omch2, ombh2, 100 * h, logA, ns] + [flat[:, k] for k in extra_idx]
    labs = LABELS + [BIAS_LABELS[BIAS_NAMES.index(p)] for p in MASS_EXTRA_PARAMS]
    return MCSamples(samples=np.column_stack(cols), names=PARAMS + MASS_EXTRA_PARAMS, labels=labs, label=label,
                     settings=GETDIST_SETTINGS)


mass_rows = []
for case in ('mock', 'dr1'):
    schemes = [s for s, (names, *_rest) in MASS_SCHEMES.items() if names[case] in INFO]
    missing = [MASS_SCHEMES[s][0][case] for s in MASS_SCHEMES if s not in schemes]
    if missing:
        print(f'{case}: chains not found (skipped): {missing}')
    if len(schemes) < 2:
        continue
    samples = [mass_samples(MASS_SCHEMES[s][0][case], MASS_SCHEMES[s][3]) for s in schemes]
    for s, smp in zip(schemes, samples):
        mass_rows.append({'case': case, 'scheme': s, **{f'{p} mean': smp.mean(p) for p in PARAMS + MASS_EXTRA_PARAMS},
                          **{f'{p} std': smp.std(p) for p in PARAMS + MASS_EXTRA_PARAMS}})
    g = plots.get_subplot_plotter()
    g.triangle_plot(samples, params=PARAMS + MASS_EXTRA_PARAMS, filled=False,
                    contour_colors=[MASS_SCHEMES[s][1] for s in schemes], contour_ls=[MASS_SCHEMES[s][2] for s in schemes],
                    contour_lws=[1.4] * len(schemes), legend_labels=[sm.getLabel() for sm in samples],
                    markers=TRUTH if case == 'mock' else None)
    g.export(str(PLOT_DIR / f'posteriors_{case}_mass_schemes.pdf'))
    g.export(str(PLOT_DIR / f'posteriors_{case}_mass_schemes.png'))
    plt.show()

if mass_rows:
    display(pd.DataFrame(mass_rows).set_index(['case', 'scheme']).round(4))


#%% [markdown]
# ## 9. Likelihood comparison: compact (skew-t) vs Gaussian
# Baseline joint fits (mean mocks, DR1) at s_min = 20 and 40; both models have the same free parameters, data, covariance and priors, so
# the posterior values are directly comparable. Definitions: $\Delta\ln P=\ln P_{\rm compact}-\ln P_{\rm Gauss}$ and
# $\Delta\chi^2=\chi^2_{\rm Gauss}-\chi^2_{\rm compact}$ (positive favours the compact model), per tracer and in total, at each chain's MAP
# (its highest-posterior sample). The MAP is a noisy statistic (the best sample sits a few ln P units below the true maximum), so the
# table also gives the posterior-mean $\langle\ln P\rangle$, which has much smaller sampling error.
 
#%%
LLH_SMINS = (20, 40)
SMIN_COLOR = globals().get('SMIN_COLOR') or {20: 'tab:blue', 40: 'tab:purple'}     # same colours as section 5
 
 
def llh_chain(model, case, s):
    return f'{model}_{"mean" if case == "mock" else "dr1"}_joint_smin{s}'
 
 
def map_chi2(name):
    # (ln P at the MAP, [(tracer, n_data, chi2), ...], posterior-mean ln P) for one baseline joint chain
    info = INFO[name]
    spec = next(sp for sp in fl.build_specs() if sp['name'] == name)
    problem = fl.FitProblem(spec)
    theta, lp = map_theta(info)
    h, ombh2, omch2, lnAs, ns, biases = problem.unpack(theta)
    aps = fl.ap_multi(h, ombh2, omch2, ns, problem.z_effs)
    rows = []
    for key, d, (b1, b2), (qp, ql) in zip(spec['keys'], problem.data, biases, aps):
        model = fl.model_one(d['emus'], [b1, b2, ombh2, omch2, h, lnAs, ns], d['s_data'], d['W'], qp, ql)
        resid = d['data_vec'] - model
        rows.append((TRACER_SHORT[fl.kname(key)], resid.size, float(resid @ d['cov_inv'] @ resid)))
    with h5py.File(info['path'], 'r') as f:
        lp_all = f['mcmc/log_prob'][info['burn']:info['iterations']]
    return lp, rows, float(np.mean(lp_all[np.isfinite(lp_all)]))
 
 
llh = {}      # (case, s_min) -> {'compact': (lnP, rows, mean lnP), 'gauss': ...}
for case in ('mock', 'dr1'):
    for s in LLH_SMINS:
        names = {m: llh_chain(m, case, s) for m in ('compact', 'gauss')}
        if all(n in INFO for n in names.values()):
            llh[(case, s)] = {m: map_chi2(n) for m, n in names.items()}
        else:
            print(f'{case}, s_min={s}: chains missing: {[n for n in names.values() if n not in INFO]}')
 
table = []
for (case, s), r in llh.items():
    (lc, rows_c, mc), (lg, rows_g, mg) = r['compact'], r['gauss']
    chi2_c, chi2_g = sum(x[2] for x in rows_c), sum(x[2] for x in rows_g)
    table.append({'case': case, 's_min': s, 'n_data': sum(x[1] for x in rows_c),
                  'lnP compact (MAP)': lc, 'lnP Gauss (MAP)': lg, 'Delta lnP (MAP)': lc - lg,
                  '<lnP> compact': mc, '<lnP> Gauss': mg, 'Delta <lnP>': mc - mg,
                  'chi2 compact': chi2_c, 'chi2 Gauss': chi2_g, 'Delta chi2 (Gauss - compact)': chi2_g - chi2_c})
if table:
    display(pd.DataFrame(table).set_index(['case', 's_min']).round(2))
    tr_rows = []
    for (case, s), r in llh.items():
        for (t, n, c_c), (_, _, c_g) in zip(r['compact'][1], r['gauss'][1]):
            tr_rows.append({'case': case, 's_min': s, 'tracer': t, 'n_data': n, 'chi2 compact': c_c, 'chi2 Gauss': c_g, 'Delta chi2': c_g - c_c})
    display(pd.DataFrame(tr_rows).set_index(['case', 's_min', 'tracer']).round(2))
 
    # per-tracer and total Delta chi^2 (Gauss - compact): positive = compact fits better
    cases = [c for c in ('mock', 'dr1') if any(k[0] == c for k in llh)]
    fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 3.4), squeeze=False)
    labels_x = TRACER_NAMES + ['Total']
    for ax, case in zip(axes[0], cases):
        smins = [s for s in LLH_SMINS if (case, s) in llh]
        width = 0.8 / len(smins)
        for k, s in enumerate(smins):
            r = llh[(case, s)]
            d = [g[2] - c[2] for c, g in zip(r['compact'][1], r['gauss'][1])]
            vals = d + [sum(d)]
            ax.bar(np.arange(len(vals)) + (k - (len(smins) - 1) / 2) * width, vals, width * 0.92, color=SMIN_COLOR[s],
                   label=rf'$s_{{\rm min}}={s}$')
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xticks(range(len(labels_x)))
        ax.set_xticklabels(labels_x, fontsize=8)
        ax.set_ylabel(r'$\Delta\chi^2=\chi^2_{\rm Gauss}-\chi^2_{\rm compact}$')
        ax.text(0.03, 0.95, 'Mean mocks' if case == 'mock' else 'DR1', transform=ax.transAxes, va='top', fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc='upper right')
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'llh_compact_vs_gauss.pdf', bbox_inches='tight')
    fig.savefig(PLOT_DIR / 'llh_compact_vs_gauss.png', dpi=150, bbox_inches='tight')
    plt.show()


 #%% [markdown]
# ## 10. χ² of the baseline DR1 fits (s_min = 20)
# χ² is evaluated at each chain's MAP (its highest-posterior sample) against the DR1 data vector and covariance used in the fit (windowed model),
# per tracer and in total, for the joint fits (compact and Gaussian) and the six individual-tracer compact fits. Degrees of freedom are
# N_data − N_parameters (17 for a joint fit, 7 for an individual one); they are approximate because ω_b and n_s are prior-dominated and the model is
# nonlinear. The last table compares the joint fit with the individual fits tracer by tracer: Σ[χ²_joint − χ²_indiv] over the six tracers, against the
# 6×7 − 17 = 25 extra parameters of the individual fits, tests whether the tracers are consistent with one cosmology.
 
#%%
from scipy.stats import chi2 as chi2_dist
 
DR1_JOINT = {'compact': 'compact_dr1_joint_smin20', 'gauss': 'gauss_dr1_joint_smin20'}
DR1_INDIV = 'compact_dr1_indiv_{}_smin20'
 
 
def chi2_by_tracer(name):
    # (rows, n_params, ln P at the MAP); rows = [(tracer, n_data, chi2, max |pull|), ...]
    info = INFO[name]
    spec = next(sp for sp in fl.build_specs() if sp['name'] == name)
    problem = fl.FitProblem(spec)
    theta, lp = map_theta(info)
    h, ombh2, omch2, lnAs, ns, biases = problem.unpack(theta)
    aps = fl.ap_multi(h, ombh2, omch2, ns, problem.z_effs)
    rows = []
    for key, d, (b1, b2), (qp, ql) in zip(spec['keys'], problem.data, biases, aps):
        model = fl.model_one(d['emus'], [b1, b2, ombh2, omch2, h, lnAs, ns], d['s_data'], d['W'], qp, ql)
        resid = d['data_vec'] - model
        sig = np.sqrt(np.diag(np.linalg.inv(d['cov_inv'])))
        rows.append((TRACER_SHORT[fl.kname(key)], resid.size, float(resid @ d['cov_inv'] @ resid), float(np.max(np.abs(resid) / sig))))
    return rows, problem.ndim, lp
 
 
def summarize(rows, ndim):
    n, chi2 = sum(r[1] for r in rows), sum(r[2] for r in rows)
    return n, chi2, n - ndim, chi2 / (n - ndim), chi2_dist.sf(chi2, n - ndim)
 
 
joint_res = {m: chi2_by_tracer(n) for m, n in DR1_JOINT.items() if n in INFO}
indiv_res = {k: chi2_by_tracer(DR1_INDIV.format(fl.kname(k))) for k in fl.FIT_KEYS if DR1_INDIV.format(fl.kname(k)) in INFO}
 
# ---- joint fits
tab = []
for m, (rows, ndim, lp) in joint_res.items():
    for t, n, c, mp in rows:
        dof_t = n - ndim / len(rows)          # the joint parameters are split evenly across tracers (sums to N - ndim)
        tab.append({'fit': f'joint {m}', 'tracer': t, 'n_data': n, 'chi2': c, 'chi2/n': c / n, 'max |pull|': mp,
                    'dof': dof_t, 'chi2/dof': c / dof_t})
    n, c, dof, red, p = summarize(rows, ndim)
    tab.append({'fit': f'joint {m}', 'tracer': 'TOTAL', 'n_data': n, 'chi2': c, 'chi2/n': c / n, 'dof': dof, 'chi2/dof': red, 'p-value': p})
if tab:
    display(pd.DataFrame(tab).set_index(['fit', 'tracer']).round(3))
 
# ---- individual fits, with the same tracer's joint-fit chi2 alongside
if indiv_res:
    tab, tot_ind, tot_joint, k_ind = [], 0.0, 0.0, 0
    for key, (rows, ndim, lp) in indiv_res.items():
        t, n, c, mp = rows[0]
        dof = n - ndim
        row = {'tracer': t, 'n_data': n, 'chi2 indiv': c, 'dof': dof, 'chi2/dof': c / dof, 'p-value': chi2_dist.sf(c, dof), 'max |pull|': mp}
        if 'compact' in joint_res:
            cj = next(r[2] for r in joint_res['compact'][0] if r[0] == t)
            row.update({'chi2 joint': cj, 'Delta chi2 (joint - indiv)': cj - c})
            tot_joint += cj
        tot_ind += c
        k_ind += ndim
        tab.append(row)
    display(pd.DataFrame(tab).set_index('tracer').round(3))
    if 'compact' in joint_res and len(indiv_res) == len(fl.FIT_KEYS):
        d_chi2, d_dof = tot_joint - tot_ind, k_ind - joint_res['compact'][1]
        print(f'joint vs individual (compact): sum Delta chi2 = {d_chi2:.1f} for {d_dof} extra parameters in the individual fits; '
              f'p = {chi2_dist.sf(d_chi2, d_dof):.3f}   (small p = tracers pull the shared cosmology in different directions)')
 
# ---- figure: reduced chi2 (chi2 / dof) per tracer and in total. Joint fits: per-tracer dof = n_data - N_params/6 (the 17 parameters split
# evenly across the six tracers, so the dofs sum to N - 17); individual fits: n_data - 7. Error bars = expected 1-sigma scatter, sqrt(2/dof).
if joint_res or indiv_res:
    labels_x = TRACER_NAMES + ['Total']
    series = []
    for m, (rows, ndim, _) in joint_res.items():
        vals, errs = {}, {}
        for t, n, c, _mp in rows:
            dof_t = n - ndim / len(rows)
            vals[t], errs[t] = c / dof_t, np.sqrt(2 / dof_t)
        n_tot, c_tot, dof_tot, red_tot, _p = summarize(rows, ndim)
        vals['Total'], errs['Total'] = red_tot, np.sqrt(2 / dof_tot)
        series.append((f'Joint {m}', vals, errs, MODEL_COLOR[m]))
    if indiv_res:
        vals, errs, c_sum, dof_sum = {}, {}, 0.0, 0
        for rows, ndim, _ in indiv_res.values():
            t, n, c, _mp = rows[0]
            vals[t], errs[t] = c / (n - ndim), np.sqrt(2 / (n - ndim))
            c_sum += c
            dof_sum += n - ndim
        vals['Total'], errs['Total'] = c_sum / dof_sum, np.sqrt(2 / dof_sum)
        series.append(('Individual (compact)', vals, errs, '0.55'))
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    width = 0.8 / len(series)
    for k, (lab, vals, errs, col) in enumerate(series):
        x = np.arange(len(labels_x)) + (k - (len(series) - 1) / 2) * width
        ax.bar(x, [vals.get(t, np.nan) for t in labels_x], width * 0.92, color=col, label=lab,
               yerr=[errs.get(t, np.nan) for t in labels_x], error_kw=dict(ecolor='k', elinewidth=0.8, capsize=1.5))
    ax.axhline(1.0, color='k', lw=0.8, ls='--')
    ax.axvline(len(TRACER_NAMES) - 0.5, color='0.7', lw=0.8)
    ax.set_xticks(range(len(labels_x)))
    ax.set_xticklabels(labels_x)
    ax.set_ylabel(r'reduced $\chi^2=\chi^2/{\rm dof}$')
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'chi2_dr1_baseline.pdf', bbox_inches='tight')
    fig.savefig(PLOT_DIR / 'chi2_dr1_baseline.png', dpi=150, bbox_inches='tight')
    plt.show()
 
#%% [markdown]
# ### 10b. Reduced χ² of the baseline DR1 fits, without error bars
# Same bars as the figure above (uses `joint_res`, `indiv_res` and `summarize` from section 10).
 
#%%
if joint_res or indiv_res:
    labels_x = TRACER_NAMES + ['Total']
    series = []
    for m, (rows, ndim, _) in joint_res.items():
        vals = {t: c / (n - ndim / len(rows)) for t, n, c, _mp in rows}
        vals['Total'] = summarize(rows, ndim)[3]
        series.append((f'Joint {m}', vals, MODEL_COLOR[m]))
    if indiv_res:
        vals = {rows[0][0]: rows[0][2] / (rows[0][1] - ndim) for rows, ndim, _ in indiv_res.values()}
        vals['Total'] = sum(rows[0][2] for rows, _, _ in indiv_res.values()) / sum(rows[0][1] - ndim for rows, ndim, _ in indiv_res.values())
        series.append(('Individual (compact)', vals, '0.55'))
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    width = 0.8 / len(series)
    for k, (lab, vals, col) in enumerate(series):
        ax.bar(np.arange(len(labels_x)) + (k - (len(series) - 1) / 2) * width, [vals.get(t, np.nan) for t in labels_x],
               width * 0.92, color=col, label=lab)
    ax.axhline(1.0, color='k', lw=0.8, ls='--')
    ax.axvline(len(TRACER_NAMES) - 0.5, color='0.7', lw=0.8)
    ax.set_xticks(range(len(labels_x)))
    ax.set_xticklabels(labels_x)
    ax.set_ylabel(r'reduced $\chi^2=\chi^2/{\rm dof}$')
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'chi2_dr1_baseline_noerr.pdf', bbox_inches='tight')
    fig.savefig(PLOT_DIR / 'chi2_dr1_baseline_noerr.png', dpi=150, bbox_inches='tight')
    plt.show()