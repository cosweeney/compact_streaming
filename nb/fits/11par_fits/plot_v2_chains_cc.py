"""
Comparison plots for the AP-fixed refits in fits_11par_apfix_v2 (ChainConsumer >= 1.0), run cell by cell (#%% markers) or as a script, compact environment.
Ours are red (filled where it is the reference chain); everything else is a DESI chain (filled, DESI-paper colours) or a variant fit (outlines). Figures are saved to
CHAIN_ROOT/plots as PNG + PDF; chains that have not finished (no .status.json) are skipped with a message. Burn-in and thinning are the `discard` / `thin` stored in
each chain's status.json, and parameter columns are identified by the `param_names` stored there, so ns-fixed / w0wa layouts need no special handling.

  1. baseline vs DESI FS and BAO        6. single-mock fits per parameter + mean offset     (DR1 and mock-mean baselines, separately)
  2. baseline vs s_min                  7. DR1 baseline shifted by minus the offset in 6, vs the uncorrected one and DESI
  3. baseline vs Gaussian model         8. bias parameters of the DR1 and mock-mean baselines
  4. baseline vs mass schemes           9. w0wa baseline vs LCDM baseline
  5. baseline vs individual tracers    10. w0wa fits at s_min = 20 and 30
Plots 1-5 and 9-10 are made for DR1 and for the mean-of-mocks fits (the latter with the c000 truth marked). Contours of our chains use MODE = 'resample' by default: a
variance-preserving smoothed bootstrap (200k draws with Gaussian jitter of 0.4 x the sample covariance, shrunk to keep the mean and covariance) fed to ChainConsumer's histogram
with light smoothing. On synthetic emcee-like chains this was ~200x faster than ChainConsumer's own KDE (0.02 s vs 4-6 s per panel) and smoother (contour roughness 1.02-1.04 vs
1.2-1.6, where 1 = as smooth as the true Gaussian) with the 68% / 95% areas within 2% / 3% of the truth. MODE = 'kde' uses ChainConsumer's KDE; 'hist' the raw histogram (jagged).
Set PLOTS_TO_RUN to a subset to save time.
"""
#%%
import glob
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from chainconsumer import Chain, ChainConsumer, PlotConfig, Truth

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False})

CHAIN_ROOT = Path('/spiffball/cosweeney/data/chains/fits_11par_apfix_v2')
PLOT_DIR = CHAIN_ROOT / 'plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
DESI_FS = '/spiffball/cosweeney/DESI/DR1/full-shape-cosmo-params/v1.0/cobaya/base/desi-reptvelocileptors-fs-all_schoneberg2024-bbn_planck2018-ns10/'
DESI_BAO = '/spiffball/cosweeney/DESI/DR1/bao-cosmo-params/v1.0/cobaya/base/desi-bao-all_schoneberg2024-bbn/'      # adjust to your local directory name

PLOTS_TO_RUN = set(range(1, 11))
MODE = 'resample'       # contours of our chains: 'resample' (smoothed bootstrap + histogram: fast and smooth), 'kde' (ChainConsumer's KDE: slow), 'hist' (raw histogram: jagged)
RESAMPLE_N = 200_000    # draws of the smoothed bootstrap per chain
RESAMPLE_H = 0.4        # jitter bandwidth in units of the sample covariance (variance-preserving, so it does not broaden the contours)
MAX_KDE = 3000          # 'kde' mode only: random subsample per chain
MAX_KDE_BIAS = 2000     # 'kde' mode only: same for the 12-parameter bias triangle
DESI_SMOOTH = 3         # histogram smoothing (bins) for the huge DESI chains
FILL_VARIANTS = False   # True: fill the variant chains in plots 2-4 and 9-10 too (the baseline is always filled)
SMINS = (20, 25, 30, 35, 40)
W_NU = 0.0006442        # omega_nu of one massive 0.06 eV species, so Omega_m includes neutrinos like CAMB's
DATASETS = {'dr1': 'DR1', 'mean': 'mean of mocks'}

SAMPLED = ['h', 'ombh2', 'omch2', 'logA', 'ns']
PARAMS = ['omegam', 'H0', 'ombh2', 'omch2', 'logA', 'ns']
COMPARE_PARAMS = ['omegam', 'H0', 'omch2', 'logA', 'ns']            # omega_b is prior-dominated: dropped from the multi-chain comparisons to save time
W0WA_PARAMS = ['omegam', 'H0', 'omch2', 'logA', 'ns', 'w0', 'wa']
LABELS = {'omegam': r'$\Omega_m$', 'H0': r'$H_0$', 'ombh2': r'$\omega_b$', 'omch2': r'$\omega_{\rm cdm}$', 'logA': r'$\ln(10^{10}A_s)$', 'ns': r'$n_s$',
          'w0': r'$w_0$', 'wa': r'$w_a$'}
TRUTH_S = {'h': 0.6736, 'ombh2': 0.02237, 'omch2': 0.12, 'logA': float(np.log(1e10 * 2.0830e-9)), 'ns': 0.9649}              # AbacusSummit c000, sampled parameters
TRUTH = {**TRUTH_S, 'omegam': (0.02237 + 0.12 + W_NU) / 0.6736 ** 2, 'H0': 67.36, 'w0': -1.0, 'wa': 0.0}
TRACERS = [('BGS', 'BGS', 'yellowgreen'), ('LRG1', 'LRG_0.4', 'orange'), ('LRG2', 'LRG_0.6', 'orangered'), ('LRG3', 'LRG_0.8', 'firebrick'),
           ('ELG2', 'ELG', 'steelblue'), ('QSO', 'QSO', 'seagreen')]      # (short name, substring of the chain name, DESI colour)
RED, ORANGE, PURPLE = 'red', '#FF8C1A', '#D58BCB'
for s, key, _c in TRACERS:
    LABELS.update({f'b1_{s}': rf'$b_1^{{\rm {s}}}$', f'b2_{s}': rf'$b_2^{{\rm {s}}}$'})

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


def full(name):
    """(DataFrame with every sampled parameter, status dict), post-burn-in and thinned as saved in status.json."""
    if name not in _cache:
        root = CHAIN_ROOT / model_dir(name) / name
        st = json.load(open(f'{root}.status.json'))
        with h5py.File(f'{root}.h5', 'r') as f:
            it = int(f['mcmc'].attrs['iteration'])
            c = f['mcmc/chain'][st['discard']:it:st['thin'], :, :]
        _cache[name] = (pd.DataFrame(c.reshape(-1, c.shape[-1]), columns=st['param_names']), st)
    return _cache[name]


def derive(df):
    out = df.copy()
    out['H0'] = 100 * out['h']
    out['omegam'] = (out['ombh2'] + out['omch2'] + W_NU) / out['h'] ** 2
    return out


def cosmo(name, cap=MAX_KDE):
    """Cosmological columns (+ derived Omega_m, H0) of a chain, randomly subsampled to `cap` rows (None = all)."""
    df, st = full(name)
    d = derive(df[[c for c in SAMPLED + ['w0', 'wa'] if c in df.columns]])
    return d.sample(cap, random_state=0) if MODE == 'kde' and cap and len(d) > cap else d          # the cap only matters for the KDE


def load_cobaya(path, skip=0.3):
    dfs = []
    for fn in sorted(glob.glob(path + 'chain.[0-9]*.txt')) or glob.glob(path + 'chain.txt'):
        names = open(fn).readline().lstrip('#').split()
        d = pd.read_csv(fn, sep=r'\s+', comment='#', header=None, names=names)
        dfs.append(d.iloc[int(skip * len(d)):])
    return pd.concat(dfs, ignore_index=True)


def desi(path):
    try:
        d = load_cobaya(path)
    except (ValueError, FileNotFoundError) as e:
        print(f'DESI chain not found at {path}: {e}')
        return None
    h, ob = d['H0'] / 100, d['ombh2'] if 'ombh2' in d else d['omegabh2']
    c = pd.DataFrame({'H0': d['H0'], 'ombh2': ob})
    c['omch2'] = d['omch2'] if 'omch2' in d else d['omegam'] * h ** 2 - ob - W_NU
    c['omegam'] = d['omegam'] if 'omegam' in d else (ob + c['omch2'] + W_NU) / h ** 2
    for k in ('logA', 'ns'):                                                                   # the BAO chains have no logA / ns
        if k in d:
            c[k] = d[k]
    c['weight'] = d['weight']
    return c[[p for p in PARAMS if p in c] + ['weight']]


def n_eff(name):
    st = full(name)[1]
    return f"{name}: ~{st['nwalkers'] * (st['iterations'] - st['discard']) / st['tau_max']:.0f} independent samples (N/tau_max = {st['n_over_tau_max']:.0f}, converged = {st['converged']})"


#%% [markdown]
# ## Figure helpers

#%%
def smooth_resample(df, n=None, h=None, seed=0):
    """Variance-preserving smoothed bootstrap (Jones 1991): resample rows, add Gaussian jitter with covariance h^2 x the sample covariance, shrink toward the mean by 1/sqrt(1+h^2),
    so the mean and covariance are unchanged. Equivalent to a Gaussian KDE evaluated by sampling; a histogram of the draws is then smooth. Uses an eigendecomposition because
    derived columns (H0 = 100 h) make the covariance singular."""
    n, h = n or RESAMPLE_N, h if h is not None else RESAMPLE_H
    x = df.to_numpy(float)
    m = x.mean(0)
    w, V = np.linalg.eigh(np.cov(x.T))
    A = V * np.sqrt(np.clip(w, 0, None))
    r = np.random.default_rng(seed)
    y = m + ((x[r.integers(0, len(x), n)] - m) + h * r.normal(size=(n, x.shape[1])) @ A.T) / np.sqrt(1 + h ** 2)
    return pd.DataFrame(y, columns=df.columns)


def mk(df, name, color, filled=False, ls='-', z=5, kde=True):
    """ChainConsumer chain. kde=True: one of our chains, drawn according to MODE; kde=False: a huge DESI chain, drawn as a smoothed histogram."""
    if kde and MODE == 'resample':
        df, kw = smooth_resample(df), dict(kde=False, smooth=1)
    elif kde and MODE == 'kde':
        kw = dict(kde=True, smooth=None)
    else:
        kw = dict(kde=False, smooth=DESI_SMOOTH)
    return Chain(samples=df, name=name, color=to_hex(color), linestyle=ls, shade=filled, bar_shade=False, zorder=z, **kw)         # ChainConsumer needs hex colours


def figure(fname, entries, columns, truth=None, size=None):
    """Triangle plot of the (non-None) chains; saves PNG + PDF to PLOT_DIR."""
    entries = [e for e in entries if e is not None]
    if len(entries) < 2 and not fname.startswith('v2_8'):
        print(f'{fname}: nothing to compare, skipped')
        return
    c = ChainConsumer()
    for e in entries:
        c.add_chain(e)
    present = {col for e in entries for col in e.samples.columns}
    columns = [col for col in columns if col in present]
    if truth:
        c.add_truth(Truth(location={k: v for k, v in truth.items() if k in columns}, color='0.4', line_style='--'))
    c.set_plot_config(PlotConfig(labels=LABELS, summarise=False, serif=True, max_ticks=4))
    s = size or 1.55 * len(columns) + 0.5
    fig = c.plotter.plot(columns=columns, figsize=(s, s))
    fig.savefig(PLOT_DIR / f'{fname}.png', dpi=200, bbox_inches='tight')
    fig.savefig(PLOT_DIR / f'{fname}.pdf', bbox_inches='tight')
    plt.show()
    plt.close(fig)


def variant(name, label, color, ls='-', z=5):
    return mk(cosmo(name), label, color, filled=FILL_VARIANTS, ls=ls, z=z) if have(name) else None


def baseline(tag, label=None):
    name = f'compact_{tag}_joint_smin20'
    return mk(cosmo(name), label or f'Baseline, {DATASETS[tag]}', RED, filled=True, z=10) if have(name) else None


TRUTH_FOR = {'dr1': None, 'mean': TRUTH}

#%% [markdown]
# ## 1. Baseline vs DESI FS-only and BAO

#%%
if 1 in PLOTS_TO_RUN:
    fs, bao = desi(DESI_FS), desi(DESI_BAO)
    for tag in DATASETS:
        entries = [mk(bao, 'DESI DR1 BAO', ORANGE, True, z=1, kde=False) if bao is not None else None,
                   mk(fs, 'DESI DR1 FS-only', PURPLE, True, z=2, kde=False) if fs is not None else None, baseline(tag)]
        figure(f'v2_1_baseline_vs_desi_{tag}', entries, PARAMS, TRUTH_FOR[tag])

#%% [markdown]
# ## 2. Baseline vs different s_min

#%%
if 2 in PLOTS_TO_RUN:
    colors = {25: 'tab:orange', 30: 'tab:green', 35: 'tab:blue', 40: 'tab:purple'}
    for tag in DATASETS:
        entries = [baseline(tag, r'$s_{\rm min}=20$ (baseline)')] + [variant(f'compact_{tag}_joint_smin{s}', rf'$s_{{\rm min}}={s}$', colors[s]) for s in SMINS[1:]]
        figure(f'v2_2_smin_{tag}', entries, COMPARE_PARAMS, TRUTH_FOR[tag])

#%% [markdown]
# ## 3. Baseline vs Gaussian model

#%%
if 3 in PLOTS_TO_RUN:
    for tag in DATASETS:
        figure(f'v2_3_gaussian_{tag}', [baseline(tag, 'Compact (11-par), baseline'), variant(f'gauss_{tag}_joint_smin20', 'Gaussian model', 'tab:blue')],
               COMPARE_PARAMS, TRUTH_FOR[tag])

#%% [markdown]
# ## 4. Baseline vs mass schemes

#%%
if 4 in PLOTS_TO_RUN:
    for tag in DATASETS:
        figure(f'v2_4_mass_{tag}', [baseline(tag, r'avg $M$ (baseline)'), variant(f'compact_lowM_{tag}_joint_smin20', r'low $M$', 'tab:blue'),
                                    variant(f'compact_highM_{tag}_joint_smin20', r'high $M$', 'tab:green')], COMPARE_PARAMS, TRUTH_FOR[tag])

#%% [markdown]
# ## 5. Baseline vs individual-tracer fits
# Individual fits are filled in the DESI tracer colours (as in the DESI paper); the joint baseline is red and on top.

#%%
if 5 in PLOTS_TO_RUN:
    for tag in DATASETS:
        files = sorted(Path(p).name[:-len('.status.json')] for p in glob.glob(str(CHAIN_ROOT / 'compact' / f'compact_{tag}_indiv_*_smin20.status.json')))
        entries = []
        for i, (short, key, color) in enumerate(TRACERS):
            match = [f for f in files if f'_indiv_{key}' in f]
            entries.append(mk(cosmo(match[0]), short, color, filled=True, z=1 + i) if match else None)
        figure(f'v2_5_individual_{tag}', entries + [baseline(tag, 'Joint baseline')], COMPARE_PARAMS, TRUTH_FOR[tag])

#%% [markdown]
# ## 6. Single-mock fits: per-parameter scatter and the mean offset
# Posterior mean +- std of each single-mock baseline fit (s_min = 20); the red line and band are the mean of the means +- its standard error, the dashed line is the c000
# truth. `shift` is mean(means) - truth for the sampled parameters (h, omega_b, omega_cdm, ln A_s, n_s); Omega_m and H0 are shown for reference.

#%%
SHIFT, MOCK_TABLE = None, None
if 6 in PLOTS_TO_RUN or 7 in PLOTS_TO_RUN:
    mock_names = sorted(Path(p).name[:-len('.status.json')] for p in glob.glob(str(CHAIN_ROOT / 'compact' / 'compact_mock*_joint_smin20.status.json')))
    means, stds = [], []
    for n in mock_names:
        d = cosmo(n, cap=None)
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
        SHIFT = {p: float(M[p].mean() - TRUTH_S[p]) for p in SAMPLED}                 # added to the baseline samples with a minus sign in plot 7
        print('shift (mean - truth) of the sampled parameters:', {k: round(v, 5) for k, v in SHIFT.items()})

#%%
if 6 in PLOTS_TO_RUN and MOCK_TABLE is not None:
    mean_fit = 'compact_mean_joint_smin20'
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
    for ax, p in zip(axes.ravel(), PARAMS):
        ax.errorbar(np.arange(N), M[p], yerr=S[p], fmt='o', ms=4, color='k', elinewidth=1.0, capsize=0)
        ax.axhline(TRUTH[p], ls='--', color='0.4', lw=1.0)
        r = MOCK_TABLE.loc[p]
        ax.axhline(r['mean of posterior means'], color=RED, lw=1.2)
        ax.axhspan(r['mean of posterior means'] - r['offset s.e.'], r['mean of posterior means'] + r['offset s.e.'], color=RED, alpha=0.25, lw=0)
        if have(mean_fit):
            d = cosmo(mean_fit, cap=None)
            ax.errorbar(N + 0.5, d[p].mean(), yerr=d[p].std(), fmt='D', ms=5, color='tab:blue', elinewidth=1.0, capsize=0)
        ax.set_ylabel(LABELS[p])
        ax.text(0.03, 0.95, rf"offset = ${r['offset']:+.3g}\pm{r['offset s.e.']:.2g}$ ({r['offset [mean posterior std]']:+.2f}$\sigma$)", transform=ax.transAxes, va='top', fontsize=8,
                bbox=dict(facecolor='w', alpha=0.8, lw=0))
    for ax in axes[1]:
        ax.set_xlabel('mock index')
    fig.legend(handles=[plt.Line2D([], [], marker='o', color='k', ls='none', label='single-mock fit'), plt.Line2D([], [], color=RED, label='mean offset $\\pm$ s.e.'),
                        plt.Line2D([], [], color='0.4', ls='--', label='c000 truth'), plt.Line2D([], [], marker='D', color='tab:blue', ls='none', label='mean-of-mocks fit')],
               frameon=False, fontsize=8, loc='lower center', ncol=4)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(PLOT_DIR / 'v2_6_mock_scatter.png', dpi=200, bbox_inches='tight')
    fig.savefig(PLOT_DIR / 'v2_6_mock_scatter.pdf', bbox_inches='tight')
    plt.show()
    plt.close(fig)

#%% [markdown]
# ## 7. Shift-corrected DR1 baseline
# The DR1 baseline samples with the offset from plot 6 subtracted from each sampled parameter (Omega_m and H0 re-derived). Red outline = uncorrected, red fill = corrected.
# The offsets are estimated from a finite number of mocks, so they carry the standard errors in the table above; `SHIFT_APPLY` selects which parameters are shifted.

#%%
SHIFT_APPLY = SAMPLED
if 7 in PLOTS_TO_RUN and SHIFT is not None and have('compact_dr1_joint_smin20'):
    raw = cosmo('compact_dr1_joint_smin20')
    corr = raw[SAMPLED].copy()
    for p in SHIFT_APPLY:
        corr[p] = corr[p] - SHIFT[p]
    corr = derive(corr)
    fs, bao = desi(DESI_FS), desi(DESI_BAO)
    entries = [mk(bao, 'DESI DR1 BAO', ORANGE, True, z=1, kde=False) if bao is not None else None, mk(fs, 'DESI DR1 FS-only', PURPLE, True, z=2, kde=False) if fs is not None else None,
               mk(raw, 'Baseline, DR1 (uncorrected)', RED, filled=False, ls='--', z=8), mk(corr, 'Baseline, DR1 (shift-corrected)', RED, filled=True, z=10)]
    figure('v2_7_shift_corrected', entries, PARAMS)
    print(pd.DataFrame({'uncorrected mean': raw.mean(), 'corrected mean': corr.mean(), 'std': raw.std()}).loc[PARAMS].round(4).to_string())

#%% [markdown]
# ## 8. Bias parameters of the baseline fits
# b1, b2 for the six tracer bins: DR1 baseline (red, filled) and mean-of-mocks baseline (blue outline).

#%%
def bias_df(name, cap=MAX_KDE_BIAS):
    df, st = full(name)
    b1, b2 = [c for c in df.columns if c.startswith('b1_')], [c for c in df.columns if c.startswith('b2_')]
    out = pd.DataFrame({f'{b}_{s}': df[c].to_numpy() for s, c1, c2 in zip([t[0] for t in TRACERS], b1, b2) for b, c in (('b1', c1), ('b2', c2))})
    return out.sample(cap, random_state=0) if MODE == 'kde' and cap and len(out) > cap else out


if 8 in PLOTS_TO_RUN:
    cols = [f'{b}_{t[0]}' for t in TRACERS for b in ('b1', 'b2')]
    entries = [mk(bias_df('compact_mean_joint_smin20'), 'Baseline, mean of mocks', 'tab:blue', filled=FILL_VARIANTS, z=5) if have('compact_mean_joint_smin20') else None,
               mk(bias_df('compact_dr1_joint_smin20'), 'Baseline, DR1', RED, filled=True, z=10) if have('compact_dr1_joint_smin20') else None]
    figure('v2_8_bias', entries, cols, size=15)

#%% [markdown]
# ## 9. w0wa baseline vs LCDM baseline
# w0wa baseline (s_min = 20, red, filled) against the LCDM baseline (outline) in the shared parameters plus w0, wa; the c000 truth (w0, wa) = (-1, 0) is marked for mocks.
# w0 and wa are limited to the NN training box by the prior (w0 in [-1.271, -0.726], wa in [-0.628, 0.621]).

#%%
if 9 in PLOTS_TO_RUN:
    for tag in DATASETS:
        name = f'compact_w0wa_{tag}_joint_smin20'
        figure(f'v2_9_w0wa_{tag}', [variant(f'compact_{tag}_joint_smin20', 'LCDM baseline', 'tab:blue', z=5),
                                    mk(cosmo(name), 'w0waCDM baseline', RED, filled=True, z=10) if have(name) else None], W0WA_PARAMS, TRUTH_FOR[tag])

#%% [markdown]
# ## 10. w0wa fits at s_min = 20 and 30

#%%
if 10 in PLOTS_TO_RUN:
    for tag in DATASETS:
        a, b = f'compact_w0wa_{tag}_joint_smin20', f'compact_w0wa_{tag}_joint_smin30'
        figure(f'v2_10_w0wa_smin_{tag}', [variant(b, r'w0waCDM, $s_{\rm min}=30$', 'tab:blue', z=5),
                                          mk(cosmo(a), r'w0waCDM, $s_{\rm min}=20$ (baseline)', RED, filled=True, z=10) if have(a) else None], W0WA_PARAMS, TRUTH_FOR[tag])