"""
theta_* check: DESI DR1 2024 full-shape-only chain vs our DR1 joint fits (s_min = 20).

Run cell by cell (#%% markers) or as a script, in the compact environment (needs camb, getdist, h5py, joblib).
Converted-notebook style, same conventions as inspect_chains_11par.py.
"""

#%% [markdown]
# # theta_* conditional-Gaussian check
# For each chain we compute 100 theta_* with CAMB for a subsample of the posterior, form
# Q = (omega_b, n_s, ln 10^10 A_s, X, 100 theta_*) with X = h or omega_m (two variants, since theta_* replaces one of
# them), and take its mean mu and 5x5 covariance C. For the DESI FS-only chain we then condition on theta_* = t0:
#   mu_c = mu_a + C_at (t0 - mu_t) / C_tt,    C_c = C_aa - C_at C_at^T / C_tt,
# with a = (omega_b, n_s, ln A_s, X) and t = theta_*. These are compared with the marginal DESI posterior and with our own
# posteriors. If our widths match DESI | theta_* rather than DESI, our fits behave as if theta_* were pinned.
#
# Notes / assumptions
#  * omega_m = omega_b + omega_cdm (no neutrino term; a constant offset does not change C or the conditional).
#  * theta_* is CAMB's 'thetastar' (100 theta_*), flat LCDM, mnu = 0.06 eV, for every chain, so the offset between CAMB and
#    CLASS conventions cancels. The conditioning value t0 is the mean theta_* of our compact chain (set COND_THETA to a
#    number to override).
#  * The conditioning is Gaussian: DESI posteriors truncated by the flat h / omega_cdm bounds are not exactly Gaussian.
#  * Our chains: post-burn samples with the `discard`/`thin` stored in the chain's .status.json (5 tau_max, 0.5 tau_max),
#    first 5 joint parameters (h, ombh2, omch2, logA, ns). DESI chain: loaded exactly as in inspect_chains_11par.py.
#  * Priors on omega_b and n_s should match between the two analyses (BBN + ns10 for DESI); check this before interpreting widths.
#  * theta_* is cached to CACHE_DIR; with LOAD_CACHE = True the CAMB step is skipped. Delete the cache or set it False to redo.

#%%
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):   # CAMB must stay 1-thread under joblib
    os.environ[_k] = '1'
import json
import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import camb
import psutil
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from getdist import MCSamples, loadMCSamples, plots
try:
    from IPython.display import display
except ImportError:
    display = print

try:
    psutil.Process().nice(10)
except Exception:
    pass

warnings.filterwarnings('ignore', message='.*tight_layout.*')
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
                     'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False,
                     'axes.xmargin': 0, 'figure.dpi': 110})

# ---------------- configuration ----------------
CHAIN_ROOT = Path('/spiffball/cosweeney/data/chains/fits_11par')
PLOT_DIR = CHAIN_ROOT / 'plots' / 'theta_star'
CACHE_DIR = PLOT_DIR / 'cache'
for _d in (PLOT_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
DESI_BASE = '/spiffball/cosweeney/DESI/DR1/full-shape-cosmo-params/v1.0/cobaya/base/desi-reptvelocileptors-fs-all_schoneberg2024-bbn_planck2018-ns10/'

OURS = {   # label: (model directory, chain name, colour)
    'Compact (11-par)': ('compact', 'compact_dr1_joint_smin20', 'tab:blue'),
}
COND_THETA = 'ours'       # 'ours' = mean 100 theta_* of our chain, or a float
N_SAMPLES = 5000          # posterior samples per chain pushed through CAMB
N_JOBS = 16
SEED = 0
LOAD_CACHE = True
N_DRAW = 200_000          # draws from the conditional Gaussian for the triangle plot
GETDIST_SETTINGS = {'smooth_scale_1D': 0.5, 'smooth_scale_2D': 0.5}
TRUTH_C000 = dict(ombh2=0.02237, omch2=0.12, h=0.6736)    # AbacusSummit c000
ABACUS_CSV = Path('cosmologies.csv')    # local copy of the AbacusSummit table (e.g. nb/calibration/runs/auto_runs/cosmologies.csv); the URL below is used if it is missing
ABACUS_CSV_URL = 'https://raw.githubusercontent.com/abacusorg/AbacusSummit/main/Cosmologies/cosmologies.csv'
GRID_IDS = range(130, 182)              # emulator-grid sims c130-c181 (52 rows in the table; edit to drop any)

BASE_NAMES = ['ombh2', 'ns', 'logA']
BASE_LABELS = [r'\omega_b', 'n_s', r'\ln(10^{10}A_s)']
X_VARIANTS = {            # variant: (getdist label, function of the sample dict)
    'h':      ('h', lambda s: s['h']),
    'omegam': (r'\omega_m', lambda s: s['ombh2'] + s['omch2']),
}

#%% [markdown]
# ## 1. theta_* for the posterior samples

#%%
def theta100(ombh2, omch2, h):
    """CAMB 'thetastar' = 100 theta_*, flat LCDM, one massive neutrino (0.06 eV)."""
    try:
        pars = camb.set_params(H0=100 * h, ombh2=ombh2, omch2=omch2, mnu=0.06, omk=0)
        return camb.get_background(pars).get_derived_params()['thetastar']
    except Exception:
        return np.nan


def load_desi_columns():
    smp = loadMCSamples(DESI_BASE + 'chain', settings=GETDIST_SETTINGS)
    names = smp.getParamNames().list()
    missing = [p for p in ('ombh2', 'omch2', 'H0', 'ns', 'logA') if p not in names]
    assert not missing, f'DESI chain lacks {missing}; available: {names[:30]}'
    col = lambda n: smp.samples[:, names.index(n)]
    return dict(ombh2=col('ombh2'), omch2=col('omch2'), h=col('H0') / 100, ns=col('ns'), logA=col('logA')), smp.weights


def load_our_columns(model, name):
    st = json.load(open(CHAIN_ROOT / model / f'{name}.status.json'))
    with h5py.File(CHAIN_ROOT / model / f'{name}.h5', 'r') as f:
        it = int(f['mcmc'].attrs['iteration'])
        flat = f['mcmc/chain'][st['discard']:it:st['thin'], :, :5].reshape(-1, 5)    # shared cosmology = first 5 joint parameters
    h, ombh2, omch2, logA, ns = flat.T
    return dict(ombh2=ombh2, omch2=omch2, h=h, ns=ns, logA=logA), None


def with_theta(tag, loader):
    """Subsample a chain (weighted resample if it has weights), add 100 theta_*, cache to disk."""
    path = CACHE_DIR / f'{tag}_N{N_SAMPLES}.npz'
    if LOAD_CACHE and path.exists():
        return dict(np.load(path))
    cols, w = loader()
    n = len(cols['h'])
    rng = np.random.default_rng(SEED)
    idx = (rng.choice(n, N_SAMPLES, replace=True, p=w / w.sum()) if w is not None
           else rng.choice(n, min(N_SAMPLES, n), replace=False))
    sub = {k: v[idx] for k, v in cols.items()}
    sub['theta'] = np.array(Parallel(n_jobs=N_JOBS)(
        delayed(theta100)(sub['ombh2'][i], sub['omch2'][i], sub['h'][i]) for i in range(len(idx))))
    ok = np.isfinite(sub['theta'])
    if not ok.all():
        print(f'{tag}: dropped {(~ok).sum()} samples with failed CAMB calls')
    sub = {k: v[ok] for k, v in sub.items()}
    np.savez(path, **sub)
    return sub


DESI_S = with_theta('desi_fs', load_desi_columns)
OUR_S = {lab: with_theta(f'ours_{name}', lambda m=model, n=name: load_our_columns(m, n))
         for lab, (model, name, _c) in OURS.items()}
T_C000 = theta100(TRUTH_C000['ombh2'], TRUTH_C000['omch2'], TRUTH_C000['h'])
T_COND = float(OUR_S[next(iter(OURS))]['theta'].mean()) if COND_THETA == 'ours' else float(COND_THETA)
print(f'100 theta_*: DESI {DESI_S["theta"].mean():.5f} +- {DESI_S["theta"].std():.5f};  '
      + ';  '.join(f'{lab} {s["theta"].mean():.5f} +- {s["theta"].std():.5f}' for lab, s in OUR_S.items())
      + f';  c000 (CAMB) {T_C000:.5f};  conditioning value t0 = {T_COND:.5f} '
      f'({(T_COND - DESI_S["theta"].mean()) / DESI_S["theta"].std():+.2f} sigma from the DESI mean)')

# ---- Abacus emulator-grid sims: 100 theta_* from CAMB with each sim's own w0, wa, N_ur, n_s (Abacus neutrino convention: massless N_ur + one massive species)
def load_abacus_grid():
    src = ABACUS_CSV if Path(ABACUS_CSV).exists() else ABACUS_CSV_URL
    df = pd.read_csv(src, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df['n'] = df['root'].str.strip().str.extract(r'(\d+)$')[0].astype(int)
    df = df[df['n'].isin(GRID_IDS)].copy()
    df['omega_ncdm'] = pd.to_numeric(df['omega_ncdm'].astype(str).str.strip(), errors='coerce')
    return df.reset_index(drop=True)


def theta100_abacus(r):
    """CAMB 100 theta_* for one row of the Abacus table (w0wa PPF dark energy, N_ur massless + N_ncdm massive species with omega_ncdm)."""
    try:
        cp = camb.set_params(H0=100 * r.h, ombh2=r.omega_b, omch2=r.omega_cdm, omnuh2=r.omega_ncdm, w=r.w0_fld, wa=r.wa_fld, ns=r.n_s, As=r.A_s,
                             num_nu_massless=r.N_ur, num_nu_massive=int(r.N_ncdm), WantTransfer=False, WantDerivedParameters=True,
                             dark_energy_model='ppf')
        return camb.get_results(cp).get_derived_params()['thetastar']
    except Exception:
        return np.nan


GRID_CACHE = CACHE_DIR / f'abacus_grid_theta_c{GRID_IDS[0]}-c{GRID_IDS[-1]}_n{len(GRID_IDS)}.npz'
if LOAD_CACHE and GRID_CACHE.exists():
    GRID = dict(np.load(GRID_CACHE))
else:
    _g = load_abacus_grid()
    GRID = dict(root=_g['root'].str.strip().to_numpy(dtype=str), h=_g['h'].to_numpy(),
                theta=np.array(Parallel(n_jobs=N_JOBS)(delayed(theta100_abacus)(r) for r in _g.itertuples())))
    _ok = np.isfinite(GRID['theta'])
    GRID = {k: v[_ok] for k, v in GRID.items()}
    np.savez(GRID_CACHE, **GRID)
T_C000_ABACUS = theta100_abacus(pd.Series(dict(h=0.6736, omega_b=0.02237, omega_cdm=0.12, omega_ncdm=0.0006442, w0_fld=-1.0, wa_fld=0.0,
                                               n_s=0.9649, A_s=2.083e-9, N_ur=2.0328, N_ncdm=1)))
print(f'Abacus grid ({len(GRID["theta"])} sims): 100 theta_* = {GRID["theta"].mean():.5f} +- {GRID["theta"].std():.5f} '
      f'(range {GRID["theta"].min():.5f}-{GRID["theta"].max():.5f}), h range {GRID["h"].min():.4f}-{GRID["h"].max():.4f}; '
      f'c000 in the same convention {T_C000_ABACUS:.5f} (chain convention {T_C000:.5f})')

#%% [markdown]
# ## 2. Mean and covariance of Q, and the conditional Gaussian

#%%
def Q_matrix(s, xkey):
    return np.column_stack([s['ombh2'], s['ns'], s['logA'], X_VARIANTS[xkey][1](s), s['theta']])


def condition(mu, C, t0):
    """Gaussian conditional of a = Q[:-1] given t = Q[-1] = t0."""
    mu_a, mu_t = mu[:-1], mu[-1]
    C_aa, C_at, C_tt = C[:-1, :-1], C[:-1, -1], C[-1, -1]
    return mu_a + C_at * (t0 - mu_t) / C_tt, C_aa - np.outer(C_at, C_at) / C_tt


RES = {}
for xkey, (xlabel, _f) in X_VARIANTS.items():
    qnames = BASE_NAMES + [xkey, 'theta']
    Qd = Q_matrix(DESI_S, xkey)
    mu_d, C_d = Qd.mean(0), np.cov(Qd.T)
    mu_c, C_c = condition(mu_d, C_d, T_COND)
    ours = {}
    for lab, s in OUR_S.items():
        Qo = Q_matrix(s, xkey)
        ours[lab] = (Qo.mean(0), np.cov(Qo.T), Qo)
    RES[xkey] = dict(qnames=qnames, Qd=Qd, mu_d=mu_d, C_d=C_d, mu_c=mu_c, C_c=C_c, ours=ours)
    print(f'\n=== X = {xkey} ===  DESI mean of Q: {np.round(mu_d, 5)}')
    sd = np.sqrt(np.diag(C_d))
    display(pd.DataFrame(C_d / np.outer(sd, sd), index=qnames, columns=qnames).round(3))   # DESI correlation matrix of Q

#%% [markdown]
# ## 3. Table: widths and means, ours vs DESI vs DESI | theta_*
# `ours/DESI std` and `ours/(DESI|theta*) std` are posterior-width ratios; `pull` is (ours - DESI|theta*) mean over the conditional std.

#%%
rows = []
for xkey, r in RES.items():
    names4 = BASE_NAMES + [xkey]
    for lab, (mu_o, C_o, _Q) in r['ours'].items():
        for k, p in enumerate(names4 + ['theta']):
            sd_d, sd_o = np.sqrt(r['C_d'][k, k]), np.sqrt(C_o[k, k])
            is_t = p == 'theta'
            sd_c = np.nan if is_t else np.sqrt(r['C_c'][k, k])
            mu_c = np.nan if is_t else r['mu_c'][k]
            rows.append({'X': xkey, 'model': lab, 'parameter': p,
                         'DESI mean': r['mu_d'][k], 'DESI std': sd_d, 'DESI|theta* mean': mu_c, 'DESI|theta* std': sd_c,
                         'ours mean': mu_o[k], 'ours std': sd_o,
                         'ours/DESI std': sd_o / sd_d, 'ours/(DESI|theta*) std': sd_o / sd_c,
                         'pull': (mu_o[k] - mu_c) / sd_c})
TABLE = pd.DataFrame(rows).set_index(['X', 'model', 'parameter'])
TABLE.to_csv(PLOT_DIR / 'theta_star_conditional_table.csv')
display(TABLE.round(4))

#%% [markdown]
# ## 4. Triangle plots: DESI, DESI | theta_* (dashed), and our fits
# One figure per variant (X = h or omega_m). The dashed contours are the conditional Gaussian N(mu_c, C_c) at
# theta_* = t0, drawn as samples.

#%%
rng = np.random.default_rng(SEED + 1)
for xkey, (xlabel, _f) in X_VARIANTS.items():
    r = RES[xkey]
    names4, labels4 = BASE_NAMES + [xkey], BASE_LABELS + [xlabel]
    mk = lambda a, lab: MCSamples(samples=a, names=names4, labels=labels4, label=lab, settings=GETDIST_SETTINGS)
    sets = [mk(r['Qd'][:, :4], 'DESI DR1 FS-only'),
            mk(rng.multivariate_normal(r['mu_c'], r['C_c'], N_DRAW), rf'DESI | $100\theta_*={T_COND:.4f}$')]
    sets += [mk(Q[:, :4], lab) for lab, (_m, _C, Q) in r['ours'].items()]
    colors = ['k', 'tab:red'] + [c for (_m, _n, c) in OURS.values()]
    g = plots.get_subplot_plotter()
    g.triangle_plot(sets, params=names4, filled=False, contour_colors=colors,
                    contour_ls=['-', '--'] + ['-'] * len(OURS), contour_lws=[1.4, 1.4] + [1.4] * len(OURS),
                    legend_labels=[s.getLabel() for s in sets])
    g.export(str(PLOT_DIR / f'theta_star_triangle_{xkey}.pdf'))
    g.export(str(PLOT_DIR / f'theta_star_triangle_{xkey}.png'))
    plt.show()


#%% [markdown]
# ## 5. theta_* posteriors
# Each histogram is normalized to unit peak, so the very narrow Abacus-grid distribution (grey) can share the axis with the posteriors.
 
#%%
def peak_normalized(ax, x, bins, **kw):
    c, e = np.histogram(x, bins=bins)
    ax.stairs(c / c.max(), e, **kw)
 
 
fig, ax = plt.subplots(figsize=(5.0, 3.4))
bins = np.linspace(min(DESI_S['theta'].min(), *(s['theta'].min() for s in OUR_S.values())),
                   max(DESI_S['theta'].max(), *(s['theta'].max() for s in OUR_S.values())), 50)
peak_normalized(ax, DESI_S['theta'], bins, color='k', lw=1.4, label='DESI DR1 FS-only')
for lab, s in OUR_S.items():
    peak_normalized(ax, s['theta'], bins, color=OURS[lab][2], lw=1.4, label=lab)
peak_normalized(ax, GRID['theta'], np.linspace(GRID['theta'].min(), GRID['theta'].max(), 13), color='0.4', fill=True, alpha=0.6,
                label=rf'Abacus grid ($n={len(GRID["theta"])}$)')
ax.axvline(T_COND, color='tab:red', ls=':', lw=1.2, label=r'conditioning $t_0$')
ax.set_xlabel(r'$100\,\theta_*$')
ax.set_ylabel('normalized (peak = 1)')
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'theta_star_posteriors.pdf', bbox_inches='tight')
fig.savefig(PLOT_DIR / 'theta_star_posteriors.png', dpi=150, bbox_inches='tight')
plt.show()


#%% [markdown]
# ## 6. Width ratios
# Left: our posterior std over the DESI marginal std. Right: our std over the DESI | theta_* std (a ratio near 1 means our fit is as tight
# as DESI would be with theta_* pinned). Both variants (X = h and X = omega_m) are shown for the fourth parameter.

#%%
for xkey, (xlabel, _f) in X_VARIANTS.items():
    names4, labels4 = BASE_NAMES + [xkey], BASE_LABELS + [xlabel]
    sub = TABLE.xs(xkey, level='X')
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
    width = 0.8 / len(OURS)
    for k, lab in enumerate(OURS):
        x = np.arange(len(names4)) + (k - (len(OURS) - 1) / 2) * width
        for ax, col in zip(axes, ('ours/DESI std', 'ours/(DESI|theta*) std')):
            ax.bar(x, [sub.loc[(lab, p), col] for p in names4], width * 0.92, color=OURS[lab][2], label=lab)
    for ax, ylab in zip(axes, (r'$\sigma_{\rm ours}/\sigma_{\rm DESI}$', r'$\sigma_{\rm ours}/\sigma_{{\rm DESI}|\theta_*}$')):
        ax.axhline(1.0, color='k', lw=0.8, ls='--')
        ax.set_xticks(range(len(names4)))
        ax.set_xticklabels([f'${l}$' for l in labels4])
        ax.set_ylabel(ylab)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f'theta_star_width_ratios_{xkey}.pdf', bbox_inches='tight')
    fig.savefig(PLOT_DIR / f'theta_star_width_ratios_{xkey}.png', dpi=150, bbox_inches='tight')
    plt.show()

#%% [markdown]
# ## 7. Single-mock scatter vs posterior width
# Calibration check of the cosmology error bars, independent of DESI. For the s_min = 20 group (`MOCK_SMINS`), the single-mock joint fits
# (`compact_mock*_joint_smin20`, all 24 mocks, fit at the AbacusSummit c000 truth) should scatter about the truth by about their own posterior width.
# Per parameter: `scatter / std` = std over mocks of the
# posterior means divided by the mean posterior std (expected 1 +- 1/sqrt(2(n-1)): +-0.15 for 24 mocks, +-0.35 for 5); `mean pull` = mean of
# (posterior mean - truth)/posterior std (expected 0 +- 1/sqrt(n)); `rms pull` = sqrt(mean(pull^2)) (expected ~1; combines scatter and bias). omega_b and n_s have
# priors centred on the truth, so their pulls are not informative. Posterior means are used rather than MAPs (the MAP of a chain is noisy). Caveats: c000 lies
# on the Abacus theta_* surface, so this cannot detect a missing off-surface h dependence; and the joint fit neglects cross-tracer covariance (QSO overlaps
# LRG3/ELG2), which would also show up as scatter / std > 1. Chains without a .status.json are skipped. Uses `load_our_columns` from section 1.

#%%
MOCK_PARAMS = ['omch2', 'ombh2', 'H0', 'logA', 'ns']
MOCK_LABELS = [r'\omega_{\rm cdm}', r'\omega_b', 'H_0', r'\ln(10^{10}A_s)', 'n_s']
MOCK_TRUTH = np.array([0.12, 0.02237, 67.36, 3.0363943, 0.9649])       # AbacusSummit c000
SMIN_COLOR = {20: 'tab:blue', 25: 'tab:orange', 30: 'tab:green', 35: 'tab:red', 40: 'tab:purple'}
MOCK_SMINS = (20,)   # s_min = 20 has every finished mock (24); the other s_min groups have only 4-5 mocks each (add them here to include them)

MOCK_STATS = {}      # s_min -> (posterior means, posterior stds), each (n_mocks, len(MOCK_PARAMS))
for s in MOCK_SMINS:
    names = sorted(p.stem for p in (CHAIN_ROOT / 'compact').glob(f'compact_mock*_joint_smin{s}.h5')
                   if re.fullmatch(rf'compact_mock\d+_joint_smin{s}', p.stem)
                   and (CHAIN_ROOT / 'compact' / f'{p.stem}.status.json').exists())
    if len(names) < 3:
        continue
    means, stds = [], []
    for n in names:
        c = load_our_columns('compact', n)[0]
        v = np.column_stack([c['omch2'], c['ombh2'], 100 * c['h'], c['logA'], c['ns']])
        means.append(v.mean(0))
        stds.append(v.std(0))
    MOCK_STATS[s] = (np.array(means), np.array(stds))
print('mocks per s_min: ' + ', '.join(f'{s}: {len(v[0])}' for s, v in MOCK_STATS.items()))

rows = []
for s, (mu, sd) in MOCK_STATS.items():
    n = len(mu)
    pull = (mu - MOCK_TRUTH) / sd
    for k, p in enumerate(MOCK_PARAMS):
        rows.append({'s_min': s, 'parameter': p, 'n_mocks': n,
                     'scatter of means': mu[:, k].std(ddof=1), 'mean posterior std': sd[:, k].mean(),
                     'scatter / std': mu[:, k].std(ddof=1) / sd[:, k].mean(), 'scatter / std +-': 1 / np.sqrt(2 * (n - 1)),
                     'mean pull': pull[:, k].mean(), 'mean pull +-': 1 / np.sqrt(n),
                     'rms pull': np.sqrt(np.mean(pull[:, k] ** 2))})
MOCK_SCATTER = pd.DataFrame(rows).set_index(['s_min', 'parameter'])
MOCK_SCATTER.to_csv(PLOT_DIR / 'mock_scatter_vs_posterior.csv')
display(MOCK_SCATTER.round(4))

fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))
smins = list(MOCK_STATS)
width = 0.8 / len(smins)
ekw = dict(ecolor='k', elinewidth=0.8, capsize=1.5)
for k, s in enumerate(smins):
    mu, sd = MOCK_STATS[s]
    n = len(mu)
    x = np.arange(len(MOCK_PARAMS)) + (k - (len(smins) - 1) / 2) * width
    ratio = mu.std(0, ddof=1) / sd.mean(0)
    axes[0].bar(x, ratio, width * 0.92, color=SMIN_COLOR[s], yerr=ratio / np.sqrt(2 * (n - 1)), error_kw=ekw,
                label=rf'$s_{{\rm min}}={s}$ ($n={n}$)')
    axes[1].bar(x, ((mu - MOCK_TRUTH) / sd).mean(0), width * 0.92, color=SMIN_COLOR[s], yerr=1 / np.sqrt(n), error_kw=ekw)
axes[0].axhline(1.0, color='k', lw=0.8, ls='--')
axes[1].axhline(0.0, color='k', lw=0.8, ls='--')
axes[0].set_ylabel(r'scatter of means / mean posterior std')
axes[1].set_ylabel(r'mean pull $=\langle(\mu-\mu_{\rm true})/\sigma\rangle$')
for ax in axes:
    ax.set_xticks(range(len(MOCK_PARAMS)))
    ax.set_xticklabels([f'${l}$' for l in MOCK_LABELS])
axes[0].legend(frameon=False, fontsize=7, ncol=2, loc='upper left')
fig.tight_layout()
fig.savefig(PLOT_DIR / 'mock_scatter_vs_posterior.pdf', bbox_inches='tight')
fig.savefig(PLOT_DIR / 'mock_scatter_vs_posterior.png', dpi=150, bbox_inches='tight')
plt.show()
 

#%% [markdown]
# ## 8. Our chain conditioned on theta_*
# Same Gaussian conditioning as for DESI (section 2), applied to our own chain's Q at t0 = T_C000, the c000 value of 100 theta_* in the chain
# convention (the Abacus surface; the spread of theta_* over the grid is ~1e-4, negligible next to the posterior widths). DESI is conditioned at the same t0
# here, so the comparison is like for like. `std ratio` = conditional std / marginal std = sqrt(1 - rho^2), with rho the correlation of the
# parameter with theta_* in our chain: close to 1 means pinning theta_* would not change that parameter's constraint; well below 1 means it would.
# `mean shift` is the move of the mean in units of the marginal std. Gaussian approximation; a narrow-window reweighting of the chain would check it.
 
#%%
rows = []
COND8 = {}      # (xkey, label) -> ((mu_oc, C_oc), (mu_dc, C_dc))
for xkey in X_VARIANTS:
    r = RES[xkey]
    names4 = BASE_NAMES + [xkey]
    mu_dc, C_dc = condition(r['mu_d'], r['C_d'], T_C000)
    for lab, (mu_o, C_o, _Q) in r['ours'].items():
        mu_oc, C_oc = condition(mu_o, C_o, T_C000)
        COND8[(xkey, lab)] = ((mu_oc, C_oc), (mu_dc, C_dc))
        print(f'X = {xkey}, {lab}: t0 = {T_C000:.5f} is {(T_C000 - mu_o[-1]) / np.sqrt(C_o[-1, -1]):+.2f} sigma from our mean theta_*')
        sd_o, sd_oc, sd_dc = np.sqrt(np.diag(C_o)[:-1]), np.sqrt(np.diag(C_oc)), np.sqrt(np.diag(C_dc))
        rho = C_o[:-1, -1] / (sd_o * np.sqrt(C_o[-1, -1]))
        for k, p in enumerate(names4):
            rows.append({'X': xkey, 'model': lab, 'parameter': p, 'corr with theta*': rho[k],
                         'ours mean': mu_o[k], 'ours std': sd_o[k], 'ours|theta* mean': mu_oc[k], 'ours|theta* std': sd_oc[k],
                         'std ratio': sd_oc[k] / sd_o[k], 'mean shift [sigma]': (mu_oc[k] - mu_o[k]) / sd_o[k],
                         'DESI|theta* std': sd_dc[k], 'ours|theta* / DESI|theta* std': sd_oc[k] / sd_dc[k]})
TABLE8 = pd.DataFrame(rows).set_index(['X', 'model', 'parameter'])
TABLE8.to_csv(PLOT_DIR / 'theta_star_ours_conditioned_table.csv')
display(TABLE8.round(4))
 
#%%
rng = np.random.default_rng(SEED + 2)
for xkey, (xlabel, _f) in X_VARIANTS.items():
    r = RES[xkey]
    names4, labels4 = BASE_NAMES + [xkey], BASE_LABELS + [xlabel]
    mk = lambda a, lab: MCSamples(samples=a, names=names4, labels=labels4, label=lab, settings=GETDIST_SETTINGS)
    t0s = rf'$100\theta_*={T_C000:.4f}$'
    for lab, (_m, _C, Q_o) in r['ours'].items():
        (mu_oc, C_oc), _ = COND8[(xkey, lab)]
        sets = [mk(Q_o[:, :4], lab), mk(rng.multivariate_normal(mu_oc, C_oc, N_DRAW), f'{lab} | {t0s}')]
        c = OURS[lab][2]
        g = plots.get_subplot_plotter()
        g.triangle_plot(sets, params=names4, filled=False, contour_colors=[c, c], contour_ls=['-', '--'],
                        contour_lws=[1.4] * 2, legend_labels=[s.getLabel() for s in sets])
        g.export(str(PLOT_DIR / f'theta_star_ours_conditioned_{xkey}.pdf'))
        g.export(str(PLOT_DIR / f'theta_star_ours_conditioned_{xkey}.png'))
        plt.show()

#%% [markdown]
# ## 9. Fisher decomposition: geometry (AP) vs emulator channel
# Where does the constraint on each parameter come from? In the likelihood, h, omega_b, omega_cdm and n_s enter twice: through the AP / geometry
# (`fl.ap_multi`, one CAMB background for all tracers) and through the emulator input vector; ln A_s only enters the emulator. Each of those four
# parameters is split into an `_ap` and an `_emu` copy; the Jacobian of the stacked windowed multipoles is taken by central differences at the MAP of
# the compact DR1 joint chain (step = STEP_FRACS[0] x the posterior std), and F = sum over tracers of J^T C^-1 J, plus the Gaussian priors on
# omega_b and n_s (the flat bounds are ignored). From F:
#   * both channels  = the two copies merged into one parameter (the baseline fit; should reproduce the chain widths),
#   * AP-only        = emulator copy of that parameter frozen (its constraint comes from the geometry alone),
#   * emulator-only  = AP copy frozen (its constraint comes from the shape alone).
# `AP-only / both` and `emulator-only / both` are >= 1; a value near 1 means that channel carries almost all of the information. Checks printed:
# Fisher vs chain std for the baseline (a large mismatch means the posterior is too non-Gaussian or prior-limited for this to be trusted) and the
# stability of the baseline widths under the second step size. Needs fit_lib_11par.py (set FITLIB_DIR) and the compact environment.
 
#%%
import sys

FITLIB_DIR = Path('.')     # folder containing fit_lib_11par.py
FISHER_MODEL, FISHER_CHAIN = 'compact', 'compact_dr1_joint_smin20'
STEP_FRACS = (0.1, 0.3)    # finite-difference step as a fraction of the posterior std; the second only checks stability
SPLITS = ('h', 'ombh2', 'omch2', 'ns')
 
sys.path.insert(0, str(FITLIB_DIR))
import fit_lib_11par as fl
 
problem = fl.FitProblem(next(sp for sp in fl.build_specs() if sp['name'] == FISHER_CHAIN))
nb = len(problem.data)
 
st = json.load(open(CHAIN_ROOT / FISHER_MODEL / f'{FISHER_CHAIN}.status.json'))
with h5py.File(CHAIN_ROOT / FISHER_MODEL / f'{FISHER_CHAIN}.h5', 'r') as f:
    it = int(f['mcmc'].attrs['iteration'])
    lp = f['mcmc/log_prob'][st['discard']:it]
    t, w = np.unravel_index(np.nanargmax(np.where(np.isfinite(lp), lp, -np.inf)), lp.shape)
    theta_map = f['mcmc/chain'][st['discard'] + t, w, :]       # MAP = highest-posterior sample after burn-in
    sd = f['mcmc/chain'][st['discard']:it:st['thin'], :, :].reshape(-1, problem.ndim).std(0)
 
JOINT_NAMES = ['h', 'ombh2', 'omch2', 'logA', 'ns'] + [f'b{j}' for j in range(2 * nb)]      # joint-fit parameter order
FULL_OF = {nm: ([nm + '_ap', nm + '_emu'] if nm in SPLITS else [nm]) for nm in JOINT_NAMES}   # joint parameter -> full-parameter names
full_names = [fn for nm in JOINT_NAMES for fn in FULL_OF[nm]]
full_src = [k for k, nm in enumerate(JOINT_NAMES) for _ in FULL_OF[nm]]     # index of the joint parameter each full parameter copies
idx = {fn: i for i, fn in enumerate(full_names)}
x0, sd_full = theta_map[full_src], sd[full_src]
 
#%%
def model_vecs(x):
    """Windowed multipole vector of every tracer for a full-parameter vector x (geometry and emulator copies separate)."""
    h_a, ombh2_a, omch2_a, ns_a = (x[idx[f'{n}_ap']] for n in SPLITS)
    h_e, ombh2_e, omch2_e, ns_e = (x[idx[f'{n}_emu']] for n in SPLITS)
    aps = fl.ap_multi(h_a, ombh2_a, omch2_a, ns_a, problem.z_effs)
    return [fl.model_one(d['emus'], [x[idx[f'b{2 * i}']], x[idx[f'b{2 * i + 1}']], ombh2_e, omch2_e, h_e, x[idx['logA']], ns_e],
                         d['s_data'], d['W'], qp, ql) for i, (d, (qp, ql)) in enumerate(zip(problem.data, aps))]
 
 
def fisher(frac):
    J = [np.zeros((len(d['data_vec']), len(x0))) for d in problem.data]
    for j in range(len(x0)):
        dx = frac * sd_full[j]
        e = np.zeros(len(x0))
        e[j] = dx
        mp, mm = model_vecs(x0 + e), model_vecs(x0 - e)
        for Ji, a, b in zip(J, mp, mm):
            Ji[:, j] = (a - b) / (2 * dx)
    return sum(Ji.T @ d['cov_inv'] @ Ji for Ji, d in zip(J, problem.data))
 
 
PRIOR_PREC = {'ombh2': 1 / fl.OMBH2_SIG ** 2, 'ns': 1 / fl.NS_SIG ** 2}
 
 
def cosmo_sigmas(F, use=None):
    """Std of (h, ombh2, omch2, logA, ns); `use` maps a joint parameter to the full-parameter names that carry it (default: all of them)."""
    use = use or {}
    T = np.zeros((len(x0), len(JOINT_NAMES)))
    for r, nm in enumerate(JOINT_NAMES):
        for fn in use.get(nm, FULL_OF[nm]):
            T[idx[fn], r] = 1.0
    Fr = T.T @ F @ T
    for nm, prec in PRIOR_PREC.items():
        Fr[JOINT_NAMES.index(nm), JOINT_NAMES.index(nm)] += prec
    return np.sqrt(np.diag(np.linalg.inv(Fr)))[:5]
 
 
Fs = [fisher(f) for f in STEP_FRACS]
base = cosmo_sigmas(Fs[0])
if len(Fs) > 1:
    print(f'baseline widths, step {STEP_FRACS[0]} vs {STEP_FRACS[1]} x std: max relative difference '
          f'{np.max(np.abs(base / cosmo_sigmas(Fs[1]) - 1)):.3f}')
 
rows = []
for k, nm in enumerate(JOINT_NAMES[:5]):
    row = {'parameter': nm, 'chain std': sd[k], 'Fisher std (both)': base[k], 'Fisher / chain': base[k] / sd[k]}
    if nm in SPLITS:
        ap = cosmo_sigmas(Fs[0], {nm: [nm + '_ap']})[k]
        em = cosmo_sigmas(Fs[0], {nm: [nm + '_emu']})[k]
        row.update({'AP-only std': ap, 'emulator-only std': em, 'AP-only / both': ap / base[k], 'emulator-only / both': em / base[k]})
    rows.append(row)
FISHER_TABLE = pd.DataFrame(rows).set_index('parameter')
FISHER_TABLE.to_csv(PLOT_DIR / 'theta_star_fisher_channels.csv')
display(FISHER_TABLE.round(5))
 
#%%
FISHER_LABELS = {'h': 'h', 'ombh2': r'\omega_b', 'omch2': r'\omega_{\rm cdm}', 'ns': 'n_s'}
fig, ax = plt.subplots(figsize=(5.0, 3.4))
x = np.arange(len(SPLITS))
ax.plot(x - 0.06, [FISHER_TABLE.loc[p, 'AP-only / both'] for p in SPLITS], 'o', color='tab:blue', label='AP-only (emulator channel frozen)')
ax.plot(x + 0.06, [FISHER_TABLE.loc[p, 'emulator-only / both'] for p in SPLITS], 's', color='tab:orange', label='emulator-only (AP channel frozen)')
ax.axhline(1.0, color='k', lw=0.8, ls='--')
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels([f'${FISHER_LABELS[p]}$' for p in SPLITS])
ax.set_xlim(-0.5, len(SPLITS) - 0.5)
ax.set_ylabel(r'$\sigma_{\rm one\ channel}/\sigma_{\rm both}$')
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'theta_star_fisher_channels.pdf', bbox_inches='tight')
fig.savefig(PLOT_DIR / 'theta_star_fisher_channels.png', dpi=150, bbox_inches='tight')
plt.show()


#%% [markdown]
# ### 9b. Sensitivity of the Fisher widths to the AP implementation
# The data separations are in h_fid^-1 Mpc with the DESI fiducial cosmology (AbacusSummit c000: h = 0.6736, omega_b = 0.02237, omega_cdm = 0.12); the model
# (LPT + emulator) is in h^-1 Mpc at the sampled h. Mapping between them needs q_perp, q_par referenced to the c000 cosmology AND an isotropic factor
# h / h_fid (ratios of physical distances alone omit it). The Fisher widths are recomputed at the same MAP for three AP implementations:
# (0) the current one (reference cosmology of `common.py`, no h factor); (1) c000 reference only; (2) c000 reference + the h / h_fid factor.
# The reference and the sampled cosmology go through the same `fl.get_background`, so q = 1 exactly at c000 for (1) and (2); `max |q - 1| at c000` shows how far
# (0) is from that. The expansion point is the baseline MAP for every variant (a corrected fit would move it), so this shows the change in information
# content, not the corrected posterior. `fl.ap_multi` is patched temporarily and restored afterwards.
 
#%%
C000_REF = (0.6736, 0.02237, 0.12)     # DESI fiducial cosmology: AbacusSummit c000 (h, omega_b, omega_cdm)
AP_ORIG = fl.ap_multi
 
 
def make_ap_multi(ref, h_factor):
    ref_res = fl.get_background(*ref, fl.NS_FIXED)
    zs = set(problem.z_effs)
    da_ref = {z: ref_res.angular_diameter_distance(z) for z in zs}
    hz_ref = {z: ref_res.hubble_parameter(z) for z in zs}
 
    def ap_multi(h, ombh2, omch2, ns, z_effs):
        r = fl.get_background(h, ombh2, omch2, ns)
        f = h / ref[0] if h_factor else 1.0
        return [(f * r.angular_diameter_distance(z) / da_ref[z], f * hz_ref[z] / r.hubble_parameter(z)) for z in z_effs]
    return ap_multi
 
 
AP_VARIANTS = {'0: current (common.py reference, no h factor)': AP_ORIG,
               '1: c000 reference': make_ap_multi(C000_REF, False),
               '2: c000 reference + h/h_fid factor': make_ap_multi(C000_REF, True)}
 
rows = []
try:
    for name, ap in AP_VARIANTS.items():
        fl.ap_multi = ap
        F = fisher(STEP_FRACS[0])
        b = cosmo_sigmas(F)
        row = {'variant': name, 'max |q - 1| at c000': max(abs(q - 1) for pair in ap(*C000_REF, fl.NS_FIXED, problem.z_effs) for q in pair)}
        row.update({f'std {nm}': b[k] for k, nm in enumerate(JOINT_NAMES[:5])})
        row['h AP-only'] = cosmo_sigmas(F, {'h': ['h_ap']})[0]
        row['h emulator-only'] = cosmo_sigmas(F, {'h': ['h_emu']})[0]
        rows.append(row)
finally:
    fl.ap_multi = AP_ORIG
AP_TABLE = pd.DataFrame(rows).set_index('variant')
for nm in ('h', 'omch2'):
    AP_TABLE[f'std {nm} / variant 0'] = AP_TABLE[f'std {nm}'] / AP_TABLE[f'std {nm}'].iloc[0]
AP_TABLE.to_csv(PLOT_DIR / 'theta_star_fisher_ap_variants.csv')
display(AP_TABLE.round(5))



#%% [markdown]
# ### 9c. Does the emulator's BAO peak scale with h?
# Check of the unit question behind 9b, with no fit. At fixed omega_b, omega_cdm, ln A_s, n_s and biases (all at the MAP), the physical BAO scale r_d is
# fixed, so if the emulator's xi_0(s) is in h^-1 Mpc at the sampled h the BAO bump in s^2 xi_0 must sit at r_d h and move as
# d ln s_peak / d ln h = 1; a slope near 0 would mean the units are handled upstream (and the h / h_fid factor in 9b would then be wrong to add).
# The bump position is the vertex of a parabola through the five grid points around the maximum of s^2 xi_0 inside S_WIN, for one tracer (the one
# closest to z = 0.51, LRG1-like); xi_0 is the emulator output before any AP transformation.
# The separation grid is `common.S_MODEL` (the emulator's first return value is not the grid).
 
#%%
import common     # found through the sys.path entry fit_lib_11par adds
H_SCAN = np.linspace(-0.04, 0.04, 9)     # fractional changes of h about the MAP
S_WIN = (85.0, 130.0)                    # h^-1 Mpc window searched for the BAO bump in s^2 xi_0
i_t = min(range(nb), key=lambda i: abs(problem.data[i]['z_eff'] - 0.51))
emu0 = problem.data[i_t]['emus'][0]
h_m, ombh2_m, omch2_m, lnAs_m, ns_m = theta_map[:5]
b1_m, b2_m = theta_map[5 + 2 * i_t], theta_map[6 + 2 * i_t]
 
 
def bao_peak(h):
    _, xi0 = emu0([b1_m, b2_m, ombh2_m, omch2_m, h, lnAs_m, ns_m])     # first return is not the separation grid; that is common.S_MODEL
    s, xi0 = np.asarray(common.S_MODEL), np.asarray(xi0)
    assert len(xi0) == len(s), f'emulator output has {len(xi0)} points but S_MODEL has {len(s)}'
    y = s ** 2 * xi0
    w = np.where((s >= S_WIN[0]) & (s <= S_WIN[1]))[0]
    i = w[np.argmax(y[w])]
    if i in (w[0], w[-1]):
        print(f'warning: maximum of s^2 xi_0 at the edge of the window for h = {h:.4f}; no BAO bump found')
        return np.nan, s, y
    c = np.polyfit(s[i - 2:i + 3], y[i - 2:i + 3], 2)
    return -c[1] / (2 * c[0]), s, y
 
 
hs = h_m * (1 + H_SCAN)
scan = [bao_peak(h) for h in hs]
peaks = np.array([r[0] for r in scan])
ok = np.isfinite(peaks)
slope = np.polyfit(np.log(hs[ok]), np.log(peaks[ok]), 1)[0]
display(pd.DataFrame({'h': hs, 's_peak [h^-1 Mpc]': peaks, 'h / h_MAP': hs / h_m, 's_peak / s_peak(MAP)': peaks / peaks[len(hs) // 2]}).round(4))
print(f'tracer z_eff = {problem.data[i_t]["z_eff"]}: d ln s_peak / d ln h = {slope:.3f} at fixed omega_b, omega_cdm '
      f'(~1: model in h^-1 Mpc at the sampled h, so the h/h_fid factor is needed; ~0: units handled upstream)')
 
fig, ax = plt.subplots(figsize=(5.0, 3.4))
for k, col in zip((0, len(hs) // 2, len(hs) - 1), ('tab:blue', 'k', 'tab:red')):
    _p, s, y = scan[k]
    ax.plot(s, y, color=col, lw=1.2, label=rf'$h={hs[k]:.3f}$')
    ax.axvline(_p, color=col, ls=':', lw=0.9)
ax.set_xlim(60, 150)
ax.set_xlabel(r'$s\;[h^{-1}{\rm Mpc}]$')
ax.set_ylabel(r'$s^2\xi_0$')
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'theta_star_bao_peak_vs_h.pdf', bbox_inches='tight')
fig.savefig(PLOT_DIR / 'theta_star_bao_peak_vs_h.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
