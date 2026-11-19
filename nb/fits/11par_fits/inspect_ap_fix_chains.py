"""
Compare the AP-fixed DR1 joint chain with the old DR1 baseline chain and the DESI DR1 2024 full-shape (FS-only) chain.

Run cell by cell (#%% markers) or as a script, in the compact environment (needs getdist, h5py). Same conventions as
inspect_chains_11par.py / theta_star_conditional.py.
"""

#%% [markdown]
# # AP-fixed DR1 fit vs old baseline vs DESI DR1 FS-only
# Post-burn-in samples of `compact_dr1_joint_smin20_apfix` (AP reference = c000 and the h / h_fid unit factor; short chain) and of the old baseline
# `compact_dr1_joint_smin20`, against the DESI DR1 FS-only chain. Burn-in and thinning are the `discard` / `thin` stored in each chain's
# `.status.json` (5 tau_max, 0.5 tau_max), unless overridden in `BURN_OVERRIDE`. The AP-fixed chain is short and flagged unconverged, so its
# widths are approximate; check the trace figure for the burn-in before trusting the contours. Section 5 adds a fit-validation figure (data vs the
# MAP model of each chain, per tracer) and a chi^2 table; it needs fit_lib_11par.py (set FITLIB_DIR) and the compact environment.

#%%
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
    os.environ[_k] = '1'       # as in fit_lib_11par.py; must be set before numpy loads (section 5 imports fit_lib_11par)
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
from scipy.stats import chi2 as chi2_dist
from getdist import MCSamples, loadMCSamples, plots
try:
    from IPython.display import display
except ImportError:
    display = print

warnings.filterwarnings('ignore', message='.*tight_layout.*')
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
                     'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False,
                     'axes.xmargin': 0, 'figure.dpi': 110})

# ---------------- configuration ----------------
CHAIN_ROOT = Path('/spiffball/cosweeney/data/chains/fits_11par')
PLOT_DIR = CHAIN_ROOT / 'plots' / 'apfix'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
DESI_BASE = '/spiffball/cosweeney/DESI/DR1/full-shape-cosmo-params/v1.0/cobaya/base/desi-reptvelocileptors-fs-all_schoneberg2024-bbn_planck2018-ns10/'
GETDIST_SETTINGS = {'smooth_scale_1D': 0.5, 'smooth_scale_2D': 0.5}

CHAINS = {   # label: (model directory, chain name, colour); the first entry is the "new" chain
    'Compact, AP fixed':   ('compact_apfix', 'compact_dr1_joint_smin20_apfix', 'tab:red'),
    'Compact, old baseline': ('compact',      'compact_dr1_joint_smin20',       'tab:blue'),
}
FITLIB_DIR = Path('.')                      # folder containing fit_lib_11par.py (section 5)
SPEC_NAME = 'compact_dr1_joint_smin20'      # the fit spec (data, covariance, emulators) both chains were run with
C000_REF = (0.6736, 0.02237, 0.12)          # DESI fiducial cosmology: AbacusSummit c000 (h, omega_b, omega_cdm)
BURN_OVERRIDE = {}      # chain name -> number of steps to discard instead of the status.json value, e.g. {'compact_dr1_joint_smin20_apfix': 1500}

TRACER_NAMES = ['BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG2', 'QSO']             # fl.FIT_KEYS order
TRACER_COLORS = ['yellowgreen', 'orange', 'orangered', 'firebrick', 'steelblue', 'seagreen']

ALL_PARAMS = ['omch2', 'ombh2', 'H0', 'logA', 'ns', 'omegam']
LABELS = {'omch2': r'\omega_{\rm cdm}', 'ombh2': r'\omega_b', 'H0': 'H_0', 'logA': r'\ln(10^{10}A_s)', 'ns': 'n_s', 'omegam': r'\omega_m'}
PLOTS = {   # figure name: parameters shown
    'cosmology': ['omch2', 'ombh2', 'H0', 'logA', 'ns'],
    'omegam':    ['omegam', 'H0', 'logA', 'ns'],
}

#%% [markdown]
# ## 1. Load the chains

#%%
def load_chain(model, name):
    """Post-burn-in, thinned samples of the shared cosmology (first 5 joint parameters) and the chain bookkeeping."""
    root = CHAIN_ROOT / model / name
    st = json.load(open(f'{root}.status.json'))
    burn = BURN_OVERRIDE.get(name, st['discard'])
    with h5py.File(f'{root}.h5', 'r') as f:
        it = int(f['mcmc'].attrs['iteration'])
        flat = f['mcmc/chain'][burn:it:st['thin'], :, :5].reshape(-1, 5)
    h, ombh2, omch2, logA, ns = flat.T
    info = dict(iterations=it, burn=burn, thin=st['thin'], n_samples=len(flat), tau_max=st['tau_max'],
                n_over_tau=it / st['tau_max'], converged=st['converged'])
    return dict(omch2=omch2, ombh2=ombh2, H0=100 * h, logA=logA, ns=ns, omegam=ombh2 + omch2), None, info


def load_desi():
    smp = loadMCSamples(DESI_BASE + 'chain', settings=GETDIST_SETTINGS)
    names = smp.getParamNames().list()
    missing = [p for p in ('omch2', 'ombh2', 'H0', 'logA', 'ns') if p not in names]
    assert not missing, f'DESI chain lacks {missing}; available: {names[:30]}'
    col = lambda n: smp.samples[:, names.index(n)]
    cols = dict(omch2=col('omch2'), ombh2=col('ombh2'), H0=col('H0'), logA=col('logA'), ns=col('ns'))
    cols['omegam'] = cols['ombh2'] + cols['omch2']
    return cols, smp.weights, None


def to_mcsamples(cols, weights, label):
    return MCSamples(samples=np.column_stack([cols[p] for p in ALL_PARAMS]), weights=weights, names=ALL_PARAMS,
                     labels=[LABELS[p] for p in ALL_PARAMS], label=label, settings=GETDIST_SETTINGS)


SAMPLES, INFO = {}, {}
for label, (model, name, _c) in CHAINS.items():
    cols, w, INFO[label] = load_chain(model, name)
    SAMPLES[label] = to_mcsamples(cols, w, label)
cols, w, _ = load_desi()
SAMPLES['DESI DR1 FS-only'] = to_mcsamples(cols, w, 'DESI DR1 FS-only')
COLORS = {**{lab: c for lab, (_m, _n, c) in CHAINS.items()}, 'DESI DR1 FS-only': 'k'}

display(pd.DataFrame(INFO).T.round(1))

#%% [markdown]
# ## 2. Burn-in check for the AP-fixed chain
# Walker-median ln P (top) and h for 12 walkers (bottom) over the whole chain; the dashed line is the discarded burn-in.

#%%
label_new = next(iter(CHAINS))
model_new, name_new, _ = CHAINS[label_new]
root = CHAIN_ROOT / model_new / name_new
with h5py.File(f'{root}.h5', 'r') as f:
    it = int(f['mcmc'].attrs['iteration'])
    lp = f['mcmc/log_prob'][:it]
    hw = f['mcmc/chain'][:it, :12, 0]
burn = INFO[label_new]['burn']
steps = np.arange(it)
fig, axes = plt.subplots(2, 1, figsize=(6.0, 4.2), sharex=True)
axes[0].plot(steps, np.nanmedian(np.where(np.isfinite(lp), lp, np.nan), axis=1), color='k', lw=0.7)
axes[0].set_ylabel(r'$\ln P$ (walker median)')
axes[1].plot(steps, hw, lw=0.4, alpha=0.5)
axes[1].set_ylabel(r'$h$')
axes[1].set_xlabel('step')
for ax in axes:
    ax.axvline(burn, color='tab:red', ls='--', lw=1.0)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'apfix_burnin_check.pdf', bbox_inches='tight')
fig.savefig(PLOT_DIR / 'apfix_burnin_check.png', dpi=150, bbox_inches='tight')
plt.show()

#%% [markdown]
# ## 3. Triangle plots

#%%
labels = list(SAMPLES)
for fig_name, params in PLOTS.items():
    g = plots.get_subplot_plotter()
    g.triangle_plot([SAMPLES[l] for l in labels], params=params, filled=False,
                    contour_colors=[COLORS[l] for l in labels], contour_lws=[1.5] * len(labels), legend_labels=labels)
    g.export(str(PLOT_DIR / f'apfix_triangle_{fig_name}.pdf'))
    g.export(str(PLOT_DIR / f'apfix_triangle_{fig_name}.png'))
    plt.show()

#%% [markdown]
# ## 4. Summary table
# Means and standard deviations of the 1D marginals; `std / DESI` is each chain's width over the DESI width, and `shift vs old [sigma]` is the move of
# the AP-fixed mean relative to the old baseline mean in units of the old baseline std.

#%%
rows = []
for lab in labels:
    row = {'chain': lab}
    for p in ALL_PARAMS:
        row[f'{p} mean'], row[f'{p} std'] = SAMPLES[lab].mean(p), SAMPLES[lab].std(p)
    rows.append(row)
SUMMARY = pd.DataFrame(rows).set_index('chain')
ratio = pd.DataFrame({f'{p} std / DESI': SUMMARY[f'{p} std'] / SUMMARY.loc['DESI DR1 FS-only', f'{p} std'] for p in ALL_PARAMS})
label_old = list(CHAINS)[1]
shift = pd.Series({f'{p} shift vs old [sigma]': (SUMMARY.loc[label_new, f'{p} mean'] - SUMMARY.loc[label_old, f'{p} mean'])
                   / SUMMARY.loc[label_old, f'{p} std'] for p in ALL_PARAMS}, name=label_new)
SUMMARY.to_csv(PLOT_DIR / 'apfix_summary.csv')
display(SUMMARY.round(4))
display(ratio.round(2))
display(shift.round(2).to_frame().T)

#%% [markdown]
# ## 4b. Bias parameters of the AP-fixed chain
# Triangle plot of $b_1$, $b_2$ for the six tracer bins (joint-fit parameters 5-16, in the order BGS, LRG1-3, ELG2, QSO), AP-fixed chain only, with the
# same burn-in and thinning as above. `BIAS_FILLED = True` fills the contours. The flat priors are $0<b_1<2$ and $-5<b_2<10$.

#%%
BIAS_FILLED = False
BIAS_NAMES = [f'{b}_{t}' for t in TRACER_NAMES for b in ('b1', 'b2')]
BIAS_LABELS = [rf'b_{b[1]}^{{\rm {t}}}' for t in TRACER_NAMES for b in ('b1', 'b2')]


def load_bias(model, name):
    root = CHAIN_ROOT / model / name
    st = json.load(open(f'{root}.status.json'))
    burn = BURN_OVERRIDE.get(name, st['discard'])
    with h5py.File(f'{root}.h5', 'r') as f:
        it = int(f['mcmc'].attrs['iteration'])
        return f['mcmc/chain'][burn:it:st['thin'], :, 5:5 + len(BIAS_NAMES)].reshape(-1, len(BIAS_NAMES))


bias_smp = MCSamples(samples=load_bias(*CHAINS[label_new][:2]), names=BIAS_NAMES, labels=BIAS_LABELS, label=label_new, settings=GETDIST_SETTINGS)
g = plots.get_subplot_plotter(width_inch=11)
g.triangle_plot([bias_smp], params=BIAS_NAMES, filled=BIAS_FILLED, contour_colors=[COLORS[label_new]], contour_lws=[1.3], legend_labels=[label_new])
g.export(str(PLOT_DIR / 'apfix_triangle_bias.pdf'))
g.export(str(PLOT_DIR / 'apfix_triangle_bias.png'))
plt.show()

BIAS_TABLE = pd.DataFrame({'b1 mean': [bias_smp.mean(f'b1_{t}') for t in TRACER_NAMES], 'b1 std': [bias_smp.std(f'b1_{t}') for t in TRACER_NAMES],
                           'b2 mean': [bias_smp.mean(f'b2_{t}') for t in TRACER_NAMES], 'b2 std': [bias_smp.std(f'b2_{t}') for t in TRACER_NAMES]},
                          index=TRACER_NAMES)
BIAS_TABLE.to_csv(PLOT_DIR / 'apfix_bias_summary.csv')
display(BIAS_TABLE.round(3))

#%% [markdown]
# ## 5. Fit validation: data vs the MAP model of each chain
# Per tracer: $s^2\xi_\ell$ (monopole filled, quadrupole open) with the DR1 data and error bars from the covariance diagonal, the windowed MAP model
# of the AP-fixed chain (solid) evaluated with the fixed AP, and of the old baseline chain (dashed grey) evaluated with the old AP; bottom: the residuals
# $\Delta/\sigma=(\mathrm{data}-\mathrm{model})/\sigma$ with a $\pm1$ band. The MAP is the highest-posterior post-burn-in sample of each chain. The AP
# functions are rebuilt here (c000 reference + h / h_fid factor for the new chain; fit_lib_11par's original for the old one), so no patching of
# fit_lib_11par is needed. Builds the full FitProblem (loads data, covariances and emulators).

#%%
sys.path.insert(0, str(FITLIB_DIR))
import fit_lib_11par as fl


def make_ap_multi(ref):
    """(q_perp, q_par) for every z from ONE CAMB background call, referenced to `ref` and in h^-1 Mpc units (as in fit_dr1_apfix.py)."""
    h_ref = ref[0]
    ref_res = fl.get_background(*ref, fl.NS_FIXED)
    zs = set(fl.Z_EFF.values())
    da_ref = {z: ref_res.angular_diameter_distance(z) for z in zs}
    hz_ref = {z: ref_res.hubble_parameter(z) for z in zs}

    def ap_multi(h, ombh2, omch2, ns, z_effs):
        r = fl.get_background(h, ombh2, omch2, ns)
        f = h / h_ref
        return [(f * r.angular_diameter_distance(z) / da_ref[z], f * hz_ref[z] / r.hubble_parameter(z)) for z in z_effs]
    return ap_multi


def map_theta(model, name):
    root = CHAIN_ROOT / model / name
    st = json.load(open(f'{root}.status.json'))
    burn = BURN_OVERRIDE.get(name, st['discard'])
    with h5py.File(f'{root}.h5', 'r') as f:
        it = int(f['mcmc'].attrs['iteration'])
        lp = f['mcmc/log_prob'][burn:it]
        t, w = np.unravel_index(np.nanargmax(np.where(np.isfinite(lp), lp, -np.inf)), lp.shape)
        return f['mcmc/chain'][burn + t, w, :], float(lp[t, w])


problem = fl.FitProblem(next(sp for sp in fl.build_specs() if sp['name'] == SPEC_NAME))
AP_BY_LABEL = {label_new: make_ap_multi(C000_REF), label_old: fl.ap_multi}      # new chain: fixed AP; old chain: the AP it was fitted with
FITS = {lab: map_theta(CHAINS[lab][0], CHAINS[lab][1]) for lab in (label_new, label_old)}


def model_at(theta, ap):
    h, ombh2, omch2, lnAs, ns, biases = problem.unpack(theta)
    aps = ap(h, ombh2, omch2, ns, problem.z_effs)
    return [fl.model_one(d['emus'], [b1, b2, ombh2, omch2, h, lnAs, ns], d['s_data'], d['W'], qp, ql).reshape(2, -1)
            for d, (b1, b2), (qp, ql) in zip(problem.data, biases, aps)]


MODELS = {lab: model_at(FITS[lab][0], AP_BY_LABEL[lab]) for lab in FITS}
assert len(problem.data) == len(TRACER_NAMES), f'{len(problem.data)} tracers in the problem'
print({lab: f'h = {FITS[lab][0][0]:.4f}, lnP = {FITS[lab][1]:.2f}' for lab in FITS})

#%%
n = len(problem.data)
fig = plt.figure(figsize=(2.9 * n, 5.0))
gs = fig.add_gridspec(2, n, height_ratios=[3, 1], hspace=0.06, wspace=0.32)
for j, (d, name, color) in enumerate(zip(problem.data, TRACER_NAMES, TRACER_COLORS)):
    s_d, data = d['s_data'], d['data_vec'].reshape(2, -1)
    sig = np.sqrt(np.diag(np.linalg.inv(d['cov_inv']))).reshape(2, -1)
    ax, axr = fig.add_subplot(gs[0, j]), fig.add_subplot(gs[1, j])
    sf = np.linspace(s_d[0], s_d[-1], 300)
    for i, (marker, filled) in enumerate((('o', True), ('s', False))):
        ax.errorbar(s_d, s_d ** 2 * data[i], yerr=s_d ** 2 * sig[i], fmt=marker, ms=3.3, color=color, ecolor=color,
                    mfc=color if filled else 'white', elinewidth=1.1, capsize=0)
        for lab, ls, col in ((label_new, '-', color), (label_old, '--', '0.45')):
            m = MODELS[lab][j]
            ax.plot(sf, CubicSpline(s_d, s_d ** 2 * m[i])(sf), ls=ls, color=col, lw=1.3)
            axr.plot(s_d, (data[i] - m[i]) / sig[i], marker=marker, ms=2.8, ls=ls, lw=0.7, color=col, mfc=col if filled else 'white')
    axr.axhspan(-1, 1, color='0.85', zorder=0)
    axr.axhline(0, color='0.55', lw=0.7)
    axr.set_ylim(-5, 5)
    ax.set_title(name, fontsize=9)
    ax.tick_params(labelbottom=False, labelsize=7)
    axr.tick_params(labelsize=7)
    axr.set_xlabel(r'$s\;[h^{-1}{\rm Mpc}]$')
    if j == 0:
        ax.set_ylabel(r'$s^2\xi_\ell(s)$')
        axr.set_ylabel(r'$\Delta/\sigma$')
        ax.legend(handles=[Line2D([0], [0], color='k', lw=1.3, label=label_new), Line2D([0], [0], color='0.45', ls='--', lw=1.3, label=label_old),
                           Line2D([0], [0], color='k', marker='o', ms=3.3, ls='none', label='data')], frameon=False, fontsize=6.5, loc='upper right')
fig.text(0.5, 0.005, 'filled: monopole   open: quadrupole', ha='center', fontsize=7)
fig.savefig(PLOT_DIR / 'apfix_fit_validation.pdf', bbox_inches='tight')
fig.savefig(PLOT_DIR / 'apfix_fit_validation.png', dpi=150, bbox_inches='tight')
plt.show()

#%% [markdown]
# ### chi-squared at the MAP
# $\chi^2=r^T C^{-1} r$ per tracer for each chain's own MAP and AP implementation; the total's degrees of freedom are $N_{\rm data}-N_{\rm par}$ (17).
# $\Delta\chi^2=\chi^2_{\rm old}-\chi^2_{\rm new}$ is positive when the AP-fixed fit is better. Only differences of a few are meaningful, and the MAP of a
# short chain sits a few ln P units below the true maximum.

#%%
chi2_rows = []
for j, (d, name) in enumerate(zip(problem.data, TRACER_NAMES)):
    row = {'tracer': name, 'n_data': d['data_vec'].size}
    for lab in (label_new, label_old):
        r = (d['data_vec'] - MODELS[lab][j].ravel())
        row[f'chi2 {lab}'] = float(r @ d['cov_inv'] @ r)
    chi2_rows.append(row)
CHI2 = pd.DataFrame(chi2_rows).set_index('tracer')
CHI2.loc['TOTAL'] = CHI2.sum()
CHI2['Delta chi2 (old - new)'] = CHI2[f'chi2 {label_old}'] - CHI2[f'chi2 {label_new}']
dof = int(CHI2.loc['TOTAL', 'n_data']) - problem.ndim
for lab in (label_new, label_old):
    print(f'{lab}: total chi2 = {CHI2.loc["TOTAL", f"chi2 {lab}"]:.1f} for {dof} dof -> chi2/dof = {CHI2.loc["TOTAL", f"chi2 {lab}"] / dof:.3f}, '
          f'p = {chi2_dist.sf(CHI2.loc["TOTAL", f"chi2 {lab}"], dof):.3g}')
CHI2.to_csv(PLOT_DIR / 'apfix_chi2.csv')
display(CHI2.round(2))