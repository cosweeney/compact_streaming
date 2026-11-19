"""
Checks on the Alcock-Paczynski (AP) parameters used by the v2 fits (fits_apfix.ap_multi), designed to catch the old bug (q built from physical distance ratios WITHOUT the h / h_fid factor; the data are in
h_fid^-1 Mpc, the emulator output in h^-1 Mpc at the sampled h). Run cell by cell (#%% markers) or as a script in the compact environment.

  1. Unit checks (no chains): q = 1 at the data's fiducial cosmology (AbacusSummit c000), and invariance of q under h -> lam h, omega -> lam^2 omega (the geometry in h^-1 Mpc depends only on the Omega's).
  2. An INDEPENDENT diagnostic, the BAO peak position. The BAO feature of the emulated monopole sits at s_peak ~ r_d h (h^-1 Mpc), so after the AP the pipeline must put it at
        s_peak(theta) = s_peak(c000, same biases) x [r_d(theta) / r_d(c000)] x [D_V(c000) / D_V(theta)]       (h_fid^-1 Mpc, the units of the data)
     with r_d (drag-epoch sound horizon) and D_V from CAMB directly, not from ap_multi. R = pipeline peak / expected peak must be 1 to within TOL for a correct AP, at any cosmology; without the
     h / h_fid factor it is off by roughly h_fid / h - 1 (a few %). The bugged variant is computed alongside (the same samples with q / (h / h_fid)) to show that the diagnostic would have caught it.
     Scan in h at fixed omega's, then evaluate R over posterior samples of the baseline joint fit and of every individual-tracer fit (DR1 and mean of mocks), per tracer.
Figures and a CSV go to CHAIN_ROOT/plots_ap_check; `AP_CHECK_PASSED` summarizes the pass/fail verdict.
"""
#%%
import sys
from pathlib import Path
FITS_DIR = Path('/home/cosweeney/code/compact_streaming/nb/fits/11par_fits')
sys.path.insert(0, str(FITS_DIR))
import fits_apfix as fa                     # first: it pins the BLAS/OpenMP/TF thread counts before numpy is imported
import glob
import json
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False})
NB_DPI = 200            # resolution of figures shown in the notebook (and so of copy / paste into slides); with the 'retina' format below the inline PNG has twice this density. Saved files use DPI.
plt.rcParams['figure.dpi'] = NB_DPI
try:
    from matplotlib_inline.backend_inline import set_matplotlib_formats
    set_matplotlib_formats('retina')
except ImportError:         # not running in a Jupyter / VS Code notebook: nothing to set
    pass

CHAIN_ROOT = fa.CHAIN_ROOT
PLOT_DIR = CHAIN_ROOT / 'plots_ap_check'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
DPI = 600
TOL = 0.005              # |R - 1| below this (0.5%) counts as consistent: allows for the weak dependence of the nonlinear peak shift on the cosmology and biases
N_SAMPLES = 150          # posterior samples per chain for the diagnostic (CAMB + two model evaluations each)
DATASETS = {'dr1': 'DR1', 'mean': 'mean of mocks'}
WIDTH = 7.2
C_KMS = 299792.458
S_FINE = np.linspace(80.0, 135.0, 221)        # grid on which the peak is located (h_fid^-1 Mpc); the BAO bump sits at ~100-110
PEAK_WINDOW = (88.0, 128.0)
TRACERS = [('BGS', 'BGS', 'yellowgreen'), ('LRG1', 'LRG_0.4', 'orange'), ('LRG2', 'LRG_0.6', 'orangered'), ('LRG3', 'LRG_0.8', 'firebrick'),
           ('ELG2', 'ELG', 'steelblue'), ('QSO', 'QSO', 'seagreen')]      # (short name, substring of the chain name, DESI colour); same order as fa.FIT_KEYS
Z_OF = {short: fa.Z_EFF[key] for (short, _k, _c), key in zip(TRACERS, fa.FIT_KEYS)}
SPECS = {s['name']: s for s in fa.build_specs()}

#%% [markdown]
# ## 1. Unit checks (no chains needed)
# (a) q must be exactly 1 at the fiducial cosmology of the data. (b) Scaling h -> lam h and omega_b, omega_cdm -> lam^2 omega leaves Omega_b, Omega_cdm fixed, so q (in h^-1 Mpc units) must not change
# (to ~1e-4: radiation and neutrinos break the exact invariance). The table also gives the same numbers for the bugged variant (q without the h / h_fid factor), which fails (b) by ~|lam - 1|.

#%%
zs = [Z_OF[t[0]] for t in TRACERS]
checks = []
q_ref = np.array(fa.ap_multi(*fa.AP_REF, fa.NS_FIXED, zs))
checks.append({'check': 'max |q - 1| at the c000 reference', 'current code': float(np.abs(q_ref - 1).max()), 'without h/h_fid': float(np.abs(q_ref - 1).max()), 'tolerance': 1e-6})
for lam in (1.03, 1.05):
    h0, ob0, oc0 = fa.AP_REF
    q0 = np.array(fa.ap_multi(h0, ob0, oc0, fa.NS_FIXED, zs))
    q1 = np.array(fa.ap_multi(lam * h0, lam ** 2 * ob0, lam ** 2 * oc0, fa.NS_FIXED, zs))
    q1_bug = q1 / lam                                                                # the bugged variant divides out f = h / h_fid
    checks.append({'check': f'max |q(lam h, lam^2 omega) / q - 1|, lam = {lam}', 'current code': float(np.abs(q1 / q0 - 1).max()), 'without h/h_fid': float(np.abs(q1_bug / q0 - 1).max()), 'tolerance': 5e-4})
UNIT = pd.DataFrame(checks).set_index('check')
UNIT['current code passes'] = UNIT['current code'] < UNIT['tolerance']
UNIT['bugged variant passes'] = UNIT['without h/h_fid'] < UNIT['tolerance']
UNIT.to_csv(PLOT_DIR / 'ap_check_unit.csv')
print(UNIT.to_string(float_format=lambda x: f'{x:.3g}'))

#%% [markdown]
# ## 2. BAO-peak-position diagnostic
# R = (peak of the pipeline's s^2 xi_0, located on a fine grid) / (c000 peak x r_d / r_d^c000 x D_V^c000 / D_V), with r_d and D_V from CAMB and the c000 peak evaluated with the same (b1, b2).

#%%
def peak_position(emus, params, q_perp, q_par):
    """Location (h_fid^-1 Mpc) of the BAO bump in s^2 xi_0 of the model after the AP; NaN if there is no interior local maximum in PEAK_WINDOW."""
    y = S_FINE ** 2 * fa.model_one(emus, params, S_FINE, None, q_perp, q_par).reshape(2, -1)[0]
    i = int(np.argmax(np.where((S_FINE >= PEAK_WINDOW[0]) & (S_FINE <= PEAK_WINDOW[1]), y, -np.inf)))
    if i < 8 or i > len(S_FINE) - 9 or S_FINE[i] in PEAK_WINDOW:
        return np.nan
    sl = slice(i - 8, i + 9)
    c = np.polyfit(S_FINE[sl], y[sl], 2)
    return -c[1] / (2 * c[0])


def geometry(h, ob, oc, ns, z):
    """(r_d [Mpc], D_V [Mpc]) from CAMB, independent of ap_multi."""
    r = fa.get_background(h, ob, oc, ns)
    return r.get_derived_params()['rdrag'], ((1 + z) ** 2 * r.angular_diameter_distance(z) ** 2 * C_KMS * z / r.hubble_parameter(z)) ** (1 / 3)


_ref_geom = {}


def peak_ratio(emus, z, theta, b1, b2, bug=False):
    """R for one sample theta = (h, ombh2, omch2, lnAs, ns); bug=True removes the h / h_fid factor from the AP parameters."""
    h, ob, oc, lnA, ns = theta
    qp, ql = fa.ap_multi(h, ob, oc, ns, [z])[0]
    if bug:
        qp, ql = qp / (h / fa.AP_REF[0]), ql / (h / fa.AP_REF[0])
    P = peak_position(emus, [b1, b2, ob, oc, h, lnA, ns], qp, ql)
    h0, ob0, oc0 = fa.AP_REF
    P0 = peak_position(emus, [b1, b2, ob0, oc0, h0, fa.LNAS_C000, fa.NS_FIXED], 1.0, 1.0)
    if z not in _ref_geom:
        _ref_geom[z] = geometry(h0, ob0, oc0, fa.NS_FIXED, z)
    (rd0, dv0), (rd, dv) = _ref_geom[z], geometry(h, ob, oc, ns, z)
    return P / (P0 * (rd / rd0) * (dv0 / dv))


#%% [markdown]
# ### 2a. Scan in h at fixed omega_b, omega_cdm (c000 values) and typical biases
# The physical correlation function is the same at every h, so the pipeline's peak must track D_V^c000 / D_V exactly if the AP is right.

#%%
h0, ob0, oc0 = fa.AP_REF
h_grid = np.linspace(0.62, 0.78, 9)                      # +-15% around h_fid: the bugged peak stays inside PEAK_WINDOW over this range
SCAN = {}
for short, _k, color in TRACERS:
    z, emus = Z_OF[short], fa.get_emus('compact', Z_OF[short])
    SCAN[short] = {bug: np.array([peak_ratio(emus, z, (h, ob0, oc0, fa.LNAS_C000, fa.NS_FIXED), 1.5, 0.5, bug=bug) for h in h_grid]) for bug in (False, True)}
pd.DataFrame({f'{t} ({"bugged" if b else "current"})': SCAN[t][b] for t in SCAN for b in (False, True)}, index=h_grid).rename_axis('h').to_csv(PLOT_DIR / 'ap_check_hscan.csv')

fig, ax = plt.subplots(figsize=(WIDTH, 3.6))
ax.axhspan(-100 * TOL, 100 * TOL, color='0.85', zorder=0)
for short, _k, color in TRACERS:
    ax.plot(h_grid, 100 * (SCAN[short][False] - 1), '-', color=color, lw=1.6)
    ax.plot(h_grid, 100 * (SCAN[short][True] - 1), '--', color=color, lw=1.2)
ax.axvline(h0, color='0.4', ls=':', lw=0.9)
ax.set_xlabel(r'$h$ (at the $c000$ $\omega_b$, $\omega_{\rm cdm}$)', fontsize=10)
ax.set_ylabel(r'peak position, pipeline / expected $-\,1$  [%]', fontsize=10)
ax.tick_params(labelsize=8)
ax.legend(handles=[Line2D([0], [0], color='k', lw=1.6, label='AP as implemented'), Line2D([0], [0], color='k', ls='--', lw=1.2, label=r'without the $h/h_{\rm fid}$ factor (old bug)'), Patch(color='0.85', label=rf'$\pm${100 * TOL:.1f}% tolerance')]
          + [Line2D([0], [0], color=c, lw=1.6, label=s) for s, _k, c in TRACERS], frameon=False, fontsize=7.5, ncol=2, loc='upper left')
fig.tight_layout()
fig.savefig(PLOT_DIR / 'v2_ap_check_hscan.pdf', bbox_inches='tight')
fig.savefig(PLOT_DIR / 'v2_ap_check_hscan.png', dpi=DPI, bbox_inches='tight')
plt.show()
plt.close(fig)

#%% [markdown]
# ### 2b. The same diagnostic over the posteriors
# For every tracer, R is evaluated on N_SAMPLES posterior samples of the joint baseline fit (all six tracers share the cosmology) and of that tracer's individual fit, at the tracer's z_eff and with the
# sample's own biases. Top panel: the AP as implemented (should sit on 0); bottom panel: the same samples with the h / h_fid factor removed (what the old code would have shown). Points are medians, bars the 16-84% range.
# The mean-of-mocks fits have h ~ h_fid, where the factor is ~1, so the bug is nearly invisible for them (as it was in the old mock tests); the DR1 fits, with h away from h_fid, show it clearly.

#%%
def indiv_chain(tag, key):
    files = [Path(p).name[:-len('.status.json')] for p in glob.glob(str(CHAIN_ROOT / 'compact' / f'compact_{tag}_indiv_*_smin20.status.json'))]
    match = [f for f in files if f'_indiv_{key}' in f]
    return match[0] if match else None


def draw_samples(name, n, seed=0):
    """n random post-burn-in samples of a chain as a DataFrame (columns = param_names)."""
    root = CHAIN_ROOT / SPECS[name]['model'] / name
    st = json.load(open(f'{root}.status.json'))
    with h5py.File(f'{root}.h5', 'r') as f:
        it = int(f['mcmc'].attrs['iteration'])
        ch = f['mcmc/chain'][st['discard']:it:st['thin']]
    flat = ch.reshape(-1, ch.shape[-1])
    return pd.DataFrame(flat[np.random.default_rng(seed).choice(len(flat), min(n, len(flat)), replace=False)], columns=st['param_names'])


def ratios(df, short, key_index, joint):
    """R and R_bug for every row of df at the tracer `short`."""
    z, emus = Z_OF[short], fa.get_emus('compact', Z_OF[short])
    kn = fa.kname(fa.FIT_KEYS[key_index])
    b1c, b2c = (f'b1_{kn}', f'b2_{kn}') if joint else ('b1', 'b2')
    out = []
    for _i, r in df.iterrows():
        th = (r['h'], r['ombh2'], r['omch2'], r['logA'], r['ns'])
        out.append((peak_ratio(emus, z, th, r[b1c], r[b2c]), peak_ratio(emus, z, th, r[b1c], r[b2c], bug=True)))
    return np.array(out)


rows, RDATA = [], {}
for tag, title in DATASETS.items():
    joint_name = f'compact_{tag}_joint_smin20'
    if not (joint_name in SPECS and (CHAIN_ROOT / 'compact' / f'{joint_name}.status.json').exists()):
        print(f'{joint_name}: chain not finished, skipped')
        continue
    joint = draw_samples(joint_name, N_SAMPLES)
    for k, (short, key, color) in enumerate(TRACERS):
        ind = indiv_chain(tag, key)
        for fit, df, is_joint in (('joint baseline', joint, True), ('individual', draw_samples(ind, N_SAMPLES) if ind else None, False)):
            if df is None:
                continue
            R = ratios(df, short, k, is_joint)
            ok = np.isfinite(R).all(1)
            q = lambda col, p: float(np.percentile(R[ok, col], p)) if ok.any() else np.nan
            rows.append({'dataset': title, 'tracer': short, 'fit': fit, 'R-1 median [%]': 100 * (q(0, 50) - 1), 'R-1 16% [%]': 100 * (q(0, 16) - 1), 'R-1 84% [%]': 100 * (q(0, 84) - 1),
                         'bugged R-1 median [%]': 100 * (q(1, 50) - 1), 'bugged R-1 16% [%]': 100 * (q(1, 16) - 1), 'bugged R-1 84% [%]': 100 * (q(1, 84) - 1), 'fraction without a peak': float(1 - ok.mean())})
CHECK = pd.DataFrame(rows)
if len(CHECK):
    CHECK['current passes'] = CHECK['R-1 median [%]'].abs() < 100 * TOL
    CHECK['bug detected'] = (CHECK['bugged R-1 median [%]'].abs() > 100 * TOL)
    CHECK.to_csv(PLOT_DIR / 'ap_check_posteriors.csv', index=False)
    print(CHECK.round(3).to_string(index=False))

    for title, sub in CHECK.groupby('dataset', sort=False):
        fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 5.2), sharex=True)
        for ax, bug in zip(axes, (False, True)):
            ax.axhspan(-100 * TOL, 100 * TOL, color='0.85', zorder=0)
            ax.axhline(0, color='0.5', lw=0.8)
            pre = 'bugged R-1' if bug else 'R-1'
            for k, (short, _key, color) in enumerate(TRACERS):
                for off, fit, fc in ((-0.12, 'joint baseline', 'k'), (0.12, 'individual', color)):
                    r = sub[(sub.tracer == short) & (sub.fit == fit)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    ax.errorbar(k + off, r[f'{pre} median [%]'], yerr=[[r[f'{pre} median [%]'] - r[f'{pre} 16% [%]']], [r[f'{pre} 84% [%]'] - r[f'{pre} median [%]']]], fmt='o', ms=4.5, color=fc,
                                capsize=2.5, elinewidth=1.1)
            ax.set_ylabel(r'peak position, pipeline / expected $-\,1$  [%]' + ('\n(old bug: no $h/h_{\\rm fid}$)' if bug else '\n(AP as implemented)'), fontsize=8.5)
            ax.tick_params(labelsize=8)
        axes[-1].set_xticks(range(len(TRACERS)))
        axes[-1].set_xticklabels([t[0] for t in TRACERS], fontsize=9)
        axes[0].legend(handles=[Line2D([], [], marker='o', color='k', ls='none', label='joint baseline fit'), Line2D([], [], marker='o', color='0.5', ls='none', label='individual-tracer fit (tracer colour)'),
                                Patch(color='0.85', label=rf'$\pm${100 * TOL:.1f}% tolerance')], frameon=False, fontsize=7.5, loc='upper right')
        axes[0].set_title(title, fontsize=9)
        fig.tight_layout()
        tagf = 'dr1' if title == 'DR1' else 'mean'
        fig.savefig(PLOT_DIR / f'v2_ap_check_posteriors_{tagf}.pdf', bbox_inches='tight')
        fig.savefig(PLOT_DIR / f'v2_ap_check_posteriors_{tagf}.png', dpi=DPI, bbox_inches='tight')
        plt.show()
        plt.close(fig)

#%% [markdown]
# ## Verdict

#%%
scan_ok = all(np.nanmax(np.abs(SCAN[t][False] - 1)) < TOL and not np.isnan(SCAN[t][False]).any() for t in SCAN)
scan_detects = all(np.nanmax(np.abs(SCAN[t][True] - 1)) > 3 * TOL for t in SCAN)
post_ok = bool(CHECK['current passes'].all()) if len(CHECK) else None
dr1 = CHECK[CHECK.dataset == 'DR1'] if len(CHECK) else CHECK
post_detects = bool(dr1['bug detected'].all()) if len(dr1) else None
AP_CHECK_PASSED = bool(UNIT['current code passes'].all() and scan_ok and post_ok is not False)
print(f"unit checks pass: {bool(UNIT['current code passes'].all())} (bugged variant would pass: {bool(UNIT['bugged variant passes'].all())})")
print(f'h scan: current AP within {100 * TOL:.1f}% at every h and tracer: {scan_ok}; the bugged variant exceeds 3x the tolerance: {scan_detects}')
print(f'posterior check: current AP within tolerance for every fit: {post_ok}; bug detected in every DR1 fit: {post_detects}')
print('(the mock fits sit at h ~ h_fid, where the h / h_fid factor is ~1, so the old bug is largely invisible there; that is how it escaped the mock tests)')
print('AP_CHECK_PASSED =', AP_CHECK_PASSED)