# %% [markdown]
# Raw QSO-only fits vs the Lorentzian convolution (fit_qso_lorentz.py) vs the sigma_err variance addition (fit_qso_sigerr.py).
# Run cell by cell. Re-run the "load" cell to pick up chains that have finished since; every plot cell is independent and skips missing chains.
# The Lorentzian width w is mapped to the PAIR RMS sqrt(2) x (per-object RMS of the truncated Lorentzian), the quantity sigma_err stands for,
# so the two treatments can be compared directly.

# %% imports and settings
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from getdist import loadMCSamples, plots

CHAIN_DIR = Path('/spiffball/cosweeney/data/chains/fits_11par_apfix_v2_v3/compact_v3_noslope')
MODEL = 'compact_v3_noslope'
SMIN = 25                  # s_min of all three chains
CONV_TAG, SIG_TAG = '', ''  # suffixes of the Lorentzian / sigma_err chain names (the --tag given to the fit scripts)
OUT = None                 # directory to save figures to, e.g. Path('figs'); None = just show them

DATASETS = {'mean': 'mean of mocks', 'dr1': 'DR1'}
BASE_PARAMS = ['h', 'ombh2', 'omch2', 'logA', 'b1', 'b2', 'ns']
PRIOR_RANGES = {'b1': (0.0, 2.0), 'b2': (-5.0, 10.0)}               # fits_apfix.WIDE
KINDS = ['raw', 'conv', 'sig']
COLORS = {'raw': 'C0', 'conv': 'C3', 'sig': 'C1'}
NAMES = {'raw': 'raw (no smear)', 'conv': 'Lorentzian convolution', 'sig': r'$\sigma_{\rm err}$ variance addition'}
# Yu et al. 2023 (clustering-calibrated, truncated at 2000 km/s): (label, per-object w)
LIT = [('Yu+23 0.8<z<1.1', 101.0), ('Yu+23 1.1<z<1.6', 78.0), ('Yu+23 1.6<z<2.1', 273.0), ('Yu+23 0.8<z<3.5', 215.0)]
SIG_REF = {'eBOSS-like Gaussian RMS 300': 300.0 * np.sqrt(2.0)}      # pair sigma_err, km/s


# %% helpers
def pair_rms(w, L):
    """Pair dispersion sqrt(2) x RMS of a Lorentzian of half-width w truncated at |e| < L (exact): sqrt(2) w sqrt((L/w - arctan(L/w)) / arctan(L/w))."""
    a = np.arctan(L / w)
    return np.sqrt(2.0) * w * np.sqrt((L / w - a) / a)


def chain_roots(tag):
    return {'raw': CHAIN_DIR / f'{MODEL}_{tag}_indiv_QSO_0.8-2.1_smin{SMIN:g}',
            'conv': CHAIN_DIR / f'{MODEL}_{tag}_indiv_QSO_lorentz_smin{SMIN:g}{CONV_TAG}',
            'sig': CHAIN_DIR / f'{MODEL}_{tag}_indiv_QSO_sigerr_smin{SMIN:g}{SIG_TAG}'}


def load(root, kind):
    for ext in ('.txt', '.paramnames'):
        if not Path(f'{root}{ext}').exists():
            print(f'  missing {Path(root).name}{ext}')
            return None, {}
    spec = {}
    st = Path(f'{root}.status.json')
    if st.exists():
        s = json.load(open(st))
        spec = s.get('spec', {})
        print(f"  {Path(root).name}: {'converged' if s['converged'] else 'NOT converged'}, {s['iterations']} steps, N/tau_max = {s['n_over_tau_max']:.0f}")
    samples = loadMCSamples(str(root), settings={'ignore_rows': 0})
    samples.label = NAMES[kind]
    p = samples.getParams()
    if kind == 'conv':
        w = 10.0 ** p.log10w
        samples.addDerived(w, name='w', label=r'w\,[{\rm km\,s^{-1}}]')
        samples.addDerived(pair_rms(w, spec.get('L1', 2000.0)), name='sigerr', label=r'\sigma_{\rm pair}\,[{\rm km\,s^{-1}}]')   # same name as the sigma_err chain's
    if kind == 'sig':
        samples.addDerived(10.0 ** p.log10sigerr, name='sigerr', label=r'\sigma_{\rm err}\,[{\rm km\,s^{-1}}]')
    return samples, spec


def fmt(samples, name):
    p = samples.getMargeStats().parWithName(name)
    lo, hi = p.limits[0].lower, p.limits[0].upper                     # 1-D marginalized 68% limits, the widths the triangles show
    return f'{p.mean:9.4g}   (one-tail)' if lo is None or hi is None else f'{p.mean:9.4g} -{p.mean - lo:.3g} +{hi - p.mean:.3g}'


def edge_fraction(samples, name, frac=0.1):
    lo, hi = PRIOR_RANGES[name]
    x, w = getattr(samples.getParams(), name), samples.weights
    return float(np.sum(w * (x < lo + frac * (hi - lo))) / w.sum()), float(np.sum(w * (x > hi - frac * (hi - lo))) / w.sum())


def summary(tag):
    S = {k: STORE[k].get(tag) for k in KINDS}
    print(f'\n=== {DATASETS[tag]} ===')
    print(f"{'param':8s}" + ''.join(f'{k:>26s}' for k in KINDS))
    for n in BASE_PARAMS + ['w', 'sigerr']:
        cells = [fmt(S[k], n) if S[k] is not None and n in S[k].paramNames.list() else '' for k in KINDS]
        if any(cells):
            print(f'{n:8s}' + ''.join(f'{c:>26s}' for c in cells))
    for k in KINDS:
        for n in ('b1', 'b2') if S[k] is not None else ():
            lo, hi = edge_fraction(S[k], n)
            print(f'  {k:4s} {n}: {100 * lo:5.1f}% in the lowest 10% of its prior range, {100 * hi:5.1f}% in the highest 10%')
    best = {k: 2 * S[k].loglikes.min() for k in KINDS if S[k] is not None}
    print('  min(-2 ln P) (approx., thinned chains; conv and sig have one extra parameter): ' + ', '.join(f'{k} {v:.1f}' for k, v in best.items()))


def save(g, name):
    if OUT is not None:
        Path(OUT).mkdir(parents=True, exist_ok=True)
        g.export(str(Path(OUT) / name))


# %% load (re-run to pick up new chains)
STORE = {k: {} for k in KINDS}
SPECS = {k: {} for k in KINDS}
for tag in DATASETS:
    print(f'{DATASETS[tag]}:')
    for kind, root in chain_roots(tag).items():
        s, spec = load(root, kind)
        if s is not None:
            STORE[kind][tag], SPECS[kind][tag] = s, spec

# %% table: mock mean
summary('mean') if any('mean' in STORE[k] for k in KINDS) else print('no mock-mean chains yet')

# %% table: DR1
summary('dr1') if any('dr1' in STORE[k] for k in KINDS) else print('no DR1 chains yet')

# %% triangles: every treatment that exists for the dataset (raw vs the others)
for tag in DATASETS:
    kinds = [k for k in KINDS if tag in STORE[k]]
    if len(kinds) < 2:
        print(f'{DATASETS[tag]}: need at least two of the three chains')
        continue
    g = plots.get_subplot_plotter()
    g.triangle_plot([STORE[k][tag] for k in kinds], BASE_PARAMS, filled=True, legend_labels=[NAMES[k] for k in kinds],
                    contour_colors=[COLORS[k] for k in kinds], title_limit=1)
    g.fig.suptitle(DATASETS[tag] + ' (QSO only)', y=1.0)
    save(g, f'qso_sigerr_triangle_{tag}.png')
    plt.show()

# %% b1, b2 with the prior edges (dotted)
for tag in DATASETS:
    kinds = [k for k in KINDS if tag in STORE[k]]
    if len(kinds) >= 2:
        g = plots.get_single_plotter(width_inch=8)
        g.plots_1d([STORE[k][tag] for k in kinds], ['b1', 'b2'], nx=2, legend_labels=[NAMES[k] for k in kinds], colors=[COLORS[k] for k in kinds])
        for ax, n in zip(g.fig.axes[:2], ('b1', 'b2')):
            for v in PRIOR_RANGES[n]:
                ax.axvline(v, color='k', ls=':', lw=1)
        g.fig.suptitle(DATASETS[tag] + ': bias posteriors (dotted: prior edges)', y=1.14)
        save(g, f'qso_sigerr_b1b2_{tag}.png')
        plt.show()

# %% sigma_err degeneracies (sigma_err chain only): b2 vs log10 sigma_err is the panel of interest
for tag in DATASETS:
    if tag in STORE['sig']:
        g = plots.get_subplot_plotter()
        g.triangle_plot([STORE['sig'][tag]], ['logA', 'omch2', 'b1', 'b2', 'log10sigerr'], filled=True, contour_colors=[COLORS['sig']], title_limit=1)
        g.fig.suptitle(DATASETS[tag] + r': $\sigma_{\rm err}$ variance addition', y=1.0)
        save(g, f'qso_sigerr_degeneracies_{tag}.png')
        plt.show()

# %% the same quantity two ways: pair sigma from the Lorentzian fit (w mapped through its truncation) vs sigma_err; prior ranges dashed
for tag in DATASETS:
    if tag in STORE['conv'] and tag in STORE['sig']:
        g = plots.get_single_plotter(width_inch=7.5)
        g.plots_1d([STORE['conv'][tag], STORE['sig'][tag]], ['sigerr'], legend_labels=[NAMES['conv'] + r' ($\sigma_{\rm pair}$ from $w$)', NAMES['sig']],
                   colors=[COLORS['conv'], COLORS['sig']])
        ax = g.fig.axes[0]
        sc, ss = SPECS['conv'][tag], SPECS['sig'][tag]
        L1 = sc.get('L1', 2000.0)
        if 'wmin' in sc:
            for v in (pair_rms(sc['wmin'], L1), pair_rms(sc['wmax'], L1)):
                ax.axvline(v, color=COLORS['conv'], ls='--', lw=1)
        if 'errmin' in ss:
            for v in (ss['errmin'], ss['errmax']):
                ax.axvline(v, color=COLORS['sig'], ls='--', lw=1)
        for (name, w), c in zip(LIT, ['C2', 'C4', 'C5', 'C8']):
            ax.axvline(pair_rms(w, L1), color=c, lw=1, label=name + r' (as $\sigma_{\rm pair}$)')
        for name, v in SIG_REF.items():
            ax.axvline(v, color='k', ls='-.', lw=1, label=name)
        ax.set_xlabel(r'pair $\sigma$ [km/s]  ($\sqrt{2}\times$ per-object RMS)')
        ax.legend(fontsize=7, loc='upper right')
        g.fig.suptitle(DATASETS[tag] + ': dashed = prior ranges of each fit', y=1.02)
        save(g, f'qso_sigerr_vs_lorentz_{tag}.png')
        plt.show()
    else:
        print(f'{DATASETS[tag]}: need both the Lorentzian and the sigma_err chains')

# %% sigma_err degeneracies (sigma_err chain only): b2 vs log10 sigma_err is the panel of interest
for tag in DATASETS:
    if tag in STORE['sig']:
        g = plots.get_subplot_plotter()
        g.triangle_plot([STORE['sig'][tag]], ['logA', 'omch2', 'b1', 'b2', 'log10sigerr'], filled=True, contour_colors=[COLORS['sig']],
                        legend_labels=[f"{DATASETS[tag]}: {NAMES['sig']}"], title_limit=1)
        save(g, f'qso_sigerr_degeneracies_{tag}.png')
        plt.show()