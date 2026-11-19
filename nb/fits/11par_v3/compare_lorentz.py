# %% [markdown]
# Raw QSO-only fits vs the same fits with the free Lorentzian redshift-error convolution (fit_qso_lorentz.py).
# Run cell by cell. Re-run the "load" cell to pick up chains that have finished since (DR1 is skipped until its chains exist);
# every plot cell is independent. Per-object Lorentzian half-width w; literature bands are per-object too.

# %% imports and settings
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from getdist import loadMCSamples, plots

CHAIN_DIR = Path('/spiffball/cosweeney/data/chains/fits_11par_apfix_v2_v3/compact_v3_noslope')
MODEL = 'compact_v3_noslope'
SMIN = 25                  # s_min of both chains (20 compares with the existing raw s_min = 20 chains)
CONV_TAG = ''              # suffix of the convolution chain names, i.e. the --tag given to fit_qso_lorentz.py (e.g. '_zero')
OUT = None                 # directory to save figures to, e.g. Path('figs'); None = just show them

DATASETS = {'mean': 'mean of mocks', 'dr1': 'DR1'}
BASE_PARAMS = ['h', 'ombh2', 'omch2', 'logA', 'b1', 'b2', 'ns']
PRIOR_RANGES = {'b1': (0.0, 2.0), 'b2': (-5.0, 10.0)}               # fits_apfix.WIDE
COLORS = {'raw': 'C0', 'conv': 'C3'}
LEGEND = ['raw (no convolution)', 'with Lorentzian convolution']
# Yu et al. 2023 (clustering-calibrated, truncated at 2000 km/s, DESI One-Percent): (label, w, +err, -err); repeat observations cannot see line shifts -> lower bound
LIT = [('Yu+23, 0.8<z<1.1', 101, 87, 59), ('Yu+23, 1.1<z<1.6', 78, 35, 26), ('Yu+23, 1.6<z<2.1', 273, 151, 61), ('Yu+23, 0.8<z<3.5', 215, 24, 31)]
LIT_COLORS = ['0.55', '0.65', '0.75', '0.45']
REPEAT_OBS, HE25 = 30.2, 100.0                                        # km/s; He+25 adopt 100 km/s with a 3000 km/s truncation


# %% helpers
def chain_roots(tag):
    return (CHAIN_DIR / f'{MODEL}_{tag}_indiv_QSO_0.8-2.1_smin{SMIN:g}',
            CHAIN_DIR / f'{MODEL}_{tag}_indiv_QSO_lorentz_smin{SMIN:g}{CONV_TAG}')


def load(root, label):
    for ext in ('.txt', '.paramnames'):
        if not Path(f'{root}{ext}').exists():
            print(f'  missing {Path(root).name}{ext}')
            return None
    st = Path(f'{root}.status.json')
    if st.exists():
        s = json.load(open(st))
        print(f"  {Path(root).name}: {'converged' if s['converged'] else 'NOT converged'}, {s['iterations']} steps, N/tau_max = {s['n_over_tau_max']:.0f}")
    samples = loadMCSamples(str(root), settings={'ignore_rows': 0})
    samples.label = label
    if 'log10w' in samples.paramNames.list():
        samples.addDerived(10.0 ** samples.getParams().log10w, name='w', label=r'w\,[{\rm km\,s^{-1}}]')
    return samples


def fmt(samples, name):
    p = samples.getMargeStats().parWithName(name)
    lo, hi = p.limits[0].lower, p.limits[0].upper                     # 1-D marginalized 68% limits, the widths the triangles show
    return f'{p.mean:9.4g}   (one-tail limit)' if lo is None or hi is None else f'{p.mean:9.4g}  -{p.mean - lo:.3g} +{hi - p.mean:.3g}'


def edge_fraction(samples, name, frac=0.1):
    lo, hi = PRIOR_RANGES[name]
    x, w = getattr(samples.getParams(), name), samples.weights
    return float(np.sum(w * (x < lo + frac * (hi - lo))) / w.sum()), float(np.sum(w * (x > hi - frac * (hi - lo))) / w.sum())


def summary(tag):
    raw, conv = RAW.get(tag), CONV.get(tag)
    print(f'\n=== {DATASETS[tag]} ===')
    print(f"{'param':8s} {'raw':>28s} {'with convolution':>28s}")
    for n in BASE_PARAMS + ['w', 'log10w']:
        a = fmt(raw, n) if raw is not None and n in raw.paramNames.list() else ''
        b = fmt(conv, n) if conv is not None and n in conv.paramNames.list() else ''
        if a or b:
            print(f'{n:8s} {a:>28s} {b:>28s}')
    for label, s in (('raw', raw), ('conv', conv)):
        for n in ('b1', 'b2') if s is not None else ():
            lo, hi = edge_fraction(s, n)
            print(f'  {label:4s} {n}: {100 * lo:5.1f}% in the lowest 10% of its prior range, {100 * hi:5.1f}% in the highest 10%')
    if raw is not None and conv is not None:
        print(f'  min(-2 ln P): raw {2 * raw.loglikes.min():.1f}, conv {2 * conv.loglikes.min():.1f}  (approx., thinned chains; conv has one extra parameter)')


def save(g, name):
    if OUT is not None:
        Path(OUT).mkdir(parents=True, exist_ok=True)
        g.export(str(Path(OUT) / name))


# %% load (re-run to pick up new chains)
RAW, CONV = {}, {}
for tag in DATASETS:
    print(f'{DATASETS[tag]}:')
    r_root, c_root = chain_roots(tag)
    for store, root, label in ((RAW, r_root, 'raw'), (CONV, c_root, 'with convolution')):
        s = load(root, label)
        if s is not None:
            store[tag] = s

# %% table: mock mean
summary('mean') if ('mean' in RAW or 'mean' in CONV) else print('no mock-mean chains yet')

# %% table: DR1
summary('dr1') if ('dr1' in RAW or 'dr1' in CONV) else print('no DR1 chains yet')

# %% triangle: mock mean
tag = 'mean'
if tag in RAW and tag in CONV:
    g = plots.get_subplot_plotter()
    g.triangle_plot([RAW[tag], CONV[tag]], BASE_PARAMS, filled=True, legend_labels=LEGEND, contour_colors=[COLORS['raw'], COLORS['conv']], title_limit=1)
    g.fig.suptitle(DATASETS[tag] + ' (QSO only)', y=1.0)
    save(g, f'qso_conv_triangle_{tag}.png')
    plt.show()
else:
    print('need both the raw and the convolution mock-mean chains')

# %% triangle: DR1
tag = 'dr1'
if tag in RAW and tag in CONV:
    g = plots.get_subplot_plotter()
    g.triangle_plot([RAW[tag], CONV[tag]], BASE_PARAMS, filled=True, legend_labels=LEGEND, contour_colors=[COLORS['raw'], COLORS['conv']], title_limit=1)
    g.fig.suptitle(DATASETS[tag] + ' (QSO only)', y=1.0)
    save(g, f'qso_conv_triangle_{tag}.png')
    plt.show()
else:
    print('need both the raw and the convolution DR1 chains')

# %% b1, b2 with the prior edges (dotted): both datasets that are available
for tag in DATASETS:
    if tag in RAW and tag in CONV:
        g = plots.get_single_plotter(width_inch=8)
        g.plots_1d([RAW[tag], CONV[tag]], ['b1', 'b2'], nx=2, legend_labels=LEGEND, colors=[COLORS['raw'], COLORS['conv']])
        for ax, n in zip(g.fig.axes[:2], ('b1', 'b2')):
            for v in PRIOR_RANGES[n]:
                ax.axvline(v, color='k', ls=':', lw=1)
        g.fig.suptitle(DATASETS[tag] + ': bias posteriors (dotted: prior edges)', y=1.02)
        save(g, f'qso_conv_b1b2_{tag}.png')
        plt.show()

# %% convolution chains only: how w degenerates with the biases and amplitude (b2 vs log10 w is the panel of interest)
for tag in DATASETS:
    if tag in CONV:
        g = plots.get_subplot_plotter()
        g.triangle_plot([CONV[tag]], ['logA', 'omch2', 'b1', 'b2', 'log10w'], filled=True, contour_colors=[COLORS['conv']], title_limit=1)
        g.fig.suptitle(DATASETS[tag] + ': with Lorentzian convolution', y=1.0)
        save(g, f'qso_conv_w_degeneracies_{tag}.png')
        plt.show()

# %% width posterior vs literature: whichever convolution chains exist
have = {t: CONV[t] for t in DATASETS if t in CONV}
if have:
    g = plots.get_single_plotter(width_inch=7)
    g.plots_1d(list(have.values()), ['log10w'], legend_labels=[DATASETS[t] for t in have], colors=['C3', 'C1'][:len(have)])
    ax = g.fig.axes[0]
    for (name, w, up, dn), c in zip(LIT, LIT_COLORS):
        ax.axvspan(np.log10(w - dn), np.log10(w + up), color=c, alpha=0.25, lw=0)
        ax.axvline(np.log10(w), color=c, lw=1.2, label=name)
    ax.axvline(np.log10(REPEAT_OBS), color='k', ls='--', lw=1, label='repeat observations (lower bound)')
    ax.axvline(np.log10(HE25), color='k', ls='-.', lw=1, label='He+25 mocks')
    ax.set_xlabel(r'$\log_{10}(w\,/\,{\rm km\,s^{-1}})$, per-object Lorentzian half-width')
    ax.legend(fontsize=7, loc='upper left')
    save(g, 'qso_conv_w.png')
    plt.show()
else:
    print('no convolution chains yet')
# %%
