# %% [markdown]
# Inspect the recent chains: sigma_err QSO-only (mock mean, DR1), and the JOINT mock-mean fits (raw, Lorentzian convolution, sigma_err). Run cell by cell; re-run
# the "load" cell to pick up chains that have finished since. Everything skips chains that do not exist yet.

# %% settings and load
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from getdist import loadMCSamples, plots

CHAIN_DIR = Path('/spiffball/cosweeney/data/chains/fits_11par_apfix_v2_v3/compact_v3_noslope')
M, SMIN, OUT = 'compact_v3_noslope', 25, None                      # OUT: directory to save figures to (None = just show)
QSO = '_QSO_0.8-2.1'
CHAINS = {'joint raw': f'{M}_mean_joint_smin{SMIN}', 'joint Lorentz': f'{M}_mean_joint_lorentz_smin{SMIN}', 'joint sigma_err': f'{M}_mean_joint_sigerr_smin{SMIN}',
          'QSO Lorentz (DR1)': f'{M}_dr1_indiv_QSO_lorentz_smin{SMIN}', 'QSO sigma_err (DR1)': f'{M}_dr1_indiv_QSO_sigerr_smin{SMIN}'}
COLORS = {'joint raw': 'C0', 'joint Lorentz': 'C3', 'joint sigma_err': 'C1', 'QSO Lorentz (DR1)': 'C2', 'QSO sigma_err (DR1)': 'C4'}
PRIOR = {'b1': (0.0, 2.0), 'b2': (-5.0, 10.0)}


def pair_rms(w, L):          # pair sigma = sqrt(2) x RMS of a Lorentzian of half-width w truncated at |e| < L
    a = np.arctan(L / w)
    return np.sqrt(2.0) * w * np.sqrt((L / w - a) / a)


def load(label, name):
    root = CHAIN_DIR / name
    if not (Path(f'{root}.txt').exists() and Path(f'{root}.paramnames').exists()):
        print(f'  missing: {name}')
        return None
    spec = {}
    if Path(f'{root}.status.json').exists():
        st = json.load(open(f'{root}.status.json'))
        spec = st.get('spec', {})
        print(f"  {label:26s} {'converged' if st['converged'] else 'NOT converged'}, {st['iterations']} steps, N/tau_max = {st['n_over_tau_max']:.0f}")
    s = loadMCSamples(str(root), settings={'ignore_rows': 0})
    s.label = label
    names = s.paramNames.list()
    if 'log10w' in names:
        i = names.index('log10w')
        s.addDerived(pair_rms(10.0 ** s.samples[:, i], spec.get('L1', 2000.0)), name='sigerr', label=r'\sigma_{\rm pair}\,[{\rm km\,s^{-1}}]')
    if 'log10sigerr' in names:
        s.addDerived(10.0 ** s.samples[:, names.index('log10sigerr')], name='sigerr', label=r'\sigma_{\rm err}\,[{\rm km\,s^{-1}}]')
    return s


S = {}
for label, name in CHAINS.items():
    s = load(label, name)
    if s is not None:
        S[label] = s


def bias_names(s):           # joint chains name the QSO biases per tracer-bin, QSO-only chains plain b1, b2
    n = s.paramNames.list()
    return (f'b1{QSO}', f'b2{QSO}') if f'b2{QSO}' in n else ('b1', 'b2')


def fmt(s, name):
    if name not in s.paramNames.list():
        return ''
    p = s.getMargeStats().parWithName(name)
    lo, hi = p.limits[0].lower, p.limits[0].upper                  # 1-D marginalized 68% limits, as in the triangles
    return f'{p.mean:8.4g} (one-tail)' if lo is None or hi is None else f'{p.mean:8.4g} -{p.mean - lo:.3g} +{hi - p.mean:.3g}'


def edge(s, which):          # percent of the posterior in the lowest / highest 10% of the QSO b1 or b2 prior range
    name, (lo, hi) = bias_names(s)[0 if which == 'b1' else 1], PRIOR[which]
    x, w = s.samples[:, s.paramNames.list().index(name)], s.weights
    return 100 * np.sum(w * (x < lo + 0.1 * (hi - lo))) / w.sum(), 100 * np.sum(w * (x > hi - 0.1 * (hi - lo))) / w.sum()


# %% table
rows = ['h', 'ombh2', 'omch2', 'logA', 'ns', 'b1', 'b2', 'sigerr']
print(f"{'':8s}" + ''.join(f'{k:>30s}' for k in S))
for r in rows:
    print(f'{r:8s}' + ''.join(f"{fmt(s, {'b1': bias_names(s)[0], 'b2': bias_names(s)[1]}.get(r, r)):>30s}" for s in S.values()))
print()
for k, s in S.items():
    e1, e2 = edge(s, 'b1'), edge(s, 'b2')
    print(f'{k:26s} QSO b1 edges {e1[0]:4.1f}% / {e1[1]:4.1f}%   b2 edges {e2[0]:4.1f}% / {e2[1]:4.1f}%   min(-2 ln P) = {2 * s.loglikes.min():.1f}')

# %% joint fits: cosmology and QSO biases, raw vs smear treatments
J = [k for k in ('joint raw', 'joint Lorentz', 'joint sigma_err') if k in S]
if len(J) >= 2:
    p = ['h', 'omch2', 'logA', 'ns', f'b1{QSO}', f'b2{QSO}']
    g = plots.get_subplot_plotter()
    g.triangle_plot([S[k] for k in J], p, filled=False, legend_labels=J, contour_colors=[COLORS[k] for k in J], title_limit=1)
    if OUT:
        g.export(str(Path(OUT) / 'recent_joint_triangle.png'))
    plt.show()

# %% smear degeneracies in the joint fits (QSO b2 vs the smear parameter)
for k, par in (('joint Lorentz', 'log10w'), ('joint sigma_err', 'log10sigerr')):
    if k in S:
        g = plots.get_subplot_plotter()
        g.triangle_plot([S[k]], ['omch2', 'logA', f'b1{QSO}', f'b2{QSO}', par], filled=True, contour_colors=[COLORS[k]], title_limit=1)
        g.fig.suptitle(k, y=1.0)
        if OUT:
            g.export(str(Path(OUT) / f"recent_{k.replace(' ', '_')}_degeneracies.png"))
        plt.show()

# %% pair sigma posteriors side by side (the Lorentzian width is mapped through its truncation)
D = [k for k in ('joint Lorentz', 'joint sigma_err', 'QSO sigma_err (mock mean)', 'QSO sigma_err (DR1)') if k in S]
if D:
    g = plots.get_single_plotter(width_inch=7.5)
    g.plots_1d([S[k] for k in D], ['sigerr'], legend_labels=D, colors=[COLORS[k] for k in D])
    if OUT:
        g.export(str(Path(OUT) / 'recent_sigma_posteriors.png'))
    plt.show()

# %% pair sigma posteriors side by side, with each fit's prior limits as dashed lines
SHOW_68 = False        # True: also draw each posterior's 1-D 68% limits as dotted lines
D = [k for k in ('joint Lorentz', 'joint sigma_err', 'QSO Lorentz (DR1)', 'QSO sigma_err (DR1)') if k in S]


def prior_limits(label):
    """(low, high) of the chain's smear prior in pair-sigma units, from its recorded spec (the Lorentzian w prior mapped through its truncation)."""
    st = CHAIN_DIR / f'{CHAINS[label]}.status.json'
    if not st.exists():
        return None
    spec = json.load(open(st)).get('spec', {})
    if 'errmin' in spec:
        return spec['errmin'], spec['errmax']
    if 'wmin' in spec:
        L = spec.get('L1', 2000.0)
        return pair_rms(spec['wmin'], L), pair_rms(spec['wmax'], L)
    return None


if D:
    g = plots.get_single_plotter(width_inch=7.5)
    g.plots_1d([S[k] for k in D], ['sigerr'], legend_labels=D, colors=[COLORS[k] for k in D])
    ax = g.fig.axes[0]
    xs = []
    for k in D:
        lim = prior_limits(k)
        if lim is not None:
            for v in lim:
                ax.axvline(v, color=COLORS[k], ls='--', lw=1)
            xs += list(lim)
        if SHOW_68:
            l68 = S[k].getMargeStats().parWithName('sigerr').limits[0]
            for v in (l68.lower, l68.upper):
                if v is not None:
                    ax.axvline(v, color=COLORS[k], ls=':', lw=1)
    if xs:                                           # widen the axis so the upper prior edges are visible
        lo, hi = ax.get_xlim()
        ax.set_xlim(min(lo, min(xs)), max(hi, 1.03 * max(xs)))
    if getattr(g, 'legend', None) is not None:
        g.legend.set_title('dashed: prior limits' + (', dotted: 68%' if SHOW_68 else ''))
    if OUT:
        g.export(str(Path(OUT) / 'recent_sigma_posteriors.png'))
    plt.show()

# %% Lorentzian width w directly: posterior with the prior limits as vertical dashed lines
LZ = [k for k in S if 'Lorentz' in k]                           # every Lorentzian chain that is loaded
NBINS = 30


def w_prior(label):
    """(wmin, wmax) of the chain's prior, from its recorded spec."""
    st = CHAIN_DIR / f'{CHAINS[label]}.status.json'
    spec = json.load(open(st)).get('spec', {}) if st.exists() else {}
    return spec.get('wmin', 20.0), spec.get('wmax', 600.0)


if LZ:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xs = []
    for k in LZ:
        lo, hi = w_prior(k)
        x = S[k].samples[:, S[k].paramNames.list().index('log10w')]
        edges = np.linspace(np.log10(lo), np.log10(hi), NBINS + 1)
        h, e = np.histogram(x, bins=edges, weights=S[k].weights, density=True)       # density per decade, integrates to 1
        ax.stairs(h, 10.0 ** e, color=COLORS[k], lw=1.8, label=k)
        for v in (lo, hi):
            ax.axvline(v, color=COLORS[k], ls='--', lw=1)
        xs += [lo, hi]
    ax.plot([], [], color='k', ls='--', lw=1, label='prior limits')
    ax.set_xscale('log')
    ax.set_xlim(min(xs) / 1.3, max(xs) * 1.3)                                        # room to see the limits
    ax.set_xlabel(r'$w\ [\mathrm{km\,s^{-1}}]$  (per-object Lorentzian half-width)')
    ax.set_ylabel(r'$p(\log_{10} w)$')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if OUT:
        fig.savefig(str(Path(OUT) / 'recent_w_posteriors.png'), dpi=200)
    plt.show()