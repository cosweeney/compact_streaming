"""
Slide-size plots of the latest "no slope" fits (compact_v3_noslope, s_min = 25 h^-1 Mpc, joint fit of all six tracer-bins).
Helpers (chain loading, MAP evaluation, getdist settings) follow plot_v2_chains_gd.py / fits_apfix_11par_v3.py. Run cell by cell (#%% markers)
or as a script, compact environment (needs getdist, h5py, camb).

  1. talk_triangle_mock:     (omega_cdm, h, sigma_8, n_s) of the mean-of-mocks fit, with the c000 truth marked (omega_b is prior dominated and left out)
  2. talk_triangle_dr1:      the same for the DR1 fit, with the DESI DR1 full-shape-only and BAO-only chains
  3. talk_fit_quality_mock:  data vs MAP model, s^2 xi_0 and s^2 xi_2 of the six tracer-bins (two rows of three) with residuals; the errors and the chi2 in the
                             titles are those of the covariance the fit used (for the mean-of-mocks fit: the single-mock covariance)
  4. talk_fit_quality_dr1:   the same for DR1
  5. talk_fom_matrix_dr1:    lower-triangle matrix of the figure-of-merit ratio FoM(ours) / FoM(DESI FS-only), FoM = 1 / sqrt(det C), for the triangle-plot parameters (DR1)

sigma_8 is not sampled: it is computed from (omega_b, omega_cdm, h, ln A_s, n_s) as sqrt(A_s) x a smooth function of (omega_b, omega_cdm, h, n_s)
that is fitted to CAMB runs (CAMB's get_sigma8_0, as cobaya computes it; one massive 0.06 eV neutrino); the held-out error is printed. DESI chains
that contain their own sigma8 column use it.
"""
#%% setup
import json
import sys
from itertools import combinations_with_replacement
from pathlib import Path

import camb
import h5py
import matplotlib.pyplot as plt
import numpy as np
from getdist import MCSamples, loadMCSamples, plots
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import CubicSpline
from scipy.stats import chi2 as chi2_dist, qmc

NB_DPI = 200            # figures shown in the notebook (and so copy / paste into slides); with the 'retina' format below the inline PNG has twice this density
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False,
                     'axes.xmargin': 0, 'axes.labelsize': 22, 'xtick.labelsize': 18, 'ytick.labelsize': 18, 'figure.dpi': NB_DPI})
try:
    from matplotlib_inline.backend_inline import set_matplotlib_formats
    set_matplotlib_formats('retina')
except ImportError:         # not running in a Jupyter / VS Code notebook: nothing to set
    pass

CHAIN_ROOT = Path('/spiffball/cosweeney/data/chains/fits_11par_apfix_v2_v3')
PLOT_DIR = CHAIN_ROOT / 'plots_talk'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
DESI_FS = '/spiffball/cosweeney/DESI/DR1/full-shape-cosmo-params/v1.0/cobaya/base/desi-reptvelocileptors-fs-all_schoneberg2024-bbn_planck2018-ns10/'
DESI_BAO = '/spiffball/cosweeney/DESI/DR1/bao-cosmo-params/v1.0/cobaya/base/desi-bao-all_schoneberg2024-bbn/'

PREFIX, SMIN = 'compact_v3_noslope', 25
OUR_LABEL = rf'Non-Gaussian model, $s_{{\rm min}}={SMIN}\ h^{{-1}}$Mpc'      # legend label of our chains in all the plots
MOCK_FIT = f'mean_joint_smin{SMIN}'
DR1_FIT = f'dr1_joint_smin{SMIN}'                  # raw (not de-biased) DR1 fit, used in all the DR1 plots
DPI = 600                                          # PNG resolution (the PDFs are vector)
WIDTH_TRI = 10.0
LEGEND_FS = 21                                     # triangle-plot legend font size; shrunk if the legend would reach the first panel                                   # triangle-plot width in inches
SMOOTH = {'smooth_scale_1D': -1.25, 'smooth_scale_2D': -1.25}      # getdist KDE bandwidth = automatic x 1.25
W_NU = 0.0006442        # omega_nu of one massive 0.06 eV species

SAMPLED = ['h', 'ombh2', 'omch2', 'logA', 'ns']
TRI_PARAMS = ['omch2', 'h', 'sigma8', 'ns']          # omega_b is left out of the triangle plots and the FoM matrix: it is prior dominated
LABELS = {'h': r'h', 'ombh2': r'\omega_b', 'omch2': r'\omega_{\rm cdm}', 'logA': r'\ln(10^{10}A_s)', 'ns': r'n_s', 'sigma8': r'\sigma_8'}
TRACERS = [('BGS', 'yellowgreen'), ('LRG1', 'orange'), ('LRG2', 'orangered'), ('LRG3', 'firebrick'), ('ELG2', 'steelblue'), ('QSO', 'seagreen')]      # (name, DESI colour)
PURPLE, ORANGE, BLUE = '#D58BCB', '#FF8C1A', 'tab:blue'
C000 = dict(h=0.6736, ombh2=0.02237, omch2=0.12, logA=float(np.log(1e10 * 2.0830e-9)), ns=0.9649)       # AbacusSummit c000 (mock truth)


#%% helpers: chains
def fit_path(rest):
    return CHAIN_ROOT / PREFIX / f'{PREFIX}_{rest}'


_cache = {}


def full3(rest):
    """(array (steps, walkers, n_params), parameter names): post-burn-in and thinned as saved in status.json."""
    if rest not in _cache:
        root = fit_path(rest)
        st = json.load(open(f'{root}.status.json'))
        with h5py.File(f'{root}.h5', 'r') as f:
            it = int(f['mcmc'].attrs['iteration'])
            arr = f['mcmc/chain'][st['discard']:it:st['thin'], :, :]
        _cache[rest] = (arr, st['param_names'])
    return _cache[rest]


def columns(rest):
    """dict parameter -> array (steps, walkers) of the five sampled cosmological parameters."""
    arr, names = full3(rest)
    return {n: arr[..., i].copy() for i, n in enumerate(names) if n in SAMPLED}


def map_point(rest):
    """(highest-posterior post-burn-in sample as {parameter: value}, its log-posterior)."""
    root = fit_path(rest)
    st = json.load(open(f'{root}.status.json'))
    with h5py.File(f'{root}.h5', 'r') as f:
        it = int(f['mcmc'].attrs['iteration'])
        lp = f['mcmc/log_prob'][st['discard']:it]
        t, w = np.unravel_index(np.nanargmax(np.where(np.isfinite(lp), lp, -np.inf)), lp.shape)
        theta = f['mcmc/chain'][st['discard'] + t, w, :]
    return dict(zip(st['param_names'], theta)), float(lp[t, w])


def desi_columns(path):
    """(dict of 1-D arrays, weights) of a DESI cobaya chain: h, ombh2, omch2 and, if the chain has them, logA, ns and sigma8."""
    s = loadMCSamples(path + 'chain', settings={'ignore_rows': 0.3})
    n = s.getParamNames().list()
    has = lambda *a: any(x in n for x in a)
    col = lambda *a: s.samples[:, n.index(next(x for x in a if x in n))]
    h, ob = col('H0') / 100, col('ombh2', 'omegabh2')
    c = {'h': h, 'ombh2': ob, 'omch2': col('omch2') if has('omch2') else col('omegam') * h ** 2 - ob - W_NU}
    c.update({k: col(k) for k in ('logA', 'ns', 'sigma8') if has(k)})
    return c, s.weights


def to_mcs(c, params, label, weights=None):
    """getdist MCSamples: one chain per walker for our (steps, walkers) arrays, weighted samples for the DESI 1-D arrays."""
    use = [p for p in params if p in c]
    if weights is None:
        stack = np.stack([c[p] for p in use], axis=-1)
        kw = dict(samples=[stack[:, w] for w in range(stack.shape[1])])
    else:
        kw = dict(samples=np.column_stack([c[p] for p in use]), weights=weights)
    return MCSamples(names=use, labels=[LABELS[p] for p in use], label=label, settings=SMOOTH, **kw)


#%% helpers: sigma_8
A_REF = 2.1e-9
SIG8_PARS = ['ombh2', 'omch2', 'h', 'ns']      # sigma_8 = sqrt(A_s / A_REF) x f(these four): exact in A_s, smooth in the others


def sigma8_camb(h, ombh2, omch2, logA, ns):
    """sigma_8 at z = 0 from CAMB (linear, one massive 0.06 eV neutrino), as cobaya's get_sigma8_0."""
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100 * h, ombh2=ombh2, omch2=omch2, mnu=0.06)
    pars.InitPower.set_params(As=np.exp(logA) * 1e-10, ns=ns)
    pars.set_matter_power(redshifts=[0.0], kmax=1.0)
    pars.NonLinear = camb.model.NonLinear_none
    pars.WantCls = False
    return camb.get_results(pars).get_sigma8_0()


def poly_features(x, degree=3):
    cols = [np.ones(len(x))]
    for d in range(1, degree + 1):
        for idx in combinations_with_replacement(range(x.shape[1]), d):
            cols.append(np.prod(x[:, idx], axis=1))
    return np.stack(cols, axis=1)


def fit_sigma8(samples, n_train=128, n_test=32):
    """Cubic fit of ln sigma_8(A_REF) in the four parameters over a Sobol design covering all the `samples` (list of column dicts).
    Returns sigma8(c) for a column dict c (any array shape); prints the held-out error, with random ln A_s, of the whole construction."""
    pooled = np.stack([np.concatenate([np.ravel(c[p]) for c in samples]) for p in SIG8_PARS + ['logA']], axis=1)
    lo, hi = pooled.min(0), pooled.max(0)
    lo, hi = lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)
    design = qmc.scale(np.vstack([qmc.Sobol(d=5, seed=0).random_base2(int(np.log2(n_train))), qmc.Sobol(d=5, seed=1).random_base2(int(np.log2(n_test)))]), lo, hi)
    log_a_ref = np.log(1e10 * A_REF)
    y = np.array([np.log(sigma8_camb(h, ob, oc, log_a_ref, ns)) for ob, oc, h, ns, _ in design[:n_train]])
    unit = lambda x: 2 * (x - lo[:4]) / (hi[:4] - lo[:4]) - 1
    coef = np.linalg.lstsq(poly_features(unit(design[:n_train, :4])), y, rcond=None)[0]

    def sigma8(c):
        x = np.stack([np.ravel(c[p]) for p in SIG8_PARS], axis=1)
        out = np.exp(poly_features(unit(x)) @ coef) * np.sqrt(np.exp(np.ravel(c['logA'])) * 1e-10 / A_REF)
        return out.reshape(np.shape(c['h']))

    test = design[n_train:]
    exact = np.array([sigma8_camb(h, ob, oc, la, ns) for ob, oc, h, ns, la in test])
    approx = sigma8({'ombh2': test[:, 0], 'omch2': test[:, 1], 'h': test[:, 2], 'ns': test[:, 3], 'logA': test[:, 4]})
    print(f'sigma_8 fit: max relative error {np.max(np.abs(approx / exact - 1)):.2e} on {n_test} held-out CAMB runs (design box: {dict(zip(SIG8_PARS + ["logA"], zip(lo.round(4), hi.round(4))))})')
    return sigma8


#%% helpers: figures
def plotter(width):
    """getdist plotter with slide-size fonts (fixed, not rescaled with the width) and no legend frame."""
    st = plots.GetDistPlotSettings()
    st.axes_fontsize, st.lab_fontsize, st.legend_fontsize = 17, 25, LEGEND_FS
    st.legend_frame = False
    st.axis_tick_max_labels = 3
    return plots.get_subplot_plotter(width_inch=width, scaling=False, settings=st)


def triangle(fname, items, params, truth=None):
    """Triangle plot of items = (MCSamples, colour); all filled, drawn in order (the last is on top). `truth` = {parameter: value} drawn as dashed lines.
    The legend is drawn here, right-aligned in the empty upper-right of the grid, with the font shrunk if needed so it never reaches the first panel."""
    roots = [i[0] for i in items]
    present = {p for r in roots for p in r.getParamNames().list()}
    params = [p for p in params if p in present]
    g = plotter(WIDTH_TRI)
    g.triangle_plot(roots, params, filled=[True] * len(items), contour_colors=[i[1] for i in items], contour_lws=[2.0] * len(items), legend_labels=[],
                    markers={k: v for k, v in (truth or {}).items() if k in params}, marker_args={'ls': '--', 'lw': 1.5, 'color': 'k'})
    handles = [Patch(fc=color, ec=color, label=r.getLabel()) for r, color in items]
    if truth:
        handles.append(Line2D([], [], color='k', ls='--', lw=1.5, label='AbacusSummit c000 truth'))
    pad = 0.3 * g.fig.dpi                                    # gap (pixels) kept between the legend and the first panel
    for fs in range(LEGEND_FS, 11, -1):
        leg = g.fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=False, fontsize=fs)
        g.fig.canvas.draw()
        if fs == 12 or leg.get_window_extent().x0 > g.subplots[0, 0].get_window_extent().x1 + pad:
            break
        leg.remove()
    g.export(str(PLOT_DIR / f'{fname}.pdf'))
    g.fig.savefig(PLOT_DIR / f'{fname}.png', dpi=DPI, bbox_inches='tight')
    plt.show()
    plt.close(g.fig)
    print('saved', fname, flush=True)


def fit_quality_figure(fname, evs):
    """Data vs MAP model for the six tracer-bins in two rows of three: s^2 xi_0 (filled circles, solid line) and s^2 xi_2 (open squares, dashed line) with
    residuals (data - model) / sigma below each. evs: per tracer dict(s, data, model, sigma, chi2, n), the multipoles concatenated. The errors and the chi2 in the
    titles are those of the covariance the fit used (for the mean-of-mocks fit: the single-mock covariance). The legend sits above the panels, away from the data."""
    fig = plt.figure(figsize=(17, 11))
    outer = fig.add_gridspec(2, 1, hspace=0.5, left=0.07, right=0.99, top=0.9, bottom=0.07)
    lim = max(3.5, 1.1 * max(np.max(np.abs((ev['data'] - ev['model']) / ev['sigma'])) for ev in evs))
    for r_row in range(2):
        inner = outer[r_row].subgridspec(2, 3, height_ratios=[3, 1.2], hspace=0.08, wspace=0.34)
        for j in range(3):
            k = 3 * r_row + j
            (short, color), ev = TRACERS[k], evs[k]
            s = ev['s']
            data, model, sig = (ev[key].reshape(-1, len(s)) for key in ('data', 'model', 'sigma'))
            ax = fig.add_subplot(inner[0, j])
            axr = fig.add_subplot(inner[1, j], sharex=ax)
            sf = np.linspace(s[0], s[-1], 300)
            for i, (marker, ls, filled) in enumerate((('o', '-', True), ('s', '--', False))):
                ax.errorbar(s, s ** 2 * data[i], yerr=s ** 2 * sig[i], fmt=marker, ms=6.5, color=color, ecolor=color, mfc=color if filled else 'white', elinewidth=1.8, capsize=0)
                ax.plot(sf, CubicSpline(s, s ** 2 * model[i])(sf), ls=ls, color=color, lw=2.6)
                axr.plot(s, (data[i] - model[i]) / sig[i], marker=marker, ms=6, ls=ls, lw=1.6, color=color, mfc=color if filled else 'white')
            axr.axhspan(-1, 1, color='0.8', zorder=0)
            axr.axhline(0, color='0.55', lw=1.2)
            axr.set_ylim(-lim, lim)
            axr.set_yticks([-2, 0, 2])
            ax.set_title(rf'{short}  ($\chi^2={ev["chi2"]:.1f}$, $N={ev["n"]}$)', fontsize=20)
            ax.tick_params(labelbottom=False)
            axr.set_xlabel(r'$s\;[h^{-1}{\rm Mpc}]$')
            if j == 0:
                ax.set_ylabel(r'$s^2\xi_\ell(s)$')
                axr.set_ylabel(r'$\Delta/\sigma$')
    fig.legend(handles=[Line2D([0], [0], color='k', marker='o', ms=6.5, lw=2.6, label=r'$\ell=0$: data, model'),
                        Line2D([0], [0], color='k', marker='s', ms=6.5, mfc='white', ls='--', lw=2.6, label=r'$\ell=2$: data, model')],
               loc='lower center', bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False, fontsize=20, handlelength=3)
    fig.savefig(PLOT_DIR / f'{fname}.pdf', bbox_inches='tight')
    fig.savefig(PLOT_DIR / f'{fname}.png', dpi=DPI, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print('saved', fname, flush=True)


def sqdet(smp, names):
    """sqrt(det of the posterior covariance) of the named parameters of a getdist MCSamples: proportional to the volume of the 1-sigma ellipsoid."""
    return float(np.exp(0.5 * np.linalg.slogdet(np.atleast_2d(smp.cov(pars=names)))[1]))


def fom_matrix_figure(fname, ours, ref, params, ref_name='DESI FS-only'):
    """Lower-triangle matrix of the figure-of-merit ratio FoM(ours) / FoM(ref), FoM = 1 / sqrt(det C) with C the posterior covariance of the parameter pair
    (diagonal: the single parameter, FoM = 1 / sigma). Above 1 (blue) means ours is tighter. ours, ref: getdist MCSamples.
    The colour scale is symmetric in log10(ratio), white at 1, so its two ends are equal factors (5x tighter / 5x looser); it is widened, still symmetrically,
    only if a value would otherwise saturate."""
    n = len(params)
    R = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1):
            names = [params[i]] if i == j else [params[i], params[j]]
            R[i, j] = sqdet(ref, names) / sqdet(ours, names)
    fig, ax = plt.subplots(figsize=(9.5, 8))
    cmap = plt.get_cmap('RdBu').copy()
    cmap.set_bad('white')
    lim = max(0.7, np.ceil(10 * np.nanmax(np.abs(np.log10(R)))) / 10)
    im = ax.imshow(np.ma.masked_invalid(np.log10(R)), cmap=cmap, vmin=-lim, vmax=lim)
    for i in range(n):
        for j in range(i + 1):
            ax.text(j, i, f'{R[i, j]:.2f}', ha='center', va='center', fontsize=22, color='white' if abs(np.log10(R[i, j])) > 0.7 * lim else 'k')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f'${LABELS[p]}$' for p in params], fontsize=22)
    ax.set_yticklabels([f'${LABELS[p]}$' for p in params], fontsize=22)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    ticks = np.array([-lim, -lim / 2, 0, lim / 2, lim])
    cb.set_ticks(ticks)
    cb.set_ticklabels([f'{10 ** t:.2g}' for t in ticks])
    cb.set_label(f'FoM / FoM({ref_name})  (blue: ours tighter)', fontsize=20)
    cb.ax.tick_params(labelsize=18)
    fig.savefig(PLOT_DIR / f'{fname}.pdf', bbox_inches='tight')
    fig.savefig(PLOT_DIR / f'{fname}.png', dpi=DPI, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print('saved', fname, flush=True)


#%% load the chains and derive sigma_8
mock = columns(MOCK_FIT)
dr1 = columns(DR1_FIT)
fs, fs_w = desi_columns(DESI_FS)
bao, bao_w = desi_columns(DESI_BAO)

sigma8 = fit_sigma8([mock, dr1] + ([] if 'sigma8' in fs else [fs]))
for c in (mock, dr1):
    c['sigma8'] = sigma8(c)
if 'sigma8' not in fs:
    fs['sigma8'] = sigma8(fs)
TRUTH = {**C000, 'sigma8': sigma8_camb(**C000)}
print('c000 truth:', {k: round(v, 5) for k, v in TRUTH.items()})

#%% 1. triangle plot: mean of mocks, truth marked
triangle('talk_triangle_mock', [(to_mcs(mock, TRI_PARAMS, OUR_LABEL), BLUE)], TRI_PARAMS, TRUTH)

#%% 2. triangle plot: DR1 with DESI DR1 BAO-only and full-shape-only
bao_mcs, fs_mcs, ours_mcs = to_mcs(bao, TRI_PARAMS, 'DESI DR1 BAO', bao_w), to_mcs(fs, TRI_PARAMS, 'DESI DR1 FS-only', fs_w), to_mcs(dr1, TRI_PARAMS, OUR_LABEL)
triangle('talk_triangle_dr1', [(bao_mcs, ORANGE), (fs_mcs, PURPLE), (ours_mcs, BLUE)], TRI_PARAMS)

#%% 5. figure-of-merit ratio to DESI FS-only (DR1): lower-triangle matrix, diagonal = single parameters, off-diagonal = parameter pairs
fom_matrix_figure('talk_fom_matrix_dr1', ours_mcs, fs_mcs, TRI_PARAMS)

#%% 3-4. fit quality: data vs MAP model (needs fits_apfix_11par_v3 and its data and emulators: compact environment)
# Self-contained: it redefines fit_quality_figure (overriding the helper cell's version), so this cell alone can be pasted into a running notebook.
# The whole data vector is shown (no explicit x limits or ticks); the plotted points are the fit's own data vector (cut at s_min <= s <= s_max on the stored bin positions).
HERE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
sys.path.insert(0, str(HERE))
import fits_apfix_11par_v3 as fv                         # imports your fits_apfix.py and common.py
SPECS = {s['name']: s for s in fv.build_specs_v3()}
SMAX = float(getattr(fv.fa, 'SMAX_FIT', 130.0))          # upper limit of the fit range (130 h^-1 Mpc if fits_apfix does not expose SMAX_FIT); only used in the printout below


def evaluate_at_map(rest):
    """(FitProblem, per-tracer list of dict(s, data, model, sigma, chi2, n), MAP log-posterior) of a chain, as in plot_v3_chains_gd.py."""
    problem = fv.fa.FitProblem(SPECS[f'{PREFIX}_{rest}'])
    names = full3(rest)[1]
    mp, lp_max = map_point(rest)
    theta = np.array([mp[n] for n in names])
    biases = problem.unpack(theta)[5]
    h, ombh2, omch2, lnAs, ns = problem.model_params(theta)
    aps = fv.fa.ap_multi(h, ombh2, omch2, ns, problem.z_effs)
    out = []
    for d, (b1, b2), (q_perp, q_par) in zip(problem.data, biases, aps):
        mod = fv.fa.model_one(d['emus'], [b1, b2, ombh2, omch2, h, lnAs, ns], d['s_data'], d['W'], q_perp, q_par)
        r = d['data_vec'] - mod
        out.append(dict(s=d['s_data'], data=d['data_vec'], model=mod, sigma=np.sqrt(np.diag(np.linalg.inv(d['cov_inv']))), chi2=float(r @ d['cov_inv'] @ r), n=len(r)))
    return problem, out, lp_max


def fit_quality_figure(fname, evs):
    """Data vs MAP model for the six tracer-bins in two rows of three. Each column: s^2 xi_0 (filled circles, solid line) and s^2 xi_2 (open squares, dashed line)
    with the tracer name in its colour at the top right, then one residual panel per multipole, (data - model) / sigma with a +-1 sigma band; only the bottom
    panel carries the x axis. evs: per tracer dict(s, data, model, sigma, chi2, n), the multipoles concatenated. The errors are those of the covariance the fit
    used (for the mean-of-mocks fit: the single-mock covariance). There is no legend; the tracer name sits in headroom above the data.
    The x axes keep the default white space (5% margins) between the first / last points and the panel edges, whatever the notebook's rcParams say."""
    # Layout in inches, set from a measured render so that little whitespace is left: the margins hold the y labels (left), the last x tick label (right), the
    # top edge and the x labels (bottom); COL_GAP holds the widest y tick labels (0.64 in) of the next column plus ~0.3 in; ROW_GAP holds the x tick
    # labels and x label (~0.7 in) plus ~0.3 in.
    W_FIG, ROW_H, ROW_GAP, COL_GAP = 17.0, 4.4, 1.0, 0.95
    L_MARGIN, R_MARGIN, T_MARGIN, B_MARGIN = 1.16, 0.28, 0.15, 0.75
    H_FIG = T_MARGIN + B_MARGIN + 2 * ROW_H + ROW_GAP
    pw = (W_FIG - L_MARGIN - R_MARGIN - 2 * COL_GAP) / 3                      # panel width
    with plt.rc_context({'axes.xmargin': 0.05}):                        # matplotlib's default white space at the ends of the x axis (the notebook's rcParams use 0)
        fig = plt.figure(figsize=(W_FIG, H_FIG))
        outer = fig.add_gridspec(2, 1, hspace=ROW_GAP / ROW_H, left=L_MARGIN / W_FIG, right=1 - R_MARGIN / W_FIG, top=1 - T_MARGIN / H_FIG, bottom=B_MARGIN / H_FIG)
        lim = max(3.5, 1.1 * max(np.max(np.abs((ev['data'] - ev['model']) / ev['sigma'])) for ev in evs))
        for r_row in range(2):
            inner = outer[r_row].subgridspec(3, 3, height_ratios=[3, 1, 1], hspace=0.08, wspace=COL_GAP / pw)
            for j in range(3):
                k = 3 * r_row + j
                (short, color), ev = TRACERS[k], evs[k]
                s = ev['s']
                data, model, sig = (ev[key].reshape(-1, len(s)) for key in ('data', 'model', 'sigma'))
                ax = fig.add_subplot(inner[0, j])
                axr = [fig.add_subplot(inner[1, j], sharex=ax), fig.add_subplot(inner[2, j], sharex=ax)]      # residuals of xi_0, xi_2
                sf = np.linspace(s[0], s[-1], 300)
                for i, (marker, ls, filled) in enumerate((('o', '-', True), ('s', '--', False))):
                    ax.errorbar(s, s ** 2 * data[i], yerr=s ** 2 * sig[i], fmt=marker, ms=6.5, color=color, ecolor=color, mfc=color if filled else 'white', elinewidth=1.8, capsize=0)
                    ax.plot(sf, CubicSpline(s, s ** 2 * model[i])(sf), ls=ls, color=color, lw=2.6)
                    axr[i].plot(s, (data[i] - model[i]) / sig[i], marker=marker, ms=6, ls=ls, lw=1.6, color=color, mfc=color if filled else 'white')
                    axr[i].axhspan(-1, 1, color='0.8', zorder=0)
                    axr[i].axhline(0, color='0.55', lw=1.2)
                    axr[i].set_ylim(-lim, lim)
                    axr[i].set_yticks([-2, 0, 2])
                    axr[i].tick_params(axis='y', labelsize=16)                       # the residual panels are ~0.85 in tall: keeps -2, 0, 2 apart
                lo, hi = ax.get_ylim()
                ax.set_ylim(lo, hi + 0.25 * (hi - lo))                            # headroom above the data for the tracer name
                ax.text(0.97, 0.96, short, transform=ax.transAxes, ha='right', va='top', color=color, fontsize=24)
                ax.tick_params(labelbottom=False)
                axr[0].tick_params(labelbottom=False)
                axr[1].set_xlabel(r'$s\;[h^{-1}{\rm Mpc}]$')
                if j == 0:
                    ax.set_ylabel(r'$s^2\xi_\ell(s)$')
                    axr[0].set_ylabel(r'$\Delta_0/\sigma$', fontsize=18)
                    axr[1].set_ylabel(r'$\Delta_2/\sigma$', fontsize=18)
    fig.savefig(PLOT_DIR / f'{fname}.pdf', bbox_inches='tight')
    fig.savefig(PLOT_DIR / f'{fname}.png', dpi=DPI, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print('saved', fname, flush=True)


for tag, rest in (('mock', MOCK_FIT), ('dr1', DR1_FIT)):
    problem, evs, lp_max = evaluate_at_map(rest)
    chi2_fit, n_data = sum(e['chi2'] for e in evs), sum(e['n'] for e in evs)
    s_low, s_high = min(e['s'][0] for e in evs), max(e['s'][-1] for e in evs)
    p_txt = (f', p = {chi2_dist.sf(chi2_fit, n_data - problem.ndim):.3g}' if tag == 'dr1' else
             ' (single-mock covariance on the mean of mocks: no p-value here, see cell 9 of plot_v3_chains_gd.py)')
    print(f'{rest}: plotted s from {s_low:.1f} to {s_high:.1f} (fit range {SMIN:g}-{SMAX:g}); last three bin centres of {TRACERS[0][0]}: {np.round(evs[0]["s"][-3:], 3)}; chi2 = {chi2_fit:.1f}, N_data = {n_data}, N_par = {problem.ndim}{p_txt}; -2 max ln P = {-2 * lp_max:.1f} (agrees with chi2 up to prior and normalisation terms)')
    print('  chi2 / N per tracer: ' + ', '.join(f'{name} {e["chi2"]:.1f}/{e["n"]}' for (name, _), e in zip(TRACERS, evs)))
    if s_low < SMIN:
        print(f'WARNING: the fit data vector of {rest} has points below s_min = {SMIN}')
    if s_high < SMAX - 1:
        print(f'NOTE: the highest bin in the fit data vector of {rest} is at s = {s_high:.1f}, below s_max = {SMAX:g}')
    fit_quality_figure(f'talk_fit_quality_{tag}', evs)


#%% (Omega_m, ln A_s) plane: mock fit (red) with the c000 truth: a single 2D panel (uses mock, C000, W_NU, to_mcs from the cells above)
LABELS['omegam'] = r'\Omega_m'                                  # Omega_m includes the 0.06 eV neutrino, as everywhere else in this script
RED = 'red'
MOCK_LABEL = 'Compact skew-$t$, DR1'
om = lambda c: (c['ombh2'] + c['omch2'] + W_NU) / c['h'] ** 2
OM_TRUTH = om(C000)

st = plots.GetDistPlotSettings()
st.axes_fontsize, st.lab_fontsize, st.legend_fontsize = 18, 26, 20
st.legend_frame = False
g = plots.get_single_plotter(width_inch=7.5, ratio=1, scaling=False, settings=st)
g.plot_2d([to_mcs({**mock, 'omegam': om(mock)}, ['omegam', 'logA'], MOCK_LABEL)], 'omegam', 'logA', filled=True, colors=[RED])
ax = g.get_axes()
ax.axvline(OM_TRUTH, ls='--', lw=1.5, color='k')
ax.axhline(C000['logA'], ls='--', lw=1.5, color='k')
for get_lim, set_lim, v in ((ax.get_xlim, ax.set_xlim, OM_TRUTH), (ax.get_ylim, ax.set_ylim, C000['logA'])):      # keep the truth inside the axes
    lo, hi = get_lim()
    pad = 0.08 * (hi - lo)
    set_lim(min(lo, v - pad), max(hi, v + pad))
ax.legend(handles=[Patch(fc=RED, ec=RED, label=MOCK_LABEL), Line2D([], [], color='k', ls='--', lw=1.5, label='AbacusSummit c000 truth')],
          loc='lower center', bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=20)          # above the panel, so it never covers the contours
g.export(str(PLOT_DIR / 'talk_Om_lnAs_mock.pdf'))
g.fig.savefig(PLOT_DIR / 'talk_Om_lnAs_mock.png', dpi=DPI, bbox_inches='tight')
plt.show()
plt.close(g.fig)
print(f'c000 truth: Omega_m = {OM_TRUTH:.4f}, ln(10^10 A_s) = {C000["logA"]:.4f}')


#%% zoom-in on the (omega_cdm, sigma_8) panel of the DR1 triangle plot: same chains, colours and labels, a single panel (uses dr1, fs, fs_w, to_mcs, OUR_LABEL, PURPLE, BLUE, LEGEND_FS from the cells above)
# The BAO-only chain has no sigma_8, so (as in the triangle plot) it does not appear in this plane.
PAIR = ['omch2', 'sigma8']
desi_fs_mcs, ours_dr1_mcs = to_mcs(fs, PAIR, 'DESI DR1 FS-only', fs_w), to_mcs(dr1, PAIR, OUR_LABEL)

st = plots.GetDistPlotSettings()
st.axes_fontsize, st.lab_fontsize, st.legend_fontsize = 17, 25, LEGEND_FS
st.legend_frame = False
g = plots.get_single_plotter(width_inch=7.5, ratio=1, scaling=False, settings=st)
g.plot_2d([desi_fs_mcs, ours_dr1_mcs], PAIR[0], PAIR[1], filled=True, colors=[PURPLE, BLUE], lws=[2.0, 2.0])         # DESI first, ours on top, as in the triangle
ax = g.get_axes()
ax.legend(handles=[Patch(fc=PURPLE, ec=PURPLE, label='DESI DR1 FS-only'), Patch(fc=BLUE, ec=BLUE, label=OUR_LABEL)],
          loc='lower center', bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=20)          # above the panel, so it never covers the contours
g.export(str(PLOT_DIR / 'talk_zoom_omch2_sigma8_dr1.pdf'))
g.fig.savefig(PLOT_DIR / 'talk_zoom_omch2_sigma8_dr1.png', dpi=DPI, bbox_inches='tight')
plt.show()
plt.close(g.fig)
print('saved talk_zoom_omch2_sigma8_dr1', flush=True) 

# %%
# bias of the mock fit: (posterior mean - c000 truth) / posterior std, for the triangle-plot parameters except omega_b (prior dominated)
bias = {p: (mock[p].mean() - TRUTH[p]) / mock[p].std() for p in TRI_PARAMS if p != 'ombh2'}
worst = max(bias, key=lambda p: abs(bias[p]))
print({p: round(b, 2) for p, b in bias.items()})
print(f'greatest bias: {worst} at {bias[worst]:+.2f} sigma ({100 * (mock[worst].mean() / TRUTH[worst] - 1):+.2f}% of the truth value)')
#%%
# bias of the mock fit with A_s in place of sigma_8: (posterior mean - c000 truth) / posterior std; omega_b is left out (prior dominated)
A_S = lambda c: 1e-10 * np.exp(c['logA'])                  # A_s from the sampled ln(10^10 A_s)
mock_a, truth_a = {**mock, 'As': A_S(mock)}, {**TRUTH, 'As': A_S(TRUTH)}
bias = {p: (mock_a[p].mean() - truth_a[p]) / mock_a[p].std() for p in ['omch2', 'h', 'As', 'ns']}
worst = max(bias, key=lambda p: abs(bias[p]))
print({p: round(b, 2) for p, b in bias.items()})
print(f'greatest bias: {worst} at {bias[worst]:+.2f} sigma ({100 * (mock_a[worst].mean() / truth_a[worst] - 1):+.2f}% of the truth value)')
print(f'A_s: mean {mock_a["As"].mean():.4e} vs truth {truth_a["As"]:.4e}')

#%%
# bias of the mock fit in the sampled parameters, with ln(10^10 A_s) in place of sigma_8: (posterior mean - c000 truth) / posterior std; omega_b is left out (prior dominated)
bias = {p: (mock[p].mean() - TRUTH[p]) / mock[p].std() for p in ['omch2', 'h', 'logA', 'ns']}
worst = max(bias, key=lambda p: abs(bias[p]))
print({p: round(b, 2) for p, b in bias.items()})
print(f'greatest bias: {worst} at {bias[worst]:+.2f} sigma (mean - truth = {mock[worst].mean() - TRUTH[worst]:+.4g})')
print(f'ln(10^10 A_s): mean {mock["logA"].mean():.4f} vs truth {TRUTH["logA"]:.4f}')