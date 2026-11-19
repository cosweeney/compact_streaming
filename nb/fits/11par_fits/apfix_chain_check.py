"""Triangle plots of the AP-fixed baseline fits (red, filled, drawn on top) against the DESI DR1 FS-only (purple) and BAO (orange) chains, for all the
cosmological parameters we fit (+ Omega_m): one figure for the DR1 baseline, one for the mean-of-mocks baseline (with the c000 truth marked).
The BAO chains only constrain Omega_m, H0, omega_b (and omega_cdm through them), so they appear only in those panels."""
#%%
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from getdist import MCSamples, loadMCSamples, plots

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False})

ROOT = '/spiffball/cosweeney/data/chains/fits_11par_apfix/compact/'
DESI_FS = '/spiffball/cosweeney/DESI/DR1/full-shape-cosmo-params/v1.0/cobaya/base/desi-reptvelocileptors-fs-all_schoneberg2024-bbn_planck2018-ns10/'
DESI_BAO = '/spiffball/cosweeney/DESI/DR1/bao-cosmo-params/v1.0/cobaya/base/desi-bao-all_schoneberg2024-bbn/'      # adjust to your local directory name
W_NU = 0.0006442                                     # omega_nu of one massive 0.06 eV species, so Omega_m includes neutrinos like CAMB's
PARAMS = ['omegam', 'H0', 'ombh2', 'omch2', 'logA', 'ns']
LABELS = [r'\Omega_m', 'H_0', r'\omega_b', r'\omega_{\rm cdm}', r'\ln(10^{10}A_s)', 'n_s']
SMOOTH = {'smooth_scale_1D': -1.25, 'smooth_scale_2D': -1.25}      # getdist KDE bandwidth = automatic x 1.25 (-1 = default); larger is smoother but broadens the contours
TRUTH = {'omegam': (0.02237 + 0.12 + W_NU) / 0.6736 ** 2, 'H0': 67.36, 'ombh2': 0.02237, 'omch2': 0.12, 'logA': np.log(1e10 * 2.0830e-9), 'ns': 0.9649}   # c000
RUNS = {'dr1':   ('compact_dr1_joint_smin20',  'Compact baseline, DR1',          None),
        'mocks': ('compact_mean_joint_smin20', 'Compact baseline, mean of mocks', TRUTH)}
COLORS = {'bao': '#FFA028', 'fs': '#D58BCB', 'ours': 'red'}

#%%
def ours(root):
    st = json.load(open(root + '.status.json'))
    with h5py.File(root + '.h5', 'r') as f:
        it = int(f['mcmc'].attrs['iteration'])
        c = f['mcmc/chain'][st['discard']:it:st['thin'], :, :5]                                # (steps, walkers, 5): h, ombh2, omch2, logA, ns
    print(f"{root.split('/')[-1]}: ~{st['nwalkers'] * (it - st['discard']) / st['tau_max']:.0f} independent samples (N/tau_max = {st['n_over_tau_max']:.0f})")
    h, ob, oc, lnA, ns = (c[..., i] for i in range(5))
    cols = np.stack([(ob + oc + W_NU) / h ** 2, 100 * h, ob, oc, lnA, ns], axis=-1)
    return MCSamples(samples=[cols[:, w] for w in range(cols.shape[1])], names=PARAMS, labels=LABELS, settings=SMOOTH)   # one chain per walker


def desi(path):
    s = loadMCSamples(path + 'chain', settings={'ignore_rows': 0.3})
    n = s.getParamNames().list()
    has = lambda *a: any(x in n for x in a)
    col = lambda *a: s.samples[:, n.index(next(x for x in a if x in n))]
    h, ob = col('H0') / 100, col('ombh2', 'omegabh2')
    c = {'H0': 100 * h, 'ombh2': ob}
    c['omch2'] = col('omch2') if has('omch2') else col('omegam') * h ** 2 - ob - W_NU
    c['omegam'] = col('omegam') if has('omegam') else (ob + c['omch2'] + W_NU) / h ** 2
    c.update({k: col(k) for k in ('logA', 'ns') if has(k)})
    names = [p for p in PARAMS if p in c]                                                      # the BAO chains have no logA / ns
    return MCSamples(samples=np.column_stack([c[p] for p in names]), weights=s.weights, names=names, labels=[LABELS[PARAMS.index(p)] for p in names],
                     settings=SMOOTH)


bao, fs = desi(DESI_BAO), desi(DESI_FS)
for tag, (name, label, markers) in RUNS.items():
    g = plots.get_subplot_plotter(width_inch=9)
    g.triangle_plot([bao, fs, ours(ROOT + name)], PARAMS, filled=True, contour_colors=[COLORS['bao'], COLORS['fs'], COLORS['ours']], markers=markers,
                    legend_labels=['DESI DR1 BAO', 'DESI DR1 FS-only', label])                  # later entries are drawn on top
    g.export(f'apfix_baseline_{tag}.pdf')
    g.fig.savefig(f'apfix_baseline_{tag}.png', dpi=300, bbox_inches='tight')
    plt.show()

#=================================================================
# now using chainconsumer
#=================================================================
#%%
"""ChainConsumer (>= 1.0) version of plot_apfix_vs_desi.py: triangle plots of the AP-fixed baseline fits (red, filled, on top) against the DESI DR1 FS-only
(purple) and BAO (orange) chains, for all the cosmological parameters we fit (+ Omega_m); one figure for the DR1 baseline, one for the mean-of-mocks baseline
(with the c000 truth marked). The BAO chains only constrain Omega_m, H0, omega_b (and omega_cdm through them), so they appear only in those panels.
Does not use getdist: the DESI cobaya chains are read directly from chain.N.txt."""
import glob
import json
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chainconsumer import Chain, ChainConsumer, PlotConfig, Truth

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False})

ROOT = '/spiffball/cosweeney/data/chains/fits_11par_apfix/compact/'
DESI_FS = '/spiffball/cosweeney/DESI/DR1/full-shape-cosmo-params/v1.0/cobaya/base/desi-reptvelocileptors-fs-all_schoneberg2024-bbn_planck2018-ns10/'
DESI_BAO = '/spiffball/cosweeney/DESI/DR1/bao-cosmo-params/v1.0/cobaya/base/desi-bao-all_schoneberg2024-bbn/'      # adjust to your local directory name
W_NU = 0.0006442                                     # omega_nu of one massive 0.06 eV species, so Omega_m includes neutrinos like CAMB's
PARAMS = ['omegam', 'H0', 'ombh2', 'omch2', 'logA', 'ns']
LABELS = {'omegam': r'$\Omega_m$', 'H0': r'$H_0$', 'ombh2': r'$\omega_b$', 'omch2': r'$\omega_{\rm cdm}$', 'logA': r'$\ln(10^{10}A_s)$', 'ns': r'$n_s$'}
TRUTH = {'omegam': (0.02237 + 0.12 + W_NU) / 0.6736 ** 2, 'H0': 67.36, 'ombh2': 0.02237, 'omch2': 0.12, 'logA': np.log(1e10 * 2.0830e-9), 'ns': 0.9649}   # c000
RUNS = {'dr1':   ('compact_dr1_joint_smin20',  'Compact baseline, DR1',          None),
        'mocks': ('compact_mean_joint_smin20', 'Compact baseline, mean of mocks', TRUTH)}
COLORS = {'bao': '#FF8C1A', 'fs': '#D58BCB', 'ours': 'red'}
OURS_KDE = False        # KDE contours for ours (smooth; ~5 s per 5000-sample panel). False = histogram + smoothing, which is jagged for a few thousand samples
MAX_KDE = 5000         # random subsample of our post-burn samples used for the KDE (cost grows ~ N^2)
DESI_SMOOTH = 3        # histogram smoothing (bins) for the huge DESI chains, where a KDE is too slow

#%%
def ours(root):
    st = json.load(open(root + '.status.json'))
    with h5py.File(root + '.h5', 'r') as f:
        it = int(f['mcmc'].attrs['iteration'])
        h, ob, oc, lnA, ns = f['mcmc/chain'][st['discard']:it:st['thin'], :, :5].reshape(-1, 5).T      # h, ombh2, omch2, logA, ns
    print(f"{root.split('/')[-1]}: ~{st['nwalkers'] * (it - st['discard']) / st['tau_max']:.0f} independent samples (N/tau_max = {st['n_over_tau_max']:.0f})")
    df = pd.DataFrame({'omegam': (ob + oc + W_NU) / h ** 2, 'H0': 100 * h, 'ombh2': ob, 'omch2': oc, 'logA': lnA, 'ns': ns})
    return df.sample(MAX_KDE, random_state=0) if len(df) > MAX_KDE else df


def load_cobaya(path, skip=0.3):
    """Concatenate the cobaya chain files path/chain.N.txt (or chain.txt), dropping the first `skip` fraction of each as burn-in."""
    dfs = []
    for fn in sorted(glob.glob(path + 'chain.[0-9]*.txt')) or glob.glob(path + 'chain.txt'):
        names = open(fn).readline().lstrip('#').split()
        d = pd.read_csv(fn, sep=r'\s+', comment='#', header=None, names=names)
        dfs.append(d.iloc[int(skip * len(d)):])
    return pd.concat(dfs, ignore_index=True)


def desi(path):
    d = load_cobaya(path)
    h, ob = d['H0'] / 100, d['ombh2'] if 'ombh2' in d else d['omegabh2']
    c = pd.DataFrame({'H0': d['H0'], 'ombh2': ob})
    c['omch2'] = d['omch2'] if 'omch2' in d else d['omegam'] * h ** 2 - ob - W_NU
    c['omegam'] = d['omegam'] if 'omegam' in d else (ob + c['omch2'] + W_NU) / h ** 2
    for k in ('logA', 'ns'):                                                                   # the BAO chains have no logA / ns
        if k in d:
            c[k] = d[k]
    c['weight'] = d['weight']
    return c[[p for p in PARAMS if p in c] + ['weight']]


bao, fs = desi(DESI_BAO), desi(DESI_FS)
for tag, (name, label, truth) in RUNS.items():
    c = ChainConsumer()
    c.add_chain(Chain(samples=bao, name='DESI DR1 BAO', color=COLORS['bao'], bar_shade=False, smooth=DESI_SMOOTH, zorder=1))
    c.add_chain(Chain(samples=fs, name='DESI DR1 FS-only', color=COLORS['fs'], bar_shade=False, smooth=DESI_SMOOTH, zorder=2))
    c.add_chain(Chain(samples=ours(ROOT + name), name=label, color=COLORS['ours'], bar_shade=False, kde=OURS_KDE, zorder=10))         # highest zorder = on top
    if truth:
        c.add_truth(Truth(location=truth, color='0.4', line_style='--'))
    c.set_plot_config(PlotConfig(labels=LABELS, summarise=False, serif=True, max_ticks=4))
    fig = c.plotter.plot(columns=PARAMS, figsize=(9, 9))
    fig.savefig(f'apfix_baseline_{tag}.pdf', bbox_inches='tight')
    fig.savefig(f'apfix_baseline_{tag}.png', dpi=300, bbox_inches='tight')
    plt.show()
# %%
