"""
Quick check of how well the NN emulators extrapolate outside their training box, for n_s (LCDM emulators) and for w0, wa (w0wa emulators).
The reference is the direct model that made the training sets (LPT + ridge-calibrated skew-t parameters, via the generators' run_one), which is exact apart
from the linear continuation of the calibration, so NN - direct isolates the NN's own extrapolation error. Per scan point it reports the max over
s = 20-130 h^-1 Mpc of |NN - direct| in s^2 xi_ell, as a fraction of the max |s^2 xi_ell| and in absolute units (compare the latter with your data errors).
Compare the errors outside the box with the in-box baseline. It cannot tell you whether the linear calibration itself is right beyond the Abacus grid.

Assumptions to check: generate_multiz_11par_w0wa.py has run_one(i, theta, z, cfg, schemes) like generate_multiz_11par.py, with theta =
[b1, b2, ombh2, omch2, h, As, ns, w0, wa]; the w0wa direct model needs compact.config_w0wa.ConfigW0wa (the default Config() has no w0, wa dependence), which is
assumed to take no arguments. The direct model refuses w0 + wa > 0 (w > 0 at high z), so the wa scan is done at w0 = -1.2. Run in the compact environment.
"""
#%%
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

REPO = Path('/home/cosweeney/code/compact_streaming')
sys.path[:0] = [str(REPO / 'nb/nn_training/LoHi_mass_scheme'), str(REPO / 'nb/fits')]
from compact.config import Config
from compact.config_w0wa import ConfigW0wa
from common import Emulator
from generate_multiz_11par import run_one                          # direct model, 7-parameter (LCDM) theta
from generate_multiz_11par_w0wa import run_one as run_one_w0wa     # direct model, 9-parameter theta (ASSUMED same interface)

ZS = (0.510, 1.317)
S_RANGE = (20.0, 130.0)
N_JOBS = 8
EMU = {'lcdm': ('/spiffball/cosweeney/data/models_11par_avg_M/', '_11par', run_one),
       'w0wa': ('/spiffball/cosweeney/data/models_11par_w0wa_avg_M/', '_11par_w0wa', run_one_w0wa)}
BOX = {'ns': (0.901, 1.025), 'w0': (-1.271, -0.726), 'wa': (-0.628, 0.621)}
FID = dict(b1=1.0, b2=0.0, ombh2=0.02237, omch2=0.12, h=0.6736, As=2.083e-9, ns=0.9649, w0=-1.0, wa=0.0)      # c000 with LRG-like biases
SCANS = {'ns': (np.linspace(0.90, 1.10, 11), 'lcdm', {}), 'w0': (np.linspace(-1.5, -0.5, 11), 'w0wa', {}),
         'wa': (np.linspace(-1.2, 1.15, 13), 'w0wa', {'w0': -1.2})}               # extra fixed parameters per scan (w0 + wa must stay < 0)

CFG = {'lcdm': Config(), 'w0wa': ConfigW0wa()}
s = np.asarray(CFG['lcdm'].r)
assert len(CFG['w0wa'].r) == len(s), 'the two configs use different r grids'
mask = (s >= S_RANGE[0]) & (s <= S_RANGE[1])
_emus = {}

#%%
def nn_xi(kind, z, p):
    """(3, n_r) NN multipoles for the parameter dict p."""
    if (kind, z) not in _emus:
        d, suf, _ = EMU[kind]
        _emus[(kind, z)] = [Emulator(os.path.join(d, f"compact_xi{ell}_z{str(z).replace('.', 'p')}{suf}")) for ell in (0, 2, 4)]
    vec = [p['b1'], p['b2'], p['ombh2'], p['omch2'], p['h'], np.log(1e10 * p['As']), p['ns']] + ([p['w0'], p['wa']] if kind == 'w0wa' else [])
    return np.array([e(vec)[1] for e in _emus[(kind, z)]])


def theta(kind, p):
    return [p['b1'], p['b2'], p['ombh2'], p['omch2'], p['h'], p['As'], p['ns']] + ([p['w0'], p['wa']] if kind == 'w0wa' else [])


points = [(name, val, kind, z, {**FID, **fixed, name: val}) for name, (vals, kind, fixed) in SCANS.items() for val in vals for z in ZS]
direct = Parallel(n_jobs=N_JOBS)(delayed(EMU[kind][2])(0, theta(kind, p), z, CFG[kind], ['avg_M']) for _n, _v, kind, z, p in points)

rows = []
for (name, val, kind, z, p), (_i, out) in zip(points, direct):
    xi_d, err = out['avg_M']
    row = {'scan': name, 'z': z, 'value': val, 'in_box': BOX[name][0] <= val <= BOX[name][1], 'error': (err or '')[:45]}
    if xi_d is not None:
        xi_n = nn_xi(kind, z, p)
        for k, ell in enumerate((0, 2)):
            ref, diff = (np.asarray(xi_d)[k] * s ** 2)[mask], ((xi_n[k] - np.asarray(xi_d)[k]) * s ** 2)[mask]
            row[f'ell{ell} rel [%]'], row[f'ell{ell} abs'] = 100 * np.max(np.abs(diff)) / np.max(np.abs(ref)), np.max(np.abs(diff))
    rows.append(row)
R = pd.DataFrame(rows)
R.to_csv('nn_extrapolation_check.csv', index=False)
pd.set_option('display.width', 200)
print(R.round(4).to_string(index=False))
print('\nin-box baseline (median rel. error, %):')
print(R[R.in_box].groupby(['scan', 'z'])[['ell0 rel [%]', 'ell2 rel [%]']].median().round(3))

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
for ax, name in zip(axes, SCANS):
    for z in ZS:
        sub = R[(R.scan == name) & (R.z == z)]
        for ell, ls in ((0, '-'), (2, '--')):
            ax.plot(sub.value, sub[f'ell{ell} rel [%]'], ls, marker='o', ms=3, label=rf'$z={z}$, $\ell={ell}$')
    for edge in BOX[name]:
        ax.axvline(edge, color='k', ls=':', lw=1)
    ax.set_yscale('log')
    ax.set_xlabel({'ns': r'$n_s$', 'w0': r'$w_0$', 'wa': r'$w_a$'}[name])
    ax.set_ylabel(r'max $|{\rm NN}-{\rm direct}|$ / max $|s^2\xi_\ell|$ [%]')
axes[0].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig('nn_extrapolation_check.png', dpi=150, bbox_inches='tight')
plt.show()


#%% [markdown]
# ## w0-wa NN error over the full DESI prior ranges
# Varies only w0 and wa (everything else at c000) over DESI's priors, w0 in [-3, 1] and wa in [-3, 2] with w0 + wa < 0 (the direct model refuses w0 + wa > 0), on a
# regular grid, which samples the uniform DESI prior. Per grid point the error is the larger of the ell = 0 and ell = 2 values of max |NN - direct| / max |s^2 xi_ell|
# over s = 20-130. Points where the direct model itself fails are left blank and counted. The summary gives the share of the DESI prior area where the NN error is
# below a threshold; the figure shows the error maps with the NN box (dashed) and the w0 + wa = 0 line (dotted). Needs the definitions from the cell above.
 
#%%
from matplotlib.colors import LogNorm
DESI_W0, DESI_WA = (-3.0, 1.0), (-3.0, 2.0)
N_W0, N_WA = 17, 17
w0s, was = np.linspace(*DESI_W0, N_W0), np.linspace(*DESI_WA, N_WA)
grid = [(i, j) for i, w0 in enumerate(w0s) for j, wa in enumerate(was) if w0 + wa < 0]
pts = [(i, j, z, {**FID, 'w0': w0s[i], 'wa': was[j]}) for i, j in grid for z in ZS]
out = Parallel(n_jobs=N_JOBS)(delayed(EMU['w0wa'][2])(0, theta('w0wa', p), z, CFG['w0wa'], ['avg_M']) for _i, _j, z, p in pts)
 
 
def rel_err(xi_d, xi_n):
    return [100 * np.max(np.abs((xi_n[k] - xi_d[k]) * s ** 2)[mask]) / np.max(np.abs(xi_d[k] * s ** 2)[mask]) for k in (0, 1)]      # ell = 0, 2
 
 
E = {z: np.full((N_W0, N_WA), np.nan) for z in ZS}
G_rows, fails = [], {z: 0 for z in ZS}
for (i, j, z, p), (_k, res) in zip(pts, out):
    xi_d, err = res['avg_M']
    row = {'w0': w0s[i], 'wa': was[j], 'z': z, 'in_box': BOX['w0'][0] <= w0s[i] <= BOX['w0'][1] and BOX['wa'][0] <= was[j] <= BOX['wa'][1], 'error': (err or '')[:45]}
    if xi_d is None:
        fails[z] += 1
    else:
        row['ell0 rel [%]'], row['ell2 rel [%]'] = rel_err(np.asarray(xi_d), nn_xi('w0wa', z, p))
        E[z][i, j] = max(row['ell0 rel [%]'], row['ell2 rel [%]'])
    G_rows.append(row)
pd.DataFrame(G_rows).to_csv('nn_extrapolation_desi_grid.csv', index=False)
 
summ = []
for z in ZS:
    n_valid = len(grid)
    summ.append({'z': z, 'grid points (w0 + wa < 0)': n_valid, 'direct model failed': fails[z], **{f'error < {t}%': np.sum(E[z] < t) / n_valid for t in (0.3, 1, 3, 10)}})
print(pd.DataFrame(summ).round(3).to_string(index=False))
 
fig, axes = plt.subplots(1, len(ZS), figsize=(5.6 * len(ZS), 4.6), squeeze=False)
W0g, WAg = np.meshgrid(w0s, was, indexing='ij')
for ax, z in zip(axes[0], ZS):
    im = ax.pcolormesh(W0g, WAg, np.ma.masked_invalid(E[z]), norm=LogNorm(vmin=0.05, vmax=100), cmap='magma_r', shading='nearest')
    ax.contour(W0g, WAg, np.ma.masked_invalid(E[z]), levels=[0.3, 1, 3], colors=['w', 'c', 'lime'], linewidths=0.8)
    ax.add_patch(plt.Rectangle((BOX['w0'][0], BOX['wa'][0]), BOX['w0'][1] - BOX['w0'][0], BOX['wa'][1] - BOX['wa'][0], fill=False, ls='--', color='k', lw=1.2))
    ax.plot(w0s, -w0s, 'r:', lw=1.2)
    ax.set_xlim(*DESI_W0)
    ax.set_ylim(*DESI_WA)
    ax.set_xlabel(r'$w_0$')
    ax.set_ylabel(r'$w_a$')
    ax.text(0.03, 0.04, rf'$z={z}$', transform=ax.transAxes, color='w', bbox=dict(facecolor='k', alpha=0.6, lw=0))
fig.colorbar(im, ax=axes[0].tolist(), label=r'NN error: max over $\ell=0,2$ of max $|{\rm NN}-{\rm direct}|$ / max $|s^2\xi_\ell|$ [%]')
fig.savefig('nn_extrapolation_desi_grid.png', dpi=150, bbox_inches='tight')
plt.show()
 
