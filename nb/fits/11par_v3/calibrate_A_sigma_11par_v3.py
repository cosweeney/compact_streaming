"""
Recalibrate the cosmology emulator for the 11par_v3 model, in which row 0 is A_sigma instead of sigma_0:

    sigma_0^2 = A_sigma * sigma_{v,lin}^2          (compact/linear.py; the same code the model runs)

Why the per-sim fits are not re-run: at fixed (sim, z, bin) sigma_{v,lin} is a fixed number, so A_sigma =
sigma_0^2 / sigma_{v,lin}^2 is a one-to-one reparametrization of the saved sigma_0, and rows 1-10 are unchanged.
This script therefore starts from the saved per-sim fits (cosmo_grid_fit_results.h5, written by
nb/calibration/runs/auto_runs/cosmo_grid_building_nu.py) and

  1. computes sigma_{v,lin} for every sim and redshift from (h, omega_b, omega_cdm, A_s, n_s) with all other CAMB
     parameters at their defaults, i.e. exactly the call the model makes (w0, wa, alpha_s, N_ur, ... of the sim are NOT used);
  2. forms A_sigma and fits the ridge emulator for row 0 in the same way as the old parameters
     (predictors theta_fid - theta over [omega_b, omega_cdm, ln 1e10 A_s, n_s], MAD outlier mask, leave-one-out alpha);
  3. high-M above Z_ANCHOR_MAX: bin 0 + the mean (bin 1 - bin 0) shift over z <= Z_ANCHOR_MAX for the intercept and every
     slope (the same scheme as the Delta rows); avg_M is the mean of the bin-0 and high-M coefficients;
  4. copies rows 1-10 unchanged from the baseline compact/config.py and writes config_11par_v3.py, which defines
        Config           slopes on  (sigma0_from_linear = True, exact_growth = True)
        ConfigNoSlope    the same with every slope of every row set to zero (intercepts kept)

It also checks that the supplied fit-results file reproduces the baseline sigma_0 emulator in compact/config.py (if it
does not, the file is not the one config.py was built from -- see the printout) and writes a QA figure.

    python calibrate_A_sigma_11par_v3.py --fit-results /path/to/cosmo_grid_fit_results.h5 [--n-jobs 8]
"""
import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import h5py as h5
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from compact.config import Config as BaseConfig

HERE = os.path.dirname(os.path.abspath(__file__))
FIDUCIAL_LCDM = np.array([0.02237, 0.1200, np.log(1e10 * 2.0830e-9), 0.9649])     # c000, as in the calibration script
LCDM_NAMES = ['omega_b', 'omega_cdm', 'ln_1e10_A_s', 'n_s']
N_LCDM = 4
ALPHA_GRID = np.logspace(-3, 3, 13)
MIN_SIMS_FOR_FIT = 4
Z_ANCHOR_MAX = 0.5
ROW = 0                                                                        # the A_sigma row


# ------------------------------------------------------------------ ridge (identical to cosmo_grid_building_nu.py)
def ridge_fit_beta(delta, y, alpha):
    n_sim, n_col = delta.shape
    scale = np.std(delta, axis=0)
    scale[scale == 0] = 1.0
    X = np.column_stack([np.ones(n_sim), delta / scale])
    Gamma = np.diag([0.0] + [alpha] * n_col)
    beta_s = np.linalg.solve(X.T @ X + Gamma, X.T @ y)
    return np.concatenate([[beta_s[0]], beta_s[1:] / scale])


def ridge_cv_alpha(delta, y):
    n_sim = len(y)
    if n_sim < 2 * MIN_SIMS_FOR_FIT:
        return ALPHA_GRID[len(ALPHA_GRID) // 2]
    best_alpha, best_err = ALPHA_GRID[0], np.inf
    for alpha in ALPHA_GRID:
        se = 0.0
        for m in range(n_sim):
            mask = np.ones(n_sim, dtype=bool)
            mask[m] = False
            beta = ridge_fit_beta(delta[mask], y[mask], alpha)
            se += (beta[0] + delta[m] @ beta[1:] - y[m])**2
        if se < best_err:
            best_err, best_alpha = se, alpha
    return best_alpha


def robust_outlier_mask(y, mad_k=6.0):
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    return np.ones_like(y, dtype=bool) if mad == 0 else np.abs(y - med) <= mad_k * 1.4826 * mad


def ridge_emulator(theta, y):
    keep = robust_outlier_mask(y)
    if keep.sum() < MIN_SIMS_FOR_FIT:
        keep = np.ones_like(keep)
    delta = (FIDUCIAL_LCDM[None, :] - theta)[keep]
    return ridge_fit_beta(delta, y[keep], ridge_cv_alpha(delta, y[keep]))


def eval_beta(beta, theta):
    return beta[0] + (FIDUCIAL_LCDM[None, :] - theta) @ beta[1:]


# ------------------------------------------------------------------ load
ap = argparse.ArgumentParser()
ap.add_argument('--fit-results', default='cosmo_grid_fit_results.h5')
ap.add_argument('--cosmo-csv', default=os.path.join(HERE, '..', '..', 'calibration', 'runs', 'auto_runs', 'cosmologies.csv'))
ap.add_argument('--out-dir', default=HERE)
ap.add_argument('--n-jobs', type=int, default=8)
ap.add_argument('--max-sims', type=int, default=0, help='stop after this many new sims (testing)')
ap.add_argument('--check-only', action='store_true', help='only test whether --fit-results reproduces the baseline emulator (fast, no CAMB), then exit')
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

with h5.File(args.fit_results, 'r') as f:
    sim, zz, bn, par, cos = (f[k][()] for k in ('sim', 'z', 'bin', 'params', 'cosmo'))
    cos_names = [c.decode() if isinstance(c, bytes) else c for c in f.attrs['cosmo_names']]
    param_names = [c.decode() if isinstance(c, bytes) else c for c in f.attrs['param_names']]
assert param_names[ROW] == 'sigma0', f'row {ROW} is {param_names[ROW]}, expected sigma0'

table = pd.read_csv(args.cosmo_csv, skipinitialspace=True)
table.columns = [c.strip() for c in table.columns]
table['root'] = table['root'].str.strip()
A_s_of = table.set_index('root')['A_s']
sims = sorted(set(sim))
zs = sorted(set(zz))
col = lambda name: cos[:, cos_names.index(name)]
theta = np.column_stack([col('omega_b'), col('omega_cdm'),
                         np.log(1e10 * A_s_of.loc[[f'abacus_cosm{n:03d}' for n in sim]].values), col('n_s')])
h_row = col('h')

# ------------------------------------------------------------------ ridge emulators per bin and z node; baseline check (no CAMB needed)
def fit_rows(values):
    out = {}
    for k in (0, 1):
        for z in zs:
            sel = (zz == z) & (bn == k) & np.isfinite(values)
            if sel.sum() >= MIN_SIMS_FOR_FIT:
                out[(k, z)] = ridge_emulator(theta[sel], values[sel])
    return out


BETA_SIG0 = fit_rows(par[:, ROW])
print(f'fit results: {os.path.abspath(args.fit_results)}  ({len(sims)} sims, z = {[float(z) for z in zs]})')
print('check: how closely does the sigma_0 emulator rebuilt from this file reproduce the one in compact/config.py?')
grid_theta = np.unique(theta, axis=0)                                           # the grid cosmologies
ev = lambda b, th: b[0] + (FIDUCIAL_LCDM[None, :] - th) @ b[1:]
diff = {}                                                                       # (bin, z) -> max |sigma_0(new)/sigma_0(config) - 1| over the grid cosmologies
for (k, z), b in BETA_SIG0.items():
    emu = BaseConfig.compact_emulator.get(z)
    ref = [e for e in emu['fits_indiv']['low_M' if k == 0 else 'high_M'] if e[0] == ROW] if emu else []
    if ref:
        diff[(k, z)] = np.max(np.abs(ev(b, grid_theta) / ev(ref[0][2], grid_theta) - 1))
print('  z      max |dsigma_0/sigma_0| over the grid:  low_M      high_M')
for z in zs:
    cell = lambda k: f'{diff[(k, z)]:9.1e}' if (k, z) in diff else '   (none)'
    print(f'  {float(z):<4}                                      {cell(0)}   {cell(1)}')
missing = sorted((k, float(z)) for k in (0, 1) for z in BaseConfig.compact_emulator if BaseConfig.compact_emulator[z] and (k, z) not in BETA_SIG0)
if missing:
    print(f'  no ridge fit from this file (too few sims) for (bin, z): {missing}; the baseline has emulators there')
low = [v for (k, z), v in diff.items() if z <= Z_ANCHOR_MAX]
exact = bool(diff) and max(diff.values()) < 1e-4 and not missing
print('  RESULT:', 'exact reproduction of the baseline emulator' if exact else
      f'differs from the baseline emulator (max {max(low):.1e} at z <= {Z_ANCHOR_MAX:g}). For A_sigma, high_M above z = {Z_ANCHOR_MAX:g} is extrapolated from low_M, so only '
      f'low_M (all z) and high_M at z <= {Z_ANCHOR_MAX:g} enter; judge the size against the ~1-3% scatter of the sigma_0 ridge fits')
if args.check_only:
    sys.exit(0)

from compact.linear import camb_results, sigma_v_lin      # needs the patched compact (imported after --check-only so the check works without it)

# ------------------------------------------------------------------ sigma_{v,lin} per (sim, z): the model's CAMB call
first_row = {n: int(np.where(sim == n)[0][0]) for n in sims}


def lin_for_sim(n):
    i = first_row[n]
    res = camb_results(h_row[i], theta[i, 0], theta[i, 1], np.exp(theta[i, 2]) / 1e10, theta[i, 3], zs)
    return n, {z: sigma_v_lin(res, z, h_row[i]) for z in zs}


CACHE = os.path.join(args.out_dir, 'sigma_lin_cache_11par_v3.json')            # resumable: one entry per sim
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
todo = [n for n in sims if str(n) not in cache or any(str(z) not in cache[str(n)] for z in zs)]
print(f'sigma_v,lin for {len(sims)} sims x {len(zs)} redshifts ({len(todo)} to compute, rest from {CACHE}) ...')
if todo:
    for n_done, (n, vals) in enumerate(Parallel(n_jobs=args.n_jobs, return_as='generator_unordered')(
            delayed(lin_for_sim)(n) for n in todo), 1):
        cache[str(n)] = {str(z): float(v) for z, v in vals.items()}
        json.dump(cache, open(CACHE, 'w'))
        if args.max_sims and n_done >= args.max_sims:
            print(f'stopping after {n_done} sims (--max-sims); rerun to continue')
            sys.exit(0)
sigma_lin_of = {n: {z: cache[str(n)][str(z)] for z in zs} for n in sims}
sigma_lin = np.array([sigma_lin_of[n][z] for n, z in zip(sim, zz)])
A_sigma = par[:, ROW]**2 / sigma_lin**2
print(f'A_sigma over all fits: median {np.nanmedian(A_sigma):.4f}, 5-95%: {np.nanpercentile(A_sigma, 5):.4f}-{np.nanpercentile(A_sigma, 95):.4f}')

# ------------------------------------------------------------------ ridge emulators for A_sigma (new), per bin and z node
BETA_A_RAW = fit_rows(A_sigma)

# high-M above Z_ANCHOR_MAX: bin 0 + mean (bin 1 - bin 0) shift over z <= Z_ANCHOR_MAX (intercept and every slope)
diffs = [BETA_A_RAW[(1, z)] - BETA_A_RAW[(0, z)] for z in zs if z <= Z_ANCHOR_MAX and (1, z) in BETA_A_RAW and (0, z) in BETA_A_RAW]
shift = np.mean(diffs, axis=0)
BETA_A = {0: {z: BETA_A_RAW[(0, z)] for z in zs if (0, z) in BETA_A_RAW}, 1: {}, 'avg': {}}
for z in zs:
    if z <= Z_ANCHOR_MAX:
        if (1, z) in BETA_A_RAW:
            BETA_A[1][z] = BETA_A_RAW[(1, z)]
    elif (0, z) in BETA_A_RAW:
        BETA_A[1][z] = BETA_A_RAW[(0, z)] + shift
    if z in BETA_A[0] and z in BETA_A[1]:
        BETA_A['avg'][z] = 0.5 * (BETA_A[0][z] + BETA_A[1][z])
print(f'\nA_sigma: high-M above z = {Z_ANCHOR_MAX} = low-M + mean shift {np.round(shift, 4)} (intercept, 4 slopes)')
print('  z    bin0 intercept   high-M intercept (raw -> used)   avg intercept')
for z in zs:
    raw = BETA_A_RAW.get((1, z))
    print(f'  {z:<4} {BETA_A[0].get(z, [np.nan])[0]:8.4f}        {np.nan if raw is None else raw[0]:8.4f} -> {BETA_A[1].get(z, [np.nan])[0]:8.4f}'
          f'         {BETA_A["avg"].get(z, [np.nan])[0]:8.4f}')

# ------------------------------------------------------------------ write config_11par_v3.py
def entries(rows_baseline, new_beta):
    out = []
    for r, cols, b in rows_baseline:
        out.append((r, cols, new_beta if r == ROW else b))
    if not any(r == ROW for r, _, _ in rows_baseline) and new_beta is not None:
        out.insert(0, (ROW, list(range(N_LCDM)), new_beta))
    return [e for e in out if e[2] is not None]


A_mean = float(np.nanmedian(A_sigma))                                      # fallback value only (every node has a row-0 fit)
mean_params = np.array(BaseConfig.compact_emulator[zs[0]]['mean_params'], dtype=float)
mean_params[ROW] = A_mean


def fmt(arr):
    return 'np.array([' + ', '.join(f'{v:.10g}' for v in arr) + '])'


lines = ['"""',
         'Cosmology emulator for the 11par_v3 model (generated by calibrate_A_sigma_11par_v3.py -- do not edit by hand).',
         'Row 0 is A_sigma (sigma_0^2 = A_sigma sigma_{v,lin}^2, compact/linear.py); rows 1-10 are copied from compact/config.py.',
         f'Fit results: {os.path.abspath(args.fit_results)}',
         'Config: slopes on. ConfigNoSlope: every slope of every row set to zero (intercepts kept).',
         '"""', 'import numpy as np', 'from compact.config import Config as _BaseConfig', '',
         f'_MEAN_PARAMS = {fmt(mean_params)}', '', '',
         'class Config(_BaseConfig):', '    sigma0_from_linear = True      # row 0 is A_sigma: predictions.py sets sigma_0 = sqrt(A_sigma) sigma_{v,lin}',
         '    exact_growth = True            # CAMB f in the LPT mean velocity (CompactStreamingModel reads this from the config)', '    _MEAN_PARAMS = _MEAN_PARAMS', '    compact_emulator = {']
for z in sorted(BaseConfig.compact_emulator):
    emu = BaseConfig.compact_emulator[z]
    if emu is None:
        lines.append(f'        {z}: None,')
        continue
    lines += [f'        {z}: {{', f"            'ns_fiducial': {float(emu['ns_fiducial'])!r},", "            'mean_params': _MEAN_PARAMS,"]
    for key, rows, b_new in (('fits', emu['fits'], BETA_A['avg'].get(z)),):
        lines.append(f"            '{key}': [")
        lines += [f'                ({r}, {list(c)}, {fmt(b)}),' for r, c, b in entries(rows, b_new)]
        lines.append('            ],')
    lines.append("            'fits_indiv': {")
    for name, k in (('low_M', 0), ('high_M', 1)):
        lines.append(f"                '{name}': [")
        lines += [f'                    ({r}, {list(c)}, {fmt(b)}),' for r, c, b in entries(emu['fits_indiv'][name], BETA_A[k].get(z))]
        lines.append('                ],')
    lines += ['            },', '        },']
lines += ['    }', '', '',
          'def _zero_slopes(emu_dict):',
          '    """Same emulator with beta_1..beta_4 = 0 in every row (intercepts unchanged)."""',
          '    def z(rows):',
          '        return [(r, c, np.concatenate([b[:1], np.zeros(len(b) - 1)])) for r, c, b in rows]',
          '    return {zn: None if e is None else {**e, "fits": z(e["fits"]),',
          '                                        "fits_indiv": {k: z(v) for k, v in e["fits_indiv"].items()}}',
          '            for zn, e in emu_dict.items()}',
          '', '',
          'class ConfigNoSlope(Config):', '    compact_emulator = _zero_slopes(Config.compact_emulator)', '']
cfg_path = os.path.join(args.out_dir, 'config_11par_v3.py')
with open(cfg_path, 'w') as fh:
    fh.write('\n'.join(lines))
print(f'\nwrote {cfg_path}')

# ------------------------------------------------------------------ save emulator parameters (row 0) + per-sim A_sigma
out_h5 = os.path.join(args.out_dir, 'cosmo_emulator_params_11par_v3.h5')
with h5.File(out_h5, 'w') as hdf:
    hdf.attrs['param_names'] = ['A_sigma'] + param_names[1:]
    hdf.attrs['cosmo_names'] = LCDM_NAMES
    hdf.attrs['fiducial_cosmo'] = FIDUCIAL_LCDM
    hdf.attrs['bin1_extrapolated_above_z'] = Z_ANCHOR_MAX
    hdf.attrs['fit_results'] = os.path.abspath(args.fit_results)
    hdf.create_dataset('redshifts', data=np.array(zs))
    for name, source in (('bin0', BETA_A[0]), ('bin1', BETA_A[1]), ('average', BETA_A['avg'])):
        grp = hdf.create_group(name)
        for iz, z in enumerate(zs):
            if z in source:
                sub = grp.create_group(f'param{ROW}_z{iz}')
                sub.attrs['redshift'] = z
                sub.create_dataset('beta', data=source[z])
    per = hdf.create_group('per_sim')
    for k, v in (('sim', sim), ('z', zz), ('bin', bn), ('sigma0', par[:, ROW]), ('sigma_lin', sigma_lin), ('A_sigma', A_sigma)):
        per.create_dataset(k, data=v)
print(f'wrote {out_h5}')

# ------------------------------------------------------------------ QA figure
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
for k, c, lab in ((0, 'C0', 'low-M'), (1, 'C3', 'high-M')):
    zk = [z for z in zs if (k, z) in BETA_A_RAW]
    ax[0].plot(zk, [BETA_A_RAW[(k, z)][0] for z in zk], 'o', c=c, label=f'{lab} (own fit)')
    zu = [z for z in zs if z in BETA_A[k]]
    ax[0].plot(zu, [BETA_A[k][z][0] for z in zu], '-', c=c)
ax[0].plot(list(BETA_A['avg']), [BETA_A['avg'][z][0] for z in BETA_A['avg']], 'k--', label='avg_M')
ax[0].set_xlabel(r'$z$')
ax[0].set_ylabel(r'$A_\sigma$ at the fiducial cosmology')
ax[0].legend(fontsize=8)
for z, c in zip(zs, plt.cm.viridis(np.linspace(0, 1, len(zs)))):
    if (0, z) not in BETA_SIG0 or z not in BETA_A[0]:
        continue
    sel = (zz == z) & (bn == 0) & np.isfinite(A_sigma)
    new = np.sqrt(np.clip(eval_beta(BETA_A[0][z], theta[sel]), 0, None)) * sigma_lin[sel]
    old = eval_beta(BETA_SIG0[(0, z)], theta[sel])
    ax[1].plot(theta[sel, 2], new / old - 1, 'o', ms=3, c=c, label=f'z={z}')
ax[1].axhline(0, c='k', lw=0.8)
ax[1].set_xlabel(r'$\log(10^{10}A_s)$')
ax[1].set_ylabel(r'$\sigma_0^{\rm new}(\theta)/\sigma_0^{\rm old}(\theta) - 1$ (low-M, grid cosmologies)')
ax[1].legend(fontsize=6, ncol=2)
fig.savefig(os.path.join(args.out_dir, 'A_sigma_calibration_11par_v3.png'), dpi=300)
print('wrote A_sigma_calibration_11par_v3.png')
