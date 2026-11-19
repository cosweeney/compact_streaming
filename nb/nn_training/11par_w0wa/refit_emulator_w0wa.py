"""
Refit the compact-model ridge emulators on the AbacusSummit cosmology grid with w0 and wa as
extra predictor columns, and write compact/config_w0wa.py.

Same procedure as nb/calibration/runs/auto_runs/cosmo_grid_building_nu.py, applied to its saved
per-(sim, z, mass bin) fit results: per (bin, z, parameter) a leave-one-out-CV ridge in
Delta_theta = theta_fid - theta, MAD outlier exclusion, then the high-mass bin above
Z_ANCHOR_MAX = 0.5 for EXTRAPOLATE_ROWS taken as bin 0 + mean (bin 1 - bin 0) shift (applied to
the intercept and every slope), and the average emulator = (bin 0 + bin 1) / 2.

The per-sim velocity fits (including their N_MIN = 50 pairs per cell threshold) are NOT redone here;
only the ridge stage is, with the same settings as the builder (MIN_SIMS_FOR_FIT, ALPHA_GRID, MAD cut).

  --columns lcdm   the 4 original columns (omega_b, omega_cdm, ln 1e10 A_s, n_s). Reproduces the
                   emulator in compact/config.py; --check compares against it.
  --columns w0wa   adds w0, wa (default). All 52 grid sims are used.

The intercept/slope-vs-z figure (same style as the builder's emulator_all_params_vs_redshift.png, with
ridge errors) is written with the config, or with --plot in --check mode:

    python refit_emulator_w0wa.py --columns lcdm --check --plot --fit-results cosmo_grid_fit_results.h5   # one-to-one with the old figure
    python refit_emulator_w0wa.py --columns w0wa --fit-results cosmo_grid_fit_results.h5
"""
import argparse
import csv
import io
import os
import shutil
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO)

COSMOLOGIES = os.path.join(REPO, 'nb/calibration/runs/auto_runs/cosmologies.csv')
OUT_CONFIG = os.path.join(REPO, 'compact/config_w0wa.py')

PARAM_NAMES = ['sigma0', 'A_rho0', 'A_rho_p', 'sigma_rho', 'A_delta0', 'r_delta', 'sigma_delta',
               'beta_delta', 'alpha', 'nu_0', 'r_c']
PARAM_LABELS = [r"$\sigma_0$ [km/s]", r"$A_{\rho,0}$", r"$A_{\rho,p}$ [$h^{-1}$ Mpc]", r"$\sigma_\rho$ [$h^{-1}$ Mpc]",
                r"$A_{\Delta,0}$ [km/s]", r"$r_\Delta$ [$h^{-1}$ Mpc]", r"$\sigma_\Delta$ [$h^{-1}$ Mpc]",
                r"$\beta_\Delta$", r"$\alpha_\Delta$", r"$\nu_0$", r"$r_c$ [$h^{-1}$ Mpc]"]
COLUMN_SETS = {
    'lcdm': ['omega_b', 'omega_cdm', 'ln_1e10_A_s', 'n_s'],
    'w0wa': ['omega_b', 'omega_cdm', 'ln_1e10_A_s', 'n_s', 'w0_fld', 'wa_fld'],
}
COLUMN_LABELS = {'omega_b': r'$\omega_b$', 'omega_cdm': r'$\omega_{\mathrm{cdm}}$', 'ln_1e10_A_s': r'$\log(10^{10}A_s)$',
                 'n_s': r'$n_s$', 'w0_fld': r'$w_0$', 'wa_fld': r'$w_a$'}
FIDUCIAL = {'omega_b': 0.02237, 'omega_cdm': 0.1200, 'ln_1e10_A_s': np.log(1e10 * 2.0830e-9),
            'n_s': 0.9649, 'w0_fld': -1.0, 'wa_fld': 0.0}

MIN_SIMS_FOR_FIT = 4
ALPHA_GRID = np.logspace(-3, 3, 13)
Z_ANCHOR_MAX = 0.5
EXTRAPOLATE_ROWS = ['r_delta', 'beta_delta', 'alpha', 'r_c']
EXTRAP_IDX = [PARAM_NAMES.index(p) for p in EXTRAPOLATE_ROWS]


def ridge_fit_beta(delta, y, alpha):
    n_sim, n_col = delta.shape
    scale = np.std(delta, axis=0)
    scale[scale == 0] = 1.0
    X = np.column_stack([np.ones(n_sim), delta / scale])
    Gamma = np.diag([0.0] + [alpha] * n_col)
    beta_s = np.linalg.solve(X.T @ X + Gamma, X.T @ y)
    return np.concatenate([[beta_s[0]], beta_s[1:] / scale])


def ridge_beta_se(delta, y, alpha):
    """Standard errors of the ridge coefficients (original units), as in the builder's ridge_beta_cov."""
    n_sim, n_col = delta.shape
    scale = np.std(delta, axis=0)
    scale[scale == 0] = 1.0
    X = np.column_stack([np.ones(n_sim), delta / scale])
    Gamma = np.diag([0.0] + [alpha] * n_col)
    XtX = X.T @ X
    A_inv = np.linalg.inv(XtX + Gamma)
    beta_s = A_inv @ (X.T @ y)
    edf = np.trace(X @ A_inv @ X.T)
    sigma2 = np.sum((y - X @ beta_s) ** 2) / max(n_sim - edf, 1.0)
    D = np.diag(np.concatenate([[1.0], 1.0 / scale]))
    return np.sqrt(np.diag(D @ (sigma2 * A_inv @ XtX @ A_inv) @ D.T))


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
            se += (beta[0] + delta[m] @ beta[1:] - y[m]) ** 2
        if se < best_err:
            best_err, best_alpha = se, alpha
    return best_alpha


def robust_outlier_mask(y, mad_k=6.0):
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    if mad == 0:
        return np.ones_like(y, dtype=bool)
    return np.abs(y - med) <= mad_k * 1.4826 * mad


def load_grid(columns, fit_results):
    """(params, theta): params[(bin, z)] = {sim: 11-vector}; theta[sim] = predictor vector."""
    with h5py.File(fit_results, 'r') as f:
        sim, z, b = f['sim'][:], f['z'][:], f['bin'][:]
        P, cosmo = f['params'][:], f['cosmo'][:]
        cnames = [n if isinstance(n, str) else n.decode() for n in f.attrs['cosmo_names']]
    lines = [ln for ln in open(COSMOLOGIES).read().splitlines() if ln.strip()]
    hdr = [h.strip() for h in lines[0].split(',')]
    A_s = {}
    for row in csv.reader(io.StringIO('\n'.join(lines[1:])), skipinitialspace=True):
        row = [x.strip() for x in row]
        A_s[int(row[0][-3:])] = float(dict(zip(hdr, row))['A_s'])

    theta = {}
    for n in sorted(set(sim.tolist())):
        c = cosmo[np.where(sim == n)[0][0]]
        val = {name: c[cnames.index(name)] for name in ('omega_b', 'omega_cdm', 'n_s', 'w0_fld', 'wa_fld')}
        val['ln_1e10_A_s'] = np.log(1e10 * A_s[n])
        theta[n] = np.array([val[name] for name in columns])
    params = {}
    for i in range(len(sim)):
        params.setdefault((int(b[i]), float(z[i])), {})[int(sim[i])] = P[i]
    return params, theta


def fit_beta(sims, theta, params_kz, i, fid):
    """(beta, standard errors) for parameter row i from this (bin, z)'s own fits, or (None, None)."""
    sims = [n for n in sims if np.isfinite(params_kz[n][i])]
    if len(sims) < MIN_SIMS_FOR_FIT:
        return None, None
    y = np.array([params_kz[n][i] for n in sims])
    th = np.array([theta[n] for n in sims])
    keep = robust_outlier_mask(y)
    if keep.sum() < MIN_SIMS_FOR_FIT:
        keep = np.ones_like(keep, dtype=bool)
    y, th = y[keep], th[keep]
    delta = fid[None, :] - th
    alpha = ridge_cv_alpha(delta, y)
    return ridge_fit_beta(delta, y, alpha), ridge_beta_se(delta, y, alpha)


def build_emulators(columns, fit_results):
    """(zs, emu, se, fid). emu / se: dicts keyed by 'low_M', 'high_M' (used: extrapolated where applicable),
    'avg_M' (used), 'high_M_raw', 'avg_M_raw' (bin 1's own fits everywhere) -> {(row, z): array or None}."""
    fid = np.array([FIDUCIAL[c] for c in columns])
    params, theta = load_grid(columns, fit_results)
    zs = sorted({z for (_, z) in params})
    n_labels = len(PARAM_NAMES)

    raw, raw_se = {0: {}, 1: {}}, {0: {}, 1: {}}
    for k in (0, 1):
        for z in zs:
            kz = params.get((k, z), {})
            for i in range(n_labels):
                raw[k][(i, z)], raw_se[k][(i, z)] = fit_beta(sorted(kz), theta, kz, i, fid) if kz else (None, None)

    def combine(a, a_se, b, b_se):
        return (None, None) if a is None or b is None else ((a + b) / 2.0, np.sqrt(a_se ** 2 + b_se ** 2) / 2.0)

    beta, se = {0: dict(raw[0]), 1: {}}, {0: dict(raw_se[0]), 1: {}}
    avg, avg_se, avg_raw, avg_raw_se = {}, {}, {}, {}
    for i in range(n_labels):
        pairs = [(z, raw[1][(i, z)] - raw[0][(i, z)], np.sqrt(raw_se[0][(i, z)] ** 2 + raw_se[1][(i, z)] ** 2))
                 for z in zs if z <= Z_ANCHOR_MAX and raw[0][(i, z)] is not None and raw[1][(i, z)] is not None]
        shift = shift_err = None
        if pairs:
            diffs = np.array([p[1] for p in pairs])
            shift = diffs.mean(axis=0)
            shift_err = (diffs.std(axis=0, ddof=1) / np.sqrt(len(diffs)) if len(diffs) > 1
                         else np.mean([p[2] for p in pairs], axis=0))
        for z in zs:
            if z <= Z_ANCHOR_MAX or i not in EXTRAP_IDX:
                beta[1][(i, z)], se[1][(i, z)] = raw[1][(i, z)], raw_se[1][(i, z)]
            else:
                b0 = raw[0][(i, z)]
                if b0 is None or shift is None:
                    beta[1][(i, z)], se[1][(i, z)] = None, None
                else:
                    beta[1][(i, z)] = b0 + shift
                    se[1][(i, z)] = np.sqrt(raw_se[0][(i, z)] ** 2 + shift_err ** 2)
            avg[(i, z)], avg_se[(i, z)] = combine(beta[0][(i, z)], se[0][(i, z)], beta[1][(i, z)], se[1][(i, z)])
            avg_raw[(i, z)], avg_raw_se[(i, z)] = combine(raw[0][(i, z)], raw_se[0][(i, z)], raw[1][(i, z)], raw_se[1][(i, z)])
    emu = {'low_M': beta[0], 'high_M': beta[1], 'avg_M': avg, 'high_M_raw': raw[1], 'avg_M_raw': avg_raw}
    ses = {'low_M': se[0], 'high_M': se[1], 'avg_M': avg_se, 'high_M_raw': raw_se[1], 'avg_M_raw': avg_raw_se}
    return zs, emu, ses, fid


def to_fits(source, z, n_labels, n_cols):
    return [(i, list(range(n_cols)), source[(i, z)]) for i in range(n_labels) if source[(i, z)] is not None]


def write_config(path, zs, emu, fid, columns):
    n_cols, n_labels = len(columns), len(PARAM_NAMES)

    def fmt(fits, indent):
        pad = ' ' * indent
        return '\n'.join(f"{pad}({i}, {cols}, np.array([{', '.join(f'{v:.8g}' for v in beta)}]))," for i, cols, beta in fits)

    lines = [
        '"""',
        f'Compact-model emulator refit with columns {columns}',
        'Generated by nb/nn_training/11par_w0wa/refit_emulator_w0wa.py -- do not edit by hand.',
        'Fits are in Delta_theta = theta_fiducial - theta; predictor order: ' + ', '.join(columns) + '.',
        '"""',
        'import numpy as np',
        '',
        'from compact.config import Config',
        '',
        '',
        'class ConfigW0wa(Config):',
        '    """Config whose compact-model emulator also depends on w0, wa (theta_fid: w0=-1, wa=0)."""',
        '',
        '    exact_growth = True   # growth rate f from CAMB (fsigma8 / sigma8) instead of Omega_m(z)^0.55',
        f'    fiducial_cosmo = np.array([{", ".join(repr(float(v)) for v in fid)}])',
        '',
        '    compact_emulator = {',
    ]
    for z in zs:
        lines += [f'        {z}: {{', "            'ns_fiducial': np.float64(0.9649),", "            'mean_params': Config._MEAN_PARAMS,",
                  "            'fits': [", fmt(to_fits(emu['avg_M'], z, n_labels, n_cols), 20), '            ],',
                  "            'fits_indiv': {"]
        for scheme in ('low_M', 'high_M'):
            lines += [f"                '{scheme}': [", fmt(to_fits(emu[scheme], z, n_labels, n_cols), 20), '                ],']
        lines += ['            },', '        },']
    lines += ['    }', '']
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def robust_ylim(vals_list, pad_frac=0.15, mad_k=4.0):
    vals = np.concatenate([np.atleast_1d(v) for v in vals_list if v is not None and len(np.atleast_1d(v))])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    if mad == 0:
        spread = np.std(vals) or abs(med) * 0.1 + 1e-6
        lo, hi = med - spread, med + spread
    else:
        lo, hi = med - mad_k * 1.4826 * mad, med + mad_k * 1.4826 * mad
    lo, hi = max(lo, vals.min()), min(hi, vals.max())
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


def plot_emulators(zs, emu, se, columns, path):
    """Intercept and per-column slope of every parameter vs z, in the style of the builder's
    emulator_all_params_vs_redshift.png: bin 0 (blue), bin 1 (orange) and their average (purple) from the
    bins' own fits with ridge errors; for EXTRAPOLATE_ROWS the extrapolated bin 1 and average above
    Z_ANCHOR_MAX are thin dashed lines with open markers."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    use_tex = all(shutil.which(x) for x in ('latex', 'dvipng', 'gs'))
    if use_tex:
        matplotlib.rcParams.update({'text.usetex': True, 'font.family': 'Computer Modern Roman'})
    else:
        matplotlib.rcParams.update({'mathtext.fontset': 'cm', 'font.family': 'serif'})
    plt.rcParams['axes.labelsize'] = 20
    us = r'\_' if use_tex else '_'

    n_labels, n_cols = len(PARAM_NAMES), len(columns)
    fig, ax = plt.subplots(n_labels, 1 + n_cols, figsize=(3.0 * (1 + n_cols), 2.0 * n_labels),
                           sharex=True, layout='constrained')

    def series(scheme, i, col, zmin=-np.inf):
        pts = [(z, emu[scheme][(i, z)][col], se[scheme][(i, z)][col]) for z in zs
               if emu[scheme].get((i, z)) is not None and z >= zmin]
        return np.array(pts).T if pts else np.empty((3, 0))

    for i in range(n_labels):
        for col in range(1 + n_cols):
            a, vals = ax[i, col], []
            used_raw = [('low_M', 'C0', 'bin 0'), ('high_M_raw', 'C1', 'bin 1'), ('avg_M_raw', 'purple', 'average')]
            for scheme, c, label in used_raw:
                x, y, e = series(scheme, i, col)
                a.errorbar(x, y, yerr=e, fmt='o' + ('-' if scheme == 'avg_M_raw' else '--'), c=c, ms=4, lw=1.6,
                           capsize=2, label=label)
                vals.append(y)
            if i in EXTRAP_IDX:
                for scheme, c in (('high_M', 'C1'), ('avg_M', 'purple')):
                    x, y, e = series(scheme, i, col, zmin=Z_ANCHOR_MAX)
                    a.errorbar(x, y, yerr=e, fmt='o--', c=c, ms=3.5, mfc='none', lw=0.9, capsize=2)
                    vals.append(y)
                a.axvline(Z_ANCHOR_MAX, c='gray', ls=':', lw=0.8)
            if col > 0:
                a.axhline(0, c='gray', ls='--', lw=1)
                vals.append(np.array([0.0]))
            ylim = robust_ylim(vals, 0.25)
            if ylim is not None:
                a.set_ylim(*ylim)
            if i == n_labels - 1:
                a.set_xlabel(r'$z$')
        ax[i, 0].set_ylabel(PARAM_LABELS[i])
    ax[0, 0].set_title('intercept')
    for j, c in enumerate(columns):
        ax[0, 1 + j].set_title('slope: ' + COLUMN_LABELS[c])
    handles = [Line2D([], [], c='k', ls='--', lw=0.9, marker='o', mfc='none', ms=4, label='extrapolated (bin 0 + shift)'),
               Line2D([], [], c='C0', ls='--', marker='o', ms=4, label='bin 0'),
               Line2D([], [], c='C1', ls='--', marker='o', ms=4, label='bin 1'),
               Line2D([], [], c='purple', ls='-', marker='o', ms=4, label='average')]
    ax[0, 0].legend(handles=handles, fontsize=6)
    fig.suptitle(f'All parameters; extrapolated above $z={Z_ANCHOR_MAX:g}$ only for ' + ', '.join(p.replace('_', us) for p in EXTRAPOLATE_ROWS))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def check_against_config(zs, emu):
    """Compare the refit (--columns lcdm) with compact/config.py's compact_emulator: worst
    max-relative coefficient difference per (scheme, z); 'missing' = row not refit."""
    from compact.config import Config
    for scheme, getref in (('low_M', lambda n: n['fits_indiv']['low_M']), ('high_M', lambda n: n['fits_indiv']['high_M']),
                           ('avg_M', lambda n: n['fits'])):
        print(f'{scheme}:')
        for z in zs:
            node = Config.compact_emulator.get(z)
            if node is None:
                print(f'  z={z}: no node in config.py')
                continue
            worst, missing = 0.0, 0
            for i, cols, beta_ref in getref(node):
                beta_new = emu[scheme][(i, z)]
                if beta_new is None:
                    missing += 1
                    continue
                worst = max(worst, np.max(np.abs(beta_new - beta_ref) / (np.abs(beta_ref) + 1e-12)))
            print(f'  z={z}: worst rel diff {worst:.1e}, {missing} of {len(PARAM_NAMES)} rows missing')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--columns', choices=list(COLUMN_SETS), default='w0wa')
    ap.add_argument('--check', action='store_true', help='compare against compact/config.py (use with --columns lcdm)')
    ap.add_argument('--out', default=OUT_CONFIG)
    ap.add_argument('--fit-results', required=True,
                    help='per-(sim, z, bin) fit results h5 saved by cosmo_grid_building_nu.py (cosmo_grid_fit_results.h5); '
                         'NOT the older nb/drafts/auto_update/cosmo_grid_fit_results_numod.h5, whose bin 1 differs')
    ap.add_argument('--plot', action='store_true', help='also write the figure in --check mode (always written with the config)')
    ap.add_argument('--no-plot', action='store_true')
    ap.add_argument('--plot-out', default=None, help='default: emulator_vs_redshift_<columns>.png next to this script')
    args = ap.parse_args()
    cols = COLUMN_SETS[args.columns]
    zs, emu, se, fid = build_emulators(cols, args.fit_results)
    print(f'columns: {cols}\nredshift nodes: {zs}')
    if args.check:
        check_against_config(zs, emu)
    else:
        write_config(args.out, zs, emu, fid, cols)
        print(f'wrote {args.out}')
    if (args.plot or not args.check) and not args.no_plot:
        plot_path = args.plot_out or os.path.join(HERE, f'emulator_vs_redshift_{args.columns}.png')
        plot_emulators(zs, emu, se, cols, plot_path)
        print(f'wrote {plot_path}')
