import numpy as np
from pathlib import Path
from scipy.special import eval_legendre
from pycorr import TwoPointCorrelationFunction

# Run this once in your pycorr environment. It reads the raw allcounts .npy files
# and writes everything your non-pycorr fitting script needs (rebinned multipoles
# + theta-cut window components) to a sibling directory, same base filename,
# .npz instead of .npy.

DATA_DIR = Path('/spiffball/cosweeney/DESI/DR1/CorrFuncs/gccomb/gccomb')
EXTRACT_DIR = DATA_DIR.parent / 'gccomb_extracted'  # neighboring dir, not DATA_DIR itself
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

# Duplicated here rather than imported from common.py, since common.py pulls in
# the tensorflow-based emulator -- this script is meant to run in the pycorr-only
# environment and stay independent of that. Keep these in sync with common.py by
# hand if ZBINS/NRAN/ELL_ALL/ELLS_FIT ever change there.
ZBINS = {
    'LRG':           [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)],
    'ELG_LOPnotqso': [(1.1, 1.6)],
    'QSO':           [(0.8, 2.1)],
}
NRAN = {'LRG': 8, 'ELG_LOPnotqso': 10, 'QSO': 4}
ELL_ALL = (0, 2, 4)
ELLS_FIT = (0, 2)
SMIN, SMAX, DS = 20., 200., 4.


def dr1_poles_stem(tracer, zmin, zmax, nran, thetacut=True):
    suffix = '_thetacut0.05' if thetacut else ''
    return (f'allcounts_{tracer}_GCcomb_z{zmin}-{zmax}_default_FKP_lin_'
            f'nran{nran}_njack0_split20{suffix}')


def theta_cut_window(rr_counter, ells_obs, ells_theory=ELL_ALL):
    """W^cut_{ell_o,ell_t}(s) on the counter's native fine s-grid, per Pinon et al. Eq 3.5."""
    s_edges, mu_edges = rr_counter.edges
    mu_centers = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    dmu = np.diff(mu_edges)
    n_s = rr_counter.wcounts.shape[0]
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])

    W = np.zeros((len(ells_obs), len(ells_theory), n_s))
    for i_s in range(n_s):
        mask = rr_counter.wcounts[i_s] != 0.
        total_width = dmu[mask].sum()
        if total_width == 0.:
            continue
        Lo_all = {lo: eval_legendre(lo, mu_centers[mask]) for lo in ells_obs}
        for i_lo, lo in enumerate(ells_obs):
            for i_lt, lt in enumerate(ells_theory):
                Lt = eval_legendre(lt, mu_centers[mask])
                W[i_lo, i_lt, i_s] = (2 * lo + 1) * np.sum(Lo_all[lo] * Lt * dmu[mask]) / total_width
    return W, s_centers


def extract(tracer, zmin, zmax, nran):
    stem = dr1_poles_stem(tracer, zmin, zmax, nran)
    result = TwoPointCorrelationFunction.load(DATA_DIR / (stem + '.npy'))

    result_sel = result.select((SMIN, SMAX))
    binwidth = np.diff(result_sel.edges[0])[0]
    rebin_factor = int(round(DS / binwidth))
    if rebin_factor > 1:
        result_sel = result_sel[::rebin_factor]
    s_full, xi_full = result_sel(ells=ELL_ALL, return_sep=True)
    s_full, xi_full = np.asarray(s_full), np.asarray(xi_full)

    W_fine, s_fine = theta_cut_window(result.S1S2, ells_obs=ELLS_FIT, ells_theory=ELL_ALL)

    out_path = EXTRACT_DIR / (stem + '.npz')
    np.savez(out_path, s_full=s_full, xi_full=xi_full, W_fine=W_fine, s_fine=s_fine,
             ells_obs=np.array(ELLS_FIT), ells_theory=np.array(ELL_ALL))
    print(f'Saved {out_path}  (s_full: {s_full.shape}, xi_full: {xi_full.shape}, '
          f'W_fine: {W_fine.shape})')


if __name__ == '__main__':
    for tracer, zbins in ZBINS.items():
        nran = NRAN[tracer]
        for zmin, zmax in zbins:
            extract(tracer, zmin, zmax, nran)