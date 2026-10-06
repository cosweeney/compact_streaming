"""
Fits v_r and xi_hh SIMULTANEOUSLY (diagonal-covariance-only) in every
available mass bin, for the Salcedo base_c000 z=0.3 catalog, and plots the
result in the same style as this project's original Fig 1.

xi_hh(r) is AVERAGED over all 5 available phases (ph000-ph004) -- its
jackknife covariance is correspondingly averaged and divided by n_phases
(mean single-phase jackknife covariance / n_phases -- phases treated as
independent realizations, per this project's established convention for
scaling a mock-mean covariance). v_r stays SINGLE-PHASE (SIM below) --
same asymmetry as the earlier 10-phase-averaged AbacusSummit version of
this script, and for the same reason: only asked to average xi_hh.

Entirely self-contained EXCEPT for reading the data files it needs
(compute_xihh_salcedo_jk.py's output, and run_salcedo_batch_jk.py's v_r
output) -- NEVER imports pycorr, gather_paper_data.py, or anything
pycorr-adjacent. Defines its own copies of every model/fit function
needed, matching this project's convention that pycorr is only ever used
for MEASURING, never for FITTING.

Redshift is 0.3 here (z0p300), not 0.2 -- matching the Salcedo catalog,
unlike the original AbacusSummit-catalog version of this script.

I have not run this myself -- no cluster access -- so treat this as a
structurally-complete first draft to check over, same as everything else
in this pipeline.
"""
#%% 

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

from compact.lpt import GetLPTInputs
from compact.velocities.pairhist import PairHist

FID_COSMO = [0.6736, 0.02237, 0.1200, 2.083e-9, 0.9649]   # official AbacusSummit c000
FID_REDSHIFT_VAL = 0.3   # matches the Salcedo catalog's z0p300, NOT the 0.2
                         # used in the original AbacusSummit-catalog version
R_MIN_FIT = 15.0
N_MASS = 6
LOG_MASS_EDGES = np.array([[12, 12.01], [12.5, 12.6], [13, 13.1],
                            [13.5, 13.6], [14.0, 14.2], [14.4, 14.7]])

SIM = 'c000_ph000'   # which phase's v_r to fit -- change this
XI_PHASES = [f'c000_ph{i:03d}' for i in range(5)]   # all 5 available phases
REDSHIFT_OUT = 'z0.300'   # matches run_salcedo_batch_jk.py / compute_xihh_salcedo_jk.py

VR_JK_PATH = ('/spiffball/cosweeney/simulations/AbacusSummit_base_salcedo/halos/'
              f'AbacusSummit_base_{SIM}/halos/{REDSHIFT_OUT}/data/pairhist_allbins_jk.h5')
XI_JK_PATH_TEMPLATE = ('/spiffball/cosweeney/simulations/AbacusSummit_base_salcedo/halos/'
                        'AbacusSummit_base_{sim}/halos/' + REDSHIFT_OUT +
                        '/data/xi_hh_jk.npz')
R_EDGES_SHARED = np.arange(2.0, 142.0, 2.0)

lpt = GetLPTInputs(redshift=FID_REDSHIFT_VAL, cosmo=FID_COSMO)
lpt.get_pklin_gsm()


def vr_profile(h, k, r_edges=None, r_max_trust=None):
    """Pair-weighted <v_r>(r) and sigma_vr(r) for mass bin k. From
    ASbase_compact_calibration.ipynb -- unchanged."""
    Rc = 0.5 * (h.R_edges[1:] + h.R_edges[:-1])
    lc = 0.5 * (h.rlos_edges[1:] + h.rlos_edges[:-1])
    r3 = np.sqrt(Rc[:, None] ** 2 + lc[None, :] ** 2).ravel()
    n = h.n[k].ravel().astype(float)
    s1 = h.sum_vr[k].ravel()
    s2 = h.sum_vr2[k].ravel()
    if r_edges is None:
        top = r_max_trust if r_max_trust is not None else h.R_edges[-1]
        r_edges = np.arange(0.0, top + 2.0, 2.0)
    nb_bins = len(r_edges) - 1
    idx = np.searchsorted(r_edges, r3, side='right') - 1
    ok = (idx >= 0) & (idx < nb_bins) & (n > 0)

    def acc(w):
        return np.bincount(idx[ok], weights=w[ok], minlength=nb_bins)

    N = acc(n)
    r_eff = np.where(N > 0, acc(n * r3) / np.where(N > 0, N, 1), np.nan)
    mu = np.where(N > 0, acc(s1) / np.where(N > 0, N, 1), np.nan)
    m2 = np.where(N > 0, acc(s2) / np.where(N > 0, N, 1), np.nan)
    sd = np.sqrt(np.maximum(m2 - mu ** 2, 0.0))
    return r_eff, mu, sd, N.astype(np.int64)


def model_xihh(r, b1, b2, bs=0.0):
    pars = np.array([b1, b2, bs, 0, 0, 0, 0, 0, 0])
    return lpt.xi_real(r, pars)


def model_vr(r, b1, b2, bs=0.0):
    pars = np.array([b1, b2, bs, 0, 0, 0, 0, 0, 0])
    # NOT lpt.pwv_mean(r, pars) -- see this project's earlier debugging:
    # velocileptors' pwv_mean normalizes by a Tukey-windowed xi(r) that
    # doesn't match compute_xi_real's xi(r) (our actual xi_hh model), so
    # this recomputes the normalization consistently instead.
    lpt._ensure_cumulants(pars)
    v_int = np.interp(r, lpt.gsm.rint, lpt.gsm.veft)
    xi_consistent = lpt.gsm.compute_xi_real(r, *pars)
    return lpt.conv * lpt.f * v_int / (1 + xi_consistent)


def fit_bias(r_min=15.0, fit_bs=False, mode='joint',
             r_xi=None, xihh=None, cov_xi=None, n_jk_xi=64,
             r_vr=None, vr=None, cov_vr=None, n_jk_vr=64,
             popsize=15, maxiter=150, tol=1e-6, seed=0):
    # cov_xi/cov_vr are FULL covariance matrices. n_jk_xi/n_jk_vr are the
    # number of jackknife regions each covariance was built from (64,
    # matching N_JK_1D=4's 4x4x4 grid, for both observables in this
    # project) -- used for the Hartlap correction below, which corrects
    # the known bias in inverting a covariance estimated from a finite
    # number of jackknife/mock realizations (Hartlap et al. 2007): the
    # naive inverse systematically OVERSTATES precision (understates
    # true uncertainty) as n_bins approaches n_jk. Applied AFTER masking
    # to r_min, since n_bins there is what actually enters the inversion.
    if mode in ('joint', 'xi'):
        m_xi = r_xi > r_min
        r_xi_fit, xihh_fit = r_xi[m_xi], xihh[m_xi]
        cov_xi_fit = cov_xi[np.ix_(m_xi, m_xi)]
        n_bins_xi = m_xi.sum()
        hartlap_xi = (n_jk_xi - n_bins_xi - 2) / (n_jk_xi - 1)
        inv_cov_xi_fit = hartlap_xi * np.linalg.inv(cov_xi_fit)
    if mode in ('joint', 'vr'):
        m_vr = r_vr > r_min
        r_vr_fit, vr_fit = r_vr[m_vr], vr[m_vr]
        cov_vr_fit = cov_vr[np.ix_(m_vr, m_vr)]
        n_bins_vr = m_vr.sum()
        hartlap_vr = (n_jk_vr - n_bins_vr - 2) / (n_jk_vr - 1)
        inv_cov_vr_fit = hartlap_vr * np.linalg.inv(cov_vr_fit)

    def loss(params):
        b1, b2 = params[0], params[1]
        bs = params[2] if fit_bs else 0.0
        c = 0.0
        if mode in ('joint', 'xi'):
            res = model_xihh(r_xi_fit, b1, b2, bs) - xihh_fit
            c += res @ inv_cov_xi_fit @ res
        if mode in ('joint', 'vr'):
            res = model_vr(r_vr_fit, b1, b2, bs) - vr_fit
            c += res @ inv_cov_vr_fit @ res
        return c if np.isfinite(c) else np.inf

    bounds = [(-0.3, 5.0), (-10.0, 10.0)] + ([(-5.0, 5.0)] if fit_bs else [])
    result = differential_evolution(loss, bounds, tol=tol, seed=seed,
                                     maxiter=maxiter, popsize=popsize, polish=True)

    n_data = 0
    if mode in ('joint', 'xi'):
        n_data += len(xihh_fit)
    if mode in ('joint', 'vr'):
        n_data += len(vr_fit)
    result.dof = max(n_data - len(bounds), 1)
    result.chi2 = result.fun
    result.reduced_chi2 = result.fun / result.dof
    return result

def main():
    # average xi_hh (and its jackknife covariance) over the 5 available
    # phases; v_r stays single-phase (SIM) below.
    xi_per_bin_list = []
    cov_jk_per_bin_list = []
    r_xi = None
    for sim in XI_PHASES:
        xi_data = np.load(XI_JK_PATH_TEMPLATE.format(sim=sim))
        if r_xi is None:
            r_xi = xi_data['r_cens']
        else:
            assert np.allclose(r_xi, xi_data['r_cens']), f'{sim}: r_cens mismatch'
        xi_per_bin_list.append(xi_data['xi_per_bin'])
        cov_jk_per_bin_list.append(xi_data['cov_jk_per_bin'])

    n_phases = len(XI_PHASES)
    xi_per_bin = np.mean(xi_per_bin_list, axis=0)
    cov_jk_xi_per_bin = np.mean(cov_jk_per_bin_list, axis=0) / n_phases

    total = PairHist.load(VR_JK_PATH)
    with h5.File(VR_JK_PATH, 'r') as hdf:
        vr_jk_cov_per_bin = {k: hdf[f'vr_jk_cov_mbin{k}'][()] for k in range(N_MASS)}

    fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharex='col',
                            gridspec_kw={'height_ratios': [1.2, 1]})
    offset_step = 0.5

    for k in range(N_MASS-2):
        xihh = xi_per_bin[k]
        cov_xi = cov_jk_xi_per_bin[k] #/ 2   # full matrix now, /2 convention preserved

        r_vr, vr, sd_vr, npairs = vr_profile(total, k, r_edges=R_EDGES_SHARED)
        cov_vr_full = vr_jk_cov_per_bin[k]
        var_vr_full = np.diag(cov_vr_full)   # diag only for the good-point mask below
        good = np.isfinite(vr) & (npairs > 1000) & (var_vr_full > 0) & (r_vr < 110)
        r_vr_k, vr_k = r_vr[good], vr[good]
        cov_vr_k = cov_vr_full[np.ix_(good, good)]   # full matrix, row+col masked together

        res = fit_bias(mode='joint', r_min=R_MIN_FIT, r_xi=r_xi, xihh=xihh, cov_xi=cov_xi,
                        r_vr=r_vr_k, vr=vr_k, cov_vr=cov_vr_k)
        b1, b2 = res.x
        print(f'bin {k}: success={res.success}, nfev={res.nfev}, chi2={res.chi2:.3f}, '
              f'dof={res.dof}, reduced_chi2={res.reduced_chi2:.3f}, b1={b1:.3f}, b2={b2:.3f}')

        xi_model = model_xihh(r_xi, b1, b2)
        vr_model = model_vr(r_vr_k, b1, b2)

        xi_r2 = r_xi**2 * xihh
        xi_model_r2 = r_xi**2 * xi_model
        # diagonal only, for plotted error bands -- NOT used in the fit above
        vr_err = np.sqrt(np.diag(cov_vr_k))
        xi_err = np.sqrt(np.diag(cov_xi))

        lo, hi = LOG_MASS_EDGES[k]
        ax[0, 0].plot(r_vr_k, vr_k, 'o', ms=3, c=f'C{k}', zorder=2,
                       label=rf'$\log M \in [{lo:.2f}, {hi:.2f}]$')
        ax[0, 0].fill_between(r_vr_k, vr_k - vr_err, vr_k + vr_err,
                               color=f'C{k}', alpha=0.25, lw=0, zorder=1)
        ax[0, 0].plot(r_vr_k, vr_model, '-', c=f'C{k}')

        ax[0, 1].plot(r_xi, xi_r2, 'o', ms=3, c=f'C{k}', zorder=2)
        ax[0, 1].fill_between(r_xi, xi_r2 - r_xi**2*xi_err, xi_r2 + r_xi**2*xi_err,
                               color=f'C{k}', alpha=0.25, lw=0, zorder=1)
        ax[0, 1].plot(r_xi, xi_model_r2, '-', c=f'C{k}')

        vr_ratio = vr_model / vr_k
        vr_ratio_err = np.abs(vr_ratio) * (vr_err / np.abs(vr_k))
        baseline = 1 + k * offset_step
        ax[1, 0].axhline(baseline, c=f'C{k}', lw=0.7, alpha=0.5, zorder=0)
        ax[1, 0].fill_between(r_vr_k, baseline - 2*vr_ratio_err, baseline + 2*vr_ratio_err,
                               color=f'C{k}', alpha=0.12, lw=0, zorder=0)
        ax[1, 0].fill_between(r_vr_k, baseline - vr_ratio_err, baseline + vr_ratio_err,
                               color=f'C{k}', alpha=0.22, lw=0, zorder=0)
        ax[1, 0].plot(r_vr_k, vr_ratio + k * offset_step, 'o-', ms=3, c=f'C{k}', zorder=2)

        xi_ratio = xi_model / xihh
        xi_ratio_err = np.abs(xi_ratio) * (xi_err / np.abs(xihh))
        xi_good = np.abs(xihh) > xi_err
        ax[1, 1].axhline(baseline, c=f'C{k}', lw=0.7, alpha=0.5, zorder=0)
        ax[1, 1].fill_between(r_xi[xi_good], baseline - 2*xi_ratio_err[xi_good],
                               baseline + 2*xi_ratio_err[xi_good],
                               color=f'C{k}', alpha=0.12, lw=0, zorder=0)
        ax[1, 1].fill_between(r_xi[xi_good], baseline - xi_ratio_err[xi_good],
                               baseline + xi_ratio_err[xi_good],
                               color=f'C{k}', alpha=0.22, lw=0, zorder=0)
        ax[1, 1].plot(r_xi[xi_good], xi_ratio[xi_good] + k * offset_step,
                       'o-', ms=3, c=f'C{k}', zorder=2)

    for ax_ in ax.flat:
        ax_.axvline(R_MIN_FIT, color='gray', ls=':', lw=1, zorder=0)
        ax_.margins(x=0)

    ax[0, 0].set_ylabel(r'$\langle v_r|r\rangle$ [km/s]')
    ax[0, 1].set_ylabel(r'$r^2\,\xi_{hh}(r|M)$ [$(h^{-1}\,{\rm Mpc})^2$]')
    ax[1, 0].set_ylabel('model / data (+ offset)')
    ax[1, 1].set_ylabel('model / data (+ offset)')
    ax[1, 0].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
    ax[1, 1].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
    ax[0, 0].legend(fontsize=8, ncol=2)
    ax[0, 1].set_yscale('log')
    ax[0, 1].set_ylim((1, 100))
    ax[1, 0].set_ylim((0.5, 3.0))
    ax[1, 1].set_ylim((0.5, 3.0))
    fig.suptitle(f'Salcedo z=0.3 Fig 1 analogue: v_r from {SIM}, '
                 f'xi_hh averaged over {n_phases} phases (full covariance, Hartlap-corrected)')
    fig.tight_layout()

    # out_path = f'fig1_salcedo_avg{n_phases}phase_xi_{SIM}_fullcov.png'
    # fig.savefig(out_path, dpi=150)
    # print(f'saved {out_path}')
#%%

if __name__ == '__main__':
    main()
# %%
#%% 
# Same Fig-1-style plot as main(), but with v_r and xi_hh fit
# INDIVIDUALLY (mode='vr', mode='xi') rather than jointly -- so each
# panel's model curve reflects only that observable's own best fit,
# letting you see whether the two probes agree or disagree on b1/b2.
# xi_hh is still averaged over all 5 phases (as in main()); v_r is still
# single-phase (SIM) -- same data as main(), different fit mode only.
# Now uses FULL covariance matrices (fit_bias's actual cov_xi/cov_vr
# signature) instead of the 1D diagonal variance passed before.

xi_per_bin_list, cov_jk_per_bin_list, r_xi = [], [], None
for sim in XI_PHASES:
    xi_data = np.load(XI_JK_PATH_TEMPLATE.format(sim=sim))
    if r_xi is None:
        r_xi = xi_data['r_cens']
    xi_per_bin_list.append(xi_data['xi_per_bin'])
    cov_jk_per_bin_list.append(xi_data['cov_jk_per_bin'])
n_phases = len(XI_PHASES)
xi_per_bin = np.mean(xi_per_bin_list, axis=0)
cov_jk_xi_per_bin = np.mean(cov_jk_per_bin_list, axis=0) / n_phases

total = PairHist.load(VR_JK_PATH)
with h5.File(VR_JK_PATH, 'r') as hdf:
    vr_jk_cov_per_bin = {k: hdf[f'vr_jk_cov_mbin{k}'][()] for k in range(N_MASS)}

MASS_BINS_TO_PLOT = range(N_MASS - 2)   # matches main()'s existing subset

fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharex='col',
                        gridspec_kw={'height_ratios': [1.2, 1]})
offset_step = 0.5

for k in MASS_BINS_TO_PLOT:
    xihh = xi_per_bin[k]
    cov_xi = cov_jk_xi_per_bin[k]   # full matrix, no /2

    r_vr, vr, sd_vr, npairs = vr_profile(total, k, r_edges=R_EDGES_SHARED)
    cov_vr_full = vr_jk_cov_per_bin[k]
    good = np.isfinite(vr) & (npairs > 1000) & (np.diag(cov_vr_full) > 0) & (r_vr < 110)
    r_vr_k, vr_k = r_vr[good], vr[good]
    cov_vr_k = cov_vr_full[np.ix_(good, good)]   # full matrix, row+col masked together

    res_xi = fit_bias(mode='xi', r_min=R_MIN_FIT, r_xi=r_xi, xihh=xihh, cov_xi=cov_xi)
    res_vr = fit_bias(mode='vr', r_min=R_MIN_FIT,
                       r_vr=r_vr_k, vr=vr_k, cov_vr=cov_vr_k)
    b1_xi, b2_xi = res_xi.x
    b1_vr, b2_vr = res_vr.x
    print(f'bin {k}: xi-only  b1={b1_xi:.3f}, b2={b2_xi:.3f}, '
          f'reduced_chi2={res_xi.reduced_chi2:.3f}')
    print(f'bin {k}: vr-only  b1={b1_vr:.3f}, b2={b2_vr:.3f}, '
          f'reduced_chi2={res_vr.reduced_chi2:.3f}')

    xi_model = model_xihh(r_xi, b1_xi, b2_xi)
    vr_model = model_vr(r_vr_k, b1_vr, b2_vr)

    xi_r2 = r_xi**2 * xihh
    xi_model_r2 = r_xi**2 * xi_model
    vr_err = np.sqrt(np.diag(cov_vr_k))
    xi_err = np.sqrt(np.diag(cov_xi))

    lo, hi = LOG_MASS_EDGES[k]
    ax[0, 0].plot(r_vr_k, vr_k, 'o', ms=3, c=f'C{k}', zorder=2,
                   label=rf'$\log M \in [{lo:.2f}, {hi:.2f}]$')
    ax[0, 0].fill_between(r_vr_k, vr_k - vr_err, vr_k + vr_err,
                           color=f'C{k}', alpha=0.25, lw=0, zorder=1)
    ax[0, 0].plot(r_vr_k, vr_model, '-', c=f'C{k}')

    ax[0, 1].plot(r_xi, xi_r2, 'o', ms=3, c=f'C{k}', zorder=2)
    ax[0, 1].fill_between(r_xi, xi_r2 - r_xi**2*xi_err, xi_r2 + r_xi**2*xi_err,
                           color=f'C{k}', alpha=0.25, lw=0, zorder=1)
    ax[0, 1].plot(r_xi, xi_model_r2, '-', c=f'C{k}')

    vr_ratio = vr_model / vr_k
    vr_ratio_err = np.abs(vr_ratio) * (vr_err / np.abs(vr_k))
    baseline = 1 + k * offset_step
    ax[1, 0].axhline(baseline, c=f'C{k}', lw=0.7, alpha=0.5, zorder=0)
    ax[1, 0].fill_between(r_vr_k, baseline - 2*vr_ratio_err, baseline + 2*vr_ratio_err,
                           color=f'C{k}', alpha=0.12, lw=0, zorder=0)
    ax[1, 0].fill_between(r_vr_k, baseline - vr_ratio_err, baseline + vr_ratio_err,
                           color=f'C{k}', alpha=0.22, lw=0, zorder=0)
    ax[1, 0].plot(r_vr_k, vr_ratio + k * offset_step, 'o-', ms=3, c=f'C{k}', zorder=2)

    xi_ratio = xi_model / xihh
    xi_ratio_err = np.abs(xi_ratio) * (xi_err / np.abs(xihh))
    xi_good = np.abs(xihh) > xi_err
    ax[1, 1].axhline(baseline, c=f'C{k}', lw=0.7, alpha=0.5, zorder=0)
    ax[1, 1].fill_between(r_xi[xi_good], baseline - 2*xi_ratio_err[xi_good],
                           baseline + 2*xi_ratio_err[xi_good],
                           color=f'C{k}', alpha=0.12, lw=0, zorder=0)
    ax[1, 1].fill_between(r_xi[xi_good], baseline - xi_ratio_err[xi_good],
                           baseline + xi_ratio_err[xi_good],
                           color=f'C{k}', alpha=0.22, lw=0, zorder=0)
    ax[1, 1].plot(r_xi[xi_good], xi_ratio[xi_good] + k * offset_step,
                   'o-', ms=3, c=f'C{k}', zorder=2)

for ax_ in ax.flat:
    ax_.axvline(R_MIN_FIT, color='gray', ls=':', lw=1, zorder=0)
    ax_.margins(x=0)

ax[0, 0].set_ylabel(r'$\langle v_r|r\rangle$ [km/s]')
ax[0, 1].set_ylabel(r'$r^2\,\xi_{hh}(r|M)$ [$(h^{-1}\,{\rm Mpc})^2$]')
ax[1, 0].set_ylabel('model / data (+ offset)')
ax[1, 1].set_ylabel('model / data (+ offset)')
ax[1, 0].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
ax[1, 1].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
ax[0, 0].legend(fontsize=8, ncol=2)
ax[0, 1].set_yscale('log')
ax[0, 1].set_ylim((1, 100))
ax[1, 0].set_ylim((0.5, 3.0))
ax[1, 1].set_ylim((0.5, 3.0))
fig.suptitle(f'Individual (non-joint) fits: v_r-only vs xi_hh-only b1/b2 '
             f'({n_phases}-phase avg xi_hh, v_r from {SIM})')
fig.tight_layout()
#%%
#%% #%% 2D chi2 surface in (b1, b2) per mass bin -- v2: adaptive window + log
# color scale, since the fixed +/-1.0/+/-3.0 window was far too wide for
# how sharply peaked these surfaces actually are (chi2_min up to ~3000 at
# dof=108), collapsing every contour to an invisible point. Re-run the
# rebuild-data block from the previous cell first if xi_per_bin/total/etc.
# aren't already in the namespace.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

DELTA_CHI2_LEVELS = [2.30, 6.18, 11.8]   # 68.3%/95.4%/99.7% for 2 dof
TARGET_DCHI2 = 15.0   # window is auto-sized so this Delta-chi2 is just inside the edge
GRID_N = 61

def chi2_b1b2(b1, b2, r_xi_fit, xihh_fit, var_xi_fit,
              r_vr_fit, vr_fit, var_vr_fit):
    res_xi = model_xihh(r_xi_fit, b1, b2) - xihh_fit
    c = np.sum(res_xi**2 / var_xi_fit)
    res_vr = model_vr(r_vr_fit, b1, b2) - vr_fit
    c += np.sum(res_vr**2 / var_vr_fit)
    return c

def find_half_width(b_best, other_val, chi2_min, target, vary_b1,
                     args, lo_guess=0.02, hi_guess=5.0, n_expand=30):
    """1D scan along ONE axis (other axis fixed at its best-fit value) to
    find the +/- distance from b_best where Delta-chi2 first exceeds
    `target`. Expands geometrically from lo_guess until it brackets
    target, then returns that distance -- used to size the 2D grid window
    per-bin/per-axis instead of guessing a fixed width."""
    width = lo_guess
    for _ in range(n_expand):
        if vary_b1:
            c = chi2_b1b2(b_best + width, other_val, *args)
        else:
            c = chi2_b1b2(other_val, b_best + width, *args)
        if c - chi2_min > target:
            return width
        width *= 1.4
    return hi_guess   # fallback if it never exceeds target (very flat direction)

# --- rebuild the same averaged xi_hh + single-phase v_r data main() uses ---
xi_per_bin_list, cov_jk_per_bin_list, r_xi = [], [], None
for sim in XI_PHASES:
    xi_data = np.load(XI_JK_PATH_TEMPLATE.format(sim=sim))
    if r_xi is None:
        r_xi = xi_data['r_cens']
    xi_per_bin_list.append(xi_data['xi_per_bin'])
    cov_jk_per_bin_list.append(xi_data['cov_jk_per_bin'])
n_phases = len(XI_PHASES)
xi_per_bin = np.mean(xi_per_bin_list, axis=0)
cov_jk_xi_per_bin = np.mean(cov_jk_per_bin_list, axis=0) / n_phases

total = PairHist.load(VR_JK_PATH)
with h5.File(VR_JK_PATH, 'r') as hdf:
    vr_jk_cov_per_bin = {k: hdf[f'vr_jk_cov_mbin{k}'][()] for k in range(N_MASS)}

MASS_BINS_TO_PLOT = list(range(N_MASS - 2))

fig, axes = plt.subplots(1, len(MASS_BINS_TO_PLOT),
                          figsize=(5.2 * len(MASS_BINS_TO_PLOT), 4.5))
if len(MASS_BINS_TO_PLOT) == 1:
    axes = [axes]

for ax, k in zip(axes, MASS_BINS_TO_PLOT):
    xihh = xi_per_bin[k]
    var_xi = np.diag(cov_jk_xi_per_bin[k]) / 2
    m_xi = r_xi > R_MIN_FIT
    r_xi_fit, xihh_fit, var_xi_fit = r_xi[m_xi], xihh[m_xi], var_xi[m_xi]

    r_vr, vr, sd_vr, npairs = vr_profile(total, k, r_edges=R_EDGES_SHARED)
    var_vr_full = np.diag(vr_jk_cov_per_bin[k])
    good = np.isfinite(vr) & (npairs > 1000) & (var_vr_full > 0) & (r_vr < 110)
    r_vr_k, vr_k, var_vr_k = r_vr[good], vr[good], var_vr_full[good]
    m_vr = r_vr_k > R_MIN_FIT
    r_vr_fit, vr_fit, var_vr_fit = r_vr_k[m_vr], vr_k[m_vr], var_vr_k[m_vr]

    args = (r_xi_fit, xihh_fit, var_xi_fit, r_vr_fit, vr_fit, var_vr_fit)

    res = fit_bias(mode='joint', r_min=R_MIN_FIT, r_xi=r_xi, xihh=xihh, var_xi=var_xi,
                    r_vr=r_vr_k, vr=vr_k, var_vr=var_vr_k)
    b1_best, b2_best = res.x
    chi2_min = chi2_b1b2(b1_best, b2_best, *args)

    b1_hw = find_half_width(b1_best, b2_best, chi2_min, TARGET_DCHI2, True, args)
    b2_hw = find_half_width(b2_best, b1_best, chi2_min, TARGET_DCHI2, False, args)

    b1_grid = np.linspace(b1_best - b1_hw, b1_best + b1_hw, GRID_N)
    b2_grid = np.linspace(b2_best - b2_hw, b2_best + b2_hw, GRID_N)
    chi2_grid = np.zeros((GRID_N, GRID_N))
    for i, b1v in enumerate(b1_grid):
        for j, b2v in enumerate(b2_grid):
            chi2_grid[j, i] = chi2_b1b2(b1v, b2v, *args)
    dchi2_grid = np.maximum(chi2_grid - chi2_min, 1e-3)   # floor for LogNorm

    im = ax.pcolormesh(b1_grid, b2_grid, dchi2_grid, shading='auto',
                        cmap='viridis_r',
                        norm=LogNorm(vmin=0.1, vmax=max(dchi2_grid.max(), 20)))
    cs = ax.contour(b1_grid, b2_grid, dchi2_grid, levels=DELTA_CHI2_LEVELS,
                     colors='white', linewidths=1.2)
    ax.clabel(cs, fmt={2.30: '68%', 6.18: '95%', 11.8: '99.7%'}, fontsize=7)
    ax.plot(b1_best, b2_best, 'r*', ms=14, mec='k', mew=0.8, zorder=5)
    fig.colorbar(im, ax=ax, label=r'$\Delta\chi^2$', shrink=0.85)

    lo, hi = LOG_MASS_EDGES[k]
    ax.set_title(rf'$\log M \in [{lo:.2f},{hi:.2f}]$' +
                 f'\n(min $\\chi^2$={chi2_min:.1f}, dof={res.dof}, '
                 f'$\\chi^2_\\nu$={chi2_min/res.dof:.1f})')
    ax.set_xlabel('$b_1$')
    if k == MASS_BINS_TO_PLOT[0]:
        ax.set_ylabel('$b_2$')

fig.suptitle(f'$b_1$-$b_2$ joint likelihood surfaces '
             f'({n_phases}-phase avg $\\xi_{{hh}}$, v_r from {SIM})')
fig.tight_layout()


#%%

#%% Check: does lpt.xi_real use the windowed (gsm.xieft) or unwindowed
# (gsm.compute_xi_real) real-space correlation function? Confirmed from
# velocileptors' own source (gaussian_streaming_model_fftw.py) that these
# TWO INTERNAL QUANTITIES GENUINELY DIFFER (xieft is Tukey-windowed inside
# compute_cumulants; compute_xi_real recomputes the same kernel WITHOUT
# the window) -- this is a structural feature of the package, not a bug.
# What's unknown (lpt.xi_real is this project's own wrapper, not
# velocileptors') is which convention it forwards to -- checked directly
# here rather than assumed either way.

import numpy as np
import matplotlib.pyplot as plt

b1_test, b2_test = 0.5, 0.0   # any representative bias values; joint
                                # best-fit for a bin also works
bs_test = 0.0
pars_test = np.array([b1_test, b2_test, bs_test, 0, 0, 0, 0, 0, 0])
r_test = np.linspace(15, 130, 60)

# path 1: the project's own convenience wrapper (what model_xihh calls)
xi_wrapper = lpt.xi_real(r_test, pars_test)

# path 2: unwindowed, direct call to velocileptors' own compute_xi_real
xi_unwindowed = lpt.gsm.compute_xi_real(r_test, *pars_test)

# path 3: windowed, via the cumulants gsm.xieft populates internally --
# same object model_vr already triggers via lpt._ensure_cumulants
lpt._ensure_cumulants(pars_test)
xi_windowed = np.interp(r_test, lpt.gsm.rint, lpt.gsm.xieft)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(r_test, xi_wrapper, 'k-', lw=2.5, label='lpt.xi_real (wrapper)')
axes[0].plot(r_test, xi_unwindowed, 'b--', label='gsm.compute_xi_real (unwindowed)')
axes[0].plot(r_test, xi_windowed, 'r:', label='gsm.xieft (windowed)')
axes[0].set_yscale('log')
axes[0].set_xlabel('$r$ [$h^{-1}$Mpc]')
axes[0].set_ylabel(r'$\xi_{real}(r)$')
axes[0].legend(fontsize=9)
axes[0].set_title('Three candidate real-space $\\xi(r)$ definitions')

axes[1].plot(r_test, xi_wrapper / xi_unwindowed - 1, 'b-',
             label='wrapper / unwindowed - 1')
axes[1].plot(r_test, xi_wrapper / xi_windowed - 1, 'r-',
             label='wrapper / windowed - 1')
axes[1].axhline(0, c='gray', lw=0.7)
axes[1].set_xlabel('$r$ [$h^{-1}$Mpc]')
axes[1].set_ylabel('fractional difference from lpt.xi_real')
axes[1].legend(fontsize=9)
axes[1].set_title('Which convention does lpt.xi_real actually match?')
fig.tight_layout()

print('max |wrapper/unwindowed - 1| =', np.max(np.abs(xi_wrapper/xi_unwindowed - 1)))
print('max |wrapper/windowed - 1|   =', np.max(np.abs(xi_wrapper/xi_windowed - 1)))

#%% 
#%% From-scratch LPT prediction of xi_real AND <v_r|r>, built directly
# from velocileptors' GaussianStreamingModel public API (gsm.compute_
# cumulants, gsm.compute_xi_real, gsm.veft/gsm.rint) -- entirely
# independent of this project's own model_xihh/model_vr/lpt._ensure_
# cumulants wrappers, as a from-scratch cross-check that those wrappers
# correctly reflect what velocileptors itself computes.
#
# v_r formula matches compute_xi_rsd's own v_int line (gaussian_streaming_
# model_fftw.py) with mu=1 (pure radial pairwise velocity, no line-of-
# sight projection) and unwindowed xi in the denominator -- same
# convention model_vr already uses, just built here without reusing that
# function or its caching.

import numpy as np
import matplotlib.pyplot as plt

b1_test, b2_test = 0.5, 0.0   # match cell 1's choice for direct comparison
bs_test, b3_test = 0.0, 0.0
alpha_test = alpha_v_test = alpha_s0_test = alpha_s2_test = s2fog_test = 0.0

r_test = np.linspace(15, 130, 60)

# --- from-scratch xi_real: direct call, unwindowed, no caching ---
xi_real_scratch = lpt.gsm.compute_xi_real(
    r_test, b1_test, b2_test, bs_test, b3_test,
    alpha_test, alpha_v_test, alpha_s0_test, alpha_s2_test, s2fog_test)

# --- from-scratch v_r: recompute cumulants directly (NOT via
# lpt._ensure_cumulants), then apply the same v12(r) = f*veft(r)/(1+xi(r))
# relation used in compute_xi_rsd, with mu=1 for pure radial velocity,
# and the UNWINDOWED xi_real_scratch above (not the windowed xieft the
# cumulants call also produces) as the denominator -- matching model_vr's
# own choice to use compute_xi_real for normalization consistency.
lpt.gsm.compute_cumulants(b1_test, b2_test, bs_test, b3_test,
                           alpha_test, alpha_v_test, alpha_s0_test,
                           alpha_s2_test, s2fog_test)
v_int_scratch = np.interp(r_test, lpt.gsm.rint, lpt.gsm.veft)
vr_scratch = lpt.conv * lpt.f * v_int_scratch / (1 + xi_real_scratch)

# --- compare against the project's own wrapper functions ---
pars_test = np.array([b1_test, b2_test, bs_test, 0, 0, 0, 0, 0, 0])
xi_real_wrapper = model_xihh(r_test, b1_test, b2_test, bs_test)
vr_wrapper = model_vr(r_test, b1_test, b2_test, bs_test)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes[0,0].plot(r_test, xi_real_scratch, 'k-', lw=2.5, label='from scratch')
axes[0,0].plot(r_test, xi_real_wrapper, 'b--', label='model_xihh')
axes[0,0].set_yscale('log')
axes[0,0].set_ylabel(r'$\xi_{real}(r)$')
axes[0,0].legend(fontsize=9)

axes[0,1].plot(r_test, xi_real_wrapper/xi_real_scratch - 1, 'k-')
axes[0,1].axhline(0, c='gray', lw=0.7)
axes[0,1].set_ylabel('model_xihh / scratch - 1')

axes[1,0].plot(r_test, vr_scratch, 'k-', lw=2.5, label='from scratch')
axes[1,0].plot(r_test, vr_wrapper, 'b--', label='model_vr')
axes[1,0].set_xlabel('$r$ [$h^{-1}$Mpc]')
axes[1,0].set_ylabel(r'$\langle v_r|r\rangle$ [km/s]')
axes[1,0].legend(fontsize=9)

axes[1,1].plot(r_test, vr_wrapper/vr_scratch - 1, 'k-')
axes[1,1].axhline(0, c='gray', lw=0.7)
axes[1,1].set_xlabel('$r$ [$h^{-1}$Mpc]')
axes[1,1].set_ylabel('model_vr / scratch - 1')

fig.suptitle('From-scratch velocileptors prediction vs. project wrapper functions')
fig.tight_layout()

print('max |model_xihh/scratch - 1| =', np.max(np.abs(xi_real_wrapper/xi_real_scratch - 1)))
print('max |model_vr/scratch - 1|   =', np.max(np.abs(vr_wrapper/vr_scratch - 1)))
# %%
#%% Diagnostic: pool ALL mass bins into one combined (b1,b2) fit, using
# ONLY already-saved data (no new measurement). v_r's central value is
# pooled EXACTLY (raw pair-count sums, summed before normalizing).
# xi_hh's central value is a WEIGHTED AVERAGE of the already-computed
# per-bin curves (approximate -- exact pooling needs raw DD pair counts,
# which aren't saved). BOTH covariances are approximate: weighted sum of
# the existing per-bin covariances, ASSUMING mass bins are independent --
# they aren't (same box, same phases), so this likely understates the
# true combined covariance. Treat this as a stress-test/diagnostic, not
# a calibration product.

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

MASS_BINS_TO_POOL = list(range(N_MASS))   # matches the 4 bins already plotted;
                                                # change to range(N_MASS) for all 6

R_MIN_FIT = 20

import matplotlib
matplotlib.rcParams.update({'text.usetex': True, 
                            'font.family': 'Computer Modern Roman'})
plt.rcParams['axes.labelsize'] = 20

# --- rebuild phase-averaged xi_hh per bin, PLUS n_halos_per_bin for weighting ---
xi_per_bin_list, cov_jk_per_bin_list, nhalo_list = [], [], []
r_xi = None
for sim in XI_PHASES:
    xi_data = np.load(XI_JK_PATH_TEMPLATE.format(sim=sim))
    if r_xi is None:
        r_xi = xi_data['r_cens']
    xi_per_bin_list.append(xi_data['xi_per_bin'])
    cov_jk_per_bin_list.append(xi_data['cov_jk_per_bin'])
    nhalo_list.append(xi_data['n_halos_per_bin'])
n_phases = len(XI_PHASES)
xi_per_bin = np.mean(xi_per_bin_list, axis=0)
cov_jk_xi_per_bin = np.mean(cov_jk_per_bin_list, axis=0) / n_phases
n_halos_per_bin_avg = np.mean(nhalo_list, axis=0)   # avg halo count per bin, across phases

# --- xi_hh: weighted average across the pooled mass bins (approximate) ---
w_xi = n_halos_per_bin_avg[MASS_BINS_TO_POOL]
w_xi = w_xi / w_xi.sum()
xihh_pooled = np.sum(w_xi[:, None] * xi_per_bin[MASS_BINS_TO_POOL], axis=0)
cov_xi_pooled = np.sum((w_xi**2)[:, None, None] * cov_jk_xi_per_bin[MASS_BINS_TO_POOL], axis=0)

# --- v_r: EXACT pooling via raw sums, before any normalization ---
def vr_profile_pooled(h, ks, r_edges=None, r_max_trust=None):
    """Same binning logic as vr_profile, but sums raw n/sum_vr/sum_vr2
    across multiple mass bins ks BEFORE normalizing -- this is the exact
    pairwise-weighted <v_r|r> for the pooled halo sample, not a weighted
    average of already-normalized per-bin profiles."""
    Rc = 0.5 * (h.R_edges[1:] + h.R_edges[:-1])
    lc = 0.5 * (h.rlos_edges[1:] + h.rlos_edges[:-1])
    r3 = np.sqrt(Rc[:, None] ** 2 + lc[None, :] ** 2).ravel()
    n = sum(h.n[k] for k in ks).ravel().astype(float)
    s1 = sum(h.sum_vr[k] for k in ks).ravel()
    s2 = sum(h.sum_vr2[k] for k in ks).ravel()
    if r_edges is None:
        top = r_max_trust if r_max_trust is not None else h.R_edges[-1]
        r_edges = np.arange(0.0, top + 2.0, 2.0)
    nb_bins = len(r_edges) - 1
    idx = np.searchsorted(r_edges, r3, side='right') - 1
    ok = (idx >= 0) & (idx < nb_bins) & (n > 0)

    def acc(w):
        return np.bincount(idx[ok], weights=w[ok], minlength=nb_bins)

    N = acc(n)
    r_eff = np.where(N > 0, acc(n * r3) / np.where(N > 0, N, 1), np.nan)
    mu = np.where(N > 0, acc(s1) / np.where(N > 0, N, 1), np.nan)
    m2 = np.where(N > 0, acc(s2) / np.where(N > 0, N, 1), np.nan)
    sd = np.sqrt(np.maximum(m2 - mu ** 2, 0.0))
    return r_eff, mu, sd, N.astype(np.int64)

total = PairHist.load(VR_JK_PATH)
with h5.File(VR_JK_PATH, 'r') as hdf:
    vr_jk_cov_per_bin = {k: hdf[f'vr_jk_cov_mbin{k}'][()] for k in range(N_MASS)}

r_vr, vr_pooled, sd_vr, npairs = vr_profile_pooled(total, MASS_BINS_TO_POOL, r_edges=R_EDGES_SHARED)

# v_r COVARIANCE: approximate (weighted sum of existing per-bin covariances,
# independence assumed) -- exact pooled jackknife covariance would need the
# raw per-region PairHist pieces, which run_salcedo_batch_jk.py discards.
# Weight by each bin's own pair-count fraction at each r (more principled
# than a single scalar weight, since it matches how the exact mean above
# was actually built).
n_per_bin_r = {}
for k in MASS_BINS_TO_POOL:
    _, _, _, npairs_k = vr_profile(total, k, r_edges=R_EDGES_SHARED)
    n_per_bin_r[k] = npairs_k.astype(float)
n_total_r = sum(n_per_bin_r.values())
n_total_r[n_total_r == 0] = 1   # avoid div-by-zero; those r-bins are empty anyway

cov_vr_pooled = np.zeros((len(r_vr), len(r_vr)))
for k in MASS_BINS_TO_POOL:
    w_k_r = n_per_bin_r[k] / n_total_r   # r-dependent weight vector for bin k
    cov_vr_pooled += np.outer(w_k_r, w_k_r) * vr_jk_cov_per_bin[k]

# --- masks + fit, same conventions as main() ---
good = np.isfinite(vr_pooled) & (npairs > 1000) & (np.diag(cov_vr_pooled) > 0) & (r_vr < 110)
r_vr_k, vr_k = r_vr[good], vr_pooled[good]
cov_vr_k = cov_vr_pooled[np.ix_(good, good)]

res = fit_bias(mode='joint', r_min=R_MIN_FIT, r_xi=r_xi, xihh=xihh_pooled, cov_xi=cov_xi_pooled,
                r_vr=r_vr_k, vr=vr_k, cov_vr=cov_vr_k)
b1, b2 = res.x
print(f'POOLED bins {MASS_BINS_TO_POOL}: success={res.success}, chi2={res.chi2:.3f}, '
      f'dof={res.dof}, reduced_chi2={res.reduced_chi2:.3f}, b1={b1:.3f}, b2={b2:.3f}')

xi_model = model_xihh(r_xi, b1, b2)
vr_model = model_vr(r_vr_k, b1, b2)

fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharex='col',
                        gridspec_kw={'height_ratios': [1.2, 1]})

vr_err = np.sqrt(np.diag(cov_vr_k))
xi_err = np.sqrt(np.diag(cov_xi_pooled))
xi_r2 = r_xi**2 * xihh_pooled
xi_model_r2 = r_xi**2 * xi_model

ax[0, 0].plot(r_vr_k, r_vr_k**2*vr_k, 'o', ms=3, c='C0', zorder=2, label='pooled data')
ax[0, 0].fill_between(r_vr_k, r_vr_k**2*(vr_k - vr_err), r_vr_k**2*(vr_k + vr_err), color='C0', alpha=0.25, lw=0)
ax[0, 0].plot(r_vr_k, r_vr_k**2*vr_model, '-', c='C0', label='joint fit')
ax[0, 0].legend(fontsize=9)

ax[0, 1].plot(r_xi, xi_r2, 'o', ms=3, c='C0', zorder=2)
ax[0, 1].fill_between(r_xi, xi_r2 - r_xi**2*xi_err, xi_r2 + r_xi**2*xi_err, color='C0', alpha=0.25, lw=0)
ax[0, 1].plot(r_xi, xi_model_r2, '-', c='C0')

vr_ratio = vr_model / vr_k
vr_ratio_err = np.abs(vr_ratio) * (vr_err / np.abs(vr_k))
ax[1, 0].axhline(1, c='C0', lw=0.7, alpha=0.5)
ax[1, 0].fill_between(r_vr_k, 1 - 2*vr_ratio_err, 1 + 2*vr_ratio_err, color='C0', alpha=0.12, lw=0)
ax[1, 0].fill_between(r_vr_k, 1 - vr_ratio_err, 1 + vr_ratio_err, color='C0', alpha=0.22, lw=0)
ax[1, 0].plot(r_vr_k, vr_ratio, 'o-', ms=3, c='C0')

xi_ratio = xi_model / xihh_pooled
xi_ratio_err = np.abs(xi_ratio) * (xi_err / np.abs(xihh_pooled))
xi_good = np.abs(xihh_pooled) > xi_err
ax[1, 1].axhline(1, c='C0', lw=0.7, alpha=0.5)
ax[1, 1].fill_between(r_xi[xi_good], 1 - 2*xi_ratio_err[xi_good], 1 + 2*xi_ratio_err[xi_good], color='C0', alpha=0.12, lw=0)
ax[1, 1].fill_between(r_xi[xi_good], 1 - xi_ratio_err[xi_good], 1 + xi_ratio_err[xi_good], color='C0', alpha=0.22, lw=0)
ax[1, 1].plot(r_xi[xi_good], xi_ratio[xi_good], 'o-', ms=3, c='C0')

for ax_ in ax.flat:
    ax_.axvline(R_MIN_FIT, color='gray', ls=':', lw=1, zorder=0)
    ax_.margins(x=0)

ax[0, 0].set_ylabel(r'$\langle v_r|r\rangle$ [km/s]')
ax[0, 1].set_ylabel(r'$r^2\,\xi_{hh}(r|M)$ [$(h^{-1}\,{\rm Mpc})^2$]')
ax[1, 0].set_ylabel('model / data')
ax[1, 1].set_ylabel('model / data')
ax[1, 0].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
ax[1, 1].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
ax[0, 1].set_yscale('log')
ax[1, 0].set_ylim((0.5, 1.5))
ax[1, 1].set_ylim((0.5, 1.5))
fig.suptitle(f'Mass-pooled diagnostic fit (bins {MASS_BINS_TO_POOL}): '
             f'v_r exact pool, xi_hh approx-weighted, cov approx (independence assumed)')
fig.tight_layout()

#%%
#%% Simultaneous (joint) fit for the "allmass" (M >= 1e12, no split) data.
# xi_hh averaged over all 5 available phases (compute_xihh_salcedo_allmass_
# jk.py already has all 5, decoupled from v_r); v_r single-phase (SIM),
# since run_salcedo_batch_allmass_jk.py has only run c000_ph000 so far.
# Uses the full-covariance, Hartlap-corrected fit_bias -- same as main()'s
# current form, not the earlier diagonal-only version.

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams.update({'text.usetex': True, 
                            'font.family': 'Computer Modern Roman'})
plt.rcParams['axes.labelsize'] = 20

SIM_ALLMASS = 'c000_ph000'   # only phase run so far for v_r
XI_PHASES_ALLMASS = [f'c000_ph{i:03d}' for i in range(5)]   # all 5 -- xi_hh has no
                                                               # v_r dependency

VR_JK_PATH_ALLMASS = ('/spiffball/cosweeney/simulations/AbacusSummit_base_salcedo/halos/'
                       f'AbacusSummit_base_{SIM_ALLMASS}/halos/' + REDSHIFT_OUT +
                       '/data/pairhist_allbins_jk_allmass.h5')
XI_JK_PATH_ALLMASS_TEMPLATE = ('/spiffball/cosweeney/simulations/AbacusSummit_base_salcedo/halos/'
                                'AbacusSummit_base_{sim}/halos/' + REDSHIFT_OUT +
                                '/data/xi_hh_jk_allmass.npz')

# --- average xi_hh + covariance over available phases (flat arrays --
# xi_hh_jk_allmass.npz has NO per-mass-bin axis, unlike the split version) ---
xi_list, cov_jk_list, r_xi = [], [], None
for sim in XI_PHASES_ALLMASS:
    xi_data = np.load(XI_JK_PATH_ALLMASS_TEMPLATE.format(sim=sim))
    if r_xi is None:
        r_xi = xi_data['r_cens']
    else:
        assert np.allclose(r_xi, xi_data['r_cens']), f'{sim}: r_cens mismatch'
    xi_list.append(xi_data['xi'])
    cov_jk_list.append(xi_data['cov_jk'])

n_phases_allmass = len(XI_PHASES_ALLMASS)
xihh_allmass = np.mean(xi_list, axis=0)
cov_xi_allmass = np.mean(cov_jk_list, axis=0) / n_phases_allmass

# --- v_r: single mass bin (index 0), single phase ---
total_allmass = PairHist.load(VR_JK_PATH_ALLMASS)
with h5.File(VR_JK_PATH_ALLMASS, 'r') as hdf:
    vr_jk_cov_allmass = hdf['vr_jk_cov_mbin0'][()]

r_vr, vr, sd_vr, npairs = vr_profile(total_allmass, 0, r_edges=R_EDGES_SHARED)
good = np.isfinite(vr) & (npairs > 1000) & (np.diag(vr_jk_cov_allmass) > 0) & (r_vr < 110)
r_vr_k, vr_k = r_vr[good], vr[good]
cov_vr_k = vr_jk_cov_allmass[np.ix_(good, good)]

# --- fit (full covariance, Hartlap-corrected) ---
res = fit_bias(mode='joint', r_min=R_MIN_FIT, r_xi=r_xi, xihh=xihh_allmass, cov_xi=cov_xi_allmass,
                r_vr=r_vr_k, vr=vr_k, cov_vr=cov_vr_k)
b1, b2 = res.x
print(f'allmass (M>=1e12): success={res.success}, chi2={res.chi2:.3f}, '
      f'dof={res.dof}, reduced_chi2={res.reduced_chi2:.3f}, b1={b1:.3f}, b2={b2:.3f}')

xi_model = model_xihh(r_xi, b1, b2)
vr_model = model_vr(r_vr_k, b1, b2)

vr_err = np.sqrt(np.diag(cov_vr_k))
xi_err = np.sqrt(np.diag(cov_xi_allmass))
xi_r2 = r_xi**2 * xihh_allmass
xi_model_r2 = r_xi**2 * xi_model

fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharex='col',
                        gridspec_kw={'height_ratios': [1.2, 1]})

ax[0, 0].plot(r_vr_k, vr_k, 'o', ms=3, c='C0', zorder=2, label='data')
ax[0, 0].fill_between(r_vr_k, vr_k - vr_err, vr_k + vr_err, color='C0', alpha=0.25, lw=0)
ax[0, 0].plot(r_vr_k, vr_model, '-', c='C0', label='joint fit')
ax[0, 0].legend(fontsize=9)

ax[0, 1].plot(r_xi, xi_r2, 'o', ms=3, c='C0', zorder=2)
ax[0, 1].fill_between(r_xi, xi_r2 - r_xi**2*xi_err, xi_r2 + r_xi**2*xi_err, color='C0', alpha=0.25, lw=0)
ax[0, 1].plot(r_xi, xi_model_r2, '-', c='C0')

vr_ratio = vr_model / vr_k
vr_ratio_err = np.abs(vr_ratio) * (vr_err / np.abs(vr_k))
ax[1, 0].axhline(1, c='C0', lw=0.7, alpha=0.5)
ax[1, 0].fill_between(r_vr_k, 1 - 2*vr_ratio_err, 1 + 2*vr_ratio_err, color='C0', alpha=0.12, lw=0)
ax[1, 0].fill_between(r_vr_k, 1 - vr_ratio_err, 1 + vr_ratio_err, color='C0', alpha=0.22, lw=0)
ax[1, 0].plot(r_vr_k, vr_ratio, 'o-', ms=3, c='C0')

xi_ratio = xi_model / xihh_allmass
xi_ratio_err = np.abs(xi_ratio) * (xi_err / np.abs(xihh_allmass))
xi_good = np.abs(xihh_allmass) > xi_err
ax[1, 1].axhline(1, c='C0', lw=0.7, alpha=0.5)
ax[1, 1].fill_between(r_xi[xi_good], 1 - 2*xi_ratio_err[xi_good], 1 + 2*xi_ratio_err[xi_good], color='C0', alpha=0.12, lw=0)
ax[1, 1].fill_between(r_xi[xi_good], 1 - xi_ratio_err[xi_good], 1 + xi_ratio_err[xi_good], color='C0', alpha=0.22, lw=0)
ax[1, 1].plot(r_xi[xi_good], xi_ratio[xi_good], 'o-', ms=3, c='C0')

for ax_ in ax.flat:
    ax_.axvline(R_MIN_FIT, color='gray', ls=':', lw=1, zorder=0)
    ax_.margins(x=0)
    ax_.set_xlim((5, 110))

ax[0, 0].set_ylabel(r'$\langle v_r|r\rangle$ [km/s]')
ax[0, 1].set_ylabel(r'$r^2\,\xi_{hh}(r|M)$ [$(h^{-1}\,{\rm Mpc})^2$]')
ax[1, 0].set_ylabel('model / data')
ax[1, 1].set_ylabel('model / data')
ax[1, 0].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
ax[1, 1].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
ax[0, 1].set_yscale('log')
ax[1, 0].set_ylim((0.5, 1.5))
ax[1, 1].set_ylim((0.5, 1.5))
fig.suptitle(f'$M \\geq 10^{{12}}\\ h^{{-1}}M_\\odot$) joint fit: AbacusSummit c000_ph000'
             )
fig.tight_layout()
# %%
#%% Route A feasibility check: do the existing xi_hh / v_r jackknife files
# actually retain PER-REGION values (needed to build an empirical xi_hh<->v_r
# cross-covariance the way v12_joint.py does for xi_hm/xi_hh/v12), or only the
# already-collapsed cov_jk matrices main() currently loads? Also checks
# whether the two files even use the SAME jackknife partition (same N_JK,
# same region indexing) -- a cross-covariance is only meaningful if region i
# means the same patch of the box in both.

import numpy as np
import h5py as h5

CHECK_SIM = 'c000_ph000'
xi_npz_path = XI_JK_PATH_TEMPLATE.format(sim=CHECK_SIM)
vr_h5_path = VR_JK_PATH

print(f'--- {xi_npz_path} ---')
xi_data = np.load(xi_npz_path)
for key in xi_data.files:
    arr = xi_data[key]
    print(f'  {key:24s} shape={arr.shape} dtype={arr.dtype}')
print()

print(f'--- {vr_h5_path} ---')
with h5.File(vr_h5_path, 'r') as hdf:
    def _show(name, obj):
        if isinstance(obj, h5.Dataset):
            print(f'  {name:40s} shape={obj.shape} dtype={obj.dtype}')
        else:
            print(f'  {name:40s} (group)')
    hdf.visititems(_show)
    print('  top-level attrs:', dict(hdf.attrs))
print()

# fit_sim_jk.py currently only reads cov_jk_per_bin (xi) and vr_jk_cov_mbin{k}
# (v_r) -- both already-collapsed NxN covariance matrices, no visible
# per-region axis. If nothing above shows a leading axis of ~64 (matching
# N_JK_1D=4's 4x4x4 grid) that ISN'T n_bins or n_mass, per-region values
# were never saved and Route A needs a rerun of the jk scripts, not just a
# change to this script.
n_bins_xi = xi_data['r_cens'].shape[0]
print(f'xi_hh: r_cens has {n_bins_xi} bins -- look for an xi npz array whose '
      f'shape is (n_jk, {n_bins_xi}) or (n_jk, n_mass, {n_bins_xi}), n_jk != {n_bins_xi}')

total = PairHist.load(vr_h5_path)
print('\nPairHist object attributes (look for anything region/jk-indexed):')
for attr in sorted(dir(total)):
    if attr.startswith('_'):
        continue
    val = getattr(total, attr)
    if isinstance(val, np.ndarray):
        print(f'  {attr:24s} ndarray shape={val.shape}')
    elif isinstance(val, dict):
        shapes = {k: (np.shape(v) if hasattr(v, "shape") else type(v).__name__)
                  for k, v in list(val.items())[:3]}
        print(f'  {attr:24s} dict, {len(val)} keys, sample shapes={shapes}')

#%%
#%% Route B: joint xi_hh + v_r fit with a fitted cross-statistic model-error
# term, replacing fit_bias(mode='joint')'s implicit (and here, likely wrong)
# zero-cross-covariance assumption. C = C_data (the same block-diagonal
# jackknife covariances fit_bias already uses) + C_err(eps_xi, eps_vr, L,
# rho_cross), where C_err's diagonal blocks are the usual
# eps^2 |d_i||d_j| rho(|r_i-r_j|;L) softening and the OFF-diagonal
# (xi_hh <-> v_r) block uses the same kernel on physical separation across
# the two different r grids, scaled by rho_cross -- same construction as
# modelerr_fit_v12.py's stacked_covariance, adapted to this script's
# differential_evolution + Hartlap-corrected-covariance conventions instead
# of v12_joint.py's Nelder-Mead.
#
# rho_cross fixed at 0.8, not fit: model_xihh and model_vr share the SAME
# lpt.gsm cumulant object here (unlike v12_joint.py's hm/hh sharing LPT
# templates while v is an independent GSM calc) -- if anything the case for
# correlating xi_hh and v_r's model error is stronger than v12_joint.py's
# 0.8 for hm-hh. Leaving it fixed also avoids adding a 6th free parameter to
# a fit with only 6 mass bins x 1 phase of v_r data.
#
# L is PROFILED on a grid, not optimized jointly with (b1,b2,eps_xi,eps_vr)
# -- same reasoning as modelerr_fit_v12.py: jointly optimizing L opens a
# degenerate mode where the model-error term goes rank-one and swamps b1/b2.
# This means an inner differential_evolution call PER GRID POINT, so this is
# ~len(LGRID)x slower than fit_bias -- shrink LGRID or DE's popsize/maxiter
# if that matters for a first check.

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import differential_evolution

RHO_CROSS = 0.8
EPS_BOUNDS = (1e-6, 1) #0.05
LGRID = np.logspace(np.log10(2.0), np.log10(40.0), 10)


def kernel_matrix(D, L):
    return np.exp(-0.5*(D/L)**2)   # sqexp only -- matches modelerr_fit_v12.py's default


def _model_err_block(r_a, d_a, r_b, d_b, eps_a, eps_b, L, rho):
    D = np.abs(np.subtract.outer(r_a, r_b))
    return rho*eps_a*eps_b*np.outer(np.abs(d_a), np.abs(d_b))*kernel_matrix(D, L)


def fit_bias_modelerr(r_min, r_xi, xihh, cov_xi, r_vr, vr, cov_vr,
                       rho_cross=RHO_CROSS, popsize=15, maxiter=150,
                       tol=1e-6, seed=0):
    """Joint xi_hh + v_r fit with a fitted, cross-correlated model-error
    term on top of the block-diagonal jackknife covariance -- the Route B
    stand-in for fit_bias(mode='joint') when per-region data (Route A)
    isn't available."""
    m_xi = r_xi > r_min
    r_xi_fit, xihh_fit = r_xi[m_xi], xihh[m_xi]
    cov_xi_fit = cov_xi[np.ix_(m_xi, m_xi)]

    m_vr = r_vr > r_min
    r_vr_fit, vr_fit = r_vr[m_vr], vr[m_vr]
    cov_vr_fit = cov_vr[np.ix_(m_vr, m_vr)]

    n_xi, n_vr = m_xi.sum(), m_vr.sum()
    n_data = n_xi + n_vr
    d = np.concatenate([xihh_fit, vr_fit])
    bounds = [(-0.3, 5.0), (-10.0, 10.0), EPS_BOUNDS, EPS_BOUNDS]

    def build_cov(eps_xi, eps_vr, L):
        C = np.zeros((n_data, n_data))
        C[:n_xi, :n_xi] = cov_xi_fit + _model_err_block(
            r_xi_fit, xihh_fit, r_xi_fit, xihh_fit, eps_xi, eps_xi, L, 1.0)
        C[n_xi:, n_xi:] = cov_vr_fit + _model_err_block(
            r_vr_fit, vr_fit, r_vr_fit, vr_fit, eps_vr, eps_vr, L, 1.0)
        cross = _model_err_block(r_xi_fit, xihh_fit, r_vr_fit, vr_fit,
                                  eps_xi, eps_vr, L, rho_cross)
        C[:n_xi, n_xi:] = cross
        C[n_xi:, :n_xi] = cross.T
        return C

    def cost_at(L):
        def cost(theta, chi2_only=False):
            b1, b2, eps_xi, eps_vr = theta
            res = np.concatenate([model_xihh(r_xi_fit, b1, b2) - xihh_fit,
                                   model_vr(r_vr_fit, b1, b2) - vr_fit])
            C = build_cov(eps_xi, eps_vr, L)
            try:
                cf = cho_factor(C, lower=True, check_finite=False)
            except Exception:
                return 1e100
            chi2 = res @ cho_solve(cf, res, check_finite=False)
            if chi2_only:
                return float(chi2)
            return float(chi2 + 2*np.sum(np.log(np.abs(np.diag(cf[0])))))
        return cost

    best_L, best_theta, best_val = None, None, np.inf
    for L in LGRID:
        cost = cost_at(L)
        r_ = differential_evolution(cost, bounds, tol=tol, seed=seed,
                                     maxiter=maxiter, popsize=popsize, polish=True)
        if r_.fun < best_val:
            best_L, best_theta, best_val = L, r_.x, r_.fun

    b1, b2, eps_xi, eps_vr = best_theta
    chi2 = cost_at(best_L)(best_theta, chi2_only=True)
    npar = 4 + 1   # b1, b2, eps_xi, eps_vr, L
    dof = max(n_data - npar, 1)

    class Result:
        pass
    result = Result()
    result.x = np.array([b1, b2])
    result.eps = dict(xi=eps_xi, vr=eps_vr)
    result.L = float(best_L)
    result.chi2, result.dof = float(chi2), dof
    result.reduced_chi2 = float(chi2)/dof
    return result

#%%
#%% 
# Route B, standalone: rebuild the same per-bin data main() uses, then
# fit with fit_bias_modelerr instead of fit_bias(mode='joint').

import numpy as np
import h5py as h5

xi_per_bin_list, cov_jk_per_bin_list, r_xi = [], [], None
for sim in XI_PHASES:
    xi_data = np.load(XI_JK_PATH_TEMPLATE.format(sim=sim))
    if r_xi is None:
        r_xi = xi_data['r_cens']
    xi_per_bin_list.append(xi_data['xi_per_bin'])
    cov_jk_per_bin_list.append(xi_data['cov_jk_per_bin'])
n_phases = len(XI_PHASES)
xi_per_bin = np.mean(xi_per_bin_list, axis=0)
cov_jk_xi_per_bin = np.mean(cov_jk_per_bin_list, axis=0) / n_phases

total = PairHist.load(VR_JK_PATH)
with h5.File(VR_JK_PATH, 'r') as hdf:
    vr_jk_cov_per_bin = {kk: hdf[f'vr_jk_cov_mbin{kk}'][()] for kk in range(N_MASS)}

for k in range(N_MASS - 2):
    xihh = xi_per_bin[k]
    cov_xi = cov_jk_xi_per_bin[k] 

    r_vr, vr, sd_vr, npairs = vr_profile(total, k, r_edges=R_EDGES_SHARED)
    cov_vr_full = vr_jk_cov_per_bin[k]
    good = np.isfinite(vr) & (npairs > 1000) & (np.diag(cov_vr_full) > 0) & (r_vr < 110)
    r_vr_k, vr_k = r_vr[good], vr[good]
    cov_vr_k = cov_vr_full[np.ix_(good, good)]

    res = fit_bias_modelerr(R_MIN_FIT, r_xi, xihh, cov_xi, r_vr_k, vr_k, cov_vr_k)
    b1, b2 = res.x
    print(f'bin {k}: b1={b1:.3f}, b2={b2:.3f}, eps_xi={res.eps["xi"]:.2e}, '
          f'eps_vr={res.eps["vr"]:.2e}, L={res.L:.2f}, '
          f'reduced_chi2={res.reduced_chi2:.3f} (dof={res.dof})')
# %%
#%% fit_bias, extended with one intrinsic-scatter parameter per included
# statistic (eps_xi and/or eps_vr): C_used = C_jk/hartlap + eps^2*diag(d^2)
# -- diagonal only, no coherence length or cross-statistic term. At eps=0
# this reproduces fit_bias's hartlap*inv(cov) exactly (hartlap folded into
# the data covariance before adding the scatter term, since C now varies
# with the fit parameters). Loss carries the +log|C| term eps needs, per
# this project's established requirement (omitting it lets eps diverge to
# 0). Independent of the Route B / modelerr_fit_v12.py design: no kernel,
# no L, no rho_cross.

from scipy.linalg import cho_factor, cho_solve

#%% fit_bias_scatter, no Hartlap correction: chi2 uses the raw inverse
# covariance directly, unscaled. Isolates whether the earlier small
# reduced_chi2 values were a Hartlap-factor artifact (per last message's
# diagnosis) rather than a genuinely good fit.

from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import differential_evolution

def fit_bias_scatter(r_min=15.0, fit_bs=False, mode='joint',
                      r_xi=None, xihh=None, cov_xi=None, n_jk_xi=64,
                      r_vr=None, vr=None, cov_vr=None, n_jk_vr=64,
                      popsize=15, maxiter=150, tol=1e-6, seed=0):
    if mode in ('joint', 'xi'):
        m_xi = r_xi > r_min
        r_xi_fit, xihh_fit = r_xi[m_xi], xihh[m_xi]
        cov_xi_fit = cov_xi[np.ix_(m_xi, m_xi)]
        d2_xi = xihh_fit**2
    if mode in ('joint', 'vr'):
        m_vr = r_vr > r_min
        r_vr_fit, vr_fit = r_vr[m_vr], vr[m_vr]
        cov_vr_fit = cov_vr[np.ix_(m_vr, m_vr)]
        d2_vr = vr_fit**2

    n_bs = 1 if fit_bs else 0
    n_eps_xi = 1 if mode in ('joint', 'xi') else 0
    n_eps_vr = 1 if mode in ('joint', 'vr') else 0

    def unpack(params):
        b1, b2 = params[0], params[1]
        i = 2
        bs = params[i] if fit_bs else 0.0
        i += n_bs
        eps_xi = params[i] if n_eps_xi else None
        i += n_eps_xi
        eps_vr = params[i] if n_eps_vr else None
        return b1, b2, bs, eps_xi, eps_vr

    def cost(params, chi2_only=False):
        b1, b2, bs, eps_xi, eps_vr = unpack(params)
        c = 0.0
        if mode in ('joint', 'xi'):
            try:
                res = model_xihh(r_xi_fit, b1, b2, bs) - xihh_fit
            except Exception:
                return np.inf
            C = cov_xi_fit + eps_xi**2 * np.diag(d2_xi)
            try:
                cf, low = cho_factor(C, lower=True, check_finite=False)
            except np.linalg.LinAlgError:
                return np.inf
            c += res @ cho_solve((cf, low), res, check_finite=False)
            if not chi2_only:
                c += 2.0 * np.sum(np.log(np.abs(np.diag(cf))))
        if mode in ('joint', 'vr'):
            try:
                res = model_vr(r_vr_fit, b1, b2, bs) - vr_fit
            except Exception:
                return np.inf
            C = cov_vr_fit + eps_vr**2 * np.diag(d2_vr)
            try:
                cf, low = cho_factor(C, lower=True, check_finite=False)
            except np.linalg.LinAlgError:
                return np.inf
            c += res @ cho_solve((cf, low), res, check_finite=False)
            if not chi2_only:
                c += 2.0 * np.sum(np.log(np.abs(np.diag(cf))))
        return c if np.isfinite(c) else np.inf

    EPS_BOUNDS = (1e-6, 0.5)
    bounds = [(-0.3, 5.0), (-10.0, 10.0)]
    if fit_bs:
        bounds += [(-5.0, 5.0)]
    if n_eps_xi:
        bounds += [EPS_BOUNDS]
    if n_eps_vr:
        bounds += [EPS_BOUNDS]

    result = differential_evolution(cost, bounds, tol=tol, seed=seed,
                                     maxiter=maxiter, popsize=popsize, polish=True)

    n_data = 0
    if mode in ('joint', 'xi'):
        n_data += len(xihh_fit)
    if mode in ('joint', 'vr'):
        n_data += len(vr_fit)
    result.dof = max(n_data - len(bounds), 1)
    result.chi2 = cost(result.x, chi2_only=True)
    result.reduced_chi2 = result.chi2 / result.dof
    _, _, _, eps_xi, eps_vr = unpack(result.x)
    result.eps = {}
    if n_eps_xi:
        result.eps['xi'] = eps_xi
    if n_eps_vr:
        result.eps['vr'] = eps_vr
    return result

#%%

#%% main(), unchanged in structure (including the /2 convention on cov_xi),
# with fit_bias_scatter in place of fit_bias(mode='joint'). Note res.x now
# carries [b1, b2, eps_xi, eps_vr] (or +bs), so b1,b2 = res.x[:2], not
# res.x directly.

def main_scatter():
    xi_per_bin_list = []
    cov_jk_per_bin_list = []
    r_xi = None
    for sim in XI_PHASES:
        xi_data = np.load(XI_JK_PATH_TEMPLATE.format(sim=sim))
        if r_xi is None:
            r_xi = xi_data['r_cens']
        else:
            assert np.allclose(r_xi, xi_data['r_cens']), f'{sim}: r_cens mismatch'
        xi_per_bin_list.append(xi_data['xi_per_bin'])
        cov_jk_per_bin_list.append(xi_data['cov_jk_per_bin'])

    n_phases = len(XI_PHASES)
    xi_per_bin = np.mean(xi_per_bin_list, axis=0)
    cov_jk_xi_per_bin = np.mean(cov_jk_per_bin_list, axis=0) / n_phases

    total = PairHist.load(VR_JK_PATH)
    with h5.File(VR_JK_PATH, 'r') as hdf:
        vr_jk_cov_per_bin = {k: hdf[f'vr_jk_cov_mbin{k}'][()] for k in range(N_MASS)}

    fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharex='col',
                            gridspec_kw={'height_ratios': [1.2, 1]})
    offset_step = 0.5

    for k in range(N_MASS-2):
        xihh = xi_per_bin[k]
        cov_xi = cov_jk_xi_per_bin[k] / 2   # full matrix now, /2 convention preserved

        r_vr, vr, sd_vr, npairs = vr_profile(total, k, r_edges=R_EDGES_SHARED)
        cov_vr_full = vr_jk_cov_per_bin[k]
        var_vr_full = np.diag(cov_vr_full)
        good = np.isfinite(vr) & (npairs > 1000) & (var_vr_full > 0) & (r_vr < 110)
        r_vr_k, vr_k = r_vr[good], vr[good]
        cov_vr_k = cov_vr_full[np.ix_(good, good)]

        res = fit_bias_scatter(mode='joint', r_min=R_MIN_FIT, r_xi=r_xi, xihh=xihh, cov_xi=cov_xi,
                                r_vr=r_vr_k, vr=vr_k, cov_vr=cov_vr_k)
        b1, b2 = res.x[:2]
        print(f'bin {k}: success={res.success}, nfev={res.nfev}, chi2={res.chi2:.3f}, '
              f'dof={res.dof}, reduced_chi2={res.reduced_chi2:.3f}, b1={b1:.3f}, b2={b2:.3f}, '
              f'eps_xi={res.eps["xi"]:.2e}, eps_vr={res.eps["vr"]:.2e}')

        xi_model = model_xihh(r_xi, b1, b2)
        vr_model = model_vr(r_vr_k, b1, b2)

        xi_r2 = r_xi**2 * xihh
        xi_model_r2 = r_xi**2 * xi_model
        vr_err = np.sqrt(np.diag(cov_vr_k))
        xi_err = np.sqrt(np.diag(cov_xi))

        lo, hi = LOG_MASS_EDGES[k]
        ax[0, 0].plot(r_vr_k, vr_k, 'o', ms=3, c=f'C{k}', zorder=2,
                       label=rf'$\log M \in [{lo:.2f}, {hi:.2f}]$')
        ax[0, 0].fill_between(r_vr_k, vr_k - vr_err, vr_k + vr_err,
                               color=f'C{k}', alpha=0.25, lw=0, zorder=1)
        ax[0, 0].plot(r_vr_k, vr_model, '-', c=f'C{k}')

        ax[0, 1].plot(r_xi, xi_r2, 'o', ms=3, c=f'C{k}', zorder=2)
        ax[0, 1].fill_between(r_xi, xi_r2 - r_xi**2*xi_err, xi_r2 + r_xi**2*xi_err,
                               color=f'C{k}', alpha=0.25, lw=0, zorder=1)
        ax[0, 1].plot(r_xi, xi_model_r2, '-', c=f'C{k}')

        vr_ratio = vr_model / vr_k
        vr_ratio_err = np.abs(vr_ratio) * (vr_err / np.abs(vr_k))
        baseline = 1 + k * offset_step
        ax[1, 0].axhline(baseline, c=f'C{k}', lw=0.7, alpha=0.5, zorder=0)
        ax[1, 0].fill_between(r_vr_k, baseline - 2*vr_ratio_err, baseline + 2*vr_ratio_err,
                               color=f'C{k}', alpha=0.12, lw=0, zorder=0)
        ax[1, 0].fill_between(r_vr_k, baseline - vr_ratio_err, baseline + vr_ratio_err,
                               color=f'C{k}', alpha=0.22, lw=0, zorder=0)
        ax[1, 0].plot(r_vr_k, vr_ratio + k * offset_step, 'o-', ms=3, c=f'C{k}', zorder=2)

        xi_ratio = xi_model / xihh
        xi_ratio_err = np.abs(xi_ratio) * (xi_err / np.abs(xihh))
        xi_good = np.abs(xihh) > xi_err
        ax[1, 1].axhline(baseline, c=f'C{k}', lw=0.7, alpha=0.5, zorder=0)
        ax[1, 1].fill_between(r_xi[xi_good], baseline - 2*xi_ratio_err[xi_good],
                               baseline + 2*xi_ratio_err[xi_good],
                               color=f'C{k}', alpha=0.12, lw=0, zorder=0)
        ax[1, 1].fill_between(r_xi[xi_good], baseline - xi_ratio_err[xi_good],
                               baseline + xi_ratio_err[xi_good],
                               color=f'C{k}', alpha=0.22, lw=0, zorder=0)
        ax[1, 1].plot(r_xi[xi_good], xi_ratio[xi_good] + k * offset_step,
                       'o-', ms=3, c=f'C{k}', zorder=2)

    for ax_ in ax.flat:
        ax_.axvline(R_MIN_FIT, color='gray', ls=':', lw=1, zorder=0)
        ax_.margins(x=0)

    ax[0, 0].set_ylabel(r'$\langle v_r|r\rangle$ [km/s]')
    ax[0, 1].set_ylabel(r'$r^2\,\xi_{hh}(r|M)$ [$(h^{-1}\,{\rm Mpc})^2$]')
    ax[1, 0].set_ylabel('model / data (+ offset)')
    ax[1, 1].set_ylabel('model / data (+ offset)')
    ax[1, 0].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
    ax[1, 1].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
    ax[0, 0].legend(fontsize=8, ncol=2)
    ax[0, 1].set_yscale('log')
    ax[0, 1].set_ylim((1, 100))
    ax[1, 0].set_ylim((0.5, 3.0))
    ax[1, 1].set_ylim((0.5, 3.0))
    fig.suptitle(f'Salcedo z=0.3 Fig 1 analogue (intrinsic scatter): v_r from {SIM}, '
                 f'xi_hh averaged over {n_phases} phases (full covariance, Hartlap-corrected)')
    fig.tight_layout()


if __name__ == '__main__':
    main_scatter()

#%%

#%% Plot the fits from the pasted Route B (initial, uncorrected) run --
# hardcoded (b1, b2) from that output, no refitting. Same panel layout as
# main(). Data rebuild only (xihh, cov_xi, r_vr_k, vr_k, cov_vr_k), reusing
# the "/2"-removed cov_xi convention that run used.

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

ROUTE_B_RESULTS = {
    0: dict(b1=-0.300, b2=0.317, eps_xi=8.82e-01, eps_vr=5.19e-01, L=40.00, reduced_chi2=0.976),
    1: dict(b1=-0.300, b2=0.062, eps_xi=1.00e+00, eps_vr=4.42e-01, L=40.00, reduced_chi2=0.924),
    2: dict(b1=-0.300, b2=0.537, eps_xi=1.00e+00, eps_vr=6.02e-01, L=40.00, reduced_chi2=0.634),
    3: dict(b1=-0.300, b2=0.715, eps_xi=1.00e+00, eps_vr=5.86e-01, L=40.00, reduced_chi2=0.538),
}

xi_per_bin_list, cov_jk_per_bin_list, r_xi = [], [], None
for sim in XI_PHASES:
    xi_data = np.load(XI_JK_PATH_TEMPLATE.format(sim=sim))
    if r_xi is None:
        r_xi = xi_data['r_cens']
    xi_per_bin_list.append(xi_data['xi_per_bin'])
    cov_jk_per_bin_list.append(xi_data['cov_jk_per_bin'])
n_phases = len(XI_PHASES)
xi_per_bin = np.mean(xi_per_bin_list, axis=0)
cov_jk_xi_per_bin = np.mean(cov_jk_per_bin_list, axis=0) / n_phases   # "/2" removed

total = PairHist.load(VR_JK_PATH)
with h5.File(VR_JK_PATH, 'r') as hdf:
    vr_jk_cov_per_bin = {k: hdf[f'vr_jk_cov_mbin{k}'][()] for k in range(N_MASS)}

fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharex='col',
                        gridspec_kw={'height_ratios': [1.2, 1]})
offset_step = 0.5

for k in range(N_MASS - 2):
    xihh = xi_per_bin[k]
    cov_xi = cov_jk_xi_per_bin[k]

    r_vr, vr, sd_vr, npairs = vr_profile(total, k, r_edges=R_EDGES_SHARED)
    cov_vr_full = vr_jk_cov_per_bin[k]
    good = np.isfinite(vr) & (npairs > 1000) & (np.diag(cov_vr_full) > 0) & (r_vr < 110)
    r_vr_k, vr_k = r_vr[good], vr[good]
    cov_vr_k = cov_vr_full[np.ix_(good, good)]

    b1, b2 = ROUTE_B_RESULTS[k]['b1'], ROUTE_B_RESULTS[k]['b2']
    xi_model = model_xihh(r_xi, b1, b2)
    vr_model = model_vr(r_vr_k, b1, b2)

    xi_r2 = r_xi**2 * xihh
    xi_model_r2 = r_xi**2 * xi_model
    vr_err = np.sqrt(np.diag(cov_vr_k))
    xi_err = np.sqrt(np.diag(cov_xi))

    lo, hi = LOG_MASS_EDGES[k]
    ax[0, 0].plot(r_vr_k, vr_k, 'o', ms=3, c=f'C{k}', zorder=2,
                   label=rf'$\log M \in [{lo:.2f}, {hi:.2f}]$')
    ax[0, 0].fill_between(r_vr_k, vr_k - vr_err, vr_k + vr_err,
                           color=f'C{k}', alpha=0.25, lw=0, zorder=1)
    ax[0, 0].plot(r_vr_k, vr_model, '-', c=f'C{k}')

    ax[0, 1].plot(r_xi, xi_r2, 'o', ms=3, c=f'C{k}', zorder=2)
    ax[0, 1].fill_between(r_xi, xi_r2 - r_xi**2*xi_err, xi_r2 + r_xi**2*xi_err,
                           color=f'C{k}', alpha=0.25, lw=0, zorder=1)
    ax[0, 1].plot(r_xi, xi_model_r2, '-', c=f'C{k}')

    vr_ratio = vr_model / vr_k
    vr_ratio_err = np.abs(vr_ratio) * (vr_err / np.abs(vr_k))
    baseline = 1 + k * offset_step
    ax[1, 0].axhline(baseline, c=f'C{k}', lw=0.7, alpha=0.5, zorder=0)
    ax[1, 0].fill_between(r_vr_k, baseline - 2*vr_ratio_err, baseline + 2*vr_ratio_err,
                           color=f'C{k}', alpha=0.12, lw=0, zorder=0)
    ax[1, 0].fill_between(r_vr_k, baseline - vr_ratio_err, baseline + vr_ratio_err,
                           color=f'C{k}', alpha=0.22, lw=0, zorder=0)
    ax[1, 0].plot(r_vr_k, vr_ratio + k * offset_step, 'o-', ms=3, c=f'C{k}', zorder=2)

    xi_ratio = xi_model / xihh
    xi_ratio_err = np.abs(xi_ratio) * (xi_err / np.abs(xihh))
    xi_good = np.abs(xihh) > xi_err
    ax[1, 1].axhline(baseline, c=f'C{k}', lw=0.7, alpha=0.5, zorder=0)
    ax[1, 1].fill_between(r_xi[xi_good], baseline - 2*xi_ratio_err[xi_good],
                           baseline + 2*xi_ratio_err[xi_good],
                           color=f'C{k}', alpha=0.12, lw=0, zorder=0)
    ax[1, 1].fill_between(r_xi[xi_good], baseline - xi_ratio_err[xi_good],
                           baseline + xi_ratio_err[xi_good],
                           color=f'C{k}', alpha=0.22, lw=0, zorder=0)
    ax[1, 1].plot(r_xi[xi_good], xi_ratio[xi_good] + k * offset_step,
                   'o-', ms=3, c=f'C{k}', zorder=2)

for ax_ in ax.flat:
    ax_.axvline(R_MIN_FIT, color='gray', ls=':', lw=1, zorder=0)
    ax_.margins(x=0)

ax[0, 0].set_ylabel(r'$\langle v_r|r\rangle$ [km/s]')
ax[0, 1].set_ylabel(r'$r^2\,\xi_{hh}(r|M)$ [$(h^{-1}\,{\rm Mpc})^2$]')
ax[1, 0].set_ylabel('model / data (+ offset)')
ax[1, 1].set_ylabel('model / data (+ offset)')
ax[1, 0].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
ax[1, 1].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
ax[0, 0].legend(fontsize=8, ncol=2)
ax[0, 1].set_yscale('log')
ax[0, 1].set_ylim((1, 100))
ax[1, 0].set_ylim((0.5, 3.0))
ax[1, 1].set_ylim((0.5, 3.0))
fig.suptitle('Route B (initial, pre-fix) fit results: b1 pinned at bound, '
             'eps_xi/L at boundary in all 4 bins -- degenerate mode')
fig.tight_layout()