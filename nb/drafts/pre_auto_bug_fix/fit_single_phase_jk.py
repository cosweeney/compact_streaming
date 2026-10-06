"""
Fits v_r and xi_hh SIMULTANEOUSLY (diagonal-covariance-only) in every
available mass bin, for ONE phase, and plots the result in the same style
as this project's original Fig 1: colored/offset by mass bin, main + ratio
panels for both observables.

Entirely self-contained EXCEPT for reading the two data files it needs
(compute_xihh_jk.py's output, and run_abacus_batch_jk.py's v_r output) --
NEVER imports pycorr, gather_paper_data.py, or anything pycorr-adjacent.
Defines its own copies of every model/fit function needed, matching the
project's established convention that pycorr is only ever used for
MEASURING, never for FITTING.

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
FID_REDSHIFT_VAL = 0.2
R_MIN_FIT = 15.0
N_MASS = 6
LOG_MASS_EDGES = np.array([[12, 12.01], [12.5, 12.6], [13, 13.1],
                            [13.5, 13.6], [14.0, 14.2], [14.4, 14.7]])

SIM = 'c000_ph000'   # which phase's v_r to fit -- change this
XI_PHASES = [f'c000_ph{i:03d}' for i in range(5)]   # phases with xi_hh_jk.npz available
VR_JK_PATH = ('/spiffball/cosweeney/simulations/AbacusSummit_base/halos/'
              f'AbacusSummit_base_{SIM}/halos/z0.200/data/pairhist_allbins_jk.h5')
XI_JK_PATH_TEMPLATE = ('/spiffball/cosweeney/simulations/AbacusSummit_base/halos/'
                        'AbacusSummit_base_{sim}/halos/z0.200/data/xi_hh_jk.npz')
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
             r_xi=None, xihh=None, var_xi=None,
             r_vr=None, vr=None, var_vr=None,
             popsize=15, maxiter=150, tol=1e-6, seed=0):
    # diagonal-only: var_xi/var_vr are 1D variance arrays, not full covariance.
    if mode in ('joint', 'xi'):
        m_xi = r_xi > r_min
        r_xi_fit, xihh_fit, var_xi_fit = r_xi[m_xi], xihh[m_xi], var_xi[m_xi]
    if mode in ('joint', 'vr'):
        m_vr = r_vr > r_min
        r_vr_fit, vr_fit, var_vr_fit = r_vr[m_vr], vr[m_vr], var_vr[m_vr]

    def loss(params):
        b1, b2 = params[0], params[1]
        bs = params[2] if fit_bs else 0.0
        c = 0.0
        if mode in ('joint', 'xi'):
            res = model_xihh(r_xi_fit, b1, b2, bs) - xihh_fit
            c += np.sum(res**2 / var_xi_fit)
        if mode in ('joint', 'vr'):
            res = model_vr(r_vr_fit, b1, b2, bs) - vr_fit
            c += np.sum(res**2 / var_vr_fit)
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
    # average xi_hh (and its jackknife covariance) over the phases where
    # xi_hh_jk.npz has been computed; v_r stays single-phase (SIM) below.
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
    # covariance of the phase-averaged xi_hh: mean single-phase jackknife
    # covariance / n_phases (phases treated as independent realizations)
    cov_jk_xi_per_bin = np.mean(cov_jk_per_bin_list, axis=0) / n_phases

    total = PairHist.load(VR_JK_PATH)
    with h5.File(VR_JK_PATH, 'r') as hdf:
        vr_jk_cov_per_bin = {k: hdf[f'vr_jk_cov_mbin{k}'][()] for k in range(N_MASS)}

    fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharex='col',
                            gridspec_kw={'height_ratios': [1.2, 1]})
    offset_step = 0.5

    for k in range(N_MASS-2):
        xihh = xi_per_bin[k]
        var_xi = np.diag(cov_jk_xi_per_bin[k])/2

        r_vr, vr, sd_vr, npairs = vr_profile(total, k, r_edges=R_EDGES_SHARED)
        var_vr_full = np.diag(vr_jk_cov_per_bin[k])
        good = np.isfinite(vr) & (npairs > 1000) & (var_vr_full > 0) & (r_vr < 110)
        r_vr_k, vr_k, var_vr_k = r_vr[good], vr[good], var_vr_full[good]

        res = fit_bias(mode='joint', r_min=R_MIN_FIT, r_xi=r_xi, xihh=xihh, var_xi=var_xi,
                        r_vr=r_vr_k, vr=vr_k, var_vr=var_vr_k)
        b1, b2 = res.x
        print(f'bin {k}: success={res.success}, nfev={res.nfev}, chi2={res.chi2:.3f}, '
              f'dof={res.dof}, reduced_chi2={res.reduced_chi2:.3f}, b1={b1:.3f}, b2={b2:.3f}')

        xi_model = model_xihh(r_xi, b1, b2)
        vr_model = model_vr(r_vr_k, b1, b2)

        xi_r2 = r_xi**2 * xihh
        xi_model_r2 = r_xi**2 * xi_model
        vr_err = np.sqrt(var_vr_k)
        xi_err = np.sqrt(var_xi)

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
    fig.suptitle(f'Fig 1 analogue: v_r from {SIM}, '
                 f'xi_hh averaged over {n_phases} phases (diagonal-only)')
    fig.tight_layout()

    # out_path = f'fig1_avg{n_phases}phase_xi_{SIM}.png'
    # fig.savefig(out_path, dpi=150)
    # print(f'saved {out_path}')
#%%

if __name__ == '__main__':
    main()

#%%

test_path = ('/spiff/hengweichang/AbacusSummit_base/snapshots/'
             'AbacusSummit_base_c000_ph000/halos/z0.200/halo_combined.h5')

with h5.File(test_path, 'r') as hdf:
    hcat = hdf['halos'][:]
    print(hcat.keys())

#%%
test_path_2 = ('/spiffball/salcedo/AbacusSummit/Snapshots/'
             'base_c000/base_c000_ph000/z0p300/halo_base_c000_ph000_z0p300.h5')

with h5.File(test_path_2, 'r') as hdf2:
    hcat2 = hdf2['halos'][:]
    print(hcat2.dytpe.names)
# %%
