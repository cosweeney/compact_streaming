import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import emcee
from pathlib import Path
from multiprocessing import Pool
from common import *  # COSMO_FID, LNAS_FID, NS_FIXED, NS_MEAN, NS_SIG, OMBH2_MEAN, OMBH2_SIG,
                       # TRACERS, ZBINS, Z_EFF, SMAX_FIT, NBINS_COV, ELL_ALL, ELLS_FIT,
                       # ap_params, ap_multipoles, get_emulators_cached, save_getdist_chain, CHAIN_DIR
import psutil
psutil.Process().nice(10)

# No pycorr import here at all -- this script reads only the plain .npz files
# produced by extract_dr1_pycorr_data.py (run separately, in the pycorr env).

# --- DR1 data conventions established earlier in this project ---
EXTRACT_DIR = Path('/spiffball/cosweeney/DESI/DR1/CorrFuncs/gccomb/gccomb_extracted')
DR1_COV_DIR = Path('/spiffball/cosweeney/DESI/DR1/covariances')
NRAN = {'LRG': 8, 'ELG_LOPnotqso': 10, 'QSO': 4}  # BGS excluded -- no emulator node, per TRACERS/ZBINS
SMIN, SMAX, DS = 20., 200., 4.  # matches NBINS_COV=45 from common.py: (SMAX-SMIN)/DS == NBINS_COV


def dr1_poles_stem(tracer, zmin, zmax, nran, thetacut=True):
    suffix = '_thetacut0.05' if thetacut else ''
    return (f'allcounts_{tracer}_GCcomb_z{zmin}-{zmax}_default_FKP_lin_'
            f'nran{nran}_njack0_split20{suffix}')


def dr1_cov_path(tracer, zmin, zmax):
    return DR1_COV_DIR / (f'xi024_{tracer}_GCcomb_{zmin}_{zmax}_default_FKP_lin4_'
                           f's20-200_cov_RascalC_rescaled.txt')


def window_at(W, s_fine, s_query):
    """Linearly interpolate each W[lo, lt, :] component onto s_query.
    NOTE: this interpolates the fine-grid window onto the measurement's s-bin
    centers, rather than the RRcut-weighted rebin from 1 to 4 Mpc/h that Pinon
    et al. describe exactly. Given the validated size of this effect (~0.001-0.002
    sigma for ell=0,2 at the fiducial point checked earlier), the difference
    between these two rebinning choices should be far smaller still -- but this
    is a real simplification, not the paper's literal prescription.
    """
    n_lo, n_lt, _ = W.shape
    out = np.empty((n_lo, n_lt, len(s_query)))
    for i in range(n_lo):
        for j in range(n_lt):
            out[i, j] = np.interp(s_query, s_fine, W[i, j])
    return out


def load_dr1_tracer_arrays(tracer, zmin, zmax, nran):
    stem = dr1_poles_stem(tracer, zmin, zmax, nran)
    npz = np.load(EXTRACT_DIR / (stem + '.npz'))
    s_full, xi_full = npz['s_full'], npz['xi_full']
    W_fine, s_fine = npz['W_fine'], npz['s_fine']
    assert tuple(npz['ells_obs']) == ELLS_FIT and tuple(npz['ells_theory']) == ELL_ALL, (
        'extracted file\'s ells_obs/ells_theory do not match common.py\'s ELLS_FIT/ELL_ALL -- '
        're-run extract_dr1_pycorr_data.py if those were changed.'
    )

    cov_full = np.loadtxt(dr1_cov_path(tracer, zmin, zmax))
    mask = s_full <= SMAX_FIT
    idx = np.concatenate([ELL_ALL.index(ell) * NBINS_COV + np.where(mask)[0] for ell in ELLS_FIT])
    cov = cov_full[np.ix_(idx, idx)]
    s_data = s_full[mask]
    data_vec = np.concatenate([xi_full[ELL_ALL.index(ell)][mask] for ell in ELLS_FIT])

    W_at_sdata = window_at(W_fine, s_fine, s_data)  # (len(ELLS_FIT), len(ELL_ALL), len(s_data))

    return s_data, data_vec, cov, W_at_sdata


def model_multipoles_windowed(theta, emus, z_eff, s_data, W):
    """Same as common.py's model_multipoles, but projects theory onto ELL_ALL
    (needed as window input, since the window mixes ell=0,2,4 together) and
    applies the theta-cut window before returning the ELLS_FIT-only prediction."""
    h, ombh2, omch2, lnAs, b1, b2, ns = theta[:7]
    emu_xi0, emu_xi2, emu_xi4 = emus
    emu_params = [b1, b2, ombh2, omch2, h, lnAs, ns]
    _, xi0 = emu_xi0(emu_params)
    _, xi2 = emu_xi2(emu_params)
    _, xi4 = emu_xi4(emu_params)
    q_perp, q_par = ap_params(h, ombh2, omch2, ns, z_eff)
    xi_ap = ap_multipoles(s_data, ELL_ALL, xi0, xi2, xi4, q_perp, q_par)  # (3, len(s_data))
    xi_windowed = np.einsum('ots,ts->os', W, xi_ap)  # (len(ELLS_FIT), len(s_data))
    return xi_windowed.ravel()


# --- build the fixed list of tracer/zbin fits and preload their data/covariance/window ---
FIT_KEYS = [(tracer, zmin, zmax) for tracer, prog in TRACERS.items() for zmin, zmax in ZBINS[tracer]]
N_FITS = len(FIT_KEYS)

FIT_DATA = {}
for tracer, zmin, zmax in FIT_KEYS:
    z_eff = Z_EFF[(tracer, zmin, zmax)]
    nran = NRAN[tracer]
    s_data, data_vec, cov, W_at_sdata = load_dr1_tracer_arrays(tracer, zmin, zmax, nran)
    cov_inv = np.linalg.inv(cov)  # real single-realization covariance -- no mock-mean rescaling needed
    _, logdet = np.linalg.slogdet(2 * np.pi * cov)
    emus = get_emulators_cached(z_eff)
    FIT_DATA[(tracer, zmin, zmax)] = dict(z_eff=z_eff, s_data=s_data, data_vec=data_vec,
                                           cov_inv=cov_inv, logdet=logdet, emus=emus, W=W_at_sdata)
    print(f'{tracer} z{zmin}-{zmax}: DR1 data loaded, {len(s_data)} s-bins in fit range, '
          f'window shape {W_at_sdata.shape}')

# --- parameter vector layout (unchanged from the mock-fitting script) ---
# theta[0:5]  = shared cosmology: h, ombh2, omch2, lnAs, ns
# theta[5 + 2*i], theta[5 + 2*i + 1] = b1_i, b2_i for FIT_KEYS[i]
NDIM = 5 + 2 * N_FITS
NWALKERS, NSTEPS = 64, 5000
NPROCESS = 90  # TODO: pick based on cores available; useful up to roughly NWALKERS/2

theta0 = np.concatenate([
    [COSMO_FID[0], COSMO_FID[1], COSMO_FID[2], LNAS_FID, NS_FIXED],
    np.tile([1.0, 0.0], N_FITS),
])
scale = np.concatenate([
    [0.01, 0.0001, 0.002, 0.02, 0.01],
    np.tile([0.02, 0.1], N_FITS),
])


def log_prior(theta):
    h, ombh2, omch2, lnAs, ns = theta[:5]
    if not (0.55 < h < 0.91):     return -np.inf
    if not (0.08 < omch2 < 0.16): return -np.inf
    if not (2.0 < lnAs < 4.0):    return -np.inf
    lp = -0.5 * ((ombh2 - OMBH2_MEAN) / OMBH2_SIG) ** 2
    lp += -0.5 * ((ns - NS_MEAN) / NS_SIG) ** 2

    for i in range(N_FITS):
        b1, b2 = theta[5 + 2 * i], theta[5 + 2 * i + 1]
        if not (0.0 < b1 < 2.0):   return -np.inf
        if not (-5.0 < b2 < 10.0): return -np.inf
    return lp


def log_likelihood(theta):
    h, ombh2, omch2, lnAs, ns = theta[:5]
    total = 0.0
    for i, key in enumerate(FIT_KEYS):
        b1, b2 = theta[5 + 2 * i], theta[5 + 2 * i + 1]
        d = FIT_DATA[key]
        theta_i = [h, ombh2, omch2, lnAs, b1, b2, ns]
        try:
            model = model_multipoles_windowed(theta_i, d['emus'], d['z_eff'], d['s_data'], d['W'])
        except Exception:
            return -np.inf
        if not np.all(np.isfinite(model)):
            return -np.inf
        resid = model - d['data_vec']
        total += -0.5 * (resid @ d['cov_inv'] @ resid + d['logdet'])
    return total


def log_probability(theta):
    lp = log_prior(theta)
    return -np.inf if not np.isfinite(lp) else lp + log_likelihood(theta)


if __name__ == '__main__':
    pos = theta0 + scale * np.random.randn(NWALKERS, NDIM)
    with Pool(processes=NPROCESS) as pool:
        sampler = emcee.EnsembleSampler(NWALKERS, NDIM, log_probability, pool=pool)
        sampler.run_mcmc(pos, NSTEPS, progress=True)

    tau = sampler.get_autocorr_time(quiet=True)
    print(f'estimated autocorrelation times (steps): min={tau.min():.0f}, max={tau.max():.0f}')
    print(f'chain length / max(tau) = {NSTEPS / tau.max():.1f}  (want this well above ~50 for reliable results)')

    param_names = ['h', 'ombh2', 'omch2', 'logA', 'ns']
    param_labels = ['h', r'\omega_b', r'\omega_{cdm}', r'\ln(10^{10}A_s)', 'n_s']
    for tracer, zmin, zmax in FIT_KEYS:
        param_names += [f'b1_{tracer}_{zmin}_{zmax}', f'b2_{tracer}_{zmin}_{zmax}']
        param_labels += [f'b_1^{{{tracer}}}', f'b_2^{{{tracer}}}']

    root = CHAIN_DIR / 'joint_dr1_windowed'
    save_getdist_chain(root, sampler, param_names=param_names, param_labels=param_labels)
    print(f'joint DR1 fit done -> {root}.txt')