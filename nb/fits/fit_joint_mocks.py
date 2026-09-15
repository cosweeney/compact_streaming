import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import emcee
from multiprocessing import Pool
from common import *
import psutil
psutil.Process().nice(10)

# --- build the fixed list of tracer/zbin fits and preload their data/covariance ---
FIT_KEYS = [(tracer, zmin, zmax) for tracer, prog in TRACERS.items() for zmin, zmax in ZBINS[tracer]]
N_FITS = len(FIT_KEYS)

FIT_DATA = {}
for tracer, zmin, zmax in FIT_KEYS:
    prog = TRACERS[tracer]
    z_eff = Z_EFF[(tracer, zmin, zmax)]
    s_data, poles, mask, cov = load_tracer_arrays(tracer, prog, zmin, zmax)
    data_vec = data_vec_from(poles, mask, mock_idx=None)   # mean signal
    cov_inv = np.linalg.inv(cov)                            # unscaled -- no /nmock_avail
    _, logdet = np.linalg.slogdet(2 * np.pi * cov)
    emus = get_emulators_cached(z_eff)
    FIT_DATA[(tracer, zmin, zmax)] = dict(z_eff=z_eff, s_data=s_data, data_vec=data_vec,
                                           cov_inv=cov_inv, logdet=logdet, emus=emus)
    print(f'{tracer} z{zmin}-{zmax}: mean of {poles.shape[0]} mocks, unscaled covariance loaded')

# --- parameter vector layout ---
# theta[0:5]  = shared cosmology: h, ombh2, omch2, lnAs, ns
# theta[5 + 2*i], theta[5 + 2*i + 1] = b1_i, b2_i for FIT_KEYS[i]
NDIM = 5 + 2 * N_FITS
NWALKERS, NSTEPS = 64, 5000   # more walkers than usual given the higher dimensionality
NPROCESS = 90   # TODO: pick based on cores available; useful up to roughly NWALKERS/2

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
            model = model_multipoles(theta_i, d['emus'], d['z_eff'], d['s_data'])
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

    root = CHAIN_DIR / 'joint_mean_unscaledcov'
    save_getdist_chain(root, sampler, param_names=param_names, param_labels=param_labels)
    print(f'joint fit done -> {root}.txt')