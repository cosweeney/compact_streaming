import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import emcee
from multiprocessing import Pool
from common import *  # COSMO_FID, LNAS_FID, NS_FIXED, NS_MEAN, NS_SIG, OMBH2_MEAN, OMBH2_SIG,
                       # TRACERS, ZBINS, Z_EFF, get_emulators_cached, save_getdist_chain, CHAIN_DIR,
                       # PARAM_NAMES_7, PARAM_LABELS_7
from fit_joint_DR1_windowed_test import load_dr1_tracer_arrays, model_multipoles_windowed, NRAN
import psutil

FIT_KEYS = [(tracer, zmin, zmax) for tracer, prog in TRACERS.items() for zmin, zmax in ZBINS[tracer]]

NWALKERS, NSTEPS = 32, 2_000   # 7 params here (vs 15 for the joint fit) -- fewer walkers needed;
                              # emcee's rule of thumb is >= 2*ndim=14, 32 gives comfortable margin
N_TRACER_WORKERS = min(len(FIT_KEYS), 5)   # one OS process per tracer/z-bin -- see note above on why
                                            # there's no inner Pool for emcee itself in this script

# NOTE: common.py's save_getdist_chain defaults to BURNIN=1000, THIN=15, tuned for the
# joint fit's 5000-step chains. Discarding 1000 of 2000 steps as burn-in is a much more
# aggressive fraction (50%) -- I've lowered these below, but treat this as a starting
# guess, not a validated choice: check convergence (autocorr time printed below, or a
# trace plot) before trusting the posterior, especially for any tracer/z-bin whose
# chain looks like it hasn't settled (QSO, given the b2 boundary issue, is the one
# I'd look at most closely).
INDIVIDUAL_DISCARD, INDIVIDUAL_THIN = 1_000, 5


def log_prior(theta):
    h, ombh2, omch2, lnAs, b1, b2, ns = theta
    if not (0.55 < h < 0.91):     return -np.inf
    if not (0.08 < omch2 < 0.16): return -np.inf
    if not (2.0 < lnAs < 4.0):    return -np.inf
    if not (0.0 < b1 < 2.0):      return -np.inf
    if not (-5.0 < b2 < 10.0):    return -np.inf
    lp = -0.5 * ((ombh2 - OMBH2_MEAN) / OMBH2_SIG) ** 2
    lp += -0.5 * ((ns - NS_MEAN) / NS_SIG) ** 2
    return lp


def fit_one_tracer(key):
    psutil.Process().nice(10)  # set explicitly per worker process, not just in the parent
    tracer, zmin, zmax = key
    z_eff = Z_EFF[key]
    nran = NRAN[tracer]
    s_data, data_vec, cov, W = load_dr1_tracer_arrays(tracer, zmin, zmax, nran)
    cov_inv = np.linalg.inv(cov)
    _, logdet = np.linalg.slogdet(2 * np.pi * cov)
    emus = get_emulators_cached(z_eff)  # loaded fresh in this process, after the fork

    def log_likelihood(theta):
        try:
            model = model_multipoles_windowed(theta, emus, z_eff, s_data, W)
        except Exception:
            return -np.inf
        if not np.all(np.isfinite(model)):
            return -np.inf
        resid = model - data_vec
        return -0.5 * (resid @ cov_inv @ resid + logdet)

    def log_probability(theta):
        lp = log_prior(theta)
        return -np.inf if not np.isfinite(lp) else lp + log_likelihood(theta)

    ndim = 7
    theta0 = np.array([COSMO_FID[0], COSMO_FID[1], COSMO_FID[2], LNAS_FID, 1.0, 0.0, NS_FIXED])
    scale = np.array([0.01, 0.0001, 0.002, 0.02, 0.02, 0.1, 0.01])
    pos = theta0 + scale * np.random.randn(NWALKERS, ndim)

    # No pool= here -- this sampler runs single-threaded within its own OS process;
    # the outer Pool.map below is what provides parallelism, across tracers.
    sampler = emcee.EnsembleSampler(NWALKERS, ndim, log_probability)
    # progress=False: five simultaneous tqdm bars from separate processes interleave
    # into garbage on a shared terminal, so this is disabled rather than left on.
    sampler.run_mcmc(pos, NSTEPS, progress=False)

    try:
        tau = sampler.get_autocorr_time(quiet=True)
        tau_msg = f'autocorr min={tau.min():.0f}, max={tau.max():.0f}, N/max(tau)={NSTEPS / tau.max():.1f}'
    except Exception as e:
        tau_msg = f'autocorr estimate failed (chain may be too short/unconverged): {e}'

    root = CHAIN_DIR / f'individual_dr1_windowed_{tracer}_{zmin}_{zmax}'
    save_getdist_chain(root, sampler, discard=INDIVIDUAL_DISCARD, thin=INDIVIDUAL_THIN,
                        param_names=PARAM_NAMES_7, param_labels=PARAM_LABELS_7)
    return key, tau_msg, str(root)


if __name__ == '__main__':
    with Pool(processes=N_TRACER_WORKERS) as pool:
        results = pool.map(fit_one_tracer, FIT_KEYS)

    for key, tau_msg, root in results:
        print(f'{key}: {tau_msg} -> {root}.txt')