import os
os.environ.setdefault('OMP_NUM_THREADS', '1')  # avoid CAMB oversubscription -- we parallelize across chains instead

import numpy as np
import emcee
from multiprocessing import Pool
from common import *


import psutil
psutil.Process().nice(10)

NDIM, NWALKERS, NSTEPS = 7, 32, 2_000
NPROCESS = 90   # TODO: pick a process count, e.g. os.cpu_count()
theta0 = np.array([COSMO_FID[0], COSMO_FID[1], COSMO_FID[2], LNAS_FID, 1.0, 0.0, NS_FIXED])
scale  = np.array([0.01, 0.0001, 0.002, 0.02, 0.02, 0.1, 0.01])


def make_log_prob(emus, z_eff, s_data, data_vec, cov_inv, logdet):
    def log_likelihood(theta):
        try:
            model = model_multipoles(theta, emus, z_eff, s_data)
        except Exception:
            return -np.inf
        if not np.all(np.isfinite(model)):
            return -np.inf
        resid = model - data_vec
        return -0.5 * (resid @ cov_inv @ resid + logdet)
    def log_probability(theta):
        lp = log_prior7(theta)
        return -np.inf if not np.isfinite(lp) else lp + log_likelihood(theta)
    return log_probability
 
 
def run_one(task):
    tracer, zmin, zmax, mock_idx, z_eff, s_data, data_vec, cov_inv, logdet = task
    emus = get_emulators_cached(z_eff)
    log_probability = make_log_prob(emus, z_eff, s_data, data_vec, cov_inv, logdet)
 
    pos = theta0 + scale * np.random.randn(NWALKERS, NDIM)
    sampler = emcee.EnsembleSampler(NWALKERS, NDIM, log_probability)
    sampler.run_mcmc(pos, NSTEPS, progress=False)
 
    root = CHAIN_DIR / f'{tracer}_z{zmin}-{zmax}_mock{mock_idx}'
    save_getdist_chain(root, sampler, param_names=PARAM_NAMES_7, param_labels=PARAM_LABELS_7)
    return f'{tracer} z{zmin}-{zmax} mock{mock_idx} done -> {root}.txt'
 
 
if __name__ == '__main__':
    tasks = []
    for tracer, prog in TRACERS.items():
        for zmin, zmax in ZBINS[tracer]:
            z_eff = Z_EFF[(tracer, zmin, zmax)]
            s_data, poles, mask, cov = load_tracer_arrays(tracer, prog, zmin, zmax)
            nmock_avail = poles.shape[0]
            print(f'{tracer} z{zmin}-{zmax}: {nmock_avail} mocks available')
 
            cov_inv = np.linalg.inv(cov)
            _, logdet = np.linalg.slogdet(2 * np.pi * cov)
 
            for mock_idx in range(nmock_avail):
                data_vec = data_vec_from(poles, mask, mock_idx)
                tasks.append((tracer, zmin, zmax, mock_idx, z_eff, s_data, data_vec, cov_inv, logdet))
 
    with Pool(processes=NPROCESS) as pool:
        for msg in pool.imap_unordered(run_one, tasks):
            print(msg)