import os
os.environ.setdefault('OMP_NUM_THREADS', '1')  # avoid CAMB oversubscription -- we parallelize across chains instead

import numpy as np
import emcee
from multiprocessing import Pool
from common import *
from tqdm import tqdm
import psutil
psutil.Process().nice(10)

D_MIN, D_MAX = 0.0, 100.0   # TODO: confirm range against data_vec magnitudes
NDIM, NWALKERS, NSTEPS = 9, 32, 2_000
NPROCESS = 5   # TODO: pick a process count -- few chains here (one per tracer/zbin), cap at that count
theta0 = np.array([COSMO_FID[0], COSMO_FID[1], COSMO_FID[2], LNAS_FID, 1.0, 0.0, NS_FIXED, 5.0, 5.0])
scale  = np.array([0.01, 0.0001, 0.002, 0.02, 0.02, 0.1, 0.01, 1.0, 1.0])


def log_prior9(theta):
    lp = log_prior7(theta[:7])
    if not np.isfinite(lp):
        return -np.inf
    d0, d2 = theta[-2], theta[-1]
    if not (D_MIN < d0 < D_MAX): return -np.inf
    if not (D_MIN < d2 < D_MAX): return -np.inf
    return lp
 
 
def make_log_prob(emus, z_eff, s_data, data_vec, cov_mean, n_per_ell):
    def log_likelihood(theta):
        d0, d2 = theta[-2], theta[-1]
        extra_var = np.concatenate([np.full(n_per_ell, d0**2), np.full(n_per_ell, d2**2)])
        cov_tot = cov_mean + np.diag(extra_var)
        try:
            model = model_multipoles(theta, emus, z_eff, s_data)
        except Exception:
            return -np.inf
        if not np.all(np.isfinite(model)):
            return -np.inf
        resid = model - data_vec
        sign, logdet = np.linalg.slogdet(2 * np.pi * cov_tot)
        if sign <= 0:
            return -np.inf
        chi2 = resid @ np.linalg.solve(cov_tot, resid)
        return -0.5 * (chi2 + logdet)
    def log_probability(theta):
        lp = log_prior9(theta)
        return -np.inf if not np.isfinite(lp) else lp + log_likelihood(theta)
    return log_probability
 
 
def run_one(task):
    tracer, zmin, zmax, z_eff, s_data, data_vec, cov_mean, n_per_ell = task
    emus = get_emulators_cached(z_eff)
    log_probability = make_log_prob(emus, z_eff, s_data, data_vec, cov_mean, n_per_ell)
 
    pos = theta0 + scale * np.random.randn(NWALKERS, NDIM)
    sampler = emcee.EnsembleSampler(NWALKERS, NDIM, log_probability)
    sampler.run_mcmc(pos, NSTEPS, progress=False)
 
    root = CHAIN_DIR / f'{tracer}_z{zmin}-{zmax}_mean'
    save_getdist_chain(root, sampler, param_names=PARAM_NAMES_9, param_labels=PARAM_LABELS_9)
    return f'{tracer} z{zmin}-{zmax} done -> {root}.txt'
 
 
if __name__ == '__main__':
    tasks = []
    for tracer, prog in TRACERS.items():
        for zmin, zmax in ZBINS[tracer]:
            z_eff = Z_EFF[(tracer, zmin, zmax)]
            s_data, poles, mask, cov = load_tracer_arrays(tracer, prog, zmin, zmax)
            nmock_avail = poles.shape[0]
            n_per_ell = len(s_data)
            print(f'{tracer} z{zmin}-{zmax}: {nmock_avail} mocks used for cov scaling')
 
            data_vec = data_vec_from(poles, mask, mock_idx=None)
            cov_mean = cov / nmock_avail
            tasks.append((tracer, zmin, zmax, z_eff, s_data, data_vec, cov_mean, n_per_ell))
 
    with Pool(processes=NPROCESS) as pool:
        for msg in tqdm(pool.imap_unordered(run_one, tasks), total=len(tasks)):
            print(msg)
 