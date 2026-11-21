#!/usr/bin/env python
"""
param_shot_errors.py -- shot-noise errors on all 11 compact-model parameters for every fitted (sim, z, mass-bin).  NO refitting.

For each pairhist_auto_mbin{k}.h5 and the calibrated 11 parameters p of that cell (from cosmo_grid_fit_results.h5) it builds the
Fisher information of the stored v_los histograms under the compact model,
        F = sum_cells  J_c^T I_c J_c ,   I_c = n_c * Fisher info of the cell's skew-t PDF w.r.t. (sigma_c, Delta_c, nu_c),
        J_c = d(sigma_c, Delta_c, nu_c)/dp  (model functions from compact/model.py, central differences),
with the cell PDF evaluated on the stored velocity bins (multinomial counts), mean fixed to the cell mean, location = mean + Delta.
The parameter errors are sqrt(diag(pinv(F))) (all other parameters marginalized).

Two columns of errors are written (absolute, in parameter units):
  sigma_pairs : Cramer-Rao bound treating every pair as an independent draw  (optimistic: pairs share halos)
  sigma_halos : sigma_pairs * sqrt(N_used / n_halos)   (heuristic: effective sample size = number of halos, not pairs)
For A_sigma = sigma_0^2/sigma_vlin^2:  sigma(A_sigma)/A_sigma = 2 sigma(sigma_0)/sigma_0.

Usage (repo root, on the server):
    python param_shot_errors.py --fit-results nb/calibration/runs/cosmo_grid_fit_results.h5 --out param_shot_errors.h5 --workers 32
    python param_shot_errors.py --paths-csv paths.csv --fit-results <h5> --out out.h5        # columns: sim,z,bin,path
"""
import argparse, csv, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np, h5py
from scipy import special
from concurrent.futures import ProcessPoolExecutor
NAMES = ['sigma0', 'A_rho0', 'A_rho_p', 'sigma_rho', 'A_delta0', 'r_delta', 'sigma_delta', 'beta_delta', 'alpha', 'nu_0', 'r_c']
CHUNK = 1500


def q_of_p(p, RP, X, R3, M):
    sig = M.sigma_mod(X, p[0], M.A_rho_mod(RP, p[1], p[2]), p[3])
    dl = M.delta_mod_alpha(X, M.A_delta_mod(RP, p[4], p[5]), p[6], p[7], p[8])
    return sig, dl, M.nu_of_r(R3, p[9], p[10])


def cell_probs(sig, dl, nu, mu, vc, dv, M):
    om, al = M.skewt_params_from_delta(dl, sig, nu); xi = mu + dl
    z = (vc[None, :] - xi[:, None]) / om[:, None]; nc = nu[:, None]
    logc = special.gammaln((nc + 1) / 2) - special.gammaln(nc / 2) - 0.5 * np.log(nc * np.pi)
    tpdf = np.exp(logc - 0.5 * (nc + 1) * np.log1p(z**2 / nc))
    tcdf = special.stdtr(nc + 1, al[:, None] * z * np.sqrt((nc + 1) / (nc + z**2)))
    return 2.0 / om[:, None] * tpdf * tcdf * dv


def fisher_q(sig, dl, nu, mu, n, vc, dv, M):
    """(n_cells, 3, 3) Fisher information on (sigma, Delta, nu) from multinomial v-bin counts."""
    p0 = cell_probs(sig, dl, nu, mu, vc, dv, M); w = 1.0 / np.maximum(p0, 1e-14)
    base = [sig, dl, nu]; steps = [1e-4 * sig, 0.05 + 1e-4 * np.abs(dl), 1e-4 * nu]; D = []
    for a in range(3):
        up = list(base); dn = list(base); up[a] = base[a] + steps[a]; dn[a] = base[a] - steps[a]
        D.append((cell_probs(*up, mu, vc, dv, M) - cell_probs(*dn, mu, vc, dv, M)) / (2 * steps[a][:, None]))
    I = np.empty((len(sig), 3, 3))
    for a in range(3):
        for b in range(a, 3): I[:, a, b] = I[:, b, a] = n * (D[a] * D[b] * w).sum(1)
    return I


def jacobian(p, RP, X, R3, M):
    J = np.empty((RP.size, 3, 11))
    for j in range(11):
        h = 1e-5 * max(abs(p[j]), 1e-2); dp = np.zeros(11); dp[j] = h
        qp = q_of_p(p + dp, RP, X, R3, M); qm = q_of_p(p - dp, RP, X, R3, M)
        for a in range(3): J[:, a, j] = (qp[a] - qm[a]) / (2 * h)
    return J


def one_bin(task):
    path, p, n_min, sim, z, mbin, repo = task
    try:
        sys.path.insert(0, repo); import compact.model as M
        with h5py.File(path, 'r') as f:
            n = f['n'][0].astype(np.int64); s1 = f['sum_v'][0]; at = dict(f.attrs); Re, Le = f['R_edges'][()], f['rlos_edges'][()]
            used = n >= n_min; counts = f['counts'][0][used].astype(np.float64)
        v_lo, v_hi, nv = float(at['v_lo']), float(at['v_hi']), counts.shape[1]
        dv = (v_hi - v_lo) / nv; vc = v_lo + (np.arange(nv) + 0.5) * dv
        rp = 0.5 * (Re[1:] + Re[:-1]); rl = 0.5 * (Le[1:] + Le[:-1]); RPg, RLg = np.meshgrid(rp, rl, indexing='ij')
        RP, RL = RPg[used], RLg[used]; R3 = np.hypot(RP, RL); X = R3 - RP
        nu_ = n[used].astype(float); mu = s1[used] / nu_
        J = jacobian(np.asarray(p, float), RP, X, R3, M); sig, dl, nu = q_of_p(np.asarray(p, float), RP, X, R3, M)
        F = np.zeros((11, 11))
        for c0 in range(0, RP.size, CHUNK):
            s = slice(c0, c0 + CHUNK)
            I = fisher_q(sig[s], dl[s], nu[s], mu[s], nu_[s], vc, dv, M)
            F += np.einsum('caj,cab,cbk->jk', J[s], I, J[s])
        sc = np.maximum(np.abs(p), 1e-3); Fs = F * sc[:, None] * sc[None, :]
        ev = np.linalg.eigvalsh(Fs); cov = np.linalg.pinv(Fs, rcond=1e-12)
        sig_pairs = sc * np.sqrt(np.maximum(np.diag(cov), 0)); N_used = float(nu_.sum()); nh = float(at.get('n_halos', np.nan))
        return dict(sim=sim, z=z, bin=mbin, n_halos=nh, N_used=N_used, n_cells_used=int(used.sum()), cond=float(ev.max() / max(ev.min(), 1e-300)),
                    params=np.asarray(p, float), sigma_pairs=sig_pairs, sigma_halos=sig_pairs * np.sqrt(N_used / nh))
    except Exception as e:
        return dict(error=f'{path}: {e!r}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', default='.'); ap.add_argument('--fit-results', required=True); ap.add_argument('--paths-csv', default=None)
    ap.add_argument('--out', default='param_shot_errors.h5'); ap.add_argument('--workers', type=int, default=os.cpu_count()); ap.add_argument('--n-min', type=int, default=50)
    a = ap.parse_args(); repo = os.path.abspath(a.repo)
    with h5py.File(a.fit_results, 'r') as g: pm = {(int(s), round(float(z), 2), int(b)): p for s, z, b, p in zip(g['sim'][:], g['z'][:], g['bin'][:], g['params'][:])}
    if a.paths_csv: tasks = [(int(r['sim']), round(float(r['z']), 2), int(r['bin']), r['path']) for r in csv.DictReader(open(a.paths_csv))]
    else:
        sys.path.insert(0, repo); import nb.calibration.runs.auto_runs.abacus_cosmo_core_auto as core
        op = lambda n, z, k: os.path.join(core.OUT_DIR.format(sim=core.sim_name(n), zdir=core.zdir(z)), f'pairhist_auto_mbin{k}.h5')
        tasks = [(n, z, k, op(n, z, k)) for (n, z, k) in sorted(pm, key=lambda t: (t[2], t[1], t[0])) if os.path.exists(op(n, z, k))]
    jobs = [(t[3], pm[t[:3]], a.n_min, t[0], t[1], t[2], repo) for t in tasks if t[:3] in pm and np.all(np.isfinite(pm[t[:3]]))]
    print(f'{len(jobs)} fitted (sim, z, bin) cells; {a.workers} workers', flush=True); t0 = time.time(); res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(one_bin, jobs, chunksize=1)):
            if 'error' in r: print('FAILED', r['error'], flush=True)
            else: res.append(r)
            if (i + 1) % 25 == 0: print(f'  {i + 1}/{len(jobs)}  {time.time() - t0:.0f} s', flush=True)
    res.sort(key=lambda r: (r['bin'], r['z'], r['sim']))
    with h5py.File(a.out, 'w') as h:
        for k in ('sim', 'z', 'bin', 'n_halos', 'N_used', 'n_cells_used', 'cond'): h.create_dataset(k, data=np.array([r[k] for r in res]))
        for k in ('params', 'sigma_pairs', 'sigma_halos'): h.create_dataset(k, data=np.stack([r[k] for r in res]))
        h.attrs['param_names'] = NAMES; h.attrs['note'] = 'Fisher/Cramer-Rao shot noise, all other parameters marginalized; absolute units'
    with open(os.path.splitext(a.out)[0] + '.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['sim', 'z', 'bin', 'n_halos', 'N_used', 'cond'] + [f'p_{n}' for n in NAMES] + [f'sig_pairs_{n}' for n in NAMES] + [f'sig_halos_{n}' for n in NAMES])
        for r in res: w.writerow([r['sim'], r['z'], r['bin'], r['n_halos'], r['N_used'], f"{r['cond']:.3g}"] + list(r['params']) + list(r['sigma_pairs']) + list(r['sigma_halos']))
    print(f'done: {len(res)} cells in {time.time() - t0:.0f} s -> {a.out}')


if __name__ == '__main__':
    main()