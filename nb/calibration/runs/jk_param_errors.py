#!/usr/bin/env python
"""
jk_param_errors.py -- MEASURED (jackknife-over-halos) errors on the 11 compact-model parameters, without refitting.

Method.  At the calibrated parameters theta_hat each pair contributes a score  g = d ln p_i(c; theta)/d theta  (c = its (r_p, r_los) cell,
i = its v_los bin).  Halos are split into n blocks.  The pair search yields, for every ordered pair of blocks (seed block r_s, partner block r_p),
        M[r_s, r_p, :] = sum of scores of the pairs ,     N[r_s, r_p] = number of pairs  (in the cells the fits use).
Leaving block r out removes every pair that has a member in r (BOTH roles):  dT_r = sum_p M[r,p] + sum_s M[s,r] - M[r,r].
The leave-one-out mean scores  m_(-r) = (T - dT_r)/(N - dN_r)  give the delete-one-block jackknife covariance
        Var_JK(m) = (n-1)/n sum_r (m_(-r) - mean)(m_(-r) - mean)^T ,
and the sandwich  C_theta = Fbar^-1 Var_JK(m) Fbar^-1  (Fbar = pair-independent Fisher information per pair) is the parameter covariance.
This contains the halo-level correlation that the pair-count Fisher error  sigma_pair = sqrt(diag F^-1)  ignores.  Summary:
        kappa_theta = (sigma_JK^2 / sigma_pair^2 - 1) * N_halos / N_pairs ,    sigma^2 = sigma_pair^2 [1 + kappa N_p/N_h]
(kappa = 1 reproduces the heuristic in param_shot_errors.py).  Only 11-vectors are stored: no histograms are kept.

Driver (needs the repo and the halo catalogues; re-runs the pair search once per (sim, z, bin) -- same cost as production -- so use a subset):
    python jk_param_errors.py --repo . --fit-results nb/calibration/runs/cosmo_grid_fit_results.h5 --sims 130 140 150 --zs 0.2 1.1 1.7 --bins 0 1
"""
import argparse, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import param_shot_errors as pse          # q_of_p, cell_probs, jacobian (same model code as the shot-noise script)
CHUNK = 1500; FLUSH = 3_000_000


def prepare_cell(theta, R_edges, rlos_edges, n_full, sum_v_full, v_lo, v_hi, n_v, M, n_min=50):
    """Per-cell scores G = d ln p / d(sigma, Delta, nu) (float32), Jacobians J (3x11), Fisher matrix F, and the cell -> used-cell lookup."""
    theta = np.asarray(theta, float); used = n_full >= n_min
    rp = 0.5 * (R_edges[1:] + R_edges[:-1]); rl = 0.5 * (rlos_edges[1:] + rlos_edges[:-1])
    RPg, RLg = np.meshgrid(rp, rl, indexing='ij'); RP, RL = RPg[used], RLg[used]; R3 = np.hypot(RP, RL); X = R3 - RP
    nu_c = n_full[used].astype(float); mu = sum_v_full[used] / nu_c
    dv = (v_hi - v_lo) / n_v; vc = v_lo + (np.arange(n_v) + 0.5) * dv
    J = pse.jacobian(theta, RP, X, R3, M); sig, dl, nu = pse.q_of_p(theta, RP, X, R3, M)
    G = np.empty((RP.size, n_v, 3), np.float32); F = np.zeros((11, 11))
    for c0 in range(0, RP.size, CHUNK):
        s = slice(c0, c0 + CHUNK); a_ = (sig[s], dl[s], nu[s])
        p0 = np.maximum(pse.cell_probs(*a_, mu[s], vc, dv, M), 1e-12)
        steps = [1e-4 * a_[0], 0.05 + 1e-4 * np.abs(a_[1]), 1e-4 * a_[2]]; Gc = np.empty((p0.shape[0], n_v, 3))
        for a in range(3):
            up = list(a_); dn = list(a_); up[a] = a_[a] + steps[a]; dn[a] = a_[a] - steps[a]
            Gc[:, :, a] = (pse.cell_probs(*up, mu[s], vc, dv, M) - pse.cell_probs(*dn, mu[s], vc, dv, M)) / (2 * steps[a][:, None]) / p0
        G[s] = Gc
        I = np.einsum('ci,cia,cib->cab', p0, Gc, Gc) * nu_c[s][:, None, None]
        F += np.einsum('caj,cab,cbk->jk', J[s], I, J[s])
    cell_to_used = np.full(used.shape, -1, np.int64); cell_to_used[used] = np.arange(int(used.sum()))
    return dict(used=used, J=J, G=G, F=F, theta=theta, cell_to_used=cell_to_used, nu=int(used.sum()), N_pairs=float(nu_c.sum()),
                R_edges=np.asarray(R_edges), rlos_edges=np.asarray(rlos_edges), v_lo=float(v_lo), v_hi=float(v_hi), n_v=int(n_v))


def jk_chunk(pos_all, vel_all, seed_idx, block_of, n_blocks, prep, boxsize, r_max, rlos_max, conv, rel_coord):
    """Pair search for ONE block of seeds (same selections as the production in-memory worker).  Returns (M[n_blocks, 11], N[n_blocks]):
    summed scores / counts of this block's seeded pairs, split by the partner's block."""
    tree = cKDTree(pos_all, boxsize=boxsize); nu = prep['nu']; G = prep['G']; c2u = prep['cell_to_used']
    Re, Le = prep['R_edges'], prep['rlos_edges']; nR, nl = len(Re) - 1, len(Le) - 1; inv_dv = prep['n_v'] / (prep['v_hi'] - prep['v_lo'])
    r_query = float(np.hypot(r_max, rlos_max)); T3 = np.zeros((n_blocks, nu, 3)); Nn = np.zeros(n_blocks)
    fl, wl, bl, nb_ = [], [], [], 0

    def flush():
        nonlocal fl, wl, bl, nb_
        if not fl: return
        flat = np.concatenate(fl); w = np.concatenate(wl)
        for a in range(3): T3[:, :, a] += np.bincount(flat, weights=w[:, a], minlength=n_blocks * nu).reshape(n_blocks, nu)
        fl, wl, bl, nb_ = [], [], [], 0
    for i in seed_idx:
        nb = np.asarray(tree.query_ball_point(pos_all[i], r_query, return_sorted=False))
        if nb.size == 0: continue
        rel_pos = rel_coord(pos_all[i], pos_all[nb], boxsize); R = np.hypot(rel_pos[:, 0], rel_pos[:, 1])
        keep = (R <= r_max) & (R > 0) & (np.abs(rel_pos[:, 2]) <= rlos_max)
        if not keep.any(): continue
        nbk = nb[keep]; R = R[keep]; rl = rel_pos[keep, 2]; v = conv * (vel_all[nbk, 2] - vel_all[i, 2])
        iR = np.searchsorted(Re, R, side='right') - 1; il = np.searchsorted(Le, rl, side='right') - 1
        okc = (iR >= 0) & (iR < nR) & (il >= 0) & (il < nl); u = np.full(R.shape, -1, np.int64); u[okc] = c2u[iR[okc], il[okc]]
        um = u >= 0
        if not um.any(): continue
        Nn += np.bincount(block_of[nbk[um]], minlength=n_blocks)
        iv = np.floor((v - prep['v_lo']) * inv_dv).astype(np.int64); sel = um & (iv >= 0) & (iv < prep['n_v'])
        if not sel.any(): continue
        fl.append(block_of[nbk[sel]] * nu + u[sel]); wl.append(G[u[sel], iv[sel], :]); nb_ += int(sel.sum())
        if nb_ >= FLUSH: flush()
    flush()
    return np.einsum('rca,caj->rj', T3, prep['J']), Nn


def jk_from_blocks(prep, Mfull, Nfull, n_halos):
    """Delete-one-block jackknife of the mean score and sandwich parameter errors.  Mfull[n,n,11], Nfull[n,n]."""
    n = Mfull.shape[0]; T = Mfull.sum((0, 1)); N = Nfull.sum()
    dT = Mfull.sum(1) + Mfull.sum(0) - np.einsum('rrj->rj', Mfull); dN = Nfull.sum(1) + Nfull.sum(0) - np.diag(Nfull)
    keep = (N - dN) > 0; m = (T[None, :] - dT[keep]) / (N - dN[keep])[:, None]; nk = int(keep.sum())
    d = m - m.mean(0); varm = (nk - 1.0) / nk * d.T @ d
    theta = prep['theta']; sc = np.maximum(np.abs(theta), 1e-3); Fbar = prep['F'] / prep['N_pairs']
    Fs = Fbar * sc[:, None] * sc[None, :]; Fi = np.linalg.pinv(Fs, rcond=1e-12)
    C = sc[:, None] * (Fi @ (varm * sc[:, None] * sc[None, :]) @ Fi) * sc[None, :]
    sig_jk = np.sqrt(np.maximum(np.diag(C), 0)); Fpi = np.linalg.pinv(prep['F'] * sc[:, None] * sc[None, :], rcond=1e-12)
    sig_pair = sc * np.sqrt(np.maximum(np.diag(Fpi), 0))
    with np.errstate(divide='ignore', invalid='ignore'): kappa = (sig_jk**2 / sig_pair**2 - 1.0) * n_halos / prep['N_pairs']
    return dict(sigma_jk=sig_jk, sigma_pair=sig_pair, kappa=kappa, cov=C, mean_score=T / N, varm=varm, n_blocks=nk)


def _block_task(r, pos_all, vel_all, seed_idx, block_of, n_blocks, prep, boxsize, r_max, rlos_max, conv, repo):
    sys.path.insert(0, repo)
    from compact.catalog.accumulate import relative_coordinates
    return r, jk_chunk(pos_all, vel_all, seed_idx, block_of, n_blocks, prep, boxsize, r_max, rlos_max, conv, relative_coordinates)


# ------------------------------------------------------------------ real-data driver (needs the repo + catalogues) ------
RES_SCALAR = ('sim', 'z', 'bin', 'n_halos', 'N_pairs', 'n_blocks'); RES_VEC = ('params', 'sigma_jk', 'sigma_pair', 'kappa', 'mean_score')


def _write_results(path, out):
    import h5py
    tmp = path + '.tmp'                                   # write-then-rename so a crash never leaves a half-written file
    with h5py.File(tmp, 'w') as h:
        for key in RES_SCALAR: h.create_dataset(key, data=np.array([r[key] for r in out]))
        for key in RES_VEC: h.create_dataset(key, data=np.stack([r[key] for r in out]))
        h.create_dataset('cov', data=np.stack([r['cov'] for r in out])); h.attrs['param_names'] = pse.NAMES
    os.replace(tmp, path)


def _read_results(path):
    import h5py
    with h5py.File(path, 'r') as h:
        return [dict(**{k: h[k][i] for k in RES_SCALAR + RES_VEC}, cov=h['cov'][i]) for i in range(len(h['sim']))]


def run_cells(a):
    import h5py
    sys.path.insert(0, os.path.abspath(a.repo))
    import compact.model as M
    import nb.calibration.runs.auto_runs.abacus_cosmo_core_auto as core
    from joblib import Parallel, delayed
    with h5py.File(a.fit_results, 'r') as g:
        pm = {(int(s), round(float(z), 2), int(b)): p for s, z, b, p in zip(g['sim'][:], g['z'][:], g['bin'][:], g['params'][:])}
    out = _read_results(a.out) if (a.resume and os.path.exists(a.out)) else []
    done = {(int(r['sim']), round(float(r['z']), 2), int(r['bin'])) for r in out}
    pending = lambda n, z: [k for k in a.bins if (n, round(z, 2), k) in pm and np.all(np.isfinite(pm[(n, round(z, 2), k)])) and (n, round(z, 2), k) not in done]
    todo = [(n, z) for n in a.sims for z in a.zs if pending(n, z)]
    print(f'{len(todo)} (sim, z) catalogues to process; {len(done)} cells already done', flush=True); t0 = time.time(); n_done = 0
    for n in a.sims:
        for z in a.zs:
            ks = pending(n, z)
            if not ks: continue
            par = core.read_params(n, z); m_p = core.particle_mass(par['Omega_M'])
            col, _, _ = core.read_catalogue_selected(core.halo_path(n, z), [(lo * m_p, hi * m_p) for lo, hi in core.BIN_NPART])
            mass = col['mass']; pos = np.array([col['x'], col['y'], col['z']]).T % core.box_length; vel = np.array([col['vx'], col['vy'], col['vz']]).T
            for k in ks:
                lo_np, hi_np = core.BIN_NPART[k]; sel = (mass >= lo_np * m_p) & (mass <= hi_np * m_p)
                pos_k = np.ascontiguousarray(pos[sel]); vel_k = np.ascontiguousarray(vel[sel]); nh = len(pos_k)
                with h5py.File(core.out_path(n, z, k), 'r') as f:
                    n_full = f['n'][0].astype(np.int64); sum_v = f['sum_v'][0]; at = dict(f.attrs); n_v = int(f['counts'].shape[-1])
                    Re, Le = f['R_edges'][()], f['rlos_edges'][()]
                prep = prepare_cell(pm[(n, round(z, 2), k)], Re, Le, n_full, sum_v, float(at['v_lo']), float(at['v_hi']), n_v, M, a.n_min)
                if a.regions == 'spatial':
                    ijk = np.floor(pos_k / core.subsize).astype(int); block_of = ((ijk[:, 0] * 4 + ijk[:, 1]) * 4 + ijk[:, 2]).astype(np.int64)
                    _, block_of = np.unique(block_of, return_inverse=True)
                else:
                    block_of = np.random.default_rng(0).permutation(nh) % a.n_regions
                nb_ = int(block_of.max()) + 1; groups = [np.where(block_of == r)[0] for r in range(nb_)]
                gen = Parallel(n_jobs=a.workers or os.cpu_count(), return_as='generator_unordered')(
                    delayed(_block_task)(r, pos_k, vel_k, groups[r], block_of, nb_, prep, core.box_length, core.r_max, core.RLOS_MAX, core.CONV, os.path.abspath(a.repo))
                    for r in range(nb_))
                Mfull = np.zeros((nb_, nb_, 11)); Nfull = np.zeros((nb_, nb_))
                for r, (Mr, Nr) in gen: Mfull[r] = Mr; Nfull[r] = Nr
                res = jk_from_blocks(prep, Mfull, Nfull, nh); res.update(sim=n, z=z, bin=k, n_halos=nh, N_pairs=prep['N_pairs'], params=prep['theta'])
                out.append(res); print(f'sim {n} z {z} bin {k}: {nh} halos, {nb_} blocks, sigma0 JK/pair = {res["sigma_jk"][0]/res["sigma_pair"][0]:.1f}, '
                                       f'kappa(sigma0) = {res["kappa"][0]:.2f}  [{time.time()-t0:.0f} s]', flush=True)
                del pos_k, vel_k, prep
            del pos, vel, mass
            _write_results(a.out, out); n_done += 1                                   # checkpoint after every (sim, z)
            el = time.time() - t0; print(f'  checkpoint: {n_done}/{len(todo)} catalogues, {el/3600:.2f} h elapsed, ETA {el/n_done*(len(todo)-n_done)/3600:.2f} h', flush=True)
    if out: _write_results(a.out, out)
    print('wrote', a.out)


def launch(a):
    """Run --jobs independent worker processes on interleaved subsets of the sims, then merge their part files.
    --workers is the TOTAL core budget; each job gets workers // jobs for its catalogue read and pair search."""
    import subprocess
    J = a.jobs; total = a.workers or os.cpu_count(); W = max(1, total // J); script = a.child_script or os.path.abspath(__file__)
    groups = [a.sims[k::J] for k in range(J) if a.sims[k::J]]; base, ext = os.path.splitext(a.out); procs = []
    for k, g in enumerate(groups):
        part = f'{base}.part{k}{ext}'
        cmd = [sys.executable, script, '--repo', a.repo, '--fit-results', a.fit_results, '--out', part, '--sims', *map(str, g), '--zs', *map(str, a.zs),
               '--bins', *map(str, a.bins), '--regions', a.regions, '--n-regions', str(a.n_regions), '--workers', str(W), '--n-min', str(a.n_min), '--jobs', '1', '--resume']
        env = dict(os.environ, PAIRHIST_READ_NJOBS=str(W), PAIRHIST_SEARCH_NJOBS=str(W))
        print(f'job {k}: {len(g)} sims ({g[0]}..{g[-1]}), {W} workers -> {part}  (log: {base}.part{k}.log)', flush=True)
        if a.dry_run: print('   ', ' '.join(cmd)); continue
        log = open(f'{base}.part{k}.log', 'w'); procs.append((k, part, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))
    if a.dry_run: return
    t0 = time.time()
    while any(p.poll() is None for _, _, p, _ in procs):
        time.sleep(a.poll)
        last = []
        for k, part, p, _ in procs:
            try: ln = [l for l in open(f'{base}.part{k}.log').read().splitlines() if l.strip()][-1][:90]
            except Exception: ln = ''
            last.append(f'  job {k} [{"running" if p.poll() is None else "exit %s" % p.returncode}] {ln}')
        print(f'--- {(time.time() - t0) / 3600:.2f} h elapsed ---\n' + '\n'.join(last), flush=True)
    bad = [k for k, _, p, _ in procs if p.returncode != 0]
    if bad: print(f'WARNING: jobs {bad} exited with errors (see their .log files); merging what exists. Re-run the same command with --resume to finish.')
    parts = [part for _, part, _, _ in procs if os.path.exists(part)]
    if parts: merge(argparse.Namespace(merge=parts, out=a.out))


def merge(a):
    seen, merged = set(), []
    for f in a.merge:
        for r in _read_results(f):
            key = (int(r['sim']), round(float(r['z']), 2), int(r['bin']))
            if key not in seen: seen.add(key); merged.append(r)
    _write_results(a.out, merged); print(f'merged {len(a.merge)} files -> {len(merged)} cells in {a.out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', default='.'); ap.add_argument('--fit-results'); ap.add_argument('--out', default='jk_param_errors.h5')
    ap.add_argument('--sims', type=int, nargs='+'); ap.add_argument('--zs', type=float, nargs='+'); ap.add_argument('--bins', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--resume', action='store_true', help='skip cells already in --out'); ap.add_argument('--merge', nargs='+', help='merge result files into --out and exit')
    ap.add_argument('--jobs', type=int, default=1, help='parallel worker processes over interleaved sims (--workers = total cores, split between them)')
    ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--poll', type=float, default=300.0); ap.add_argument('--child-script', default=None, help=argparse.SUPPRESS)
    ap.add_argument('--regions', choices=['spatial', 'halo'], default='spatial', help='spatial: 4x4x4 sub-boxes (64 blocks); halo: random halo blocks')
    ap.add_argument('--n-regions', type=int, default=64); ap.add_argument('--workers', type=int, default=None, help='cores to use in total (default: all)'); ap.add_argument('--n-min', type=int, default=50)
    a = ap.parse_args()
    if a.merge: merge(a)
    elif not (a.fit_results and a.sims and a.zs): ap.error('--fit-results, --sims and --zs are required unless --merge is used')
    elif a.jobs > 1: launch(a)
    else: run_cells(a)