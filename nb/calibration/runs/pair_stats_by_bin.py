#!/usr/bin/env python
"""
pair_stats_by_bin.py -- shot-noise statistics for every (sim, z, mass-bin) pair histogram that feeds the calibration.

No refitting: for each pairhist_auto_mbin{k}.h5 this reads the stored per-cell moments and v-histogram and returns

  per bin :  n_halos (seed halos), total pairs, pairs in the cells the fits use (n >= N_MIN), number of used cells,
             off-grid pair fraction, median / min / max pairs per used cell,
             I_lnsigma  = sum_cells 1 / Var[ln sigma_cell]  (Fisher information on a global velocity-width scale),
             rel_err_lnsigma = 1/sqrt(I_lnsigma)   (fractional shot-noise error on a global sigma amplitude,
                                                    e.g. sigma_0; multiply by 2 for A_sigma = sigma_0^2 / sigma_vlin^2),
             se_mean_global = 1/sqrt(sum_cells n / sigma_cell^2)  [km/s] (shot-noise error on a global velocity offset)
  per cell (--store-cells):  n, SE[ln sigma_cell], SE[mean_cell], kurtosis   -> for your own error propagation.

SE[ln sigma] = sqrt((kappa - 1) / (4 n)) with kappa the kurtosis of the cell's v histogram (Gaussian: 1/sqrt(2n)).
The histogram covers [v_lo, v_hi] only, so kappa (and hence the SE) is slightly underestimated for very heavy tails; the
off-grid fraction is reported so you can see how much that matters.  Errors are pure shot noise: the grid sims are a single
phase (ph000), so there is no phase-to-phase information here.

Usage (repo root, e.g. on spiff):
    python pair_stats_by_bin.py --fit-results nb/calibration/runs/cosmo_grid_fit_results.h5 --out pair_stats.h5 --workers 32
    python pair_stats_by_bin.py --paths-csv paths.csv --out pair_stats.h5        # columns: sim,z,bin,path  (no repo import)
"""
import argparse, csv, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np, h5py
from concurrent.futures import ProcessPoolExecutor


def bin_stats(args):
    path, n_min, store_cells, sim, z, mbin = args           # keys come from the task list, not from file attributes
    try:
        with h5py.File(path, 'r') as f:
            n = f['n'][0].astype(np.int64)                       # (n_R, n_l); one mass bin per file
            s1, s2 = f['sum_v'][0], f['sum_v2'][0]
            at = dict(f.attrs)
            v_lo, v_hi = float(at['v_lo']), float(at['v_hi'])
            used = n >= n_min
            out = dict(sim=int(sim), z=float(z), bin=int(mbin),
                       n_halos=int(at.get('n_halos', -1)), N_tot=int(n.sum()), n_cells=int(n.size), n_cells_occupied=int((n > 0).sum()),
                       n_cells_used=int(used.sum()), N_used=int(n[used].sum()))
            if not used.any():
                out.update(frac_off_grid=np.nan, n_med=0, n_min=0, n_max=0, med_sigma=np.nan, I_lnsigma=0.0,
                           rel_err_lnsigma=np.nan, se_mean_global=np.nan); return out, None
            counts = f['counts'][0][used].astype(np.float64)       # (n_used, n_v): only the cells the fits use
        nu = n[used]; nv = counts.shape[1]
        mu = s1[used] / nu; sd = np.sqrt(np.maximum(s2[used] / nu - mu**2, 1e-12))
        vc = v_lo + (np.arange(nv) + 0.5) * (v_hi - v_lo) / nv
        cs = counts.sum(1); m = counts @ vc / cs; d = vc[None, :] - m[:, None]
        m2 = (counts * d**2).sum(1) / cs; m4 = (counts * d**4).sum(1) / cs
        kurt = m4 / np.maximum(m2**2, 1e-12)
        se_lns = np.sqrt(np.maximum(kurt - 1.0, 1e-3) / (4.0 * nu))
        se_mean = sd / np.sqrt(nu)
        I = float((1.0 / se_lns**2).sum())
        out.update(frac_off_grid=float((nu - cs).sum() / nu.sum()), n_med=float(np.median(nu)), n_min=int(nu.min()), n_max=int(nu.max()),
                   med_sigma=float(np.median(sd)), I_lnsigma=I, rel_err_lnsigma=float(I**-0.5),
                   se_mean_global=float((nu / sd**2).sum() ** -0.5))
        cells = None
        if store_cells:
            full = lambda a, dt: np.full(n.shape, np.nan, dtype=dt)
            c_n, c_se, c_sm, c_k = n.astype(np.int32), full(0, np.float32), full(0, np.float32), full(0, np.float32)
            c_se[used], c_sm[used], c_k[used] = se_lns, se_mean, kurt
            cells = (c_n, c_se, c_sm, c_k)
        return out, cells
    except Exception as e:                                             # report, keep going
        return dict(error=f'{path}: {e!r}'), None


def build_tasks(a):
    if a.paths_csv:
        rows = list(csv.DictReader(open(a.paths_csv)))
        tasks = [dict(sim=int(r['sim']), z=round(float(r['z']), 2), bin=int(r['bin']), path=r['path']) for r in rows]
    else:
        sys.path.insert(0, a.repo)
        import nb.calibration.runs.auto_runs.abacus_cosmo_core_auto as core      # same paths the builder uses
        zs = [0.2, 0.3, 0.4, 0.5, 0.8, 1.1, 1.4, 1.7]
        op = lambda n, z, k: os.path.join(core.OUT_DIR.format(sim=core.sim_name(n), zdir=core.zdir(z)), f'pairhist_auto_mbin{k}.h5')
        tasks = [dict(sim=n, z=z, bin=k, path=op(n, z, k)) for n in core.SIM_NUMS for z in zs for k in range(len(core.BIN_NPART))
                 if os.path.exists(op(n, z, k))]
    if a.fit_results:                                                  # only the cells that actually have a fit
        with h5py.File(a.fit_results, 'r') as g:
            keep = set(zip(map(int, g['sim'][:]), np.round(g['z'][:], 2), map(int, g['bin'][:])))
        tasks = [t for t in tasks if (t['sim'], round(t['z'], 2), t['bin']) in keep]
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', default='.'); ap.add_argument('--fit-results', default=None); ap.add_argument('--paths-csv', default=None)
    ap.add_argument('--out', default='pair_stats.h5'); ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--n-min', type=int, default=50, help='pairs per (r_perp, r_los) cell; must match the fits (N_MIN)')
    ap.add_argument('--store-cells', action='store_true', help='also save per-cell n, SE[ln sigma], SE[mean], kurtosis')
    a = ap.parse_args()
    tasks = build_tasks(a); print(f'{len(tasks)} (sim, z, bin) histograms; {a.workers} workers', flush=True)
    t0 = time.time(); results = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (r, cells) in enumerate(ex.map(bin_stats, [(t['path'], a.n_min, a.store_cells, t['sim'], t['z'], t['bin']) for t in tasks], chunksize=2)):
            if 'error' in r: print('FAILED', r['error'], flush=True); continue
            results.append((r, cells))
            if (i + 1) % 50 == 0: print(f'  {i + 1}/{len(tasks)}  {time.time() - t0:.0f} s', flush=True)
    results.sort(key=lambda rc: (rc[0]['bin'], rc[0]['z'], rc[0]['sim']))
    keys = ['sim', 'z', 'bin', 'n_halos', 'N_tot', 'N_used', 'n_cells', 'n_cells_occupied', 'n_cells_used', 'frac_off_grid', 'n_min', 'n_med', 'n_max',
            'med_sigma', 'I_lnsigma', 'rel_err_lnsigma', 'se_mean_global']
    with h5py.File(a.out, 'w') as h:
        for k in keys: h.create_dataset(k, data=np.array([r[k] for r, _ in results]))
        h.attrs.update(n_min=a.n_min, note='shot-noise only; single phase')
        if a.store_cells:
            for name, j in (('cell_n', 0), ('cell_se_lnsigma', 1), ('cell_se_mean', 2), ('cell_kurtosis', 3)):
                h.create_dataset(name, data=np.stack([c[j] for _, c in results]), compression='gzip')
    with open(os.path.splitext(a.out)[0] + '.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(keys)
        for r, _ in results: w.writerow([r[k] for k in keys])
    print(f'done: {len(results)} bins in {time.time() - t0:.0f} s -> {a.out}')
    print('median over sims:   z   bin   N_used   cells_used   rel_err_lnsigma')
    for k in sorted({r['bin'] for r, _ in results}):
        for z in sorted({r['z'] for r, _ in results}):
            rr = [r for r, _ in results if r['bin'] == k and r['z'] == z]
            if rr: print(f"{z:20.1f}  {k:4d}  {np.median([r['N_used'] for r in rr]):10.3g}  {np.median([r['n_cells_used'] for r in rr]):8.0f}  {np.median([r['rel_err_lnsigma'] for r in rr]):10.2e}  (n={len(rr)})")


if __name__ == '__main__':
    main()