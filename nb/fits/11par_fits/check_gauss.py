"""
Verify that the Gaussian-model fits really used Gaussian-PVD predictions (not skew-t ones).
Run from the fits folder in the compact environment:  python check_gauss_pipeline.py

 1. Training data: xi_test of the Gaussian (gauss_avg_M) and skew-t (11par_avg_M) training sets at identical thetas.
    Expected: different -- small in xi_0, larger in xi_2 and xi_4. 'IDENTICAL' means the Gaussian data came from the skew-t.
 2. Emulators: each emulator vs each data set. A Gaussian emulator trained on Gaussian data should match the Gaussian data better
    than the skew-t data (and vice versa), most clearly in xi_2 and xi_4.
 3. Fits: the two baseline joint fits (mean mocks, DR1) are evaluated under BOTH models. If the models differ, the log-posteriors and
    data-vector predictions differ; the printed chi2 of (model_gauss - model_compact) says by how much, in data-covariance units.
 4. Provenance: the model name recorded in each chain's status file.
"""
import os

import h5py
import numpy as np

import fit_lib_11par as fl

TRAIN = {'compact': '/spiffball/cosweeney/data/training/compact_multipole_training_set_11par_avg_M/',
         'gauss':   '/spiffball/cosweeney/data/training/compact_multipole_training_set_gauss_avg_M/'}
REDSHIFTS = [0.295, 0.510, 0.706, 0.919, 1.317, 1.491]
N_TEST = 100
FITS = {'mean mocks': ('compact_mean_joint_smin20', 'gauss_mean_joint_smin20'),
        'DR1':        ('compact_dr1_joint_smin20',  'gauss_dr1_joint_smin20')}


def soft(a, b, sig):
    return np.abs(a - b) / np.sqrt(b ** 2 + (0.3 * sig) ** 2)      # softened fractional difference (a vs reference b)


# ------------------------------------------------------------------ 1 and 2: training data and emulators
print('1/2. TRAINING DATA AND EMULATORS (median / 95th-percentile softened fractional difference, per multipole)')
for z in REDSHIFTS:
    dirs = {m: os.path.join(TRAIN[m], fl.zdir(z)) for m in TRAIN}
    try:
        th = {m: np.load(os.path.join(dirs[m], 'thetas_test.npy'))[:N_TEST] for m in TRAIN}
        xi = {m: np.load(os.path.join(dirs[m], 'xi_test.npy'))[:N_TEST] for m in TRAIN}
    except FileNotFoundError as e:
        print(f'z={z}: missing file {e.filename}')
        continue
    if not np.allclose(th['compact'], th['gauss']):
        print(f'z={z}: thetas_test differ between the two sets -- cannot compare point by point')
        continue
    ok = np.isfinite(xi['compact']).all(axis=(1, 2)) & np.isfinite(xi['gauss']).all(axis=(1, 2))
    th_ok, xc, xg = th['compact'][ok], xi['compact'][ok], xi['gauss'][ok]
    sig = np.std(xc, axis=0)
    maxdiff = np.abs(xg - xc).max()
    d = soft(xg, xc, sig)
    print(f'z={z}: training data  gauss vs skew-t: max |diff| = {maxdiff:.2e}' + ('   <-- IDENTICAL (suspicious)' if maxdiff < 1e-12 else '') + '   ' +
          '  '.join(f'xi{l}: {np.median(d[:, i]):.3f}/{np.percentile(d[:, i], 95):.3f}' for i, l in enumerate((0, 2, 4))))
    # emulators at the same thetas (emulator inputs: [b1, b2, ombh2, omch2, h, ln(1e10 As), ns])
    p = th_ok.copy()
    p[:, 5] = np.log(1e10 * th_ok[:, 5])
    pred = {}
    for m in ('compact', 'gauss'):
        emus = fl.get_emus(m, z)
        pred[m] = np.array([[e(list(row))[1] for e in emus] for row in p])
    emu_diff = np.abs(pred['gauss'] - pred['compact']).max()
    print(f'        emulators      gauss vs skew-t: max |diff| = {emu_diff:.2e}' + ('   <-- IDENTICAL (suspicious)' if emu_diff < 1e-12 else ''))
    for i, l in enumerate((0, 2, 4)):
        a = np.median(soft(pred['gauss'][:, i], xg[:, i], sig[i]))      # gauss emulator vs gauss data
        b = np.median(soft(pred['gauss'][:, i], xc[:, i], sig[i]))      # gauss emulator vs skew-t data
        c = np.median(soft(pred['compact'][:, i], xc[:, i], sig[i]))    # skew-t emulator vs skew-t data
        dd = np.median(soft(pred['compact'][:, i], xg[:, i], sig[i]))   # skew-t emulator vs gauss data
        verdict = 'ok' if (a < b and c < dd) else 'CHECK'
        print(f'        xi{l}: gauss-emu: vs gauss data {a:.3f}, vs skew-t data {b:.3f} | skew-t-emu: vs skew-t data {c:.3f}, vs gauss data {dd:.3f}   [{verdict}]')


# ------------------------------------------------------------------ 3 and 4: the fits
def best_sample(spec):
    path = fl.CHAIN_ROOT / spec['model'] / f"{spec['name']}.h5"
    with h5py.File(path, 'r') as f:
        it = int(f['mcmc'].attrs['iteration'])
        lp = f['mcmc/log_prob'][it // 2:it]                       # second half of the chain
        t, w = np.unravel_index(np.nanargmax(np.where(np.isfinite(lp), lp, -np.inf)), lp.shape)
        return f['mcmc/chain'][it // 2 + t, w, :]


def model_vectors(problem, theta):
    h, ombh2, omch2, lnAs, ns, biases = problem.unpack(theta)
    aps = fl.ap_multi(h, ombh2, omch2, ns, problem.z_effs)
    return [fl.model_one(d['emus'], [b1, b2, ombh2, omch2, h, lnAs, ns], d['s_data'], d['W'], qp, ql)
            for d, (b1, b2), (qp, ql) in zip(problem.data, biases, aps)]


print('\n3/4. FITS (baseline joint fits evaluated under both models)')
specs = {s['name']: s for s in fl.build_specs()}
for case, (name_c, name_g) in FITS.items():
    sc, sg = specs[name_c], specs[name_g]
    pc, pg = fl.FitProblem(sc), fl.FitProblem(sg)
    tc, tg = best_sample(sc), best_sample(sg)
    print(f'{case}: model recorded for chains: {sc["model"]} -> {fl.MODELS[sc["model"]][0]}{"*" + fl.MODELS[sc["model"]][1]}, '
          f'{sg["model"]} -> {fl.MODELS[sg["model"]][0]}{"*" + fl.MODELS[sg["model"]][1]}')
    print(f'   ln P  at compact MAP: compact model {pc(tc):.2f}, gauss model {pg(tc):.2f}   |   at gauss MAP: compact model {pc(tg):.2f}, gauss model {pg(tg):.2f}')
    mc, mg = model_vectors(pc, tc), model_vectors(pg, tc)           # both models at the SAME parameters
    rows = []
    for d, a, b in zip(pc.data, mc, mg):
        delta = b - a
        rows.append((float(delta @ d['cov_inv'] @ delta), float(np.max(np.abs(delta) / np.sqrt(np.diag(np.linalg.inv(d['cov_inv'])))))))
    tot = sum(r[0] for r in rows)
    print('   chi2 of (model_gauss - model_compact) at the compact MAP, per tracer: ' + ', '.join(f'{c:.2f}' for c, _ in rows) +
          f'; total {tot:.2f}; largest single-bin |diff|/sigma: {max(r[1] for r in rows):.2f}' + ('   <-- IDENTICAL (suspicious)' if tot < 1e-8 else ''))