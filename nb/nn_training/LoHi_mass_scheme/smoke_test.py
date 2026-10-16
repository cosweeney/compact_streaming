"""
Smoke test for the 11-parameter training-data pipeline. Run this BEFORE generate_multiz_11par.py.

Uses the real compact/config.py and the real generation code path on a handful of thetas, then checks:
  1. preflight: config is the 11-parameter emulator and covers the requested redshifts (hard fail);
     rows with no fit that would silently fall back to _MEAN_PARAMS are listed (warning);
  2. generate_multiz_11par's shared-LPT path reproduces CompactStreamingModel.predict() (hard fail);
  3. every prediction is finite and passes the same bad-sample flag used in training (hard fail >5%
     flagged; warning >0%);
  4. at a fiducial-like theta: multipoles are sensible (xi_0 positive at small s, no wild amplitudes)
     and the three schemes are mutually consistent (avg_M within the low_M/high_M envelope);
  5. if the previous 9-parameter training sets exist, the same thetas (identical Sobol seeds) are
     compared against the stored xi_test -- the model update should change predictions modestly,
     not wholesale;
  6. wall time per theta, and the projected full-run time.
Plots (s^2 xi_ell for all schemes/redshifts, scheme differences, old-vs-new) go to OUT_DIR.

    python smoke_test_11par.py [--n-theta 16] [--n-jobs 16] [--redshifts 0.295 0.510 ...]
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_multiz_11par as gen
from train_11par import flag
from compact.predictions import CompactStreamingModel

OLD_BASE_DIR_TEMPLATE = '/spiffball/cosweeney/data/training/compact_multipole_training_set_free_ns_{scheme}/'
OUT_DIR = '/spiffball/cosweeney/data/smoke_11par/'
THETA_FID = np.array([1.2, 0.5, 0.02237, 0.12, 0.6736, 2.083e-9, 0.9649])   # b1, b2, wb, wc, h, As, ns
N_FULL_PER_Z = sum(n for n, _ in gen.SETS.values())

warnings.filterwarnings('ignore')


def soft_frac(pred, truth, floor=0.3):
    sig = np.std(truth, axis=0) if truth.ndim > 1 and truth.shape[0] > 1 else np.std(truth)
    return np.abs(pred - truth) / np.sqrt(truth ** 2 + (floor * sig) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-theta', type=int, default=16)
    ap.add_argument('--n-jobs', type=int, default=16)
    ap.add_argument('--redshifts', type=float, nargs='+', default=gen.REDSHIFTS)
    ap.add_argument('--out-dir', default=OUT_DIR)
    args = ap.parse_args()
    zs, schemes = args.redshifts, gen.MASS_BIN_SCHEMES
    run_dir = os.path.join(args.out_dir, time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)
    tmpl = os.path.join(run_dir, 'training_{scheme}') + '/'
    fails, warns = [], []

    # ---- 1. preflight
    cfg = gen.Config()
    try:
        for w in gen.preflight(cfg, zs, schemes):
            warns.append('preflight: ' + w)
    except Exception as e:
        print('FAIL preflight:', e)
        sys.exit(1)
    print(f'preflight ok: {len(cfg.compact_emulator)} nodes, {gen.N_COMPACT_PARAMS} parameters')

    # ---- 2/4/6. serial predictions at a fiducial-like theta: timing, sanity, scheme consistency
    fid, t_serial = {}, {}
    for z in zs:
        model = CompactStreamingModel(redshift=z, config=cfg)
        t0 = time.time()
        _, res = gen.run_one(0, THETA_FID, z, cfg, schemes)
        t_serial[z] = time.time() - t0
        for s in schemes:
            xi, err = res[s]
            if err is not None:
                fails.append(f'z={z} {s}: prediction failed: {err}')
                continue
            fid[(z, s)] = xi
            ref = np.asarray(model.predict(THETA_FID, ells=gen.ELLS, mass_bin=s))
            if not np.allclose(xi, ref, rtol=0, atol=1e-10):
                fails.append(f'z={z} {s}: shared-LPT path differs from predict() by {np.abs(xi - ref).max():.2e}')
            if xi[0, 0] <= 0 or np.abs(xi).max() > 10 or not np.isfinite(xi).all():
                fails.append(f'z={z} {s}: implausible multipoles (xi0[0]={xi[0, 0]:.3f}, max|xi|={np.abs(xi).max():.3f})')
        if all((z, s) in fid for s in schemes):
            lo, hi, av = (fid[(z, s)] for s in ('low_M', 'high_M', 'avg_M'))
            # xi is nonlinear in the compact parameters, so avg_M need not lie strictly between low_M and
            # high_M, and the two curves can cross (envelope width -> 0, e.g. xi_4 at small s); flag only if
            # it leaves the envelope by more than half the spread plus 3% of the multipole's amplitude
            tol = 0.5 * np.abs(hi - lo) + 3e-2 * np.abs(av).max(axis=-1, keepdims=True)
            excess = np.maximum(np.minimum(lo, hi) - av, av - np.maximum(lo, hi)) - tol
            if (excess > 0).any():
                warns.append(f'z={z}: avg_M xi lies well outside the low_M/high_M envelope (worst excess {excess.max():.2e})')
        print(f'z={z}: {t_serial[z]:.1f} s per theta for {len(schemes)} schemes')

    # ---- 3. the real generation path on a small test-seed set (same seed/prefix as the old test sets)
    sets = {'test': (args.n_theta, gen.SETS['test'][1])}
    t0 = time.time()
    gen.generate_all(schemes=schemes, redshifts=zs, sets=sets, base_dir_template=tmpl, n_jobs=args.n_jobs, cfg=cfg)
    print(f'generation path: {time.time() - t0:.0f} s wall for {len(zs)} z x {args.n_theta} thetas x {len(schemes)} schemes')

    report = []
    for z in zs:
        for s in schemes:
            d = os.path.join(tmpl.format(scheme=s), gen.zdir(z))
            xi = np.load(os.path.join(d, 'xi_test.npy'))
            th = np.load(os.path.join(d, 'thetas_test.npy'))
            n_nan = int(np.isnan(xi).any(axis=(1, 2)).sum())
            if n_nan:
                fails.append(f'z={z} {s}: {n_nan}/{len(xi)} predictions NaN/failed (see errors_test.json)')
                continue
            bad, _ = flag(xi)
            if bad.mean() > 0.05:
                fails.append(f'z={z} {s}: {bad.mean():.0%} of samples flagged as bad by the training filter')
            elif bad.any():
                warns.append(f'z={z} {s}: {bad.sum()}/{len(bad)} samples flagged by the training filter')

            row = dict(z=z, scheme=s, flagged=int(bad.sum()))
            old_d = os.path.join(OLD_BASE_DIR_TEMPLATE.format(scheme=s), gen.zdir(z))
            if os.path.exists(os.path.join(old_d, 'xi_test.npy')):
                th_old = np.load(os.path.join(old_d, 'thetas_test.npy'))[:args.n_theta]
                xi_old = np.load(os.path.join(old_d, 'xi_test.npy'))[:args.n_theta]
                ok_rows = np.isfinite(xi_old).all(axis=(1, 2))      # old sets store NaN for failed predictions
                if np.allclose(th_old, th) and ok_rows.any():
                    old_train = os.path.join(old_d, 'xi_train.npy')
                    if os.path.exists(old_train):
                        xtr = np.load(old_train)
                        sig = np.std(xtr[~flag(xtr)[0]], axis=0)     # same bad-sample filter as validation
                    else:
                        sig = np.std(xi_old[ok_rows], axis=0)
                    row['n_old_compared'] = int(ok_rows.sum())
                    for i, ell in enumerate(gen.ELLS):
                        fr = np.abs(xi[ok_rows, i] - xi_old[ok_rows, i]) / np.sqrt(xi_old[ok_rows, i] ** 2 + (0.3 * sig[i]) ** 2)
                        row[f'old_vs_new_xi{ell}_median'] = float(np.median(fr))
                        row[f'old_vs_new_xi{ell}_p95'] = float(np.percentile(fr, 95))
            report.append(row)

    print('\nper-(z, scheme) summary (softened fractional difference new vs old 9-parameter prediction at identical thetas):')
    for r in report:
        extra = '  '.join(f"xi{l}: med {r[f'old_vs_new_xi{l}_median']:.3f} p95 {r[f'old_vs_new_xi{l}_p95']:.3f}"
                          for l in gen.ELLS if f'old_vs_new_xi{l}_median' in r)
        n_cmp = f"[{r['n_old_compared']} thetas]  " if 'n_old_compared' in r else ''
        print(f"  z={r['z']:<6} {r['scheme']:<6} flagged {r['flagged']}   {n_cmp}{extra if extra else '(no old set to compare)'}")

    # ---- plots at the fiducial-like theta
    fig, ax = plt.subplots(3, len(zs), figsize=(3.2 * len(zs), 7.5), sharex='col', squeeze=False)
    ls = {'low_M': '--', 'high_M': ':', 'avg_M': '-'}
    for j, z in enumerate(zs):
        for i, ell in enumerate(gen.ELLS):
            for s in schemes:
                if (z, s) in fid:
                    ax[i, j].plot(cfg.r, cfg.r ** 2 * fid[(z, s)][i], ls[s], label=s)
            ax[i, j].set_title(f'z={z}  ell={ell}', fontsize=9)
        ax[2, j].set_xlabel('s [Mpc/h]')
    ax[0, 0].legend(fontsize=7)
    fig.suptitle('s^2 xi_ell at a fiducial-like theta, all schemes')
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, 'smoke_multipoles.png'), dpi=120)

    fig2, ax2 = plt.subplots(1, 3, figsize=(14, 4))
    for i, ell in enumerate(gen.ELLS):
        for z in zs:
            if all((z, s) in fid for s in schemes):
                scale = np.abs(fid[(z, 'avg_M')][i]).max()
                ax2[i].plot(cfg.r, (fid[(z, 'high_M')][i] - fid[(z, 'low_M')][i]) / scale, label=f'z={z}')
        ax2[i].set_title(f'(high_M - low_M) / max|avg_M| , ell={ell}', fontsize=9)
        ax2[i].set_xlabel('s [Mpc/h]')
    ax2[0].legend(fontsize=7)
    fig2.tight_layout()
    fig2.savefig(os.path.join(run_dir, 'smoke_scheme_differences.png'), dpi=120)

    # ---- timing projection
    per_theta_core_s = float(np.mean(list(t_serial.values())))
    core_hours = per_theta_core_s * N_FULL_PER_Z * len(gen.REDSHIFTS) / 3600
    print(f"\ntiming: {per_theta_core_s:.1f} core-s per theta (all 3 schemes, this machine, serial) -> full run "
          f"{N_FULL_PER_Z * len(gen.REDSHIFTS)} thetas = {core_hours:.0f} core-hours = "
          f"{core_hours / gen.N_JOBS:.1f} h on {gen.N_JOBS} workers (assuming similar per-core speed)")

    print(f'\nplots and data in {run_dir}')
    for w in warns:
        print('WARN', w)
    for f in fails:
        print('FAIL', f)
    print('SMOKE TEST', 'FAILED' if fails else 'PASSED' + (' with warnings' if warns else ''))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()