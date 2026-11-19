"""
Smoke test for the 11par_v3 training-data pipeline. Run this BEFORE generate_multiz_11par_v3.py.
Derived from smoke_test.py; uses config_11par_v3.py (from calibrate_A_sigma_11par_v3.py) and the real generation path.

Checks (hard fail unless noted):
  1. preflight for both variants (11 parameters, redshift nodes, s grid); rows with no fit are listed (warning);
  2. generate_multiz_11par_v3.run_one (shared LPT, both variants) reproduces CompactStreamingModel.predict() for each variant;
  3. finite, plausible multipoles at a fiducial-like theta;
  4. the real generate_all path on a few thetas writes finite sets that pass the training bad-sample filter
     (hard fail > 5% flagged, warning > 0%);
  5. (informational) size of the change: softened fractional difference of the new multipoles vs the BASELINE 11par avg_M
     model, and slopes-on vs slopes-zero, at the same thetas; plus a plot of s^2 xi_ell at the fiducial theta;
  6. wall time per theta and the projected full-run time.

    python smoke_test_11par_v3.py [--n-theta 16] [--n-jobs 16] [--redshifts 0.295 0.510 ...]
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import generate_multiz_11par_v3 as gen
from train_11par_v3 import flag
from compact.predictions import CompactStreamingModel
from compact.config import Config as BaseConfig

OUT_DIR = '/spiffball/cosweeney/data/smoke_11par_v3/'
THETA_FID = np.array([1.2, 0.5, 0.02237, 0.12, 0.6736, 2.083e-9, 0.9649])   # b1, b2, wb, wc, h, As, ns
N_FULL_PER_Z = sum(n for n, _ in gen.SETS.values())

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 200, 'savefig.dpi': 300})


def soft_frac(pred, truth, sig, floor=0.3):
    return np.abs(pred - truth) / np.sqrt(truth ** 2 + (floor * sig) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-theta', type=int, default=16)
    ap.add_argument('--n-jobs', type=int, default=16)
    ap.add_argument('--redshifts', type=float, nargs='+', default=gen.REDSHIFTS)
    ap.add_argument('--out-dir', default=OUT_DIR)
    args = ap.parse_args()
    zs, schemes = args.redshifts, gen.MASS_BIN_SCHEMES
    cfgs = dict(gen.VARIANTS)
    outputs = gen.make_outputs(cfgs, schemes)
    tags = list(outputs)
    run_dir = os.path.join(args.out_dir, time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)
    tmpl = os.path.join(run_dir, 'training_{tag}') + '/'
    fails, warns = [], []

    # ---- 1. preflight
    for v, cfg in cfgs.items():
        try:
            for w in gen.preflight(cfg, zs, schemes):
                warns.append(f'preflight [{v}]: {w}')
        except Exception as e:
            print(f'FAIL preflight [{v}]:', e)
            sys.exit(1)
        print(f'preflight ok [{v}]: {len(cfg.compact_emulator)} nodes, {gen.N_COMPACT_PARAMS} parameters, '
              f'sigma0_from_linear = {cfg.sigma0_from_linear}, exact_growth = {cfg.exact_growth}')

    # ---- 2/3/6. serial predictions at a fiducial-like theta: plumbing, sanity, timing
    fid, base, t_serial = {}, {}, {}
    for z in zs:
        t0 = time.time()
        _, res = gen.run_one(0, THETA_FID, z, cfgs, outputs, tags)
        t_serial[z] = time.time() - t0
        for t in tags:
            xi, err = res[t]
            if err is not None:
                fails.append(f'z={z} {t}: prediction failed: {err}')
                continue
            fid[(z, t)] = xi
            variant, scheme = outputs[t]
            ref = np.asarray(CompactStreamingModel(redshift=z, config=cfgs[variant]).predict(THETA_FID, ells=gen.ELLS, mass_bin=scheme))
            if not np.allclose(xi, ref, rtol=0, atol=1e-10):
                fails.append(f'z={z} {t}: shared-LPT path differs from predict() by {np.abs(xi - ref).max():.2e}')
            if xi[0, 0] <= 0 or np.abs(xi).max() > 10 or not np.isfinite(xi).all():
                fails.append(f'z={z} {t}: implausible multipoles (xi0[0]={xi[0, 0]:.3f}, max|xi|={np.abs(xi).max():.3f})')
        base[z] = np.asarray(CompactStreamingModel(redshift=z, config=BaseConfig).predict(THETA_FID, ells=gen.ELLS, mass_bin='avg_M'))
        print(f'z={z}: {t_serial[z]:.1f} s per theta for {len(tags)} outputs')

    # ---- 4. the real generation path on a small test-seed set
    t0 = time.time()
    gen.generate_all(variants=cfgs, schemes=schemes, redshifts=zs, sets={'test': (args.n_theta, gen.SETS['test'][1])},
                     base_dir_template=tmpl, n_jobs=args.n_jobs)
    print(f'generation path: {time.time() - t0:.0f} s wall for {len(zs)} z x {args.n_theta} thetas x {len(tags)} outputs')

    print('\nper-z softened fractional difference at identical thetas (median / 95th percentile over thetas):')
    for z in zs:
        d = {t: os.path.join(tmpl.format(tag=t), gen.zdir(z)) for t in tags}
        xi = {t: np.load(os.path.join(d[t], 'xi_test.npy')) for t in tags}
        th = np.load(os.path.join(d[tags[0]], 'thetas_test.npy'))
        for t in tags:
            n_nan = int(np.isnan(xi[t]).any(axis=(1, 2)).sum())
            if n_nan:
                fails.append(f'z={z} {t}: {n_nan}/{len(xi[t])} predictions NaN/failed (see errors_test.json)')
                continue
            bad, _ = flag(xi[t])
            if bad.mean() > 0.05:
                fails.append(f'z={z} {t}: {bad.mean():.0%} of samples flagged as bad by the training filter')
            elif bad.any():
                warns.append(f'z={z} {t}: {bad.sum()}/{len(bad)} samples flagged by the training filter')
        # baseline avg_M multipoles at the same thetas (baseline config, same code path)
        base_xi = np.array([np.asarray(CompactStreamingModel(redshift=z, config=BaseConfig).predict(th_i, ells=gen.ELLS, mass_bin='avg_M'))
                            for th_i in th])
        sig = np.std(base_xi, axis=0)
        a, b = (xi[t] for t in tags[:2])
        for i, ell in enumerate(gen.ELLS):
            fb = [soft_frac(xi[t][:, i], base_xi[:, i], sig[i]) for t in tags[:2]]
            fv = soft_frac(a[:, i], b[:, i], sig[i])
            print(f"  z={z:<6} xi{ell}: v3 vs baseline {np.median(fb[0]):.4f} / {np.percentile(fb[0], 95):.4f} | "
                  f"v3_noslope vs baseline {np.median(fb[1]):.4f} / {np.percentile(fb[1], 95):.4f} | "
                  f"v3 vs v3_noslope {np.median(fv):.4f} / {np.percentile(fv, 95):.4f}")

    # ---- plots at the fiducial-like theta
    r = BaseConfig.r
    fig, ax = plt.subplots(3, len(zs), figsize=(3.2 * len(zs), 7.5), sharex='col', squeeze=False)
    for j, z in enumerate(zs):
        for i, ell in enumerate(gen.ELLS):
            ax[i, j].plot(r, r ** 2 * base[z][i], 'k-', label='baseline 11par')
            for t, ls in zip(tags, ('-', '--')):
                if (z, t) in fid:
                    ax[i, j].plot(r, r ** 2 * fid[(z, t)][i], ls, label=t)
            ax[i, j].set_title(f'z={z}  ell={ell}', fontsize=9)
            ax[i, j].margins(x=0)
        ax[2, j].set_xlabel(r'$s\ [h^{-1}{\rm Mpc}]$')
    ax[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, 'smoke_multipoles_11par_v3.png'))

    # ---- timing projection
    per_theta = float(np.mean(list(t_serial.values())))
    core_hours = per_theta * N_FULL_PER_Z * len(gen.REDSHIFTS) / 3600
    print(f"\ntiming: {per_theta:.1f} core-s per theta (all {len(tags)} outputs, serial) -> full run "
          f"{N_FULL_PER_Z * len(gen.REDSHIFTS)} thetas = {core_hours:.0f} core-hours = {core_hours / gen.N_JOBS:.1f} h on {gen.N_JOBS} workers")

    print(f'\nplots and data in {run_dir}')
    for w in warns:
        print('WARN', w)
    for f in fails:
        print('FAIL', f)
    print('SMOKE TEST', 'FAILED' if fails else 'PASSED' + (' with warnings' if warns else ''))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
