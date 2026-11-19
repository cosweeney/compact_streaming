"""
Smoke test for the w0wa training-data pipeline. Run this BEFORE generate_multiz_11par_w0wa.py.

Uses the real compact/config_w0wa.py and the real generation code path on a few thetas from the first
rows of the training Sobol sequence, and checks:
  1. preflight: config is the 6-column w0, wa emulator and covers the requested redshifts (hard fail);
     rows with no fit and worst-case AP separations beyond Config.r are printed (warnings);
  2. generate_multiz_11par_w0wa.run_one (shared-LPT path) reproduces CompactStreamingModel.predict()
     (hard fail above 1e-9 of the peak amplitude; run-to-run FFT round-off is ~1e-15);
  3. every prediction is finite and passes train_11par_w0wa.flag (hard fail if >5% flagged);
  4. wall time per theta and the projected full-run time for the requested schemes.

    python smoke_test_w0wa.py [--n-theta 8] [--schemes avg_M] [--redshifts 0.295 0.510 ...]
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_multiz_11par_w0wa as gen
from train_11par_w0wa import flag
from compact.predictions import CompactStreamingModel

warnings.filterwarnings('ignore')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-theta', type=int, default=8)
    ap.add_argument('--schemes', nargs='+', default=gen.MASS_BIN_SCHEMES)
    ap.add_argument('--redshifts', type=float, nargs='+', default=gen.REDSHIFTS)
    args = ap.parse_args()
    fails = []

    cfg = gen.ConfigW0wa()
    try:
        warns = gen.preflight(cfg, args.redshifts, args.schemes)
    except Exception as e:
        sys.exit(f'FAIL preflight: {e}')
    for w in warns:
        print('WARNING preflight:', w)
    print(f'preflight ok: {len(cfg.compact_emulator)} nodes, growth rate from CAMB: {cfg.exact_growth}')

    n_full = sum(n for n, _ in gen.SETS.values())     # planned rows per z
    n_now = sum(gen.N_COMPUTE.values())               # rows computed by default
    total = 0.0
    for z in args.redshifts:
        seed = gen.SETS['train'][1] + 1000 * gen.REDSHIFTS.index(z)
        thetas = gen.make_thetas(args.n_theta, seed)

        t0 = time.time()
        out = [gen.run_one(i, th, z, cfg, args.schemes)[1] for i, th in enumerate(thetas)]
        t_theta = (time.time() - t0) / len(thetas)
        total += t_theta

        model = CompactStreamingModel(redshift=z, config=cfg)
        for s in args.schemes:
            xi, errs = zip(*(o[s] for o in out))
            if any(e is not None for e in errs):
                fails.append(f'z={z} {s}: {sum(e is not None for e in errs)} failed, first: {next(e for e in errs if e)}')
                continue
            xi = np.array(xi)
            direct = model.predict(thetas[0], ells=gen.ELLS, mass_bin=s)
            err = np.max(np.abs(direct - xi[0])) / np.max(np.abs(direct))
            bad, _ = flag(xi)
            print(f'z={z} {s}: {t_theta:.1f} s/theta serial | run_one vs predict {err:.1e} | '
                  f'flagged {bad.sum()}/{len(bad)} | xi0[s={cfg.r[0]:.0f}] = {xi[0, 0, 0]:.3f}')
            if err > 1e-9:
                fails.append(f'z={z} {s}: shared-LPT path differs from predict() by {err:.1e}')
            if bad.mean() > 0.05:
                fails.append(f'z={z} {s}: {bad.sum()}/{len(bad)} samples flagged bad')

    per_z = total / len(args.redshifts)
    print(f'\nprojected wall time at N_JOBS={gen.N_JOBS}: '
          f'{per_z * n_now * len(gen.REDSHIFTS) / gen.N_JOBS / 3600:.0f} h for the default {n_now} rows/z '
          f'x {len(gen.REDSHIFTS)} z; {per_z * n_full * len(gen.REDSHIFTS) / gen.N_JOBS / 3600:.0f} h for all {n_full} planned rows '
          f'(serial {per_z:.1f} s/theta, {len(args.schemes)} scheme(s); assumes perfect scaling)')
    if fails:
        sys.exit('FAILED:\n  ' + '\n  '.join(fails))
    print('smoke test passed')


if __name__ == '__main__':
    main()
