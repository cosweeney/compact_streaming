"""
Check that the high_M emulator entries above Z_ANCHOR_MAX follow the extrapolation scheme, for any config:
    high_M(z) = low_M(z) + mean over z <= Z_ANCHOR_MAX of (high_M - low_M),
applied elementwise to the intercept and every slope, for the extrapolated rows only. Also checks that the avg_M
emulator ('fits') is the mean of low_M and high_M at every node.

    python check_highM_extrapolation_11par_v3.py                          # baseline compact.config.Config
    python check_highM_extrapolation_11par_v3.py --module config_11par_v3 --cls Config   # the new emulator (run from this folder)
"""
import argparse
import importlib
import warnings

import numpy as np

warnings.filterwarnings('ignore')
ap = argparse.ArgumentParser()
ap.add_argument('--module', default='compact.config')
ap.add_argument('--cls', default='Config')
ap.add_argument('--anchor', type=float, default=0.5)
ap.add_argument('--tol', type=float, default=1e-5)
args = ap.parse_args()

C = getattr(importlib.import_module(args.module), args.cls)
names = ['sigma0/A_sigma', 'A_rho0', 'A_rho_p', 'sigma_rho', 'A_delta0', 'r_delta', 'sigma_delta', 'beta_delta', 'alpha', 'nu_0', 'r_c']
nodes = sorted(z for z, e in C.compact_emulator.items() if e is not None)
lo_z = [z for z in nodes if z <= args.anchor]
hi_z = [z for z in nodes if z > args.anchor]
get = lambda z, which, row: [e[2] for e in C.compact_emulator[z]['fits_indiv'][which] if e[0] == row][0]

print(f'{args.module}.{args.cls}: nodes {nodes}')
print(f'row                | max relative deviation of high_M(z) from low_M(z) + mean_(z<={args.anchor:g})(high_M - low_M), at z = {hi_z}')
follows = []
for row, nm in enumerate(names):
    shift = np.mean([get(z, 'high_M', row) - get(z, 'low_M', row) for z in lo_z], axis=0)
    dev = [np.max(np.abs(get(z, 'high_M', row) - (get(z, 'low_M', row) + shift)) / np.maximum(np.abs(get(z, 'high_M', row)), 1e-9)) for z in hi_z]
    ok = max(dev) < args.tol
    if ok:
        follows.append(nm)
    print(f'{row:2d} {nm:14s} | ' + '  '.join(f'{d:9.2e}' for d in dev) + ('   <- follows the scheme' if ok else ''))
print('rows that follow the extrapolation scheme:', follows)

avg_dev = max(np.max(np.abs([e[2] for e in C.compact_emulator[z]['fits'] if e[0] == r][0]
                          - 0.5 * (get(z, 'low_M', r) + get(z, 'high_M', r)))
                     / np.maximum(np.abs([e[2] for e in C.compact_emulator[z]['fits'] if e[0] == r][0]), 1e-9))
              for z in nodes for r in range(11))
print(f'avg_M fits == (low_M + high_M) / 2 at every node and row: max relative deviation {avg_dev:.1e}')
