"""
Joint fit like the baseline `compact_dr1_joint_smin20`, with the two AP fixes, as a SHORT chain to preview the corrected constraints.

Fixes (applied by replacing `fit_lib_11par.ap_multi`; fit_lib_11par.py itself is not edited):
  1. the AP reference cosmology is the DESI fiducial (AbacusSummit c000: h = 0.6736, omega_b = 0.02237, omega_cdm = 0.12),
     not `common.COSMO_FID`;
  2. both q_perp and q_par carry the unit factor h / h_fid. The data separations are in h_fid^-1 Mpc and the emulator is in h^-1 Mpc at the
     sampled h, so the physical-distance ratios D_A/D_A^fid and H^fid/H must be multiplied by h / h_fid (this is what desilike/cosmoprimo
     do implicitly by working in Mpc/h).

Usage (compact environment, from the folder containing fit_lib_11par.py):
    python fit_dr1_apfix.py [base_spec_name] [max_steps]
e.g. `python fit_dr1_apfix.py` (DR1 joint, s_min = 20, MAX_STEPS) or `python fit_dr1_apfix.py compact_mean_joint_smin20 3000` (the same fix on the
mean mocks). The chain is written to <CHAIN_ROOT>/compact_apfix/<base_spec_name>_apfix.{h5,txt,status.json}. Re-running with a larger max_steps
resumes it. The walkers start in a ball around the baseline chain (mean, INIT_SCALE x std) if that chain exists, so a short run is usable; a run
this short will end flagged converged = false, and its widths are only approximate.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
    os.environ[_k] = '1'                      # must be set before ANY import that loads numpy (h5py does): the fork-based emcee pool needs 1 thread
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_lib_11par as fl
import h5py                                   # after fit_lib_11par, never before

BASE_SPEC = sys.argv[1] if len(sys.argv) > 1 else 'compact_dr1_joint_smin20'
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
NCORES = 24                                   # emcee uses at most ~nwalkers/2 = 26 cores for the 17-parameter joint fit
INIT_SCALE = 2.0                              # initial walker ball = baseline posterior std x this
OUT_DIR = fl.CHAIN_ROOT / 'compact_apfix'
C000_REF = (0.6736, 0.02237, 0.12)            # DESI fiducial cosmology: AbacusSummit c000 (h, omega_b, omega_cdm)


def make_ap_multi(ref):
    """(q_perp, q_par) for every z from ONE CAMB background call, referenced to `ref` and in h^-1 Mpc units."""
    h_ref = ref[0]
    ref_res = fl.get_background(*ref, fl.NS_FIXED)
    zs = set(fl.Z_EFF.values())
    da_ref = {z: ref_res.angular_diameter_distance(z) for z in zs}
    hz_ref = {z: ref_res.hubble_parameter(z) for z in zs}

    def ap_multi(h, ombh2, omch2, ns, z_effs):
        r = fl.get_background(h, ombh2, omch2, ns)
        f = h / h_ref
        return [(f * r.angular_diameter_distance(z) / da_ref[z], f * hz_ref[z] / r.hubble_parameter(z)) for z in z_effs]
    return ap_multi


def init_from_baseline(problem, base_spec):
    """Start the walkers around the baseline chain (posterior mean, INIT_SCALE x std); keep fit_lib's default start if it is missing."""
    root = fl.CHAIN_ROOT / base_spec['model'] / base_spec['name']
    h5_path, st_path = Path(f'{root}.h5'), Path(f'{root}.status.json')
    if not (h5_path.exists() and st_path.exists()):
        print(f'baseline chain {root} not found: using fit_lib_11par default start', flush=True)
        return
    st = json.load(open(st_path))
    with h5py.File(h5_path, 'r') as f:
        it = int(f['mcmc'].attrs['iteration'])
        flat = f['mcmc/chain'][st['discard']:it:st['thin'], :, :].reshape(-1, problem.ndim)
    problem.theta0, problem.scale = flat.mean(0), INIT_SCALE * flat.std(0)
    print(f'walkers start around the baseline chain: h = {flat[:, 0].mean():.4f} +- {flat[:, 0].std():.4f}', flush=True)


if __name__ == '__main__':
    fl.lower_priority()
    fl.ap_multi = make_ap_multi(C000_REF)     # FitProblem.log_likelihood looks ap_multi up in fit_lib's namespace at call time
    z0 = next(iter(fl.Z_EFF.values()))
    q_perp, q_par = fl.ap_multi(*C000_REF, fl.NS_FIXED, [z0])[0]
    assert abs(q_perp - 1) < 1e-9 and abs(q_par - 1) < 1e-9, f'q != 1 at the reference cosmology: {q_perp}, {q_par}'
    print(f'AP fix active: reference = c000 {C000_REF}, h/h_fid factor on; q = 1 at c000 (z = {z0})', flush=True)

    base = next(s for s in fl.build_specs() if s['name'] == BASE_SPEC)
    spec = dict(base, name=BASE_SPEC + '_apfix')
    problem = fl.FitProblem(spec)
    init_from_baseline(problem, base)
    fl.run_chain(problem, OUT_DIR / spec['name'], NCORES, max_steps=MAX_STEPS, tag=f"[{spec['name']}] ")