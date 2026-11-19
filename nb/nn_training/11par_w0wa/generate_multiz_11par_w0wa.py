"""
Generate train/test/val sets of xi_0(s), xi_2(s), xi_4(s) predictions from CompactStreamingModel
across six redshifts, with w0, wa varied on top of the 7-parameter set:
theta = [b1, b2, omega_b, omega_cdm, h, As, ns, w0, wa].

Derived from generate_multiz_11par.py. Differences:
  * compact.config_w0wa.ConfigW0wa: ridge emulator with w0, wa columns, growth rate f from CAMB.
  * w0, wa priors are the AbacusSummit grid hypercube, w0 in [-1.271, -0.726], wa in [-0.628, 0.621].
  * Default scheme is avg_M only (--schemes avg_M low_M high_M computes more; the LPT inputs are
    shared only between schemes run in the same invocation).
  * Sobol sequences are planned at their full length (train 150k, test/val 10k) and only the leading
    rows are computed (--n-train, default 100k). Raising --n-train later continues the SAME scrambled
    sequence, so the computed rows are always a Sobol prefix. Do not interrupt a run if the set is
    to be used: a prefix is only space-filling once the run completes.
  * The AP-padding check is a warning computed for the fit range 20-130 h^-1 Mpc over the prior box.

Output (one tree per scheme):
  .../compact_multipole_training_set_11par_w0wa_avg_M/z0p51/{thetas,xi}_{train,test,val}.npy, s.npy
Rows not yet computed (or that failed) are all-NaN in xi_*.npy and are dropped by train_11par_w0wa.flag.

Resumable: each finished (scheme, z, set) gets '<xi file>.done' recording how many leading rows were
requested; a set with a checkpointed xi file but a smaller (or no) marker is resumed, recomputing
all-NaN rows among the requested ones. thetas files are regenerated from the seed and must match
what is on disk.

    python generate_multiz_11par_w0wa.py                       # avg_M, 100k train, 10k test, 10k val
    python generate_multiz_11par_w0wa.py --n-train 150000      # extend train to the planned 150k
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import itertools
import json
import numpy as np
from scipy.stats import qmc
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import warnings
import psutil

from compact.predictions import CompactStreamingModel
from compact.config_w0wa import ConfigW0wa
from compact.lpt import GetLPTInputs

# ------------------------------------------------------------------ settings
REDSHIFTS = [0.295, 0.510, 0.706, 0.919, 1.317, 1.491]

# (planned sequence length, Sobol seed base); the seed used for redshift z is base + 1000 * index(z)
SETS = {
    'train':  (150_000, 42),
    'test':   (10_000,  43),
    'val':    (10_000,  44),
}
# leading rows computed by default
N_COMPUTE = {'train': 100_000, 'test': 10_000, 'val': 10_000}

S_FIT_RANGE = (20.0, 130.0)     # h^-1 Mpc; the range the fits use (for the AP-padding warning)
N_JOBS = 100
CHECKPOINT_EVERY = 500
ELLS = (0, 2, 4)
NICE = 19
N_COMPACT_PARAMS = 11

MASS_BIN_SCHEMES = ['avg_M']

BASE_DIR_TEMPLATE = '/spiffball/cosweeney/data/training/compact_multipole_training_set_11par_w0wa_{scheme}/'

BOUNDS = {
    'b1':          (0.0,    2.0),
    'b2':          (-5.0,   10.0),
    'omega_b':     (0.0207, 0.0243),
    'omega_cdm':   (0.104,  0.140),
    'h':           (0.55,   0.90),
    'lnAs_e10':    (2.0,    4.0),
    'ns':          (0.901,  1.025),
    'w0':          (-1.271, -0.726),
    'wa':          (-0.628, 0.621),
}
PARAM_NAMES = list(BOUNDS.keys())
LOWER = np.array([BOUNDS[p][0] for p in PARAM_NAMES])
UPPER = np.array([BOUNDS[p][1] for p in PARAM_NAMES])

warnings.filterwarnings('ignore', message='.*balance properties.*')
warnings.filterwarnings('ignore', message='.*is outside the calibrated range.*')


def zdir(z):
    return f"z{str(z).replace('.', 'p')}"


def make_thetas(n_samples, seed):
    """First n_samples points of the scrambled Sobol sequence for `seed`; prefix-consistent, i.e.
    make_thetas(n, s) == make_thetas(m, s)[:n] for n <= m."""
    sampler = qmc.Sobol(d=len(PARAM_NAMES), scramble=True, seed=seed)
    samples = qmc.scale(sampler.random(n_samples), LOWER, UPPER)
    idx = {name: i for i, name in enumerate(PARAM_NAMES)}
    return np.column_stack([
        samples[:, idx['b1']],
        samples[:, idx['b2']],
        samples[:, idx['omega_b']],
        samples[:, idx['omega_cdm']],
        samples[:, idx['h']],
        np.exp(samples[:, idx['lnAs_e10']]) / 1e10,
        samples[:, idx['ns']],
        samples[:, idx['w0']],
        samples[:, idx['wa']],
    ])


def ap_required_range(z, s_fit=S_FIT_RANGE):
    """Worst-case true-cosmology separations (s_min, s_max) [h^-1 Mpc] needed to serve the fit range
    s_fit at redshift z, over the corners of the (omega_b, omega_cdm, h, w0, wa) prior box. AP scale
    factors are q_perp = D_M h / (D_M h)_fid, q_par = (H/h)_fid / (H/h) about GetLPTInputs.cosmo_fid;
    the dilation of s lies between q_perp and q_par."""
    import camb

    def background(h, ob, oc, w0=-1.0, wa=0.0):
        pars = camb.set_params(H0=100.0 * h, ombh2=ob, omch2=oc, As=2.2e-9, ns=0.9649)
        if (w0, wa) != (-1.0, 0.0):
            pars.set_dark_energy(w=w0, wa=wa, dark_energy_model='ppf')
        return camb.get_background(pars)

    h_f, ob_f, oc_f = GetLPTInputs.cosmo_fid[:3]
    ref = background(h_f, ob_f, oc_f)
    DA_f, H_f = ref.angular_diameter_distance(z) * h_f, ref.hubble_parameter(z) / h_f
    q = []
    for ob, oc, h, w0, wa in itertools.product(*(BOUNDS[k] for k in ('omega_b', 'omega_cdm', 'h', 'w0', 'wa'))):
        res = background(h, ob, oc, w0, wa)
        q += [res.angular_diameter_distance(z) * h / DA_f, H_f / (res.hubble_parameter(z) / h)]
    return s_fit[0] * min(q), s_fit[1] * max(q)


def preflight(cfg, redshifts=REDSHIFTS, schemes=MASS_BIN_SCHEMES, check_ap=True):
    """Fail fast if the config is not the w0, wa emulator or can't serve the requested redshifts.
    Returns non-fatal warnings: rows with no fit (silent fallback to _MEAN_PARAMS in predictions.py)
    and worst-case AP separations beyond cfg.r."""
    n_cols = len(cfg.fiducial_cosmo)
    if n_cols != 6:
        raise ValueError(f"cfg.fiducial_cosmo has {n_cols} entries, expected 6 -- use compact.config_w0wa.ConfigW0wa")

    nodes = sorted(z for z, v in cfg.compact_emulator.items() if v is not None)
    if not nodes:
        raise ValueError('config.compact_emulator has no populated redshift nodes')
    if min(redshifts) < nodes[0] or max(redshifts) > nodes[-1]:
        raise ValueError(f'requested z in [{min(redshifts)}, {max(redshifts)}] is outside the '
                         f'calibrated nodes [{nodes[0]}, {nodes[-1]}]')

    warns = []
    for zn in nodes:
        emu = cfg.compact_emulator[zn]
        if len(emu['mean_params']) != N_COMPACT_PARAMS:
            raise ValueError(f"z={zn}: mean_params has {len(emu['mean_params'])} entries, expected {N_COMPACT_PARAMS}")
        for scheme in schemes:
            fits = emu['fits'] if scheme == 'avg_M' else emu['fits_indiv'][scheme]
            for row, cols, beta in fits:
                if len(beta) != 1 + n_cols or list(cols) != list(range(n_cols)):
                    raise ValueError(f'z={zn} {scheme} row {row}: expected {n_cols} predictor columns')
            rows = {r for r, _, _ in fits}
            bad = [r for r in rows if not 0 <= r < N_COMPACT_PARAMS]
            if bad:
                raise ValueError(f'z={zn} {scheme} fit rows {bad} out of range')
            miss = sorted(set(range(N_COMPACT_PARAMS)) - rows)
            if miss:
                warns.append(f'z={zn} {scheme}: no fit for parameter rows {miss} (falls back to _MEAN_PARAMS)')

    if check_ap:
        for z in redshifts:
            s_min, s_max = ap_required_range(z)
            if cfg.r[0] > s_min or cfg.r[-1] < s_max:
                warns.append(f'z={z}: worst-case prior corner needs s in [{s_min:.1f}, {s_max:.1f}] h^-1 Mpc '
                             f'for a fit range {S_FIT_RANGE}, cfg.r spans [{cfg.r[0]:.0f}, {cfg.r[-1]:.0f}]; '
                             f'fits at such corners are outside the emulated range')
    return warns


def run_one(i, theta, z, cfg, schemes):
    """One theta, every requested scheme, LPT computed once.
    Returns (i, {scheme: (xi (len(ELLS), n_r) or None, error string or None)})."""
    psutil.Process().nice(NICE)
    model = CompactStreamingModel(redshift=z, config=cfg)
    b1, b2, ombh2, omch2, h, As, ns, w0, wa = theta
    cosmo = (h, ombh2, omch2, As, ns, w0, wa)
    try:
        lpt = model._lpt_inputs(cosmo, np.array([b1, b2]))
    except Exception as e:
        return i, {s: (None, f'LPT: {e}') for s in schemes}
    out = {}
    for s in schemes:
        try:
            pdf = model._pdf(lpt, cosmo, mass_bin=s)
            out[s] = (np.asarray(model._multipoles(pdf, ells=ELLS)), None)
        except Exception as e:
            out[s] = (None, str(e))
    return i, out


def _atomic_save(path, arr):
    tmp = path + '.tmp.npy'
    np.save(tmp, arr)
    os.replace(tmp, path)


def _marker(xi_path):
    if not os.path.exists(xi_path + '.done'):
        return None
    with open(xi_path + '.done') as f:
        return json.load(f)


def generate_set(name, n_planned, seed, n_compute, z, cfg, schemes, base_dir_template, n_jobs):
    """Generate (or resume) the leading n_compute rows of one (z, set) for every scheme in `schemes`."""
    out_dirs = {s: os.path.join(base_dir_template.format(scheme=s), zdir(z)) for s in schemes}
    thetas = make_thetas(n_planned, seed)
    n_r = len(cfg.r)

    all_xi, xi_paths, todo_schemes = {}, {}, []
    for s in schemes:
        os.makedirs(out_dirs[s], exist_ok=True)
        s_path = os.path.join(out_dirs[s], 's.npy')
        if not os.path.exists(s_path):
            np.save(s_path, np.asarray(cfg.r))
        thetas_path = os.path.join(out_dirs[s], f'thetas_{name}.npy')
        xi_path = os.path.join(out_dirs[s], f'xi_{name}.npy')
        xi_paths[s] = xi_path

        if os.path.exists(thetas_path):
            saved = np.load(thetas_path)
            if saved.shape != thetas.shape or not np.array_equal(saved, thetas):
                raise RuntimeError(f"{thetas_path} exists but does not match make_thetas({n_planned}, seed={seed}) "
                                   f"-- refusing to mix sample sets")
        else:
            np.save(thetas_path, thetas)

        marker = _marker(xi_path)
        if marker is not None and marker['n_compute'] >= n_compute:
            continue                                   # finished in an earlier run
        if os.path.exists(xi_path):
            all_xi[s] = np.load(xi_path)
            assert all_xi[s].shape == (n_planned, len(ELLS), n_r), f"{xi_path}: unexpected shape"
        else:
            all_xi[s] = np.full((n_planned, len(ELLS), n_r), np.nan)
        todo_schemes.append(s)

    if not todo_schemes:
        print(f"z={z} '{name}': all schemes already done for {n_compute} rows, skipping")
        return {}

    # leading rows still needed, per scheme (all-NaN rows); one task per row computes only what's missing
    need = {i: [s for s in todo_schemes if np.isnan(all_xi[s][i]).all()] for i in range(n_compute)}
    need = {i: ss for i, ss in need.items() if ss}
    print(f"\n{'='*70}\nz={z}  '{name}': {len(need)}/{n_compute} leading samples to compute "
          f"for schemes {todo_schemes} (seed={seed}, planned {n_planned})\n{'='*70}")

    errors = {s: {} for s in todo_schemes}
    t0 = time.time()
    with Parallel(n_jobs=n_jobs, batch_size=1, return_as="generator_unordered") as par:
        stream = par(delayed(run_one)(i, thetas[i], z, cfg, ss) for i, ss in need.items())
        for done, (i, res) in enumerate(tqdm(stream, total=len(need), desc=f"z={z} {name}"), 1):
            for s, (xi_i, err) in res.items():
                if err is not None:
                    errors[s][i] = err
                else:
                    all_xi[s][i] = xi_i
            if done % CHECKPOINT_EVERY == 0:
                for s in todo_schemes:
                    _atomic_save(xi_paths[s], all_xi[s])
    for s in todo_schemes:
        _atomic_save(xi_paths[s], all_xi[s])
        n_bad = int(np.isnan(all_xi[s][:n_compute]).all(axis=(1, 2)).sum())
        with open(xi_paths[s] + '.done', 'w') as f:
            json.dump({'n_compute': n_compute, 'n_planned': n_planned, 'n_failed': n_bad, 'seed': seed}, f)
        if errors[s]:
            err_path = os.path.join(out_dirs[s], f'errors_{name}.json')
            merged = json.load(open(err_path)) if os.path.exists(err_path) else {}
            merged.update({str(k): v for k, v in errors[s].items()})
            with open(err_path, 'w') as f:
                json.dump(merged, f)

    elapsed = time.time() - t0
    print(f"z={z} '{name}' done: {len(need)} thetas x {len(todo_schemes)} schemes in {elapsed/60:.1f} min "
          f"({elapsed/max(len(need), 1):.2f} s per theta wall, {n_jobs} workers)")
    for s in todo_schemes:
        if errors[s]:
            print(f"  [{s}] {len(errors[s])} failed -- first error: {next(iter(errors[s].values()))}")
    return errors


def generate_all(schemes=MASS_BIN_SCHEMES, redshifts=REDSHIFTS, sets=SETS, n_compute=N_COMPUTE,
                 base_dir_template=BASE_DIR_TEMPLATE, n_jobs=N_JOBS, cfg=None):
    cfg = ConfigW0wa() if cfg is None else cfg
    for w in preflight(cfg, redshifts, schemes):
        print('PREFLIGHT WARNING:', w)
    for z in redshifts:
        for name, (n_planned, seed_base) in sets.items():
            seed = seed_base + 1000 * REDSHIFTS.index(z)
            generate_set(name, n_planned, seed, n_compute[name], z, cfg, list(schemes), base_dir_template, n_jobs)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--schemes', nargs='+', default=MASS_BIN_SCHEMES, choices=['avg_M', 'low_M', 'high_M'])
    ap.add_argument('--n-train', type=int, default=N_COMPUTE['train'],
                    help=f"leading train rows to compute (planned {SETS['train'][0]}; default %(default)s)")
    args = ap.parse_args()
    if not 0 < args.n_train <= SETS['train'][0]:
        raise SystemExit(f"--n-train must be in (0, {SETS['train'][0]}]")
    counts = dict(N_COMPUTE, train=args.n_train)
    print(f"schemes: {args.schemes}\nredshifts: {REDSHIFTS}\nrows: {counts}\noutput: {BASE_DIR_TEMPLATE}")
    generate_all(schemes=args.schemes, n_compute=counts)
