"""Shared machinery for the AbacusSummit c130-c181 cosmology grid.

Not run directly -- see run_abacus_c130_z05.py and run_abacus_c130_rest.py.

These catalogues differ from the earlier halo_combined.h5 files:
  * columns are 'mass', 'x'..'vz', 'rvir', 'cvir', 'vrms', 'gid'
  * there is NO particle-count column, so the particle-count bins are applied
    as mass edges recomputed per cosmology from that cosmology's m_p
  * the redshift directory is 'z0p500', not 'z0.500'
  * cosmology comes from the 'base_c<NNN>' file one level above the sim
    directory, in the omega = Omega*h^2 convention -- Omega_M is DERIVED:
        Omega_M = (omega_b + omega_cdm + omega_ncdm) / h^2
"""

import gc
import glob
import os
import re
import shutil
import time
import traceback

import numpy as np
import h5py as h5

from pairhist import PairHist
from pairhist_direct import get_pairwise_hist
from compact.catalog.accumulate import generate_sub_box_ids, split_simulation_into_sub_boxes

import psutil
psutil.Process().nice(19)

# ---------------------------------------------------------------- config ----
SIM_NUMS = list(range(130, 182))          # 52 cosmologies, ph000 only

HALO_TMPL = ('/spiffball/salcedo/AbacusSummit/Snapshots/{cosmo}/'
             'base_{sim}/{zdir}/halo_base_{sim}_{zdir}.h5')

# Cosmology file: 'base_c131' inside 'Snapshots/base_c131/'.  The per-redshift
# cosmo_<sim>_<zdir>.param is used as a fallback -- it is not always present.
PARAM_DIR = '/spiffball/salcedo/AbacusSummit/Snapshots/{cosmo}'
PARAM_GLOBS = ['cosmo_*.param', '*.param', 'header', '*header*']

# Include massive neutrinos in Omega_M?  They are ~0.5% of the total here, so
# this shifts m_p by ~0.5% -- negligible against 0.1 dex bins, but AbacusSummit
# evolves cb-only particles, so set False to use Omega_cb instead.
OMEGA_M_INCLUDES_NCDM = True

OUT_DIR = ('/spiffball/cosweeney/simulations/AbacusSummit_base/halos/'
           'cosmo_grid/{sim}/{zdir}/data/')
SB_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base/halos/'
           'cosmo_grid/{sim}/{zdir}/SBp/')
LOG_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base/logs/'
            'cosmo_grid/pairhist_{sim}_{zdir}.log')
OUT_NAME = 'pairhist_2bins.h5'

# --- radial grids (bin EDGES).  Note r_los is one-sided: +5 to +140, so pairs
# --- with rlos < 5, including every negative rlos, fall outside the grid.
r_perp = np.linspace(5, 140, 135 + 1)
r_los = np.linspace(5, 140, 135 + 1)
r_p = 0.5 * (r_perp[1:] + r_perp[:-1])           # centres, for the fit stage
r_l = 0.5 * (r_los[1:] + r_los[:-1])

R_edges = r_perp
rlos_edges = r_los
RLOS_MAX = float(max(abs(rlos_edges[0]), abs(rlos_edges[-1])))    # = 140

box_length = 2000.0
subsize = box_length / 4          # 500 > r_max
r_max = 140.0
N_PART = 6912                     # particles per side, AbacusSummit base

# Mass bins as log10 M edges in the FIDUCIAL c000 cosmology -- massbins[1] and
# massbins[3] of the original six-bin list.  Converted once to particle counts;
# it is the PARTICLE counts that are applied to every cosmology.
BIN_LOGM = np.array([[12.5, 12.6], [13.5, 13.6]])

OM_FID = 0.3152
M_PART_FID = 27.75e10 * OM_FID * box_length**3 / N_PART**3
BIN_NPART = np.rint(10**BIN_LOGM / M_PART_FID).astype(np.int64)
N_NEIGHBOUR = int(round(1e12 / M_PART_FID))       # tracer cut, in particles

V_LO, V_HI, N_V = -4000.0, 4000.0, 400
CONV = 1.0                        # verified by the sigma_1D guard below
VEL_CHECK = (50.0, 3000.0)

MIN_SEEDS = 2_000
RESUME = True
KEEP_SUBBOXES = False
N_JOBS = int(os.environ.get('PAIRHIST_NJOBS', 112))
TASKS_PER_SUB_BOX = int(os.environ.get('PAIRHIST_TASKS_PER_SB', 1))

# --- cost model, calibrated on the c000_ph000 run --------------------------
T_PER_NBR = 2.76e-7
T_PER_PAIR = 2.63e-7
OVERHEAD = 1.48
T_READ = 120.0
T_SUBBOX_PER_1E6 = 9.4
T_REDUCE = 120.0
N_EFFECTIVE_CORES = 64

# key = value, tolerating missing spaces, trailing spaces, and [section] lines
_KV_PAT = re.compile(r'^\s*([A-Za-z_][A-Za-z_0-9]*)\s*=\s*([-+0-9.eEdD]+)\s*$')
# fallback: a literal Omega_M line in an older-style header
_OM_PAT = re.compile(r'(?i)\bomega[_\s-]*m\b\s*[=:]?\s*\[?\s*([-+0-9.eEdD]+)')

_log_path = None
_OM_CACHE = {}


# ------------------------------------------------------------- helpers -----
def log(msg):
    line = '[%s] %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), msg)
    print(line, flush=True)
    if _log_path:
        with open(_log_path, 'a') as f:
            f.write(line + '\n')


def sim_name(n):
    return 'c%d_ph000' % n


def cosmo_dir(n):
    return 'base_c%d' % n


def zdir(z):
    return ('z%.3f' % z).replace('.', 'p')        # 0.5 -> 'z0p500'


def halo_path(n, z):
    return HALO_TMPL.format(cosmo=cosmo_dir(n), sim=sim_name(n), zdir=zdir(z))


def param_filename(n, z):
    return 'cosmo_%s_%s.param' % (sim_name(n), zdir(z))


def _candidate_params(d, name):
    seen, out = set(), []
    direct = os.path.join(d, name)
    if os.path.isfile(direct):
        out.append(direct)
        seen.add(direct)
    for pat in PARAM_GLOBS:
        for p in sorted(glob.glob(os.path.join(d, pat))):
            if p not in seen and os.path.isfile(p):
                out.append(p)
                seen.add(p)
    return out


def _parse_kv(path):
    """Every 'key = number' line in a params file, as a dict of floats."""
    out = {}
    with open(path, errors='ignore') as f:
        for line in f:
            if line.lstrip().startswith(('#', ';', '//', '[')):
                continue
            m = _KV_PAT.match(line)
            if m:
                try:
                    out[m.group(1)] = float(
                        m.group(2).replace('d', 'e').replace('D', 'e'))
                except ValueError:
                    pass
    return out


def _omega_m_from_params(p):
    """Omega_M from omega_b / omega_cdm / omega_ncdm / h, or a literal key."""
    for key in ('Omega_M', 'Omega_m', 'OmegaM'):
        if key in p:
            return p[key]
    if 'h' in p and ('omega_cdm' in p or 'omega_b' in p):
        w = p.get('omega_b', 0.0) + p.get('omega_cdm', 0.0)
        if OMEGA_M_INCLUDES_NCDM:
            w += p.get('omega_ncdm', 0.0)
        return w / p['h']**2
    return None


def read_params(n, z):
    """Full cosmology dict for cosmology n, with 'Omega_M' derived.

    Reads Snapshots/base_c<NNN>/base_c<NNN> first; falls back to the
    per-redshift cosmo_<sim>_<zdir>.param beside the catalogue.
    """
    if n in _OM_CACHE:
        return _OM_CACHE[n]

    tried = []
    primary = os.path.join(PARAM_DIR.format(cosmo=cosmo_dir(n)), cosmo_dir(n))
    cands = [primary] if os.path.isfile(primary) else []
    tried.append(primary)

    d = os.path.dirname(halo_path(n, z))
    cands += _candidate_params(d, param_filename(n, z))
    tried.append(os.path.join(d, param_filename(n, z)))

    for path in cands:
        p = _parse_kv(path)
        om = _omega_m_from_params(p)
        if om is None:                       # older Omega_M-only header style
            with open(path, errors='ignore') as f:
                for line in f:
                    m = _OM_PAT.search(line)
                    if m:
                        try:
                            om = float(m.group(1))
                        except ValueError:
                            om = None
                        break
        if om is not None:
            p['Omega_M'] = om
            p['_param_file'] = path
            _OM_CACHE[n] = p
            return p

    raise KeyError('no usable cosmology file for %s; tried:\n  %s'
                   % (sim_name(n), '\n  '.join(tried)))


def read_omega_m(n, z):
    return read_params(n, z)['Omega_M']


def particle_mass(om0):
    return 27.75e10 * om0 * box_length**3 / N_PART**3


def _read_cols(hdf, names):
    """These files may hold flat datasets or one compound table."""
    keys = list(hdf.keys())
    if all(nm in keys for nm in names):
        return dict((nm, hdf[nm][()]) for nm in names)
    for k in keys:
        d = hdf[k]
        dt = getattr(d, 'dtype', None)
        if dt is not None and dt.names and all(nm in dt.names for nm in names):
            arr = d[:]
            return dict((nm, arr[nm]) for nm in names)
    raise KeyError('could not find columns %s; top-level keys: %s'
                   % (names, keys))


# ------------------------------------------------------------ estimate -----
def predict_seconds(n_tracers, n_seeds):
    n_bar = n_tracers / box_length**3
    r_q = np.hypot(r_max, RLOS_MAX)
    n_nbr = n_bar * (4.0 / 3.0) * np.pi * r_q**3
    n_cyl = n_bar * np.pi * r_max**2 * 2.0 * RLOS_MAX
    t_seed = OVERHEAD * (T_PER_NBR * n_nbr + T_PER_PAIR * n_cyl)
    t_pairs = sum(n_seeds) * t_seed / N_EFFECTIVE_CORES
    t_fixed = T_READ + T_REDUCE + T_SUBBOX_PER_1E6 * n_tracers / 1e6
    return t_pairs + t_fixed, t_pairs, t_fixed


def estimate(nums, zs, n_sample=None):
    jobs = [(n, z) for n in nums for z in zs]
    sample = jobs if n_sample is None else jobs[:int(n_sample)]
    print('query radius %.1f (r_max %g, rlos_max %g)'
          % (np.hypot(r_max, RLOS_MAX), r_max, RLOS_MAX))
    print('scanning %d of %d jobs (mass column only)\n'
          % (len(sample), len(jobs)))
    header = '%14s %6s %12s' % ('sim', 'z', 'tracers')
    header += ''.join('%12s' % ('bin%d' % k) for k in range(len(BIN_NPART)))
    header += ' %8s %7s %8s' % ('pairs', 'fixed', 'total')
    print(header)

    rows, missing = [], []
    for n, z in sample:
        try:
            om0 = read_omega_m(n, z)
            with h5.File(halo_path(n, z), 'r') as hdf:
                mass = _read_cols(hdf, ['mass'])['mass']
        except (OSError, KeyError) as e:
            missing.append((n, z, '%s: %s' % (type(e).__name__, str(e)[:60])))
            continue
        m_p = particle_mass(om0)
        n_tr = int((mass >= N_NEIGHBOUR * m_p).sum())
        seeds = [int(((mass >= lo * m_p) & (mass <= hi * m_p)).sum())
                 for lo, hi in BIN_NPART]
        tot, tp, tf = predict_seconds(n_tr, seeds)
        rows.append(tot)
        line = '%14s %6.2f %12s' % (sim_name(n), z, format(n_tr, ','))
        line += ''.join('%12s' % format(s, ',') for s in seeds)
        line += ' %7.0fm %6.0fm %7.0fm' % (tp / 60, tf / 60, tot / 60)
        print(line)

    for n, z, err in missing:
        print('  MISSING %s z%s: %s' % (sim_name(n), z, err))
    if not rows:
        print('\nnothing could be read -- check HALO_TMPL and PARAM_GLOBS')
        return
    mean = float(np.mean(rows))
    print('\nscanned %d/%d, mean %.0f min each'
          % (len(rows), len(jobs), mean / 60))
    print('projected total: %.1f h = %.1f days'
          % (mean * len(jobs) / 3600, mean * len(jobs) / 86400))
    print('  (model calibrated on ph000; +/- 30%, re-check after the first run)')


# ----------------------------------------------------------- one job -------
def run_one(n, z, tag=''):
    global _log_path
    sim = sim_name(n)
    zd = zdir(z)
    _log_path = LOG_TMPL.format(sim=sim, zdir=zd)
    os.makedirs(os.path.dirname(_log_path), exist_ok=True)

    hpath = halo_path(n, z)
    sb_path = SB_TMPL.format(sim=sim, zdir=zd)
    out_dir = OUT_DIR.format(sim=sim, zdir=zd)
    out_path = os.path.join(out_dir, OUT_NAME)

    if not os.path.exists(hpath):
        raise FileNotFoundError(hpath)

    par = read_params(n, z)
    om0 = par['Omega_M']
    m_p = particle_mass(om0)
    M_edges = BIN_NPART * m_p
    M_tracer = N_NEIGHBOUR * m_p

    log('=== %s %s %s ===' % (sim, zd, tag))
    log('%s %s: Om %.4f (h %.4f, s8_m %s), m_p %.3e Msun/h'
        % (sim, zd, om0, par.get('h', float('nan')),
           par.get('sigma8_m', 'n/a'), m_p))
    log('%s %s: params from %s' % (sim, zd, par.get('_param_file', '?')))
    for k in range(len(BIN_NPART)):
        log('%s %s:   bin %d: %s-%s particles = [%.3e, %.3e] Msun/h'
            % (sim, zd, k, format(BIN_NPART[k, 0], ','),
               format(BIN_NPART[k, 1], ','), M_edges[k, 0], M_edges[k, 1]))
    log('%s %s: r_max %g, rlos grid [%g, %g] -> query radius %.1f'
        % (sim, zd, r_max, rlos_edges[0], rlos_edges[-1],
           np.hypot(r_max, RLOS_MAX)))

    log('%s %s: reading %s' % (sim, zd, hpath))
    with h5.File(hpath, 'r') as hdf:
        col = _read_cols(hdf, ['mass', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'gid'])
    mass = col['mass']
    pos = np.array([col['x'], col['y'], col['z']]).T % box_length
    vel = np.array([col['vx'], col['vy'], col['vz']]).T
    hid = col['gid']
    del col
    gc.collect()

    sig1d = float(np.std(CONV * vel[:, 2]))
    log('%s %s: 1-D velocity dispersion %.1f km/s (CONV = %g)'
        % (sim, zd, sig1d, CONV))
    if not (VEL_CHECK[0] <= sig1d <= VEL_CHECK[1]):
        raise ValueError('%s %s: sigma_1D = %.3e km/s outside %s -- check CONV '
                         'and the velocity column' % (sim, zd, sig1d, VEL_CHECK))

    nb = mass >= M_tracer
    n_bar = int(nb.sum()) / box_length**3
    log('%s %s: %s tracers (>= %s particles = %.3e Msun/h), n_bar %.3e'
        % (sim, zd, format(int(nb.sum()), ','), format(N_NEIGHBOUR, ','),
           M_tracer, n_bar))

    seed_mbin = np.full(len(mass), -1, dtype=np.int32)
    counts, logM_eff = [], []
    for k in range(len(BIN_NPART)):
        sel = (mass >= M_edges[k, 0]) & (mass <= M_edges[k, 1])
        seed_mbin[sel] = k
        counts.append(int(sel.sum()))
        logM_eff.append(float(np.mean(np.log10(mass[sel])))
                        if sel.any() else float('nan'))
        log('%s %s:   bin %d: %s seeds, <log10 M> = %.4f'
            % (sim, zd, k, format(counts[k], ','), logM_eff[k]))
    if min(counts) < MIN_SEEDS:
        raise ValueError('%s %s: a mass bin has < %d seeds (%s)'
                         % (sim, zd, MIN_SEEDS, counts))

    shutil.rmtree(sb_path, ignore_errors=True)     # append mode -> start clean
    os.makedirs(sb_path, exist_ok=True)
    chunksize = max(int(nb.sum()) // 1000, 1)
    generate_sub_box_ids(pos[nb], box_length, subsize, chunksize,
                         sb_path, name='halo')
    split_simulation_into_sub_boxes(
        positions=pos[nb], velocities=vel[nb], ids=hid[nb], upids=hid[nb],
        boxsize=box_length, subsize=subsize, chunksize=chunksize,
        dtypes=[pos.dtype, vel.dtype], path=sb_path, name='halo')

    h = get_pairwise_hist(
        pos, vel, seed_mbin,
        r_max=r_max, boxsize=box_length, subsize=subsize, path=sb_path,
        n_M=len(BIN_NPART), R_edges=R_edges, rlos_edges=rlos_edges,
        v_lo=V_LO, v_hi=V_HI, n_v=N_V,
        conv=CONV, n_jobs=N_JOBS, tasks_per_sub_box=TASKS_PER_SUB_BOX,
        rlos_max=RLOS_MAX,
    )

    off = int((h.n - h.counts.sum(-1)).sum())
    log('%s %s: pairs per bin %s, off-grid %d (%.1e)'
        % (sim, zd, h.n.sum(axis=(1, 2)).tolist(), off,
           off / max(int(h.n.sum()), 1)))
    for k in range(len(BIN_NPART)):
        occ = h.n[k][h.n[k] > 0]
        _, sd = h.mean_std(k)
        log('%s %s:   bin %d: cells %d/%d, min/median cell n %d/%.0f, '
            'median sigma %.1f km/s'
            % (sim, zd, k, int((h.n[k] > 0).sum()), h.n[k].size,
               int(occ.min()) if occ.size else 0,
               float(np.median(occ)) if occ.size else 0.0,
               float(np.nanmedian(sd))))

    os.makedirs(out_dir, exist_ok=True)
    h.save(out_path)
    with h5.File(out_path, 'a') as hdf:
        hdf.attrs.update(sim=sim, cosmo_num=int(n), redshift=float(z),
                         suite='AbacusSummit_base', source='salcedo',
                         boxsize=box_length, subsize=subsize, r_max=r_max,
                         rlos_max=RLOS_MAX, conv=CONV, Om0=om0, m_part=m_p,
                         n_neighbour_part=N_NEIGHBOUR, M_tracer=M_tracer,
                         n_tracers=int(nb.sum()), sigma1d_cat=sig1d)
        # full cosmology, so the emulator design matrix comes straight from
        # the outputs -- omega_b, omega_cdm, h, A_s, n_s, sigma8_m, w0, wa ...
        hdf.attrs.update(dict((k, v) for k, v in par.items()
                             if not k.startswith('_')))
        hdf.create_dataset('bin_npart', data=BIN_NPART)
        hdf.create_dataset('bin_medges', data=M_edges)
        hdf.create_dataset('seeds_per_bin', data=np.array(counts))
        hdf.create_dataset('logM_eff', data=np.array(logM_eff))
    log('%s %s: saved %s' % (sim, zd, out_path))

    if not KEEP_SUBBOXES:
        shutil.rmtree(sb_path, ignore_errors=True)

    del pos, vel, hid, mass, nb, seed_mbin, h
    gc.collect()


# -------------------------------------------------------------- batch -----
def run_batch(nums, zs):
    jobs = [(n, z) for n in nums for z in zs]
    failed, t0 = [], time.time()
    for i, (n, z) in enumerate(jobs, 1):
        sim = sim_name(n)
        out_path = os.path.join(OUT_DIR.format(sim=sim, zdir=zdir(z)), OUT_NAME)
        if RESUME and os.path.exists(out_path):
            print('%s z%s: exists, skipping (%d/%d)'
                  % (sim, z, i, len(jobs)), flush=True)
            continue
        t = time.time()
        try:
            run_one(n, z, tag='(%d/%d)' % (i, len(jobs)))
            el = (time.time() - t0) / 3600
            done = i - len(failed)
            eta = el / done * (len(jobs) - i) if done else float('nan')
            log('%s z%s: done in %.1f min [%d/%d, elapsed %.1f h, eta %.1f h]'
                % (sim, z, (time.time() - t) / 60, i, len(jobs), el, eta))
        except Exception:
            failed.append((sim, z))
            log('%s z%s: FAILED\n%s' % (sim, z, traceback.format_exc()))
    if len(jobs) > 1:
        log('batch done in %.1f h; %d failed: %s'
            % ((time.time() - t0) / 3600, len(failed), failed))
    return 1 if failed else 0


def main(zs, argv):
    """Shared CLI: --estimate [n], or [cosmo_num] [z]."""
    if argv and argv[0] == '--estimate':
        estimate(SIM_NUMS, zs, argv[1] if len(argv) > 1 else None)
        return 0
    nums = [int(str(argv[0]).lstrip('c').split('_')[0])] if argv else SIM_NUMS
    z_run = [float(argv[1])] if len(argv) > 1 else zs
    return run_batch(nums, z_run)