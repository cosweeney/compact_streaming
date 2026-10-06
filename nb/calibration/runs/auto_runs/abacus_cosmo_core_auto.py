"""Shared machinery for the AbacusSummit c130-c181 cosmology grid.

Not run directly -- see run_abacus_cosmo_rest.py.

REWRITTEN to collect AUTO-pairwise statistics: for mass bin k, BOTH the
seed population AND the neighbour/tracer pool are restricted to bin k's own
halos.  The previous version used seed = bin k, neighbour = every tracer
above N_NEIGHBOUR particles with no upper cutoff -- a CROSS-statistic
against a much broader, weakly-biased population, which is a different
physical quantity from what a single-tracer LPT bias expansion predicts.
Consequences of that change here:
  * get_pairwise_hist is called once PER MASS BIN (n_M=1 each time), not
    once per catalogue, and the sub-box files are rebuilt per bin from that
    bin's own halos.
  * N_NEIGHBOUR / M_tracer are gone entirely -- there is no separate tracer
    pool any more.
  * output is one file per (sim, z, bin): pairhist_auto_mbin{k}.h5, so
    RESUME works per bin (important: 52 cosmologies x 8 redshifts x 6 bins
    is 2496 individual measurements).
  * a bin with too few halos is SKIPPED and logged, not raised -- at
    z >= 1.4 the high-mass bins will genuinely run dry, and that should not
    kill the rest of the job.

These catalogues differ from the earlier halo_combined.h5 files:
  * columns are 'mass', 'x'..'vz', 'rvir', 'cvir', 'vrms', 'gid'
  * there is NO particle-count column, so the particle-count bins are applied
    as mass edges recomputed per cosmology from that cosmology's m_p
  * the redshift directory is 'z0p500', not 'z0.500'
  * cosmology comes from the 'base_c<NNN>' file one level above the sim
    directory, in the omega = Omega*h^2 convention -- Omega_M is DERIVED:
        Omega_M = (omega_b + omega_cdm + omega_ncdm) / h^2

I have not run this -- no cluster access -- so treat it as a
structurally-complete draft to check over.
"""

import gc
import glob
import os
import re
import shutil
import sys
import time
import traceback

import numpy as np
import h5py as h5
from joblib import Parallel, delayed
from scipy.spatial import cKDTree

# compact/velocities/pairhist_direct.py does a bare `from pairhist import
# PairHist` (its line 28), which only resolves if compact/velocities is
# itself on sys.path.  Put it there BEFORE importing pairhist_direct.
#   Why sys.path and not the sys.modules alias used elsewhere in this
#   project: get_pairwise_hist runs its workers through joblib's loky
#   backend, which SPAWNS fresh interpreters.  Those children are handed
#   the parent's sys.path as part of multiprocessing's bootstrap data, but
#   never its sys.modules -- so a sys.modules alias fixes the parent and
#   still dies inside every worker, while a sys.path entry fixes both.
import compact.velocities as _compact_velocities
_CV_DIR = os.path.dirname(_compact_velocities.__file__)
if _CV_DIR not in sys.path:
    sys.path.insert(0, _CV_DIR)

from compact.velocities.pairhist import PairHist
from compact.velocities.pairhist_direct import get_pairwise_hist
from compact.catalog.accumulate import (generate_sub_box_ids, split_simulation_into_sub_boxes,
                                        relative_coordinates, get_zw13_vr_vt)

import psutil
psutil.Process().nice(19)

# ---------------------------------------------------------------- config ----
SIM_NUMS = list(range(130, 182))          # 52 cosmologies, ph000 only

HALO_TMPL = ('/spiffball/salcedo/AbacusSummit/Snapshots/{cosmo}/'
             'base_{sim}/{zdir}/halo_base_{sim}_{zdir}.h5')

# Cosmology file: 'base_c131' inside 'Snapshots/base_c131/'.  The per-redshift
# cosmo_<sim>_<zdir>.param is used as a fallback -- it is not always present.
PARAM_DIR = '/spiffball/salcedo/AbacusSummit/Snapshots/{cosmo}'

# The authoritative source: one .param file per sim, pulled from
# AbacusSummit's own published cosmology table by
# fetch_abacus_cosmologies.py, rather than relying on a per-sim file that
# may or may not exist under salcedo's tree. Checked FIRST, since it's
# always present once generated and doesn't depend on which redshift
# happens to have a param file sitting beside its catalogue.
PULLED_PARAM_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base/halos/'
                     'cosmo_grid/{sim}/base_c{n}.param')
PARAM_GLOBS = ['cosmo_*.param', '*.param', 'header', '*header*']

# Include massive neutrinos in the density used for m_p?
# VERIFIED against the AbacusSummit suite paper (Maksimova et al. 2021,
# arXiv:2110.11398 Sec 2.2): "We model massive neutrinos as a smooth
# component, including their effect in the Hubble expansion rate but
# ignoring the gravitational forces from clustered neutrinos."  The
# particles therefore carry CDM + baryons only, so Omega_cb is the
# consistent density for m_p, and this is False (the original file had it
# True, contradicting its own comment).
#   Numerical check for fiducial c000 (omega_b=0.02237, omega_c=0.12,
#   h=0.6736, Omega_nu=0.00064420):
#       Omega_cb        = 0.313772 -> m_p = 2.109e9
#       Omega_m incl nu = 0.314416 -> m_p = 2.114e9
#   Both round to the documented M_part = 2.1e9 Msun/h, so this choice is
#   a 0.2% effect -- real but small.  (Papers quoting "2e9" are rounding
#   the same number.)
OMEGA_M_INCLUDES_NCDM = False

OUT_DIR = ('/spiffball/cosweeney/simulations/AbacusSummit_base/halos/'
           'cosmo_grid/{sim}/{zdir}/data/')
# per-bin sub-box scratch, RAM-backed: these are transient and much smaller
# than the old full-tracer staging, and /dev/shm avoids the repeated
# network round-trips load_halos makes when reading neighbour sub-boxes.
# Check `df -h /dev/shm` before a long run.
SB_TMPL = '/dev/shm/{sim}_{zdir}_bin{k}_SBp_auto/'
LOG_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base/logs/'
            'cosmo_grid/pairhist_auto_{sim}_{zdir}.log')
OUT_NAME_TMPL = 'pairhist_auto_mbin{k}.h5'

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

# The same two mass bins the cross-statistic version used: massbins[1] and
# massbins[3] of the original six-bin list, i.e. one low and one high bin.
BIN_LOGM = np.array([[12.5, 12.6], [13.5, 13.6]])

OM_FID = 0.3152
M_PART_FID = 27.75e10 * OM_FID * box_length**3 / N_PART**3
BIN_NPART = np.rint(10**BIN_LOGM / M_PART_FID).astype(np.int64)

V_LO, V_HI, N_V = -4000.0, 4000.0, 400
CONV = 1.0                        # verified by the sigma_1D guard below
VEL_CHECK = (50.0, 3000.0)

# Minimum halos in a bin for that bin to be measured at all.  Set to 100 to
# match gather_auto_stats.py's own MIN_HALOS, NOT the old 2000: auto-
# statistics are deliberately sparser than the cross-statistics this floor
# was originally tuned for, and the point of the grid is to reach the
# high-mass/high-z corners where counts are low.  Every bin's halo count is
# logged regardless of whether it passes, so a bin skipped here is visible
# in the log rather than silent.
MIN_HALOS = 100

# Catalogue reads are CPU-bound on gzip decompression, not I/O: the files
# are compound tables (so a mass-first two-stage read saves nothing --
# measured at 119.3 s vs 121.0 s) compressed in ~6000-row independent
# chunks.  Reading row ranges in parallel processes, and selecting inside
# each worker, measured ~35x faster on two separate cosmologies (121 s ->
# ~3.5 s), including one that had never been read (so not a page-cache
# artefact).  It also avoids materializing the ~19 GB of full-length
# pos/vel arrays the previous serial path built and then discarded 99.4%
# of.  56 was the best of (8, 16, 32, 56); still improving slightly there,
# but 3.5 s is already negligible.
READ_N_JOBS = int(os.environ.get('PAIRHIST_READ_NJOBS', 56))

# Pair search: in-memory (method B) instead of the sub-box file machinery.
# That machinery exists so workers never hold the full catalogue, but a
# single narrow mass bin is tiny (the biggest here is ~2M halos = 24 MB of
# positions), so every worker can hold it and the sub-box build becomes
# pure overhead.  Measured on c130_ph000 z0p500: the build cost 43.6 s for
# bin 0 and 31.4 s for bin 1 -- 29% of the whole job, and for bin 1 it was
# 31.4 s of setup to support 12 s of actual search, because
# chunksize = max(n//1000, 1) forces 1000 chunks however small the bin is.
# Set USE_INMEM = False to fall back to the sub-box path.
USE_INMEM = True
# 56 was the best of (8, 16, 32, 56) in the benchmark; 112 was SLOWER.
SEARCH_N_JOBS = int(os.environ.get('PAIRHIST_SEARCH_NJOBS', 56))
INMEM_FLUSH = 5_000_000          # pairs buffered per worker before binning
RESUME = True
KEEP_SUBBOXES = False
N_JOBS = int(os.environ.get('PAIRHIST_NJOBS', 112))
TASKS_PER_SUB_BOX = int(os.environ.get('PAIRHIST_TASKS_PER_SB', 1))

# --- cost model ------------------------------------------------------------
# Same per-pair/per-neighbour constants as the cross-statistic version, but
# applied to each BIN'S OWN number density rather than the full tracer pool.
# Auto-statistics are far cheaper: pair work scales roughly as n_bar_bin *
# n_seeds, and n_bar_bin is orders of magnitude below the old tracer n_bar
# for the high-mass bins.  These constants were calibrated on the
# cross-statistic c000_ph000 run, so treat the per-bin numbers as rough.
T_PER_NBR = 2.76e-7
T_PER_PAIR = 2.63e-7
OVERHEAD = 1.48
T_READ = 120.0
T_SUBBOX_PER_1E6 = 9.4
T_REDUCE = 20.0                   # per bin now, not per catalogue
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


def out_path(n, z, k):
    return os.path.join(OUT_DIR.format(sim=sim_name(n), zdir=zdir(z)),
                        OUT_NAME_TMPL.format(k=k))


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


def _omega_for_mpart(p):
    """The Omega that sets the particle mass: Omega_cb by default, since
    AbacusSummit's particles carry CDM + baryons only and treat massive
    neutrinos as a smooth background."""
    if 'h' in p and ('omega_cdm' in p or 'omega_b' in p):
        w = p.get('omega_b', 0.0) + p.get('omega_cdm', 0.0)
        if OMEGA_M_INCLUDES_NCDM:
            w += p.get('omega_ncdm', 0.0)
        return w / p['h']**2
    for key in ('Omega_M', 'Omega_m', 'OmegaM'):
        if key in p:
            return p[key]
    return None


def read_params(n, z):
    """Full cosmology dict for cosmology n, with 'Omega_M' derived.

    Tries the pulled table (fetch_abacus_cosmologies.py's output) first --
    one file per sim, always present once generated, not tied to any
    particular redshift's catalogue directory. Falls back to
    Snapshots/base_c<NNN>/base_c<NNN>, then the per-redshift
    cosmo_<sim>_<zdir>.param beside the catalogue, for sims this hasn't
    been run for yet.
    """
    if n in _OM_CACHE:
        return _OM_CACHE[n]

    tried = []
    pulled = PULLED_PARAM_TMPL.format(sim=sim_name(n), n=n)
    cands = [pulled] if os.path.isfile(pulled) else []
    tried.append(pulled)

    primary = os.path.join(PARAM_DIR.format(cosmo=cosmo_dir(n)), cosmo_dir(n))
    if os.path.isfile(primary):
        cands.append(primary)
    tried.append(primary)

    d = os.path.dirname(halo_path(n, z))
    cands += _candidate_params(d, param_filename(n, z))
    tried.append(os.path.join(d, param_filename(n, z)))

    for path in cands:
        p = _parse_kv(path)
        om = _omega_for_mpart(p)
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


CAT_COLS = ['mass', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'gid']


def _inmem_chunk(pos_all, vel_all, seed_idx, r_max_, rlos_max_, boxsize,
                 R_edges_, rlos_edges_, v_lo, v_hi, n_v, conv):
    """One worker of the in-memory search: build a tree over the whole
    (small) bin, then process this chunk of seeds.  No file I/O at all --
    joblib memmaps pos_all/vel_all, so the arrays are shared rather than
    copied per worker.

    Identical physics to pairhist_direct's own per-sub-box worker: same
    relative_coordinates, same r_perp/r_los cuts, same get_zw13_vr_vt, same
    conv applied to both v_los and v_r at binning time.
    """
    tree = cKDTree(pos_all, boxsize=boxsize)
    h = PairHist(1, R_edges_, rlos_edges_, v_lo, v_hi, n_v)
    r_query = float(np.hypot(r_max_, rlos_max_))

    mb, mR, ml, mv, mvr, nbuf = [], [], [], [], [], 0
    for i in seed_idx:
        nb = tree.query_ball_point(pos_all[i], r_query, return_sorted=False)
        if not nb:
            continue
        nb = np.asarray(nb)
        rel_pos = relative_coordinates(pos_all[i], pos_all[nb], boxsize)
        R = np.sqrt(rel_pos[:, 0] ** 2 + rel_pos[:, 1] ** 2)
        keep = (R <= r_max_) & (R > 0) & (np.abs(rel_pos[:, 2]) <= rlos_max_)
        if not keep.any():
            continue
        rel_pos = rel_pos[keep]
        rel_vel = vel_all[nb[keep]] - vel_all[i]
        vr, _ = get_zw13_vr_vt(rel_pos, rel_vel)

        nk = int(keep.sum())
        mb.append(np.zeros(nk, dtype=np.int32))
        mR.append(R[keep].astype(np.float32))
        ml.append(rel_pos[:, 2].astype(np.float32))
        mv.append(rel_vel[:, 2].astype(np.float32))
        mvr.append(vr.astype(np.float32))
        nbuf += nk
        if nbuf >= INMEM_FLUSH:
            h.add(np.concatenate(mb), np.concatenate(mR), np.concatenate(ml),
                  conv * np.concatenate(mv), conv * np.concatenate(mvr))
            mb, mR, ml, mv, mvr, nbuf = [], [], [], [], [], 0

    if nbuf:
        h.add(np.concatenate(mb), np.concatenate(mR), np.concatenate(ml),
              conv * np.concatenate(mv), conv * np.concatenate(mvr))
    return h


def get_pairwise_hist_inmem(pos_k, vel_k, n_jobs=None):
    """Method B: auto-pairwise PairHist for one already-mass-selected
    population, with no sub-box files.  Drop-in replacement for the
    get_pairwise_hist call in run_one_bin.

    Each worker returns a PairHist (~59 MB of counts here), so ~3.3 GB
    comes back across 56 workers -- that cost is included in the benchmark
    that showed this path faster overall, but it is the reason for not
    simply raising n_jobs further.
    """
    n_jobs = n_jobs or SEARCH_N_JOBS
    chunks = [c for c in np.array_split(np.arange(len(pos_k)), n_jobs) if c.size]
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_inmem_chunk)(pos_k, vel_k, c, r_max, RLOS_MAX, box_length,
                              R_edges, rlos_edges, V_LO, V_HI, N_V, CONV)
        for c in chunks)
    total = parts[0]
    for p in parts[1:]:
        total += p
    return total


def _read_slab(hpath, key, i0, i1, m_edges):
    """One worker: decompress its own row range, keep only the rows landing
    in any mass bin, and return those plus the running v_z moments over the
    WHOLE slab -- so the sigma_1D guard still sees the full population, not
    just the selected halos."""
    with h5.File(hpath, 'r') as hdf:
        sub = hdf[key][i0:i1]
    vz = sub['vz'].astype(np.float64)
    moments = (len(sub), float(vz.sum()), float((vz ** 2).sum()))
    del vz

    mass = sub['mass']
    keep = np.zeros(len(mass), dtype=bool)
    for lo, hi in m_edges:
        keep |= (mass >= lo) & (mass <= hi)
    if not keep.any():
        return None, moments
    sel = sub[keep]
    return dict((nm, np.ascontiguousarray(sel[nm])) for nm in CAT_COLS), moments


def read_catalogue_selected(hpath, m_edges, n_jobs=None):
    """Parallel, selective catalogue read.  Returns (columns, sigma_1D,
    n_rows_total): only the halos inside any mass bin, plus the 1-D
    velocity dispersion of the full catalogue for the CONV guard.

    Slabs are aligned to the file's own chunk boundaries -- a slab starting
    mid-chunk would force that chunk to be decompressed twice.
    """
    n_jobs = n_jobs or READ_N_JOBS
    with h5.File(hpath, 'r') as hdf:
        key = next(k for k in hdf.keys()
                   if getattr(hdf[k], 'dtype', None) is not None
                   and hdf[k].dtype.names
                   and all(nm in hdf[k].dtype.names for nm in CAT_COLS))
        d = hdf[key]
        n_rows = d.shape[0]
        chunk = d.chunks[0] if d.chunks else 1

    chunks_total = int(np.ceil(n_rows / chunk))
    bounds = np.linspace(0, chunks_total, n_jobs + 1).astype(int) * chunk
    bounds[-1] = n_rows
    slabs = [(int(bounds[i]), int(bounds[i + 1])) for i in range(n_jobs)
             if bounds[i + 1] > bounds[i]]

    parts = Parallel(n_jobs=n_jobs)(
        delayed(_read_slab)(hpath, key, i0, i1, m_edges) for i0, i1 in slabs)

    n_tot = sum(m[0] for _, m in parts)
    s1 = sum(m[1] for _, m in parts)
    s2 = sum(m[2] for _, m in parts)
    mean_vz = s1 / max(n_tot, 1)
    sig1d = CONV * float(np.sqrt(max(s2 / max(n_tot, 1) - mean_vz ** 2, 0.0)))

    cols = [c for c, _ in parts if c is not None]
    if not cols:
        return dict((nm, np.empty(0)) for nm in CAT_COLS), sig1d, n_tot
    out = dict((nm, np.concatenate([c[nm] for c in cols])) for nm in CAT_COLS)
    return out, sig1d, n_tot


def hpath_safe(n, z):
    """halo_path, but raising OSError (not returning a bad path) when the
    catalogue is absent, so estimate() can report it as missing."""
    p = halo_path(n, z)
    if not os.path.exists(p):
        raise OSError('missing: %s' % p)
    return p


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
def predict_bin_seconds(n_seeds):
    """Auto-statistics: the neighbour pool IS the bin, so both the tree
    query and the pair work scale with this bin's own number density."""
    if n_seeds <= 0:
        return 0.0
    n_bar = n_seeds / box_length**3
    r_q = np.hypot(r_max, RLOS_MAX)
    n_nbr = n_bar * (4.0 / 3.0) * np.pi * r_q**3
    n_cyl = n_bar * np.pi * r_max**2 * 2.0 * RLOS_MAX
    t_seed = OVERHEAD * (T_PER_NBR * n_nbr + T_PER_PAIR * n_cyl)
    t_pairs = n_seeds * t_seed / N_EFFECTIVE_CORES
    t_fixed = T_REDUCE + T_SUBBOX_PER_1E6 * n_seeds / 1e6
    return t_pairs + t_fixed


def estimate(nums, zs, n_sample=None):
    jobs = [(n, z) for n in nums for z in zs]
    sample = jobs if n_sample is None else jobs[:int(n_sample)]
    print('AUTO-statistics: neighbour pool = the mass bin itself')
    print('query radius %.1f (r_max %g, rlos_max %g)'
          % (np.hypot(r_max, RLOS_MAX), r_max, RLOS_MAX))
    print('scanning %d of %d jobs (mass column only)\n'
          % (len(sample), len(jobs)))
    header = '%14s %6s' % ('sim', 'z')
    header += ''.join('%12s' % ('bin%d' % k) for k in range(len(BIN_NPART)))
    header += ' %9s' % 'total'
    print(header)

    rows, missing = [], []
    for n, z in sample:
        try:
            om0 = read_omega_m(n, z)
            m_p = particle_mass(om0)
            m_edges = [(lo * m_p, hi * m_p) for lo, hi in BIN_NPART]
            # same parallel selective read as the real run -- a serial full
            # read here would cost ~120 s PER SAMPLED JOB just to count halos
            col, _, _ = read_catalogue_selected(hpath_safe(n, z), m_edges)
            mass = col['mass']
            del col
        except (OSError, KeyError, StopIteration) as e:
            missing.append((n, z, '%s: %s' % (type(e).__name__, str(e)[:60])))
            continue
        seeds = [int(((mass >= lo) & (mass <= hi)).sum()) for lo, hi in m_edges]
        tot = T_READ + sum(predict_bin_seconds(s) if s >= MIN_HALOS else 0.0
                           for s in seeds)
        rows.append(tot)
        line = '%14s %6.2f' % (sim_name(n), z)
        line += ''.join('%12s' % format(s, ',') for s in seeds)
        line += ' %8.0fm' % (tot / 60)
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
    print('  (constants calibrated on the CROSS-statistic ph000 run; the '
          'per-bin auto numbers are extrapolations, so re-check after the '
          'first real job)')


# ----------------------------------------------------------- one job -------
def run_one_bin(n, z, k, mass, pos, vel, hid, m_p, par):
    """Auto-statistics for ONE mass bin: the sub-box pool is built from this
    bin's halos only, so the neighbours found are bin members too."""
    sim, zd = sim_name(n), zdir(z)
    op = out_path(n, z, k)
    if RESUME and os.path.exists(op):
        log('%s %s bin %d: exists, skipping' % (sim, zd, k))
        return True

    lo_np, hi_np = BIN_NPART[k]
    M_lo, M_hi = lo_np * m_p, hi_np * m_p
    sel = (mass >= M_lo) & (mass <= M_hi)
    n_seeds = int(sel.sum())
    logM_eff = float(np.mean(np.log10(mass[sel]))) if n_seeds else float('nan')

    log('%s %s bin %d: %s-%s particles = [%.3e, %.3e] Msun/h, %s halos, '
        '<log10 M> = %.4f'
        % (sim, zd, k, format(lo_np, ','), format(hi_np, ','), M_lo, M_hi,
           format(n_seeds, ','), logM_eff))

    if n_seeds < MIN_HALOS:
        log('%s %s bin %d: %s halos < MIN_HALOS=%d, skipping this bin'
            % (sim, zd, k, format(n_seeds, ','), MIN_HALOS))
        return False

    pos_k = np.ascontiguousarray(pos[sel])
    vel_k = np.ascontiguousarray(vel[sel])
    hid_k = hid[sel]

    sb_path = None
    if USE_INMEM:
        t0 = time.time()
        h = get_pairwise_hist_inmem(pos_k, vel_k)
    else:
        seed_mbin = np.zeros(n_seeds, dtype=np.int32)   # already mass-restricted
        sb_path = SB_TMPL.format(sim=sim, zdir=zd, k=k)
        shutil.rmtree(sb_path, ignore_errors=True)
        os.makedirs(sb_path, exist_ok=True)
        chunksize = max(n_seeds // 1000, 1)
        t_sb = time.time()
        generate_sub_box_ids(pos_k, box_length, subsize, chunksize,
                             sb_path, name='halo')
        split_simulation_into_sub_boxes(
            positions=pos_k, velocities=vel_k, ids=hid_k, upids=hid_k,
            boxsize=box_length, subsize=subsize, chunksize=chunksize,
            dtypes=[pos_k.dtype, vel_k.dtype], path=sb_path, name='halo')
        log('%s %s bin %d: sub-box build %.1f s' % (sim, zd, k, time.time() - t_sb))

        t0 = time.time()
        h = get_pairwise_hist(
            pos_k, vel_k, seed_mbin,
            r_max=r_max, boxsize=box_length, subsize=subsize, path=sb_path,
            n_M=1, R_edges=R_edges, rlos_edges=rlos_edges,
            v_lo=V_LO, v_hi=V_HI, n_v=N_V,
            conv=CONV, n_jobs=N_JOBS, tasks_per_sub_box=TASKS_PER_SUB_BOX,
            rlos_max=RLOS_MAX,
        )

    off = int((h.n - h.counts.sum(-1)).sum())
    occ = h.n[0][h.n[0] > 0]
    _, sd = h.mean_std(0)
    log('%s %s bin %d: %s pairs, off-grid %d (%.1e), cells %d/%d, '
        'min/median cell n %d/%.0f, median sigma %.1f km/s, %.1f min'
        % (sim, zd, k, format(int(h.n.sum()), ','), off,
           off / max(int(h.n.sum()), 1),
           int((h.n[0] > 0).sum()), h.n[0].size,
           int(occ.min()) if occ.size else 0,
           float(np.median(occ)) if occ.size else 0.0,
           float(np.nanmedian(sd)), (time.time() - t0) / 60))

    os.makedirs(os.path.dirname(op), exist_ok=True)
    h.save(op)
    with h5.File(op, 'a') as hdf:
        hdf.attrs.update(sim=sim, cosmo_num=int(n), redshift=float(z),
                         suite='AbacusSummit_base', source='salcedo',
                         boxsize=box_length, subsize=subsize, r_max=r_max,
                         rlos_max=RLOS_MAX, conv=CONV, Om0=par['Omega_M'],
                         m_part=m_p, mass_bin=int(k),
                         bin_npart_lo=int(lo_np), bin_npart_hi=int(hi_np),
                         M_lo=M_lo, M_hi=M_hi, n_halos=n_seeds,
                         logM_eff=logM_eff,
                         statistic='auto (halo-halo, same mass bin for '
                                   'both seed and neighbour)',
                         search_method='in-memory' if USE_INMEM else 'sub-box files')
        # full cosmology, so the emulator design matrix comes straight from
        # the outputs -- omega_b, omega_cdm, h, A_s, n_s, sigma8_m, w0, wa ...
        hdf.attrs.update(dict((kk, vv) for kk, vv in par.items()
                              if not kk.startswith('_')))
        hdf.create_dataset('bin_npart', data=BIN_NPART[k])
        hdf.create_dataset('bin_medges', data=np.array([M_lo, M_hi]))
    log('%s %s bin %d: saved %s' % (sim, zd, k, op))

    if sb_path is not None and not KEEP_SUBBOXES:
        shutil.rmtree(sb_path, ignore_errors=True)

    del pos_k, vel_k, hid_k, h
    gc.collect()
    return True


def run_one(n, z, tag=''):
    global _log_path
    sim = sim_name(n)
    zd = zdir(z)
    _log_path = LOG_TMPL.format(sim=sim, zdir=zd)
    os.makedirs(os.path.dirname(_log_path), exist_ok=True)

    hpath = halo_path(n, z)
    if not os.path.exists(hpath):
        raise FileNotFoundError(hpath)

    if RESUME and all(os.path.exists(out_path(n, z, k))
                      for k in range(len(BIN_NPART))):
        log('%s %s: all %d bins exist, skipping' % (sim, zd, len(BIN_NPART)))
        return

    par = read_params(n, z)
    om0 = par['Omega_M']
    m_p = particle_mass(om0)

    log('=== %s %s %s (auto-statistics) ===' % (sim, zd, tag))
    log('%s %s: Om_for_mpart %.4f (ncdm included: %s, h %.4f, s8_m %s), '
        'm_p %.3e Msun/h'
        % (sim, zd, om0, OMEGA_M_INCLUDES_NCDM, par.get('h', float('nan')),
           par.get('sigma8_m', 'n/a'), m_p))
    log('%s %s: params from %s' % (sim, zd, par.get('_param_file', '?')))
    log('%s %s: r_max %g, rlos grid [%g, %g] -> query radius %.1f'
        % (sim, zd, r_max, rlos_edges[0], rlos_edges[-1],
           np.hypot(r_max, RLOS_MAX)))

    log('%s %s: reading %s (parallel, selective, n_jobs=%d)'
        % (sim, zd, hpath, READ_N_JOBS))
    t_read = time.time()
    m_edges = [(lo * m_p, hi * m_p) for lo, hi in BIN_NPART]
    col, sig1d, n_rows_total = read_catalogue_selected(hpath, m_edges)
    mass = col['mass']
    pos = np.array([col['x'], col['y'], col['z']]).T % box_length
    vel = np.array([col['vx'], col['vy'], col['vz']]).T
    hid = col['gid']
    del col
    gc.collect()
    log('%s %s: read %s of %s rows (%.3f%%) in %.1f s'
        % (sim, zd, format(len(mass), ','), format(n_rows_total, ','),
           100.0 * len(mass) / max(n_rows_total, 1), time.time() - t_read))

    # sigma_1D is computed over the FULL catalogue inside the readers, not
    # just the selected halos, so this guard means the same thing it did
    # when the whole catalogue was loaded.
    log('%s %s: 1-D velocity dispersion %.1f km/s (CONV = %g, full catalogue)'
        % (sim, zd, sig1d, CONV))
    if not (VEL_CHECK[0] <= sig1d <= VEL_CHECK[1]):
        raise ValueError('%s %s: sigma_1D = %.3e km/s outside %s -- check CONV '
                         'and the velocity column' % (sim, zd, sig1d, VEL_CHECK))

    # halo counts for EVERY bin up front, whether or not each one ends up
    # being measured -- these are the numbers you want when judging how far
    # into the high-mass / high-z corners the grid actually reaches.
    log('%s %s: halos per bin (m_p = %.4e Msun/h):' % (sim, zd, m_p))
    for k in range(len(BIN_NPART)):
        lo_np, hi_np = BIN_NPART[k]
        sel_k = (mass >= lo_np * m_p) & (mass <= hi_np * m_p)
        n_k = int(sel_k.sum())
        logM_k = float(np.mean(np.log10(mass[sel_k]))) if n_k else float('nan')
        log('%s %s:   bin %d [%s-%s part = %.3e-%.3e Msun/h]: %s halos, '
            '<log10 M> = %.4f%s'
            % (sim, zd, k, format(lo_np, ','), format(hi_np, ','),
               lo_np * m_p, hi_np * m_p, format(n_k, ','), logM_k,
               '' if n_k >= MIN_HALOS else '  <-- below MIN_HALOS, will skip'))

    n_done = 0
    for k in range(len(BIN_NPART)):
        if run_one_bin(n, z, k, mass, pos, vel, hid, m_p, par):
            n_done += 1
    log('%s %s: %d/%d bins produced output' % (sim, zd, n_done, len(BIN_NPART)))

    del pos, vel, hid, mass
    gc.collect()


# -------------------------------------------------------------- batch -----
def run_batch(nums, zs):
    jobs = [(n, z) for n in nums for z in zs]
    failed, t0 = [], time.time()
    for i, (n, z) in enumerate(jobs, 1):
        sim = sim_name(n)
        if RESUME and all(os.path.exists(out_path(n, z, k))
                          for k in range(len(BIN_NPART))):
            print('%s z%s: all bins exist, skipping (%d/%d)'
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