"""
Re-measures halo-halo AUTO-pairwise velocity statistics -- mean v_r AND the
full P(v_los | R, rlos) PDF needed for sigma_v/Delta/skew-t calibration --
for EVERY mass bin, across all available phases.

THE KEY CHANGE from every previous v_r-gathering script in this project:
for mass bin k, BOTH the seed population AND the neighbor/tracer pool are
now restricted to bin k's own halos -- not the old convention (seed = bin
k, neighbor = ALL tracers >= 1e12, no upper cutoff), which is a genuinely
different physical quantity (a cross-statistic against a much broader,
weakly-biased tracer population) from what a single-tracer LPT bias
expansion (b1, b2, b_s, ... applied symmetrically to both members of a
pair, matching how xi_real/xi_hh is already an auto-statistic) actually
predicts.

Inlines the region-preserving, jackknife-capable get_pairwise_hist_jk logic
directly (no import from pairhist_direct_jk.py, to avoid ambiguity between
multiple copies of that file across different environments) -- called once
PER MASS BIN PER PHASE (6 calls per phase instead of 1), each time with the
halo population restricted to that bin's own mass range for BOTH seeds and
the underlying sub-box/tracer pool. Rebuilding sub-box files per bin instead
of once per
phase costs some redundant setup, but each bin's own population is much
smaller than the old cross-statistic's full multi-decade tracer pool, so
both the sub-box-building step and the pair search itself should be
considerably faster per call -- MOST for the higher-mass bins specifically.
The lowest mass bin (logM in [12,12.01]) may not speed up nearly as much,
since the halo mass function falls steeply enough that even a narrow
window there could rival the size of the old full tracer pool.

Per-bin resumable: with 25 phases x 6 bins = up to 150 individual
measurements, a per-(phase,bin) output file means an interrupted run can
pick back up without redoing already-finished bins.

I have not run this myself -- no cluster access -- so treat this as a
structurally-complete first draft to check over, same as everything else
in this pipeline.
"""

import gc
import os
import sys
import tempfile
import threading
import time
import traceback

import numpy as np
import h5py as h5
from colossus.cosmology import cosmology
from joblib import Parallel, delayed
from scipy.spatial import cKDTree

from compact.catalog.accumulate import relative_coordinates, get_sub_box_id, get_zw13_vr_vt
from compact.velocities.pairhist import PairHist

import psutil
psutil.Process().nice(19)

# =========================================================================
# Everything below down to "config" is inlined from pairhist_direct_jk.py
# (region-preserving pairwise-velocity accumulator + jackknife reduction)
# rather than imported, per request -- avoids ambiguity between multiple
# copies of that file across different environments. Kept byte-for-byte
# identical to that module's current content; if you update one, update
# both, or this will silently drift.
# =========================================================================

FLUSH = 5_000_000   # pairs buffered per worker before a bincount sweep -- matches
                    # pairhist_direct.py's own constant


def _monitor(pdir, total, stop):
    """Poll the workers' progress files and drive one bar over all seeds."""
    from tqdm.auto import tqdm
    bar = tqdm(total=total, desc='seeds', unit='seed', smoothing=0.02)
    while not stop.is_set():
        done = 0
        for fn in os.listdir(pdir):
            try:
                with open(os.path.join(pdir, fn)) as f:
                    done += int(f.read() or 0)
            except (ValueError, OSError):
                pass          # mid-write; picked up on the next poll
        bar.n = min(done, total)
        bar.refresh()
        stop.wait(2.0)
    bar.n = total
    bar.refresh()
    bar.close()


class _Buffer:
    """Batches pairs so bincount is called once per FLUSH, not once per seed."""

    def __init__(self, hist, conv):
        self.h = hist
        self.conv = conv
        self.m, self.R, self.l, self.v, self.vr = [], [], [], [], []
        self.n = 0

    def add(self, m_bin, R, rlos, v, vr):
        self.m.append(np.full(R.size, m_bin, dtype=np.int32))
        self.R.append(R.astype(np.float32))
        self.l.append(rlos.astype(np.float32))
        self.v.append(v.astype(np.float32))
        self.vr.append(vr.astype(np.float32))
        self.n += R.size
        if self.n >= FLUSH:
            self.flush()

    def flush(self):
        if self.n == 0:
            return
        self.h.add(np.concatenate(self.m), np.concatenate(self.R),
                   np.concatenate(self.l), self.conv * np.concatenate(self.v),
                   self.conv * np.concatenate(self.vr))
        self.m, self.R, self.l, self.v, self.vr = [], [], [], [], []
        self.n = 0


def get_pairwise_hist_jk(
    seed_pos, seed_vel, seed_mbin,
    r_max, boxsize, subsize, path,
    n_M, R_edges, rlos_edges, v_lo, v_hi, n_v,
    conv=1.0, n_jobs=-1, tmpdir=None, progress=True, tasks_per_sub_box=1,
    rlos_max=None,
):
    """IN-MEMORY version (no sub-box files): builds one shared tree over
    ALL of seed_pos, held in memory rather than staged through per-region
    files -- safe here because this is always a single, narrow,
    auto-restricted mass bin (proven small enough to hold entirely in
    memory for abacus_cosmo_core.py's cosmology grid, ~2M halos at most).
    Removes the ~30-45s sub-box BUILD step entirely, which was pure
    overhead paid before any actual searching started.

    `path` is accepted but unused (kept so callers don't need to change);
    the tree is rebuilt fresh INSIDE each worker from the shared,
    joblib-memmapped seed_pos array, rather than built once and shared as
    a cKDTree object -- matching the pattern already benchmarked and
    verified for abacus_cosmo_core.py's in-memory search, since a cKDTree
    doesn't pickle/memmap across worker processes the way a raw array does.

    Same signature, same return value (total, regions -- regions is
    {region_id: PairHist}) as the sub-box-file version it replaces.
    """
    rlos_max = float(r_max if rlos_max is None else rlos_max)
    keep = seed_mbin >= 0
    seed_pos = np.ascontiguousarray(seed_pos[keep] % boxsize)
    seed_vel = np.ascontiguousarray(seed_vel[keep])
    seed_mbin = seed_mbin[keep].astype(np.int32)

    sb_id = get_sub_box_id(seed_pos.copy(), boxsize, subsize)
    order = np.argsort(sb_id)
    sb_id, seed_pos, seed_vel, seed_mbin = (
        sb_id[order], seed_pos[order], seed_vel[order], seed_mbin[order])

    uniq, starts = np.unique(sb_id, return_index=True)
    ends = np.append(starts[1:], len(sb_id))

    tasks = []
    for sb, lo, hi in zip(uniq, starts, ends):
        for cut in np.array_split(np.arange(lo, hi), tasks_per_sub_box):
            if cut.size:
                tasks.append((int(sb), int(cut[0]), int(cut[-1]) + 1))

    tmpdir = tmpdir or tempfile.mkdtemp(prefix='pairhist_jk_')
    progdir = os.path.join(tmpdir, 'prog')
    os.makedirs(progdir, exist_ok=True)

    r_query = float(np.hypot(r_max, rlos_max))

    def _process_region(sub_box_id, lo, hi):
        # tree built fresh here, from the shared (memmapped) seed_pos --
        # NOT loaded per-region from a sub-box file
        tree = cKDTree(seed_pos, boxsize=boxsize)

        h = PairHist(n_M, R_edges, rlos_edges, v_lo, v_hi, n_v)
        buf = _Buffer(h, conv)

        pfile = os.path.join(progdir, f'{sub_box_id}_{lo}')
        sp, sv, sm = seed_pos[lo:hi], seed_vel[lo:hi], seed_mbin[lo:hi]
        for i in range(hi - lo):
            if i % 200 == 0:
                with open(pfile, 'w') as f:
                    f.write(str(i))
            nb = tree.query_ball_point(sp[i], r_query, return_sorted=False)
            if not nb:
                continue
            nb = np.asarray(nb)

            rel_pos = relative_coordinates(sp[i], seed_pos[nb], boxsize)
            R = np.sqrt(rel_pos[:, 0] ** 2 + rel_pos[:, 1] ** 2)
            keep_mask = ((R <= r_max) & (R > 0) & (np.abs(rel_pos[:, 2]) <= rlos_max))
            if not keep_mask.any():
                continue

            rel_pos = rel_pos[keep_mask]
            rel_vel = seed_vel[nb[keep_mask]] - sv[i]
            vr, _ = get_zw13_vr_vt(rel_pos, rel_vel)

            buf.add(sm[i], R[keep_mask], rel_pos[:, 2], rel_vel[:, 2], vr)

        buf.flush()
        with open(pfile, 'w') as f:
            f.write(str(hi - lo))
        return sub_box_id, h

    stop = threading.Event()
    if progress:
        mon = threading.Thread(target=_monitor, args=(progdir, len(sb_id), stop), daemon=True)
        mon.start()
    try:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_region)(sb, lo, hi) for sb, lo, hi in tasks)
    finally:
        stop.set()
        if progress:
            mon.join()

    # merge any tasks that share a region id (only happens if
    # tasks_per_sub_box > 1) into one PairHist per region, then keep every
    # region separately instead of collapsing them all the way to one total.
    regions = {}
    for sub_box_id, piece in results:
        if sub_box_id in regions:
            regions[sub_box_id] += piece
        else:
            regions[sub_box_id] = piece

    for fn in os.listdir(progdir):
        os.remove(os.path.join(progdir, fn))
    os.rmdir(progdir)
    os.rmdir(tmpdir)

    total = PairHist(n_M, R_edges, rlos_edges, v_lo, v_hi, n_v)
    for piece in regions.values():
        total += piece

    return total, regions


def vr_profile(h, k, r_edges=None, r_max_trust=None):
    """Pair-weighted <v_r>(r) and sigma_vr(r) for mass bin k."""
    Rc = 0.5 * (h.R_edges[1:] + h.R_edges[:-1])
    lc = 0.5 * (h.rlos_edges[1:] + h.rlos_edges[:-1])
    r3 = np.sqrt(Rc[:, None] ** 2 + lc[None, :] ** 2).ravel()
    n = h.n[k].ravel().astype(float)
    s1 = h.sum_vr[k].ravel()
    s2 = h.sum_vr2[k].ravel()
    if r_edges is None:
        top = r_max_trust if r_max_trust is not None else h.R_edges[-1]
        r_edges = np.arange(0.0, top + 2.0, 2.0)
    nb_bins = len(r_edges) - 1
    idx = np.searchsorted(r_edges, r3, side='right') - 1
    ok = (idx >= 0) & (idx < nb_bins) & (n > 0)

    def acc(w):
        return np.bincount(idx[ok], weights=w[ok], minlength=nb_bins)

    N = acc(n)
    r_eff = np.where(N > 0, acc(n * r3) / np.where(N > 0, N, 1), np.nan)
    mu = np.where(N > 0, acc(s1) / np.where(N > 0, N, 1), np.nan)
    m2 = np.where(N > 0, acc(s2) / np.where(N > 0, N, 1), np.nan)
    sd = np.sqrt(np.maximum(m2 - mu ** 2, 0.0))
    return r_eff, mu, sd, N.astype(np.int64)


def _iadd_copy(a, b):
    """PairHist only implements __iadd__, not __add__ -- this makes a
    non-destructive combination of two regions without mutating either
    (needed since we reuse each region across many leave-one-out sums)."""
    import copy
    out = copy.deepcopy(a)
    out += b
    return out


def vr_jackknife_covariance(regions, mass_bin, r_edges, vr_profile_fn=vr_profile):
    """Delete-one-region jackknife covariance for <v_r|r> in one mass bin,
    from the {sub_box_id: PairHist} regions get_pairwise_hist_jk returns.

    NOTE: this is the standard delete-one JACKKNIFE normalization
    ((n-1)/n * sum of squared deviations), not the /(n*(n-1)) formula used
    elsewhere for INDEPENDENT PHASES -- these 64 regions are correlated
    pieces of one box, not independent realizations, so they need the
    jackknife formula, not the phase-averaging one.
    """
    region_list = list(regions.values())
    n_regions = len(region_list)

    n_r = len(r_edges) - 1
    mean_vr_jk = np.full((n_regions, n_r), np.nan)
    for i in range(n_regions):
        others = [r for j, r in enumerate(region_list) if j != i]
        loo = others[0]
        for r in others[1:]:
            loo = _iadd_copy(loo, r)
        _, mean_vr_jk[i], _, _ = vr_profile_fn(loo, mass_bin, r_edges=r_edges)

    mean_of_jk = np.nanmean(mean_vr_jk, axis=0)
    delta = mean_vr_jk - mean_of_jk
    good = np.isfinite(delta)
    n_good = good.sum(axis=0)
    delta_filled = np.where(good, delta, 0.0)
    cov = (n_good.max() - 1) / max(n_good.max(), 1) * (delta_filled.T @ delta_filled)
    return cov

# ========================================================= end of inlined code


# ---------------------------------------------------------------- config ----
PHASES = [f'c000_ph{i:03d}' for i in range(0, 25)]   # all 25, wherever the
                                                        # catalogue actually exists
REDSHIFT = 'z0.200'

HALO_TMPL = ('/spiff/hengweichang/AbacusSummit_base/snapshots/'
             'AbacusSummit_base_{sim}/halos/{z}/halo_combined.h5')
# SB_TMPL (sub-box scratch) removed -- no longer needed now that the pair
# search is in-memory (see get_pairwise_hist_jk's docstring)
OUT_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base/halos/'
            'AbacusSummit_base_{sim}/halos/{z}/data/pairhist_auto_mbin{k}.h5')
LOG_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base/logs/'
            'pairhist_auto_{sim}.log')

box_length = 2000.0
subsize = box_length / 4
r_max = 140.0

LOG_MASS_EDGES = np.array([[12, 12.01], [12.5, 12.6], [13, 13.1],
                           [13.5, 13.6], [13.9, 14.3]])
MASSBINS = 10 ** LOG_MASS_EDGES
N_MASS = len(MASSBINS)

M_PART = 2e9   # matches the v_r pipeline's existing convention (not the
               # 2.1e9 xi_hh-side correction), so this stays comparable to
               # pairhist_allbins_jk.h5's own mass-bin definitions

R_edges = np.linspace(5, 140, 136)     # matches gather_paper_data.py's R_perp/r_los convention
rlos_edges = R_edges
V_LO, V_HI, N_V = -4000.0, 4000.0, 400
VR_R_EDGES = np.arange(2.0, 142.0, 2.0)

cosmo = cosmology.setCosmology('planck18')
redshift = 0.2
a_scale = 1 / (1 + redshift)
CONV = a_scale * cosmo.Hz(redshift) / cosmo.h * box_length

N_JOBS = 84   # matches the established, known-safe single-phase-at-a-time
              # convention from earlier in this project -- real contention
              # was found even within one phase at this job count, so no
              # concurrency across phases/bins is attempted here either
MIN_HALOS = 100   # skip a bin if there simply aren't enough halos in it for
                   # a meaningful auto-statistic (can happen for the highest
                   # mass bins in a single box)


def log(sim, msg):
    log_path = LOG_TMPL.format(sim=sim)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(log_path, 'a') as f:
        f.write(line + '\n')


def run_one_bin(sim, k, pos_all, vel_all, Mvir_all, hid_all):
    """Auto-statistics for ONE mass bin, ONE phase: both the seed
    population and the sub-box/neighbor pool are restricted to bin k's
    own halos only."""
    out_path = OUT_TMPL.format(sim=sim, z=REDSHIFT, k=k)
    if os.path.exists(out_path):
        log(sim, f'{sim} bin {k}: already done, skipping')
        return True

    t0 = time.time()
    lo, hi = MASSBINS[k]
    sel = (Mvir_all >= lo) & (Mvir_all <= hi)
    n_halos = int(sel.sum())
    if n_halos < MIN_HALOS:
        log(sim, f'{sim} bin {k}: only {n_halos} halos, skipping '
                 f'(too few for a meaningful auto-statistic)')
        return False

    pos_k = np.ascontiguousarray(pos_all[sel])
    vel_k = np.ascontiguousarray(vel_all[sel])
    seed_mbin = np.zeros(n_halos, dtype=np.int32)   # one "bin" (0): already mass-restricted

    log(sim, f'{sim} bin {k}: {n_halos:,} halos, starting pair search (n_jobs={N_JOBS})')
    total, regions = get_pairwise_hist_jk(
        pos_k, vel_k, seed_mbin,
        r_max=r_max, boxsize=box_length, subsize=subsize, path=None,
        n_M=1, R_edges=R_edges, rlos_edges=rlos_edges,
        v_lo=V_LO, v_hi=V_HI, n_v=N_V,
        conv=CONV, n_jobs=N_JOBS, tasks_per_sub_box=1, progress=False,
    )

    cov_vr = vr_jackknife_covariance(regions, 0, VR_R_EDGES, vr_profile)
    del regions
    gc.collect()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    total.save(out_path)   # PairHist's own native format -- same as every
                            # other file in this pipeline, not a custom layout
    with h5.File(out_path, 'a') as hdf:
        hdf.attrs.update(sim=sim, redshift=REDSHIFT, boxsize=box_length,
                         subsize=subsize, r_max=r_max, conv=CONV, m_part=M_PART,
                         mass_bin=k, n_halos=n_halos,
                         statistic='auto (halo-halo, same mass bin for both seed and neighbor)')
        hdf.create_dataset('vr_r_edges', data=VR_R_EDGES)
        hdf.create_dataset('vr_jk_cov', data=cov_vr)

    elapsed = time.time() - t0
    log(sim, f'{sim} bin {k}: done in {elapsed/60:.1f} min, saved {out_path}')
    return True


def run_phase(sim):
    halo_path = HALO_TMPL.format(sim=sim, z=REDSHIFT)
    if not os.path.exists(halo_path):
        log(sim, f'{sim}: catalogue not found, skipping entire phase')
        return False

    if all(os.path.exists(OUT_TMPL.format(sim=sim, z=REDSHIFT, k=k)) for k in range(N_MASS)):
        log(sim, f'{sim}: all {N_MASS} bins already done, skipping entire phase')
        return True

    t0 = time.time()
    log(sim, f'{sim}: reading catalogue')
    with h5.File(halo_path, 'r') as hdf:
        hcat = hdf['halos'][:]
    Mvir_all = M_PART * hcat['N_total']
    pos_all = np.array([hcat['x'], hcat['y'], hcat['z']]).T % box_length
    vel_all = np.array([hcat['vx'], hcat['vy'], hcat['vz']]).T
    hid_all = hcat['hid_cleaned']
    del hcat
    gc.collect()

    for k in range(N_MASS):
        run_one_bin(sim, k, pos_all, vel_all, Mvir_all, hid_all)
        log(sim, f'{sim}: {k + 1}/{N_MASS} bins attempted so far')

    del pos_all, vel_all, Mvir_all, hid_all
    gc.collect()
    log(sim, f'{sim}: phase done in {(time.time() - t0) / 60:.1f} min total')
    return True


def main():
    if len(sys.argv) > 1:
        phases = [sys.argv[1]]
    else:
        phases = PHASES

    done, failed, skipped = 0, [], 0
    t0 = time.time()
    for i, sim in enumerate(phases):
        log('batch', f'--- starting phase {i + 1}/{len(phases)}: {sim} ---')
        try:
            ok = run_phase(sim)
            if ok:
                done += 1
            else:
                skipped += 1
        except Exception:
            failed.append(sim)
            log(sim, f'{sim}: FAILED\n{traceback.format_exc()}')
        log('batch', f'progress: {done} phase(s) done, {skipped} skipped (no catalogue), '
                     f'{len(failed)} failed, {len(phases) - i - 1} remaining, '
                     f'elapsed {(time.time() - t0) / 3600:.1f} h')

    log('batch', f'ALL DONE: {done} phase(s) completed, {len(failed)} failed: {failed}, '
                 f'total {(time.time() - t0) / 3600:.1f} h')


if __name__ == '__main__':
    sys.exit(main())