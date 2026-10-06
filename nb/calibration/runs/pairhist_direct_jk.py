"""
Region-preserving version of compact.velocities.pairhist_direct.get_pairwise_hist.

Does NOT edit that module. _Buffer and _monitor are copied locally below
(both are tiny, ~15-20 lines) rather than imported from it -- importing
compact.velocities.pairhist_direct at all triggers ITS bare
`from pairhist import PairHist` (its line 28), which only resolves in
whatever process first imports it. joblib's default 'loky' backend spawns
genuinely fresh worker processes (not simple fork() copies that would
inherit an already-patched sys.modules), so that import fails inside every
worker even if a sys.modules workaround was applied in the main process.
Avoiding the import entirely sidesteps this rather than trying to make a
workaround propagate across a process-spawn boundary it can't reach. Same
reasoning applies to PairHist itself below: imported via its real
`compact.velocities.pairhist` path, not the bare `pairhist` name.

The per-task search itself -- the actual expensive part (cKDTree build,
neighbor query, velocity moments) -- is UNCHANGED, reusing the same catalog
helper functions the original module also imports. Only the reduction step
differs, so the runtime cost of this version should be close to
get_pairwise_hist's, plus whatever the bookkeeping below adds (a modest
number of extra small-array adds/writes, not more of the expensive search).

Since subsize = box_length/4 in this project's existing convention (see
abacus_cosmo_core.py, run_abacus_batch.py), the 4x4x4 = 64 sub-box grid
already used to PARALLELIZE the search doubles as the jackknife region
grid -- no separate region definition needed. tasks_per_sub_box should stay
at 1 (matching the production scripts) for the region-level breakdown to
correspond 1:1 with sub-boxes; with >1, tasks sharing a sub_box_id just get
merged into that one region below, so it still works, just repeats the
per-slice catalog load/tree build unnecessarily (the original module's own
documented ~13s-per-slice cost).

PairHist only implements __iadd__ (+=), not subtraction -- confirmed by
reading compact/velocities/pairhist.py directly rather than assuming --  so
each leave-one-region-out estimate below is built by summing the OTHER
regions, not by subtracting one from the total.
"""

import os
import tempfile
import threading

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial import cKDTree

from compact.velocities.pairhist import PairHist
from compact.catalog.accumulate import load_halos, relative_coordinates, get_sub_box_id, get_zw13_vr_vt

FLUSH = 5_000_000   # pairs buffered per worker before a bincount sweep -- matches
                    # pairhist_direct.py's own constant


def _monitor(pdir, total, stop):
    """Poll the workers' progress files and drive one bar over all seeds.
    Copied from compact/velocities/pairhist_direct.py -- see module
    docstring for why it's copied rather than imported."""
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
    """Batches pairs so bincount is called once per FLUSH, not once per seed.
    Copied from compact/velocities/pairhist_direct.py -- see module
    docstring for why it's copied rather than imported."""

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
    """Same signature and same per-task search as get_pairwise_hist. Returns
    (total, regions) instead of just total -- regions is {sub_box_id:
    PairHist}, one per jackknife region, still summing to total.
    """
    rlos_max = float(r_max if rlos_max is None else rlos_max)
    keep = seed_mbin >= 0
    seed_pos = seed_pos[keep] % boxsize
    seed_vel = seed_vel[keep]
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

    def _process_sub_box(sub_box_id, lo, hi):
        pos, vel, _, _, _ = load_halos(sub_box_id, boxsize, subsize, path)
        pos = np.ascontiguousarray(pos % boxsize)
        tree = cKDTree(pos, boxsize=boxsize)

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

            rel_pos = relative_coordinates(sp[i], pos[nb], boxsize)
            R = np.sqrt(rel_pos[:, 0] ** 2 + rel_pos[:, 1] ** 2)
            keep_mask = ((R <= r_max) & (R > 0) & (np.abs(rel_pos[:, 2]) <= rlos_max))
            if not keep_mask.any():
                continue

            rel_pos = rel_pos[keep_mask]
            rel_vel = vel[nb[keep_mask]] - sv[i]
            vr, _ = get_zw13_vr_vt(rel_pos, rel_vel)

            buf.add(sm[i], R[keep_mask], rel_pos[:, 2], rel_vel[:, 2], vr)

        buf.flush()
        with open(pfile, 'w') as f:
            f.write(str(hi - lo))
        out = os.path.join(tmpdir, f'sb_{sub_box_id}_{lo}.h5')
        h.save(out)
        return sub_box_id, out

    stop = threading.Event()
    if progress:
        mon = threading.Thread(target=_monitor, args=(progdir, len(sb_id), stop), daemon=True)
        mon.start()
    try:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_sub_box)(sb, lo, hi) for sb, lo, hi in tasks)
    finally:
        stop.set()
        if progress:
            mon.join()

    # merge any tasks that share a sub_box_id (only happens if
    # tasks_per_sub_box > 1) into one PairHist per region, then keep every
    # region separately instead of collapsing them all the way to one total.
    regions = {}
    for sub_box_id, path_ in results:
        piece = PairHist.load(path_)
        os.remove(path_)
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


def vr_jackknife_covariance(regions, mass_bin, r_edges, vr_profile_fn):
    """Delete-one-region jackknife covariance for <v_r|r> in one mass bin,
    from the {sub_box_id: PairHist} regions get_pairwise_hist_jk returns.

    vr_profile_fn must be gather_paper_data.py's vr_profile (passed in
    rather than imported, so this file doesn't hard-depend on that module's
    location) -- reused as-is so the jackknife reduction matches exactly
    how <v_r|r> is computed everywhere else in this project.

    NOTE: this is the standard delete-one JACKKNIFE normalization
    ((n-1)/n * sum of squared deviations), not the /(n*(n-1)) formula used
    elsewhere in gather_paper_data.py for INDEPENDENT PHASES -- these 64
    regions are correlated pieces of one box, not independent realizations,
    so they need the jackknife formula, not the phase-averaging one.
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


def _iadd_copy(a, b):
    """PairHist only implements __iadd__, not __add__ -- this makes a
    non-destructive combination of two regions without mutating either
    (needed since we reuse each region across many leave-one-out sums)."""
    import copy
    out = copy.deepcopy(a)
    out += b
    return out