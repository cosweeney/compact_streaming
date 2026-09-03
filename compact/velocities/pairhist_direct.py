"""Pair search that accumulates into a PairHist instead of returning pairs.

Drop-in replacement for get_pairwise_halo_pw_quants.  Differences:

  * takes seed arrays, not a file path -- the caller reads the catalogue once
    instead of once per mass bin
  * takes a per-seed mass-bin index, so ALL mass bins are done in ONE pair
    search rather than one search per bin
  * neighbours come from a per-sub-box cKDTree (periodic via boxsize=) instead
    of scanning the whole loaded region per seed
  * each sub-box's seeds are split into `tasks_per_sub_box` slices, so there
    are more tasks than cores and joblib can keep every core busy
  * workers write their accumulator to a temp file and return the path, so the
    parent never holds every accumulator at once
  * v_r is computed and accumulated as per-cell moments (sum_vr, sum_vr2);
    v_t is not -- say so if step 5 needs it

"""

import os
import tempfile
import threading

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial import cKDTree

from pairhist import PairHist

from compact.catalog.accumulate import (load_halos, relative_coordinates, get_sub_box_id,  # noqa
                        get_zw13_vr_vt)

FLUSH = 5_000_000        # pairs buffered per worker before a bincount sweep
PROG_EVERY = 200         # seeds between progress-file updates


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


def get_pairwise_hist(
    seed_pos, seed_vel, seed_mbin,
    r_max, boxsize, subsize, path,
    n_M, R_edges, rlos_edges, v_lo, v_hi, n_v,
    conv=1.0, n_jobs=-1, tmpdir=None, progress=True, tasks_per_sub_box=1,
    rlos_max=None,
):
    """seed_mbin : int array, mass-bin index per seed (use -1 to skip a seed).

    rlos_max : half-length of the search cylinder along the line of sight.
        Defaults to r_max.  Set it to rlos_edges[-1] -- pairs beyond the grid
        are binned and discarded, and the query cost goes as the volume of the
        bounding sphere, sqrt(r_max^2 + rlos_max^2)^3.
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

    # slice boundaries so each worker gets only its own seeds
    uniq, starts = np.unique(sb_id, return_index=True)
    ends = np.append(starts[1:], len(sb_id))

    # split each sub-box into several tasks so tasks >> cores.  load_halos and
    # the tree build are repeated per slice (~13 s against ~10 min of work).
    tasks = []
    for sb, lo, hi in zip(uniq, starts, ends):
        for cut in np.array_split(np.arange(lo, hi), tasks_per_sub_box):
            if cut.size:
                tasks.append((int(sb), int(cut[0]), int(cut[-1]) + 1))

    tmpdir = tmpdir or tempfile.mkdtemp(prefix='pairhist_')
    progdir = os.path.join(tmpdir, 'prog')
    os.makedirs(progdir, exist_ok=True)

    # bounding sphere of the cylinder (R <= r_max, |rlos| <= rlos_max)
    r_query = float(np.hypot(r_max, rlos_max))

    def _process_sub_box(sub_box_id, lo, hi):
        pos, vel, _, _, _ = load_halos(sub_box_id, boxsize, subsize, path)
        pos = np.ascontiguousarray(pos % boxsize)   # cKDTree needs [0, boxsize)
        tree = cKDTree(pos, boxsize=boxsize)

        h = PairHist(n_M, R_edges, rlos_edges, v_lo, v_hi, n_v)
        buf = _Buffer(h, conv)

        pfile = os.path.join(progdir, f'{sub_box_id}_{lo}')
        sp, sv, sm = seed_pos[lo:hi], seed_vel[lo:hi], seed_mbin[lo:hi]
        for i in range(hi - lo):
            if i % PROG_EVERY == 0:
                with open(pfile, 'w') as f:
                    f.write(str(i))
            nb = tree.query_ball_point(sp[i], r_query, return_sorted=False)
            if not nb:
                continue
            nb = np.asarray(nb)

            rel_pos = relative_coordinates(sp[i], pos[nb], boxsize)
            R = np.sqrt(rel_pos[:, 0]**2 + rel_pos[:, 1]**2)
            # R <= r_max implies |rel_x|, |rel_y| <= r_max, so only z is extra
            keep = ((R <= r_max) & (R > 0)        # R > 0 drops the self-pair
                    & (np.abs(rel_pos[:, 2]) <= rlos_max))
            if not keep.any():
                continue

            rel_pos = rel_pos[keep]
            # index velocities on the survivors only -- fancy indexing the full
            # neighbour list is the single most expensive step in this loop
            rel_vel = vel[nb[keep]] - sv[i]
            vr, _ = get_zw13_vr_vt(rel_pos, rel_vel)

            buf.add(sm[i], R[keep], rel_pos[:, 2], rel_vel[:, 2], vr)

        buf.flush()
        with open(pfile, 'w') as f:
            f.write(str(hi - lo))
        out = os.path.join(tmpdir, f'sb_{sub_box_id}_{lo}.h5')
        h.save(out)
        return out

    stop = threading.Event()
    if progress:
        mon = threading.Thread(target=_monitor,
                               args=(progdir, len(sb_id), stop), daemon=True)
        mon.start()
    try:
        paths = Parallel(n_jobs=n_jobs)(
            delayed(_process_sub_box)(sb, lo, hi) for sb, lo, hi in tasks)
    finally:
        stop.set()
        if progress:
            mon.join()

    total = PairHist(n_M, R_edges, rlos_edges, v_lo, v_hi, n_v)
    reducing = paths
    if progress:
        from tqdm.auto import tqdm
        reducing = tqdm(paths, desc='reducing', unit='task')
    for p in reducing:
        total += PairHist.load(p)
        os.remove(p)
    for fn in os.listdir(progdir):
        os.remove(os.path.join(progdir, fn))
    os.rmdir(progdir)
    os.rmdir(tmpdir)
    return total