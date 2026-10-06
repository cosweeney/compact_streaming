"""
The sub-box and in-memory searches disagree by ~0.13-0.20% on pair counts
(in-memory finds FEWER).  Neither is obviously right, so this establishes
ground truth by brute force.

For a small random sample of seeds, it computes every pair exactly --
minimum-image separation against the entire bin, no tree, no sub-boxes --
and compares that against what each search path reports for those SAME
seeds.  Whichever matches brute force is correct.

Brute force is O(n_test x n_bin), so n_test stays small; 200 seeds against
~2M halos is a few hundred million distance evaluations, which numpy does
in well under a minute when chunked.

Usage:
    python diagnose_pair_mismatch.py            # defaults below
    python diagnose_pair_mismatch.py 130 0.5 1  # sim, z, bin (bin 1 is faster)
"""

import os
import shutil
import sys
import time

import numpy as np

import nb.calibration.runs.auto_runs.abacus_cosmo_core_auto as core
from compact.velocities.pairhist import PairHist
from compact.velocities.pairhist_direct import get_pairwise_hist
from compact.catalog.accumulate import generate_sub_box_ids, split_simulation_into_sub_boxes

SIM_NUM, Z, K_BIN = 130, 0.5, 1     # bin 1 by default: same physics, far fewer halos
N_TEST = 200
CHUNK = 200_000                     # halos per brute-force chunk


def brute_force_counts(pos_all, vel_all, seed_idx, boxsize, R_edges, rlos_edges):
    """Exact pair count for these seeds, by minimum-image over the whole bin.

    CORRECTED: the first version of this used |d_z| <= rlos_max and no lower
    cut, which counted BOTH line-of-sight signs and every pair inside the
    innermost edge -- roughly a factor of two too many, and the reason it
    reported both real methods as ~52% low.

    The grid is ONE-SIDED: rlos_edges runs +5 to +140, so PairHist.add's
    il = searchsorted(rlos_edges, rlos, 'right') - 1 is negative for every
    rlos < 5 (including all negative rlos) and those pairs are dropped.
    Same on the perpendicular axis.  Rather than restate that as
    inequalities and risk drifting from it again, this applies PairHist's
    own searchsorted logic directly.

    Returns (n_pairs, sum_vr, n_pairs_twosided) -- the last so the factor
    of two stays visible rather than silently assumed.
    """
    n_R, n_l = len(R_edges) - 1, len(rlos_edges) - 1
    n_pairs, sum_vr, n_two = 0, 0.0, 0
    for i in seed_idx:
        sp, sv = pos_all[i], vel_all[i]
        for j0 in range(0, len(pos_all), CHUNK):
            blk = pos_all[j0:j0 + CHUNK]
            d = blk - sp
            d -= boxsize * np.round(d / boxsize)          # minimum image
            R = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
            rlos = d[:, 2]

            # exactly PairHist.add's acceptance test
            iR = np.searchsorted(R_edges, R, side='right') - 1
            il = np.searchsorted(rlos_edges, rlos, side='right') - 1
            keep = (iR >= 0) & (iR < n_R) & (il >= 0) & (il < n_l)

            # same but accepting either sign of rlos, for reference only
            il2 = np.searchsorted(rlos_edges, np.abs(rlos), side='right') - 1
            n_two += int(((iR >= 0) & (iR < n_R) & (il2 >= 0) & (il2 < n_l)).sum())

            if not keep.any():
                continue
            dk = d[keep].astype(np.float64)
            rv = (vel_all[j0:j0 + CHUNK][keep] - sv).astype(np.float64)
            r3 = np.sqrt((dk ** 2).sum(axis=1))
            vr = (dk * rv).sum(axis=1) / r3
            n_pairs += int(keep.sum())
            sum_vr += float(vr.sum())
    return n_pairs, sum_vr, n_two


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else SIM_NUM
    z = float(sys.argv[2]) if len(sys.argv) > 2 else Z
    k = int(sys.argv[3]) if len(sys.argv) > 3 else K_BIN

    par = core.read_params(n, z)
    m_p = core.particle_mass(par['Omega_M'])
    m_edges = [(lo * m_p, hi * m_p) for lo, hi in core.BIN_NPART]

    print(f'reading {core.sim_name(n)} {core.zdir(z)} ...')
    col, _, _ = core.read_catalogue_selected(core.halo_path(n, z), m_edges)
    mass = col['mass']
    pos = np.ascontiguousarray(np.array([col['x'], col['y'], col['z']]).T % core.box_length)
    vel = np.ascontiguousarray(np.array([col['vx'], col['vy'], col['vz']]).T)
    hid = col['gid']
    del col

    lo_np, hi_np = core.BIN_NPART[k]
    sel = (mass >= lo_np * m_p) & (mass <= hi_np * m_p)
    pos_k = np.ascontiguousarray(pos[sel])
    vel_k = np.ascontiguousarray(vel[sel])
    hid_k = hid[sel]
    n_bin = len(pos_k)
    print(f'bin {k}: {n_bin:,} halos\n')
    del pos, vel, hid, mass

    rng = np.random.default_rng(0)
    seed_idx = np.sort(rng.choice(n_bin, size=N_TEST, replace=False))

    # ---- ground truth
    print(f'brute force over {N_TEST} seeds x {n_bin:,} halos ...')
    t0 = time.time()
    bf_pairs, bf_sumvr, bf_two = brute_force_counts(
        pos_k, vel_k, seed_idx, core.box_length, core.R_edges, core.rlos_edges)
    print(f'  BRUTE FORCE : {bf_pairs:,} pairs, sum_vr = {bf_sumvr:.6e} '
          f'({time.time()-t0:.1f} s)')
    print(f'    (for reference, accepting BOTH rlos signs would give '
          f'{bf_two:,} = {bf_two/max(bf_pairs,1):.3f}x -- the grid is one-sided, '
          f'so the one-sided number above is the right comparison)\n')

    # ---- in-memory path, same seeds
    from nb.calibration.runs.auto_runs.abacus_cosmo_core_auto import _inmem_chunk
    h_mem = _inmem_chunk(pos_k, vel_k, seed_idx, core.r_max, core.RLOS_MAX,
                         core.box_length, core.R_edges, core.rlos_edges,
                         core.V_LO, core.V_HI, core.N_V, core.CONV)
    mem_pairs = int(h_mem.n.sum())
    mem_sumvr = float(h_mem.sum_vr.sum())
    print(f'  IN-MEMORY   : {mem_pairs:,} pairs, sum_vr = {mem_sumvr:.6e}  '
          f'({(mem_pairs/bf_pairs - 1)*100:+.4f}% vs brute force)')

    # ---- sub-box path, same seeds
    sb_path = core.SB_TMPL.format(sim=core.sim_name(n), zdir=core.zdir(z), k=k)
    shutil.rmtree(sb_path, ignore_errors=True)
    os.makedirs(sb_path, exist_ok=True)
    chunksize = max(n_bin // 1000, 1)
    generate_sub_box_ids(pos_k, core.box_length, core.subsize, chunksize, sb_path, name='halo')
    split_simulation_into_sub_boxes(
        positions=pos_k, velocities=vel_k, ids=hid_k, upids=hid_k,
        boxsize=core.box_length, subsize=core.subsize, chunksize=chunksize,
        dtypes=[pos_k.dtype, vel_k.dtype], path=sb_path, name='halo')

    seed_mbin = np.full(n_bin, -1, dtype=np.int32)
    seed_mbin[seed_idx] = 0
    h_sb = get_pairwise_hist(
        pos_k, vel_k, seed_mbin,
        r_max=core.r_max, boxsize=core.box_length, subsize=core.subsize,
        path=sb_path, n_M=1, R_edges=core.R_edges, rlos_edges=core.rlos_edges,
        v_lo=core.V_LO, v_hi=core.V_HI, n_v=core.N_V, conv=core.CONV,
        n_jobs=core.N_JOBS, tasks_per_sub_box=1, rlos_max=core.RLOS_MAX,
        progress=False)
    sb_pairs = int(h_sb.n.sum())
    sb_sumvr = float(h_sb.sum_vr.sum())
    print(f'  SUB-BOX     : {sb_pairs:,} pairs, sum_vr = {sb_sumvr:.6e}  '
          f'({(sb_pairs/bf_pairs - 1)*100:+.4f}% vs brute force)')
    shutil.rmtree(sb_path, ignore_errors=True)

    # ---- where does the sub-box excess live, in r?
    # Aggregated numbers cannot say whether the spurious pairs sit below
    # R_MIN_FIT (harmless -- never entered any fit) or across fitted
    # scales (a real systematic in everything derived from v_r).
    # must reach sqrt(2)*140 ~ 198: r3 = sqrt(R^2 + rlos^2) exceeds 140 for
    # the grid corners, and stopping at 140 silently drops ~22% of the pairs
    # -- exactly the large-r cells that matter most for the fitted range.
    R_CMP_EDGES = np.arange(0.0, 210.0, 10.0)

    def profile(h):
        """<v_r>(r) and pair counts from a PairHist, on coarse r bins."""
        Rc = 0.5 * (h.R_edges[1:] + h.R_edges[:-1])
        lc = 0.5 * (h.rlos_edges[1:] + h.rlos_edges[:-1])
        r3 = np.sqrt(Rc[:, None] ** 2 + lc[None, :] ** 2).ravel()
        n = h.n[0].ravel().astype(float)
        s1 = h.sum_vr[0].ravel()
        nb = len(R_CMP_EDGES) - 1
        idx = np.searchsorted(R_CMP_EDGES, r3, side='right') - 1
        ok = (idx >= 0) & (idx < nb) & (n > 0)
        N = np.bincount(idx[ok], weights=n[ok], minlength=nb)
        S = np.bincount(idx[ok], weights=s1[ok], minlength=nb)
        return N, np.where(N > 0, S / np.where(N > 0, N, 1), np.nan)

    N_mem, vr_mem = profile(h_mem)
    N_sb, vr_sb = profile(h_sb)

    print('\n--- per-radius comparison (sub-box vs the verified-correct in-memory) ---')
    print(f'{"r range":>14s} {"pairs(mem)":>11s} {"excess":>8s} {"excess%":>9s} '
          f'{"<vr>mem":>9s} {"<vr>sb":>9s} {"dvr%":>8s}')
    for b in range(len(R_CMP_EDGES) - 1):
        if N_mem[b] == 0 and N_sb[b] == 0:
            continue
        exc = N_sb[b] - N_mem[b]
        exc_pct = 100.0 * exc / N_mem[b] if N_mem[b] else float('nan')
        dvr_pct = (100.0 * (vr_sb[b] - vr_mem[b]) / abs(vr_mem[b])
                   if np.isfinite(vr_mem[b]) and vr_mem[b] != 0 else float('nan'))
        mark = ''
        if np.isfinite(exc_pct) and abs(exc_pct) > 0.5:
            mark = '  <-- '
        print(f'{R_CMP_EDGES[b]:6.0f}-{R_CMP_EDGES[b+1]:<7.0f} {N_mem[b]:11,.0f} '
              f'{exc:8,.0f} {exc_pct:8.3f}% {vr_mem[b]:9.2f} {vr_sb[b]:9.2f} '
              f'{dvr_pct:7.2f}%{mark}')

    R_MIN_FIT = 15.0
    below = slice(0, int(np.searchsorted(R_CMP_EDGES, R_MIN_FIT)) - 1)
    n_below = N_sb[below].sum() - N_mem[below].sum()
    n_total_exc = N_sb.sum() - N_mem.sum()
    if n_total_exc:
        print(f'\nof {n_total_exc:,.0f} excess pairs, {n_below:,.0f} '
              f'({100.0*n_below/n_total_exc:.1f}%) sit below r = {R_MIN_FIT:g} '
              f'h^-1 Mpc, i.e. below the usual fitting cut')
        if n_below / n_total_exc > 0.8:
            print('-> the excess is essentially confined to scales that never enter '
                  'the fits: the existing base-suite data is very likely fine as-is.')
        else:
            print('-> a substantial share of the excess is at FITTED scales, so the '
                  'existing sub-box-derived v_r data carries a real systematic and is '
                  'worth regenerating with the in-memory path.')

    print('\n--- verdict ---')
    for name, p in (('in-memory', mem_pairs), ('sub-box', sb_pairs)):
        d = p - bf_pairs
        flag = 'MATCHES brute force' if d == 0 else f'differs by {d:+,} ({d/bf_pairs*100:+.4f}%)'
        print(f'  {name:10s}: {flag}')
    if mem_pairs == bf_pairs and sb_pairs != bf_pairs:
        print('\n-> the IN-MEMORY path is correct; the sub-box path has been '
              'OVER-counting, so every earlier result from it is slightly biased.')
    elif sb_pairs == bf_pairs and mem_pairs != bf_pairs:
        print('\n-> the SUB-BOX path is correct; the in-memory path is losing pairs '
              'and must not be used until that is understood.')
    elif mem_pairs == bf_pairs and sb_pairs == bf_pairs:
        print('\n-> both match on this sample: the discrepancy is seed-dependent, '
              'so re-run with a larger N_TEST.')
    else:
        print('\n-> NEITHER matches: something is wrong in a place common to both, '
              'or in the brute-force reference itself. Do not start the long run.')


if __name__ == '__main__':
    main()