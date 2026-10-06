"""
Diagnoses the ~2x pair/halo count discrepancy between the current
(in-memory) gather_auto_stats.py and the previous (sub-box file) version,
on REAL data rather than small synthetic data.

Checks three things separately, since "2x more halos" could mean
different things:
  1. Halo SELECTION count (n_halos) -- purely a mass cut, untouched by the
     in-memory edit. If THIS differs 2x, the cause is unrelated to the
     pair-search rewrite entirely (something else changed, or two
     different phases/catalogues are being compared).
  2. Pair count from the current in-memory get_pairwise_hist_jk, checked
     directly against a brute-force reference -- on a random SUBSET of
     REAL halos (not a subset of SEEDS within the full bin): every halo
     in the test subset is a seed AND a neighbor, matching exactly how
     run_one_bin actually calls this function (every halo in a bin gets
     seed_mbin=0). Sampling seeds from a MUCH larger neighbor pool does
     NOT work here -- get_pairwise_hist_jk filters seed_pos down to
     seed_mbin>=0 right at the start, and that SAME filtered array is
     what the shared neighbor tree gets built from, so marking most
     halos "not a seed" shrinks the neighbor pool too.
  3. Whether the r_los sign specifically is implicated: reports the
     brute-force count under the grid's actual one-sided rule (r_los in
     [5,140) only) alongside the two-sided count (both signs), so if the
     current code matches the wrong one, that's immediately visible
     rather than inferred.

Usage:
    python diagnose_gather_auto_mismatch.py c000_ph000 0
"""

import sys

import numpy as np
import h5py as h5

import nb.drafts.auto_update.gather_auto_halo_stats as gas

SIM = sys.argv[1] if len(sys.argv) > 1 else 'c000_ph000'
K_BIN = int(sys.argv[2]) if len(sys.argv) > 2 else 0
N_SUBSET = 4000   # small enough to brute-force exactly (O(n^2) ~ 1.6e7)


def brute_force_counts(pos_all, vel_all, boxsize, R_edges, rlos_edges):
    """Exact ALL-PAIRS count over this (small) population -- every halo is
    both a seed and a neighbor, matching production semantics exactly.
    Reports both the grid's actual one-sided rule and the two-sided
    (both r_los signs) count, so a sign bug would be immediately visible."""
    n = len(pos_all)
    n_R, n_l = len(R_edges) - 1, len(rlos_edges) - 1
    n_pairs_1sided, n_pairs_2sided, sum_vr = 0, 0, 0.0
    for i in range(n):
        d = pos_all - pos_all[i]
        d -= boxsize * np.round(d / boxsize)
        R = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        rlos = d[:, 2]

        iR = np.searchsorted(R_edges, R, side='right') - 1
        il_1s = np.searchsorted(rlos_edges, rlos, side='right') - 1
        keep_1s = (iR >= 0) & (iR < n_R) & (il_1s >= 0) & (il_1s < n_l)

        il_2s = np.searchsorted(rlos_edges, np.abs(rlos), side='right') - 1
        keep_2s = (iR >= 0) & (iR < n_R) & (il_2s >= 0) & (il_2s < n_l)

        n_pairs_1sided += int(keep_1s.sum())
        n_pairs_2sided += int(keep_2s.sum())
        if keep_1s.any():
            dk = d[keep_1s].astype(np.float64)
            rv = (vel_all - vel_all[i])[keep_1s].astype(np.float64)
            r3 = np.sqrt((dk ** 2).sum(axis=1))
            sum_vr += float(((dk * rv).sum(axis=1) / r3).sum())
    return n_pairs_1sided, n_pairs_2sided, sum_vr


def main():
    print(f'reading catalogue for {SIM}...')
    with h5.File(gas.HALO_TMPL.format(sim=SIM, z=gas.REDSHIFT), 'r') as hdf:
        hcat = hdf['halos'][:]
    Mvir = gas.M_PART * hcat['N_total']
    pos = np.array([hcat['x'], hcat['y'], hcat['z']]).T % gas.box_length
    vel = np.array([hcat['vx'], hcat['vy'], hcat['vz']]).T

    lo, hi = gas.MASSBINS[K_BIN]
    sel = (Mvir >= lo) & (Mvir <= hi)
    n_halos = int(sel.sum())
    print(f'\n1. HALO SELECTION: bin {K_BIN} [{lo:.3e}, {hi:.3e}] Msun/h -> '
          f'{n_halos:,} halos selected (mass cut only, unrelated to the pair-search edit)')
    print(f'   compare this number directly against the OLD code\'s logged n_halos '
          f'for the same (sim, bin) -- if these two numbers already differ, the cause '
          f'is NOT the pair-search rewrite (mass selection is untouched).')

    pos_full = np.ascontiguousarray(pos[sel])
    vel_full = np.ascontiguousarray(vel[sel])

    # a random SUBSET of the bin's REAL halos, used as a SELF-CONTAINED
    # population (every member is both seed and neighbor) -- NOT a subset
    # of seeds drawn from the full bin with the full bin as neighbors; see
    # the module docstring for why that doesn't work with this function
    rng = np.random.default_rng(0)
    n_sub = min(N_SUBSET, n_halos)
    idx = rng.choice(n_halos, size=n_sub, replace=False)
    pos_k = np.ascontiguousarray(pos_full[idx])
    vel_k = np.ascontiguousarray(vel_full[idx])
    print(f'\n2. testing on a random SELF-CONTAINED subset: {n_sub:,} of {n_halos:,} real halos '
          f'(preserves real clustering, unlike a uniform-random synthetic test)')

    print(f'   brute force over all {n_sub:,} x {n_sub:,} pairs...')
    bf_1s, bf_2s, bf_sumvr = brute_force_counts(pos_k, vel_k, gas.box_length, gas.R_edges, gas.rlos_edges)
    print(f'   one-sided (r_los in [5,140), the GRID\'s own rule): {bf_1s:,} pairs')
    print(f'   two-sided (|r_los| in [5,140), BOTH signs)        : {bf_2s:,} pairs')
    print(f'   ratio two-sided/one-sided: {bf_2s/max(bf_1s,1):.3f} '
          f'(should be close to 2.0 -- confirms the grid genuinely halves the count by design)')

    print(f'\n3. current in-memory code on the SAME subset (seed_mbin=0 for all, matching '
          f'production exactly)...')
    seed_mbin = np.zeros(n_sub, dtype=np.int32)
    total, regions = gas.get_pairwise_hist_jk(
        pos_k, vel_k, seed_mbin,
        r_max=gas.r_max, boxsize=gas.box_length, subsize=gas.subsize, path=None,
        n_M=1, R_edges=gas.R_edges, rlos_edges=gas.rlos_edges,
        v_lo=gas.V_LO, v_hi=gas.V_HI, n_v=gas.N_V,
        conv=gas.CONV, n_jobs=gas.N_JOBS, progress=False)
    code_pairs = int(total.n.sum())
    code_sumvr = float(total.sum_vr.sum())
    print(f'   code result: {code_pairs:,} pairs, sum_vr={code_sumvr:.6e}')

    print('\n--- verdict ---')
    print(f'code vs one-sided brute force : {code_pairs:,} vs {bf_1s:,}  '
          f'({"MATCH" if code_pairs == bf_1s else f"differ by {code_pairs - bf_1s:+,}"})')
    print(f'code vs two-sided brute force : {code_pairs:,} vs {bf_2s:,}  '
          f'({"MATCH" if code_pairs == bf_2s else f"differ by {code_pairs - bf_2s:+,}"})')
    print(f'sum_vr: code={code_sumvr:.6e} vs brute force (one-sided)={bf_sumvr:.6e} '
          f'({"MATCH" if np.isclose(code_sumvr, bf_sumvr) else "DIFFER"})')

    if code_pairs == bf_1s:
        print('\n-> code matches the correct, one-sided grid convention exactly on REAL, '
              'clustered halos. The pair-search logic itself is NOT the source of your '
              '~2x discrepancy -- look at halo selection (#1 above), or double check '
              'which OLD-code number (which sim/bin/log line) you\'re actually comparing '
              'against, since it may not be an apples-to-apples comparison.')
    elif code_pairs == bf_2s:
        print('\n-> code is counting BOTH r_los signs -- this IS the bug you suspected. '
              'Needs a fix to get_pairwise_hist_jk or PairHist.add\'s binning.')
    else:
        print('\n-> code matches NEITHER reference -- something else is wrong, '
              'not a simple sign issue. Needs further investigation before trusting '
              'either number.')


if __name__ == '__main__':
    main()