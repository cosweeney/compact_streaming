#!/bin/bash
# Regenerates everything affected by the mass-bin change (6 bins -> 5 bins,
# new bin 4 = logM in [13.9, 14.3] replacing the old bins 4 and 5).
#
# Bins 0-3 are untouched (same edges, same indices) -- only bin 4's data,
# and the per-phase xi_hh files that bundle all bins together, need
# regenerating.
#
# gather_auto_stats.py and compute_xihh_jk.py CANNOT run in the same conda
# environment -- pycorr and compact/colossus/velocileptors are mutually
# incompatible (established earlier in this project). This script runs
# each in its own environment via `conda run`, per this project's own
# established split: scienv (compact/colossus, no pycorr) for
# gather_auto_stats.py, tpcf (pycorr) for compute_xihh_jk.py. Edit
# SCIENV/TPCF below if your actual environment names differ.
#
# Usage:
#   bash regenerate_bin4.sh
#
# Safe to interrupt and re-run: gather_auto_stats.py resumes per (phase,
# bin), so anything it already finished this run won't be redone.

set -e   # stop on first error, rather than continuing past a failed step

BASE_DIR="/spiffball/cosweeney/simulations/AbacusSummit_base/halos"
SCIENV="scienv"
TPCF="tpcf"

echo "=== checking both conda environments exist before doing anything ==="
for env in "$SCIENV" "$TPCF"; do
    if ! conda env list | awk '{print $1}' | grep -qx "$env"; then
        echo "ERROR: conda environment '$env' not found -- edit SCIENV/TPCF at the top of this script"
        exit 1
    fi
done

echo ""
echo "=== deleting stale bin 4/5 auto-stats (old bin definitions) ==="
for k in 4 5; do
    find "$BASE_DIR" -name "pairhist_auto_mbin${k}.h5" -print -delete
done

echo ""
echo "=== deleting stale xi_hh_jk.npz (computed for all bins together per phase) ==="
find "$BASE_DIR" -name "xi_hh_jk.npz" -print -delete

echo ""
echo "=== deleting stale Fig 5 fit cache (built from the old 6-bin scheme) ==="
rm -fv figures/fig5_fit_cache.h5

echo ""
echo "=== re-running gather_auto_halo_stats.py in '$SCIENV' (regenerates only bin 4; bins 0-3 resume/skip) ==="
conda run -n "$SCIENV" --no-capture-output python3 gather_auto_halo_stats.py

echo ""
echo "=== re-running compute_xihh_jk.py in '$TPCF' (regenerates xi_hh for all bins, all phases) ==="
conda run -n "$TPCF" --no-capture-output python3 compute_xihh_jk.py

echo ""
echo "=== done -- make_paper_figures_auto.py will pick up the new bin 4 on its next run ==="