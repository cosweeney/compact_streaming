#!/usr/bin/env bash
# Run the fit-results check (calibrate_A_sigma_11par_v3.py --check-only) on every candidate cosmo_grid_fit_results file.
# No CAMB, no patch needed, a few seconds per file. Run from anywhere:
#
#     bash nb/fits/11par_v3/check_fit_results_11par_v3.sh                 # the four candidates found on calvinball
#     bash nb/fits/11par_v3/check_fit_results_11par_v3.sh a.h5 b.h5       # or your own list
#
# Needs: the scienv python (numpy, pandas, h5py, matplotlib, joblib), the repo's compact/config.py (unmodified baseline),
# and nb/calibration/runs/auto_runs/cosmologies.csv (found automatically when this folder sits at nb/fits/11par_v3/).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../../.." && pwd)}"                 # repo root (nb/fits/11par_v3 -> three levels up)
export PYTHONPATH="$REPO:$PYTHONPATH"

CANDIDATES=("$@")
if [ ${#CANDIDATES[@]} -eq 0 ]; then
  CANDIDATES=(
    /home/cosweeney/code/compact_streaming/nb/calibration/runs/figures/cosmo_grid_fit_results.h5
    /home/cosweeney/code/compact_streaming/nb/calibration/runs/auto_runs/figures/cosmo_grid_fit_results.h5
    /home/cosweeney/code/compact_streaming/nb/calibration/runs/auto_runs/figures/cosmo_grid_fit_results_numod.h5
    /home/cosweeney/code/compact_streaming/nb/drafts/auto_update/cosmo_grid_fit_results_numod.h5
  )
fi

for f in "${CANDIDATES[@]}"; do
  echo "=================== $f"
  if [ ! -f "$f" ]; then echo "  (not found)"; continue; fi
  ls -l --time-style=long-iso "$f" | awk '{print "  modified", $6, $7, " size", $5, "bytes"}'
  echo "  sha256 $(sha256sum "$f" | cut -c1-16)..."
  python "$HERE/calibrate_A_sigma_11par_v3.py" --fit-results "$f" --out-dir /tmp/v3_fit_check --check-only 2>&1 | tail -14
done
