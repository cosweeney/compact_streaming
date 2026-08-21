#!/bin/bash
# Run all 23 phases serially on calvinball.  Resumable: rerun after any
# interruption and completed phases are skipped.
#
# Preferred:  tmux new -s pairhist   then   ./run_all.sh --fg
# Detached:   ./run_all.sh
set -uo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# per-task accumulators go to LOCAL disk, not spiffball
export TMPDIR=/tmp

cd "$(dirname "$0")"

echo "TMPDIR free: $(df -h "$TMPDIR" | awk 'NR==2{print $4}')"
echo "cores: $(nproc)   free RAM: $(free -g | awk 'NR==2{print $7}') GB"

if [[ "${1:-}" == "--fg" ]]; then
    # inside tmux: live progress bars, log still written by the script itself
    python -u run_abacus_batch.py
else
    # detached: timestamped lines to .out, tqdm bars to .progress
    nohup python -u run_abacus_batch.py \
        > pairhist_batch.out 2> pairhist_batch.progress &
    echo "started, pid $!"
    echo "  tail -f pairhist_batch.out        # phase-level log"
    echo "  tail -f pairhist_batch.progress   # live bars"
fi