"""Auto-pairwise PairHist for AbacusSummit c130-c181, all requested redshifts.

Collects AUTO-statistics (seed and neighbour pool both restricted to the
same mass bin) for all six mass bins -- see abacus_cosmo_core.py's module
docstring for what changed from the old cross-statistic version.

RESUME is per (sim, z, mass bin), so re-running after an interruption, or
after more catalogues land, is cheap.

    python run_abacus_cosmo_rest.py               # all sims, all z below
    python run_abacus_cosmo_rest.py 137 0.8       # one sim, one z
    python run_abacus_cosmo_rest.py --estimate 6  # sample 6 jobs
"""

import sys
import nb.calibration.runs.auto_runs.abacus_cosmo_core_auto as core

REDSHIFTS = [0.2, 0.3, 0.4, 0.5, 0.8, 1.1, 1.4, 1.7]

if __name__ == '__main__':
    sys.exit(core.main(REDSHIFTS, sys.argv[1:]))