"""PairHist for AbacusSummit c130-c181 at the remaining redshifts.

Edit REDSHIFTS as the catalogues land; RESUME means re-running is cheap.

    python run_abacus_c130_rest.py               # all sims, all z below
    python run_abacus_c130_rest.py 137 0.8       # one sim, one z
    python run_abacus_c130_rest.py --estimate 6  # sample 6 jobs
"""

import sys
import abacus_cosmo_core as core

REDSHIFTS = [1.1, 0.8, 0.3]

if __name__ == '__main__':
    sys.exit(core.main(REDSHIFTS, sys.argv[1:]))