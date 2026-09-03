"""PairHist for AbacusSummit c130-c181 at z = 0.5.

    python run_abacus_c130_z05.py                # all 52 sims
    python run_abacus_c130_z05.py 137            # one sim
    python run_abacus_c130_z05.py --estimate     # runtime estimate, no work
"""

import sys
import abacus_cosmo_core as core

if __name__ == '__main__':
    sys.exit(core.main([0.5], sys.argv[1:]))