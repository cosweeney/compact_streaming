"""Run one fit by name:  python run_fit.py compact_dr1_joint_smin20   (resumes automatically if interrupted)."""
import sys
import fit_lib_11par as fl

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    fl.run_spec(sys.argv[1])