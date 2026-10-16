"""
Launch the fits as independent processes (each fit = one process with its own emcee pool of `ncores` workers),
respecting a total core budget. Headline fits (priority 0: DR1 and mean-mock, both models) start first; the
single-mock fits fill the remaining cores. Logs go to LOG_DIR/<name>.log; rerunning skips converged fits and
resumes interrupted ones.

    python launch_fits.py --dry-run                 # list the fits and core allocation
    python launch_fits.py --max-cores 112
    python launch_fits.py --only dr1 --max-cores 112     # substring filter on the fit name
    python launch_fits.py --only lowM                    # e.g. only the low_M mass-scheme fits
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import fit_lib_11par as fl

LOG_DIR = Path('/spiffball/cosweeney/data/chains/fits_11par/logs')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-cores', type=int, default=112)
    ap.add_argument('--only', default=None, help='run only fits whose name contains this substring')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--skip-check', action='store_true', help='launch even if data files are missing')
    ap.add_argument('--rerun', action='store_true', help='also launch fits that already converged (default: skip them)')
    args = ap.parse_args()

    missing = fl.missing_files()
    if missing:
        print(f'{len(missing)} required data files are missing:')
        for p in missing:
            print('  ', p)
        if not args.skip_check:
            sys.exit('aborting (use --skip-check to ignore)')
    specs = sorted(fl.build_specs(), key=lambda s: s['priority'])
    if args.only:
        specs = [s for s in specs if args.only in s['name']]
    if not args.rerun:
        done = [s for s in specs if fl.is_converged(s)]
        specs = [s for s in specs if s not in done]
        print(f'skipping {len(done)} fits that already converged (use --rerun to include them)')
    total = sum(s['ncores'] for s in specs)
    print(f'{len(specs)} fits, {sum(s["mode"] == "joint" for s in specs)} joint / '
          f'{sum(s["mode"] == "indiv" for s in specs)} individual, {total} cores if all run at once '
          f'(budget {args.max_cores})')
    for s in specs:
        print(f'  [{s["priority"]}] {s["name"]:<48} cores={s["ncores"]}')
    if args.dry_run:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pending, running, free = list(specs), [], args.max_cores
    while pending or running:
        for item in list(running):
            if item[0].poll() is not None:
                running.remove(item)
                free += item[1]['ncores']
                print(f'{time.strftime("%H:%M:%S")} finished {item[1]["name"]} (exit {item[0].returncode}); '
                      f'{len(pending)} pending, {len(running)} running', flush=True)
        started = True
        while started and pending:
            started = False
            for s in pending:                       # first pending fit that fits in the free cores
                if s['ncores'] <= free:
                    log = open(LOG_DIR / f'{s["name"]}.log', 'a')
                    p = subprocess.Popen([sys.executable, str(Path(__file__).with_name('run_fit.py')), s['name']],
                                         stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ))
                    running.append((p, s))
                    pending.remove(s)
                    free -= s['ncores']
                    started = True
                    print(f'{time.strftime("%H:%M:%S")} started {s["name"]} ({s["ncores"]} cores)', flush=True)
                    break
        time.sleep(30)


if __name__ == '__main__':
    main()