"""
Run the whole set of QSO fits in parallel with one command (put this next to fit_qso_lorentz.py and fit_qso_sigerr.py):

    python run_qso_fits.py                # everything below, at s_min = 25, in parallel (default core budget 32)
    python run_qso_fits.py --dry-run      # print the plan (and what exists already) and exit

Fits (each is its own process with its own chain; names as written by the fit scripts):
    raw baseline  QSO-only, mock mean and DR1   fit_qso_lorentz.py --baseline   1 core
    Lorentzian    QSO-only, mock mean and DR1   fit_qso_lorentz.py              4 cores
    sigma_err     QSO-only, mock mean and DR1   fit_qso_sigerr.py               4 cores    (skipped, with a message, until the trained emulators exist)
    joint         mock mean, Lorentzian and sigma_err (six tracer-bins, smear on the QSO)   4 cores each

BEFORE LAUNCHING, every fit's chain is looked up in the chain directory (its <name>.status.json) and classified:
    converged, settings match   -> skipped (printed), never started, no cores used
    converged, settings differ  -> STALE: not started. The fit script would skip it by name, so the chain would look finished while having been run with other
                                   settings. The recorded settings checked are the ones this launcher passes explicitly (Lorentzian: L1, extrap, wmin, wmax;
                                   sigma_err: errmin, errmax). A chain whose status.json does not record them (an older script) cannot be verified and is
                                   treated as STALE too. Use --refit-stale to move such chains aside (to <chain dir>/stale/<timestamp>/, nothing is deleted)
                                   and refit them with the current settings.
    not converged               -> resumed from its saved chain (a status.json with the right settings, or only an .h5 file: that case is flagged, because
                                   the settings of a partial chain cannot be checked)
    nothing there               -> started
The default set uses 26 cores, so with --cores 32 all ten can start at once; with a smaller budget the rest start as earlier ones finish. Never run the same fit
twice at the same time. Logs go to ./logs/<name>.log (append). Ctrl-C stops the running fits (rerun to resume them).

    python run_qso_fits.py --only sigerr          # a subset, by regex on the fit name
    python run_qso_fits.py --cores 16 --smin 25
    python run_qso_fits.py --refit-stale          # move STALE chains aside and refit them
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAIN_DIR = Path('/spiffball/cosweeney/data/chains/fits_11par_apfix_v2_v3/compact_v3_noslope')
MODEL = 'compact_v3_noslope'
SIGERR_EMU_DIR = '/spiffball/cosweeney/data/models_11par_v3_noslope_sigerr_avg_M/'
SIGERR_EMU_SUFFIX = '_11par_v3_noslope_sigerr'
Z_TAG = 'z1p491'
TAGS = {'dr1': 'dr1', 'mock_mean': 'mean'}

# settings passed explicitly to the fit scripts AND compared with what an existing chain recorded in its status.json spec
LORENTZ = dict(L1=2000.0, extrap='constant', wmin=20.0, wmax=600.0)
SIGERR = dict(errmin=20.0, errmax=1000.0)
CLI = {'L1': '--L1', 'extrap': '--extrap', 'wmin': '--wmin', 'wmax': '--wmax', 'errmin': '--errmin', 'errmax': '--errmax'}


def chain_name(kind, data, joint, smin):
    tag = TAGS[data]
    if kind == 'raw':
        return f'{MODEL}_{tag}_' + (f'joint_smin{smin:g}' if joint else f'indiv_QSO_0.8-2.1_smin{smin:g}')
    word = 'lorentz' if kind == 'lorentz' else 'sigerr'
    return f'{MODEL}_{tag}_' + (f'joint_{word}_smin{smin:g}' if joint else f'indiv_QSO_{word}_smin{smin:g}')


def build_fits(smin):
    """Each fit: dict(name, kind, script, args, cores, chain, expected)."""
    specs = []
    for d in ('mock_mean', 'dr1'):
        specs += [('raw', d, False), ('lorentz', d, False), ('sigerr', d, False)]
    specs += [('lorentz', 'mock_mean', True), ('sigerr', 'mock_mean', True)]
    fits = []
    for kind, d, joint in specs:
        script = 'fit_qso_sigerr.py' if kind == 'sigerr' else 'fit_qso_lorentz.py'
        expected = {'raw': {}, 'lorentz': LORENTZ, 'sigerr': SIGERR}[kind]
        args = [d] + (['--baseline'] if kind == 'raw' else []) + (['--joint'] if joint else [])
        for k, v in expected.items():
            args += [CLI[k], str(v)]
        name = (f'{kind}_joint_{d}' if joint else f'{kind}_{d}')
        fits.append(dict(name=name, kind=kind, script=script, args=args, cores=1 if kind == 'raw' else 4,
                         chain=chain_name(kind, d, joint, smin), expected=expected))
    return fits


def emulators_ready(emu_dir, suffix):
    return all((Path(emu_dir) / f'compact_xi{ell}_{Z_TAG}{suffix}.json').exists() for ell in (0, 2, 4))


def classify(fit, chain_dir):
    """('converged' | 'stale' | 'resume' | 'partial' | 'new', detail string)."""
    root = chain_dir / fit['chain']
    status = Path(f'{root}.status.json')
    if not status.exists():
        return ('partial', 'chain file exists but no status.json: its settings cannot be checked') if Path(f'{root}.h5').exists() else ('new', '')
    st = json.load(open(status))
    spec = st.get('spec', {})
    missing = [k for k in fit['expected'] if k not in spec]
    differ = [f'{k}: recorded {spec[k]!r}, current {v!r}' for k, v in fit['expected'].items() if k in spec and spec[k] != v]
    if differ or missing:
        why = '; '.join(differ + ([f"not recorded (older script): {', '.join(missing)}"] if missing else []))
        return 'stale', why
    return ('converged' if st.get('converged') else 'resume'), f"{st.get('iterations', '?')} steps"


def move_aside(fit, chain_dir):
    stamp = time.strftime('%Y%m%d_%H%M%S')
    dest = chain_dir / 'stale' / stamp
    dest.mkdir(parents=True, exist_ok=True)
    moved = [p.name for p in sorted(chain_dir.glob(f"{fit['chain']}.*"))]            # '<name>.' prefix: never matches a longer name such as <name>_zero
    for n in moved:
        shutil.move(str(chain_dir / n), str(dest / n))
    return dest, moved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cores', type=int, default=32, help='core budget for all running fits together')
    ap.add_argument('--smin', type=float, default=25)
    ap.add_argument('--only', default=None, help='regex on the fit name')
    ap.add_argument('--chain-dir', default=str(CHAIN_DIR))
    ap.add_argument('--emu-dir', default=SIGERR_EMU_DIR)
    ap.add_argument('--emu-suffix', default=SIGERR_EMU_SUFFIX)
    ap.add_argument('--log-dir', default='logs')
    ap.add_argument('--refit-stale', action='store_true', help='move STALE chains aside (stale/<timestamp>/) and refit them with the current settings')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    chain_dir = Path(args.chain_dir)

    fits = [f for f in build_fits(args.smin) if args.only is None or re.search(args.only, f['name'])]
    if any(f['script'] == 'fit_qso_sigerr.py' for f in fits) and not emulators_ready(args.emu_dir, args.emu_suffix):
        skipped = [f['name'] for f in fits if f['script'] == 'fit_qso_sigerr.py']
        fits = [f for f in fits if f['script'] != 'fit_qso_sigerr.py']
        print(f'skipping {len(skipped)} sigma_err fits: no trained emulators in {args.emu_dir} (rerun when training has finished): {skipped}', flush=True)

    pending, stale = [], []
    print(f'\nexisting chains in {chain_dir}:', flush=True)
    for f in fits:
        state, detail = classify(f, chain_dir)
        if state == 'stale' and args.refit_stale and not args.dry_run:
            dest, moved = move_aside(f, chain_dir)
            print(f"  MOVED    {f['name']:26s} {f['chain']}.* ({len(moved)} files) -> {dest}   [{detail}]", flush=True)
            state, detail = 'new', ''
        tag = {'converged': 'SKIP', 'stale': 'STALE', 'resume': 'resume', 'partial': 'PARTIAL', 'new': 'new'}[state]
        print(f"  {tag:8s} {f['name']:26s} {f['chain']}" + (f'   ({detail})' if detail else ''), flush=True)
        if state == 'converged':
            continue
        if state == 'stale':
            stale.append(f['name'])
            continue
        pending.append(f)
    if stale:
        print(f"\n{len(stale)} STALE chain(s) NOT launched: {stale}\n  They are converged (or unverifiable) but were run with settings other than the current ones, and the fit script would skip them\n"
              f"  by name. Rerun with --refit-stale to move them aside (nothing is deleted) and refit, or keep them by ignoring this message.", flush=True)

    too_big = [f['name'] for f in pending if f['cores'] > args.cores]
    if too_big:
        sys.exit(f'--cores {args.cores} is smaller than the cores needed by {too_big}')

    def command(f):
        cmd = [sys.executable, str(HERE / f['script']), *f['args'], '--smin', f'{args.smin:g}', '--cores', str(f['cores'])]
        if f['script'] == 'fit_qso_sigerr.py':
            cmd += ['--emu-dir', args.emu_dir, '--emu-suffix', args.emu_suffix]
        return cmd

    print(f"\n{len(pending)} fits to run, core budget {args.cores}, s_min = {args.smin:g}", flush=True)
    if args.dry_run:
        for f in pending:
            print(f"  {f['cores']} cores  {f['name']:26s} {' '.join(command(f)[1:])}")
        print(f"total cores if all ran at once: {sum(f['cores'] for f in pending)}")
        return

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    running, used, results = [], 0, {}
    try:
        while pending or running:
            for item in running[:]:
                proc, f, lf = item
                if proc.poll() is not None:
                    running.remove(item)
                    used -= f['cores']
                    lf.close()
                    results[f['name']] = proc.returncode
                    print(f"{time.strftime('%H:%M:%S')} finished {f['name']} (rc={proc.returncode}); {len(pending)} waiting, {len(running)} running", flush=True)
            while True:
                nxt = next((f for f in pending if used + f['cores'] <= args.cores), None)      # first waiting fit that fits: small ones backfill
                if nxt is None:
                    break
                pending.remove(nxt)
                lf = open(log_dir / f"{nxt['name']}.log", 'a')
                proc = subprocess.Popen(command(nxt), stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
                running.append((proc, nxt, lf))
                used += nxt['cores']
                print(f"{time.strftime('%H:%M:%S')} started {nxt['name']} ({nxt['cores']} cores; {used}/{args.cores} used) -> {log_dir / (nxt['name'] + '.log')}", flush=True)
            time.sleep(10)
    except KeyboardInterrupt:
        print('interrupted: stopping the running fits (rerun the same command to resume them)', flush=True)
        for proc, _f, lf in running:
            proc.terminate()
            lf.close()
        return
    bad = {k: v for k, v in results.items() if v != 0}
    print('\nall done.' + (f' NONZERO exit codes (see the logs): {bad}' if bad else ' every fit exited cleanly.'), flush=True)


if __name__ == '__main__':
    main()