"""
download_desi_bao_chains.py -- fetch the public DESI DR1 BAO-only LambdaCDM chains (DESI 2024 VI) into a folder laid out like the full-shape chains.

Source:       https://data.desi.lbl.gov/public/dr1/vac/dr1/bao-cosmo-params/   (documented at https://data.desi.lbl.gov/doc/releases/dr1/vac/bao-cosmo-params/)
Destination:  /spiffball/cosweeney/DESI/DR1/bao-cosmo-params/v1.0/cobaya/base/<dataset>/   (same layout as .../full-shape-cosmo-params/v1.0/cobaya/base/...)

Run on a machine with network access (e.g. spiffball):
    python download_desi_bao_chains.py --list      # show the matching dataset folders on the server, download nothing
    python download_desi_bao_chains.py             # download every BAO-only LCDM dataset folder found (desi-bao-all*, no CMB or SN likelihoods)
    python download_desi_bao_chains.py --dataset desi-bao-all_schoneberg2024-bbn      # one specific folder
Existing files of the same size are skipped, so it can be re-run safely.

The server's folder names follow the convention documented on the page above (desi-bao-[X] for the BAO data, '_'-joined with priors such as schoneberg2024-bbn);
the exact name of the BAO-only folder was not visible to me when writing this, hence the --list option and the pattern match.
"""
import argparse
import re
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

BASE_URL = 'https://data.desi.lbl.gov/public/dr1/vac/dr1/bao-cosmo-params/'
DEST_ROOT = Path('/spiffball/cosweeney/DESI/DR1/bao-cosmo-params')
VERSIONS = ('v1.0/', '')                                           # the full-shape tree on spiffball has a v1.0 level; try it first
REJECT = ('planck', 'act', 'pantheon', 'union', 'desy5', 'thetastar', 'rdrag', 'add_', 'sdss')     # CMB / SN / SDSS-combination folders


def fetch(url, binary=False):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (research download script)'})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    return data if binary else data.decode('utf-8', 'replace')


def list_dir(url):
    # entries (relative hrefs) of an Apache-style directory index
    html = fetch(url)
    hrefs = re.findall(r'href="([^"]+)"', html)
    return [h for h in hrefs if not h.startswith(('?', '/', '#', 'http', '..'))]


def find_base_url():
    for v in VERSIONS:
        url = urljoin(BASE_URL, v + 'cobaya/base/')
        try:
            list_dir(url)
            return url, v
        except Exception:
            continue
    sys.exit(f'could not list {BASE_URL}(v1.0/)cobaya/base/ -- check the network, or open the URL in a browser to see the folder layout')


def candidates(entries):
    return sorted(e for e in entries if e.endswith('/') and e.startswith('desi-bao-all') and not any(w in e for w in REJECT))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--list', action='store_true', help='list matching dataset folders and exit')
    ap.add_argument('--dataset', help='download only this folder name (as listed by --list)')
    ap.add_argument('--dest', default=str(DEST_ROOT), help='destination root (default: %(default)s)')
    args = ap.parse_args()

    base, version = find_base_url()
    entries = list_dir(base)
    cands = candidates(entries)
    print(f'server folder: {base}')
    print('BAO-only candidates: ' + (', '.join(c.rstrip('/') for c in cands) if cands else 'none'))
    if not cands:
        print('all folders on the server:\n  ' + '\n  '.join(entries))
    if args.list or not cands:
        return
    chosen = [c for c in cands if c.rstrip('/') == args.dataset] if args.dataset else cands
    if not chosen:
        sys.exit(f'--dataset {args.dataset} not among the candidates')
    for d in chosen:
        url = urljoin(base, d)
        out = Path(args.dest) / (version or '') / 'cobaya' / 'base' / d
        out.mkdir(parents=True, exist_ok=True)
        for f in [e for e in list_dir(url) if not e.endswith('/')]:
            target = out / f
            data = fetch(urljoin(url, f), binary=True)
            if target.exists() and target.stat().st_size == len(data):
                print(f'  skip   {target}')
                continue
            target.write_bytes(data)
            print(f'  wrote  {target} ({len(data) / 1e6:.2f} MB)')
    print('done. In inspect_chains_11par.py set BAO_ROOT to ' + str(Path(args.dest) / (version or '') / 'cobaya' / 'base') + '/')


if __name__ == '__main__':
    main()