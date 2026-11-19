"""
Inspect the extracted DR1 files that fits_apfix.py reads (EXTRACT_DIR/<stem>.npz, one per tracer-bin) and report what is missing for fits that include the hexadecapole.

    python inspect_dr1_extracted.py [DIR]      # from nb/fits/11par_v3/ in the compact environment; DIR = a directory of extracted files (default: fits_apfix.EXTRACT_DIR)

fits_apfix.load_dr1_fit_data uses, per tracer-bin: s_full, xi_full (indexed as xi_full[ELL_ALL.index(ell)], so it needs a row per multipole of ELL_ALL = (0, 2, 4)), W_fine and s_fine (the theta-cut window,
W_fine[o, t, s] mixes the theory multipole t into the observed multipole o at separation s) and ells_obs / ells_theory (asserted equal to ELLS_FIT / ELL_ALL). A hexadecapole fit needs the observed-ell = 4 row of the window:
either the added keys W_fine_hex / ells_obs_hex (extract_dr1_data_hex.py adds them and leaves every other array as it was; the first two rows of W_fine_hex must equal W_fine) or, for a file extracted directly with (0, 2, 4), W_fine / ells_obs.
The script checks each of these, prints the keys and shapes, the covariance file shape, the window mixing matrix at small separation (where the theta cut matters most), and looks next to the
extracted files for the raw pair-count files they were made from. It changes nothing.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fits_apfix_11par_v3 as fv          # imports your fits_apfix.py (paths, tracer-bins) without running anything

fa = fv.fa
if len(sys.argv) > 1:
    fa.EXTRACT_DIR = Path(sys.argv[1])
ELL_ALL = tuple(getattr(fa, 'ELL_ALL', (0, 2, 4)))
NBINS = int(getattr(fa, 'NBINS_COV', 45))
problems = {}


def flag(key, msg):
    problems.setdefault(fa.kname(key), []).append(msg)
    print(f'   PROBLEM: {msg}')


for key in fa.FIT_KEYS:
    name = fa.kname(key)
    npz_path, cov_path = fa.dr1_paths(key)
    print(f'\n=== {name}\n   file: {npz_path}')
    if not Path(npz_path).exists():
        flag(key, 'extracted file does not exist')
        continue
    d = np.load(npz_path)
    for k in d.files:
        a = d[k]
        print(f'   {k:12s} shape {str(a.shape):16s} dtype {a.dtype}' + (f'   values {a.tolist()}' if a.size <= 6 else ''))
    ells_obs = tuple(int(e) for e in d['ells_obs']) if 'ells_obs' in d.files else ()
    ells_th = tuple(int(e) for e in d['ells_theory']) if 'ells_theory' in d.files else ()
    has_hex = 'W_fine_hex' in d.files and 'ells_obs_hex' in d.files
    print(f'   ells_obs {ells_obs}, ells_theory {ells_th}' + (f', ells_obs_hex {tuple(int(e) for e in d["ells_obs_hex"])}' if has_hex else ''))
    if has_hex:                                      # hexadecapole window added next to the unchanged (0, 2) arrays
        ells_hex = tuple(int(e) for e in d['ells_obs_hex'])
        Wh = d['W_fine_hex']
        if 4 not in ells_hex:
            flag(key, f'ells_obs_hex = {ells_hex} has no ell = 4')
        if Wh.ndim != 3 or Wh.shape[0] != len(ells_hex) or Wh.shape[1] != len(ells_th):
            flag(key, f'W_fine_hex has shape {Wh.shape}, expected (n_obs = {len(ells_hex)}, n_theory = {len(ells_th)}, n_s)')
        elif 'W_fine' in d.files and not np.array_equal(Wh[:len(ells_obs)], d['W_fine']):
            flag(key, 'the first rows of W_fine_hex differ from W_fine')
    elif 4 not in ells_obs:
        flag(key, f'no hexadecapole window: ells_obs = {ells_obs} has no ell = 4 and there is no W_fine_hex')
    if ells_th != ELL_ALL:
        flag(key, f'ells_theory = {ells_th}, expected {ELL_ALL}')
    if 'xi_full' in d.files:
        xi = d['xi_full']
        if xi.ndim != 2 or xi.shape[0] < len(ELL_ALL):
            flag(key, f'xi_full has shape {xi.shape}: fits_apfix indexes it by ELL_ALL = {ELL_ALL}, so it needs one row per multipole')
        elif not np.all(np.isfinite(xi[ELL_ALL.index(4)])) or np.all(xi[ELL_ALL.index(4)] == 0):
            flag(key, 'the ell = 4 row of xi_full is empty or not finite')
        if 's_full' in d.files and xi.ndim == 2 and xi.shape[1] != len(d['s_full']):
            flag(key, f'xi_full has {xi.shape[1]} bins but s_full has {len(d["s_full"])}')
    else:
        flag(key, 'no xi_full')
    if 'W_fine' in d.files:
        W = d['W_fine']
        if W.ndim != 3 or W.shape[0] != len(ells_obs) or W.shape[1] != len(ells_th):
            flag(key, f'W_fine has shape {W.shape}, expected (n_obs = {len(ells_obs)}, n_theory = {len(ells_th)}, n_s)')
        else:
            s_fine = d['s_fine']
            i = int(np.argmin(np.abs(s_fine - 20.0)))
            show, rows = (d['W_fine_hex'], tuple(int(e) for e in d['ells_obs_hex'])) if has_hex else (W, ells_obs)
            print(f'   window mixing matrix W[obs, theory] at s = {s_fine[i]:.1f} (rows: observed {rows}, columns: theory {ells_th}):')
            print('   ' + np.array2string(show[:, :, i], precision=4, suppress_small=True).replace('\n', '\n   '))
    else:
        flag(key, 'no W_fine')
    if cov_path is not None:
        if Path(cov_path).exists():
            n = np.loadtxt(cov_path).shape
            print(f'   covariance {cov_path.name}: shape {n}' + ('' if n == (len(ELL_ALL) * NBINS,) * 2 else f'   (expected {(len(ELL_ALL) * NBINS,) * 2})'))
            if n != (len(ELL_ALL) * NBINS,) * 2:
                flag(key, f'covariance shape {n} is not {len(ELL_ALL)} multipoles x {NBINS} bins')
        else:
            flag(key, f'covariance file {cov_path} does not exist')
    raw = sorted(p.name for p in Path(npz_path).parent.parent.glob(Path(npz_path).stem + '*') if p.is_file())
    print(f'   raw files with the same stem next to the extraction directory: {raw if raw else "none found"}')

print('\n' + '=' * 70)
if not problems:
    print('all tracer-bins have a hexadecapole row in xi_full, a hexadecapole window (W_fine_hex or ells_obs with 4) and a matching covariance: the DR1 hexadecapole fits can run.')
else:
    print('tracer-bins with problems:')
    for k, v in problems.items():
        print(f'  {k}: ' + '; '.join(v))
    print('\nrun extract_dr1_data_hex.py (in the pycorr environment) to add the hexadecapole window to the extracted files, then inspect again.')