"""
Propose the training box of the w0wa emulator from CMB(+DESI) chains, and check that the AP-shifted separations stay inside the model's s grid.

    python propose_w0wa_box.py ROOT [ROOT ...] [--lo 0.1] [--hi 99.9] [--margin 0.25] [--smin 25] [--smax 130]

ROOT = chain root of a cobaya / getdist chain, i.e. the path without the '.1.txt' suffix (several roots: the box is the union). For the SPT-3G D1 release use the w0wa
chains of CMB_SPA_DESI_Fixed.zip (CMB-SPA, and CMB-SPA + DESI DR2 BAO). The script prints the parameter names it finds; if an alias below does not match, it says so.

For each of h, omega_b, omega_cdm, ln A_s, n_s, w0, wa and w0 + wa it prints the weighted percentiles [lo, hi], the mean +- 5 std and the proposed box (percentile range widened by
`margin` x its width on each side). The w0wa training prior should be sampled in (w0, v = w0 + wa) with v < 0, so the printed box gives the w0 range and the v range (v capped at 0), and wa follows.
The AP check evaluates q_perp, q_par (reference c000, h / h_fid factor, as in fits_apfix.py) at the corners of the proposed box for the six DESI redshifts, and flags corners whose
separations s' (between smin * q and smax * q) leave the emulator grid [16, 172] h^-1 Mpc.
"""
import argparse
import itertools
import numpy as np

ALIASES = {'h': ['H0'], 'ombh2': ['ombh2', 'omegabh2'], 'omch2': ['omch2', 'omegach2'], 'logA': ['logA'], 'ns': ['ns'], 'w0': ['w', 'w0', 'w0_fld'], 'wa': ['wa', 'wa_fld']}
ZS = [0.295, 0.510, 0.706, 0.919, 1.317, 1.491]
S_GRID = (16.0, 172.0)            # compact.common S_MODEL: np.arange(16, 169 + 4, 1)
H_FID = 0.6736


def wpercentile(x, w, q):
    """Weighted percentiles q (0-100) of the samples x with weights w."""
    i = np.argsort(x)
    x, c = x[i], np.cumsum(w[i])
    return np.interp(np.asarray(q) / 100.0 * c[-1], c - 0.5 * w[i], x)


def load(root):
    from getdist import loadMCSamples
    s = loadMCSamples(root, settings={'ignore_rows': 0.3})
    names = [p.name for p in s.paramNames.names]
    print(f'{root}: {s.samples.shape[0]} samples, parameters: {names}')
    cols = {}
    for key, al in ALIASES.items():
        hit = [a for a in al if a in names]
        if not hit:
            print(f"  WARNING: none of {al} found for '{key}' in this chain; it is skipped")
            continue
        cols[key] = s.samples[:, names.index(hit[0])]
    if 'h' in cols:
        cols['h'] = cols['h'] / 100.0
    if 'w0' in cols and 'wa' in cols:
        cols['v'] = cols['w0'] + cols['wa']
    return cols, s.weights


def propose(roots, lo, hi, margin):
    box = {}
    for root in roots:
        cols, w = load(root)
        print(f'  {"parameter":8s} {"mean":>10s} {"std":>9s} {f"{lo}%":>10s} {f"{hi}%":>10s}   mean -+ 5 std')
        for k, x in cols.items():
            m = np.average(x, weights=w)
            sd = np.sqrt(np.average((x - m) ** 2, weights=w))
            a, b = wpercentile(x, w, [lo, hi])
            print(f'  {k:8s} {m:10.5g} {sd:9.3g} {a:10.5g} {b:10.5g}   [{m - 5 * sd:.5g}, {m + 5 * sd:.5g}]')
            old = box.get(k, (np.inf, -np.inf))
            box[k] = (min(old[0], a), max(old[1], b))
    out = {}
    for k, (a, b) in box.items():
        pad = margin * (b - a)
        out[k] = (a - pad, b + pad)
    if 'v' in out:
        out['v'] = (out['v'][0], min(out['v'][1], 0.0))          # w0 + wa < 0 (early matter domination, as DESI)
    return out


def ap_check(box, smin, smax, n_z=ZS):
    """q_perp, q_par at the corners of `box` (keys h, ombh2, omch2, w0, v; v = w0 + wa) for the DESI redshifts."""
    import camb
    ref = camb.get_background(camb.set_params(H0=100 * H_FID, ombh2=0.02237, omch2=0.12, ns=0.9649))
    da0, h0 = {z: ref.angular_diameter_distance(z) for z in n_z}, {z: ref.hubble_parameter(z) for z in n_z}
    worst, rows = None, []
    keys = ['h', 'ombh2', 'omch2', 'w0', 'v']
    for corner in itertools.product(*[box[k] for k in keys]):
        h, ob, oc, w0, v = corner
        if v >= 0:
            continue
        p = camb.set_params(H0=100 * h, ombh2=ob, omch2=oc, ns=0.9649, w=w0, wa=v - w0, dark_energy_model='ppf')
        r = camb.get_background(p)
        for z in n_z:
            f = h / H_FID
            qp, qa = f * r.angular_diameter_distance(z) / da0[z], f * h0[z] / r.hubble_parameter(z)
            rows.append((min(qp, qa), max(qp, qa), corner, z))
    lo = min(rows, key=lambda t: t[0])
    hi = max(rows, key=lambda t: t[1])
    print(f'AP check over {len(rows)} (corner, z) points: q in [{lo[0]:.3f}, {hi[1]:.3f}]  ->  s\' in [{smin * lo[0]:.1f}, {smax * hi[1]:.1f}]  (grid {S_GRID[0]:.0f}-{S_GRID[1]:.0f})')
    n_out = sum(1 for a, b, _, _ in rows if smin * a < S_GRID[0] or smax * b > S_GRID[1])
    print(f'  {n_out} of {len(rows)} points leave the grid; lowest q at (h, ombh2, omch2, w0, w0+wa) = {tuple(round(x, 4) for x in lo[2])}, z = {lo[3]}; highest q at {tuple(round(x, 4) for x in hi[2])}, z = {hi[3]}')
    return n_out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('roots', nargs='+')
    ap.add_argument('--lo', type=float, default=0.1)
    ap.add_argument('--hi', type=float, default=99.9)
    ap.add_argument('--margin', type=float, default=0.25)
    ap.add_argument('--smin', type=float, default=25.0)
    ap.add_argument('--smax', type=float, default=130.0)
    args = ap.parse_args()
    box = propose(args.roots, args.lo, args.hi, args.margin)
    print('\nproposed box (percentile range widened by the margin):')
    for k, (a, b) in box.items():
        print(f'  {k:6s} [{a:.5g}, {b:.5g}]')
    if 'wa' in box and 'v' in box:
        print('  (sample the Sobol pair as (w0, w0 + wa = v); wa follows)')
    if all(k in box for k in ('h', 'ombh2', 'omch2', 'w0', 'v')):
        ap_check(box, args.smin, args.smax)