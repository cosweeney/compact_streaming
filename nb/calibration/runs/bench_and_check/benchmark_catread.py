"""
Is the catalogue read worth optimizing?

B (in-memory) projects to 33.2 h over the grid, of which ~14.0 h is
catalogue reads: 416 (sim, z) jobs x ~121 s, each reading a 395M-halo
catalogue to keep the ~2.2M halos that fall in the two mass bins -- about
180x more data than is used.

This times three strategies:
  1. FULL      -- read all 8 columns, what the code does now
  2. MASS_ONLY -- read just the mass column (how much is the other 7/8?)
  3. TWO_STAGE -- mass column, build the selection, then read only the
                  SELECTED ROWS of the remaining 7 columns

Whether 3 actually wins depends entirely on the file layout, which this
prints first.  If the columns are separate flat datasets, reading one of
eight should cost roughly an eighth.  If they are a single COMPOUND table,
the fields are interleaved on disk and reading 'mass' alone still pulls
every byte -- in which case there is nothing to win here and the answer is
simply "no".  h5py fancy-indexing can also be slower than a straight read
on an unfavourably chunked file, so stage 3 is measured, not assumed.

PAGE-CACHE CONFOUND, and how it is handled: re-reading a file that is
already in the OS page cache is much faster, and dropping caches needs
root.  So the strategies are run in the order MASS_ONLY -> TWO_STAGE ->
FULL, i.e. the NEW approach runs cold and the BASELINE runs warm.  That
biases the comparison AGAINST the new approach, so a win here is a real
win; a narrow loss is inconclusive.  For a clean number, run each strategy
in its own fresh process on a file nothing has touched.

Usage:
    python benchmark_catread.py            # default sim/z below
    python benchmark_catread.py 130 0.5
"""

import os
import sys
import time

import numpy as np
import h5py as h5

import nb.calibration.runs.auto_runs.abacus_cosmo_core_auto as core

SIM_NUM = 130
Z = 0.5
COLS = ['mass', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'gid']


def describe_layout(hpath):
    """Flat datasets or one compound table?  This decides whether a
    mass-first read can save anything at all."""
    with h5.File(hpath, 'r') as hdf:
        keys = list(hdf.keys())
        print(f'top-level keys: {keys}')

        flat = all(nm in keys for nm in COLS)
        if flat:
            print('layout: FLAT datasets (one per column)')
            for nm in COLS:
                d = hdf[nm]
                print(f'  {nm:6s} shape={d.shape} dtype={d.dtype} '
                      f'chunks={d.chunks} compression={d.compression}')
            return 'flat'

        for k in keys:
            d = hdf[k]
            dt = getattr(d, 'dtype', None)
            if dt is not None and dt.names and all(nm in dt.names for nm in COLS):
                print(f'layout: COMPOUND table in "{k}"')
                print(f'  shape={d.shape} itemsize={dt.itemsize} B '
                      f'chunks={d.chunks} compression={d.compression}')
                print(f'  fields: {dt.names}')
                print('  NOTE: fields are interleaved on disk, so reading the mass '
                      'column alone still touches every byte -- expect little or no '
                      'saving from a two-stage read.')
                return 'compound'
        raise KeyError(f'could not identify layout; keys: {keys}')


def read_full(hpath):
    t0 = time.time()
    with h5.File(hpath, 'r') as hdf:
        col = core._read_cols(hdf, COLS)
    n = len(col['mass'])
    del col
    return time.time() - t0, n


def read_mass_only(hpath, layout):
    t0 = time.time()
    with h5.File(hpath, 'r') as hdf:
        if layout == 'flat':
            mass = hdf['mass'][()]
        else:
            col = core._read_cols(hdf, ['mass'])
            mass = col['mass']
    return time.time() - t0, mass


def read_two_stage(hpath, layout, m_p):
    """mass -> selection -> only the selected rows of the other columns."""
    t0 = time.time()
    with h5.File(hpath, 'r') as hdf:
        if layout == 'flat':
            mass = hdf['mass'][()]
        else:
            mass = core._read_cols(hdf, ['mass'])['mass']
        t_mass = time.time() - t0

        keep = np.zeros(len(mass), dtype=bool)
        for lo_np, hi_np in core.BIN_NPART:
            keep |= (mass >= lo_np * m_p) & (mass <= hi_np * m_p)
        idx = np.flatnonzero(keep)          # h5py needs increasing indices
        t_sel = time.time() - t0 - t_mass

        out = {'mass': mass[keep]}
        if layout == 'flat':
            for nm in COLS:
                if nm == 'mass':
                    continue
                out[nm] = hdf[nm][idx]
        else:
            key = next(k for k in hdf.keys()
                       if getattr(hdf[k], 'dtype', None) is not None
                       and hdf[k].dtype.names)
            sub = hdf[key][idx]
            for nm in COLS:
                if nm != 'mass':
                    out[nm] = sub[nm]
    t_tot = time.time() - t0
    n_sel = len(idx)
    del out
    return t_tot, t_mass, t_sel, t_tot - t_mass - t_sel, n_sel


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else SIM_NUM
    z = float(sys.argv[2]) if len(sys.argv) > 2 else Z

    hpath = core.halo_path(n, z)
    if not os.path.exists(hpath):
        raise FileNotFoundError(hpath)
    size_gb = os.path.getsize(hpath) / 1e9
    print(f'{hpath}\n  {size_gb:.1f} GB on disk\n')

    layout = describe_layout(hpath)
    par = core.read_params(n, z)
    m_p = core.particle_mass(par['Omega_M'])
    print()

    # NEW approaches first (cold), baseline last (warm) -- see module docstring
    t_mass, mass = read_mass_only(hpath, layout)
    n_halos = len(mass)
    print(f'2. MASS_ONLY : {t_mass:6.1f} s  ({n_halos:,} halos)')
    del mass

    t_two, t_m, t_s, t_rest, n_sel = read_two_stage(hpath, layout, m_p)
    print(f'3. TWO_STAGE : {t_two:6.1f} s  (mass {t_m:.1f} s + select {t_s:.1f} s '
          f'+ {n_sel:,} selected rows of 7 cols {t_rest:.1f} s)')

    t_full, _ = read_full(hpath)
    print(f'1. FULL      : {t_full:6.1f} s  <-- current, and the ONLY one that ran warm\n')

    frac = n_sel / n_halos
    print(f'selected {n_sel:,} of {n_halos:,} halos ({frac*100:.3f}%) into the two bins')

    speedup = t_full / t_two if t_two > 0 else float('nan')
    print(f'TWO_STAGE vs FULL: {speedup:.2f}x '
          f'(and FULL had the page-cache advantage, so the true gap is at least this)')

    n_reads = len(core.SIM_NUMS) * 8
    print(f'\nover the grid ({n_reads} catalogue reads):')
    print(f'  FULL     : {t_full * n_reads / 3600:5.1f} h')
    print(f'  TWO_STAGE: {t_two * n_reads / 3600:5.1f} h  '
          f'(saves {(t_full - t_two) * n_reads / 3600:.1f} h)')

    if speedup < 1.15:
        print('\n-> not worth changing: the read is dominated by something a '
              'two-stage read does not avoid (most likely an interleaved compound '
              'layout, or fancy-indexing costing as much as a straight read).')
    else:
        print('\n-> worth adopting: fold the two-stage read into '
              'abacus_cosmo_core.run_one, which already needs only the halos in '
              'the two bins.')


if __name__ == '__main__':
    main()