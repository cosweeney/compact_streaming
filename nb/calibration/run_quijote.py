"""PairHist for a single mass bin across Quijote latin_hypercube_HR sims.

    python run_quijote.py            # all sims in SIMS, serially
    python run_quijote.py 42         # one sim

ONE sample per sim: the halos in the mass bin are both the seeds and the
tracers, so this is an auto-correlation of that bin (b_seed == b_tracer).

The mass bin is defined by PARTICLE NUMBER (N_LO..N_HI particles), so the mass
edges are recomputed per sim from that sim's Omega_m.  Both the edges and the
resulting effective mass are written into each output file -- without them the
histograms are not comparable across the hypercube.

Radial grids, velocity grid, r_max and the tracer threshold are identical to
the Abacus runs so the two sets can be compared directly.
"""

import gc
import os
import shutil
import sys
import time
import traceback

import numpy as np
import h5py as h5
import pandas as pd

from pairhist import PairHist
from pairhist_direct import get_pairwise_hist
from compact.catalog.accumulate import generate_sub_box_ids, split_simulation_into_sub_boxes

# ---------------------------------------------------------------- config ----
# Only this subset of the 2000 latin hypercube sims is on spiffball.
SIMS = [
    1402, 1655, 1740, 1791, 1807, 1820, 1839, 1861, 1889, 1902, 1911, 1915, 1923,
    1930, 1936, 1945, 1952, 1962, 1966, 1974, 1979, 1984, 1988, 1992, 1996,
    1625, 1656, 1757, 1793, 1810, 1824, 1852, 1871, 1894, 1906, 1912, 1916, 1927,
    1933, 1937, 1948, 1955, 1963, 1967, 1975, 1980, 1985, 1989, 1993, 1997,
    1630, 1659, 1764, 1795, 1818, 1826, 1857, 1877, 1897, 1907, 1913, 1919, 1928,
    1934, 1943, 1949, 1956, 1964, 1970, 1976, 1982, 1986, 1990, 1994, 1998,
    1640, 1704, 1769, 1797, 1819, 1835, 1859, 1887, 1898, 1909, 1914, 1921, 1929,
    1935, 1944, 1950, 1960, 1965, 1972, 1977, 1983, 1987, 1991, 1995, 590,
]

HALO_TMPL = ('/spiffball/edgarmsc/simulations/Quijote/Halos/Rockstar/'
             'latin_hypercube_HR/{sim}/hlist_4.hdf5')
COSMO_TMPL = ('/spiffball/edgarmsc/simulations/Quijote/Snapshots/'
              'latin_hypercube_HR_cosmo/Cosmo_params_{sim}.dat')
OUT_DIR = ('/spiffball/cosweeney/simulations/Quijote/halos/'
           'latin_hypercube_HR/{sim}/data/')
SB_TMPL = ('/spiffball/cosweeney/simulations/Quijote/halos/'
           'latin_hypercube_HR/{sim}/SBp/')
LOG_TMPL = '/spiffball/cosweeney/simulations/Quijote/logs/pairhist_{sim}.log'
OUT_NAME = 'pairhist_massbin.h5'

COSMO_COLS = ['Om', 'Ob', 'h', 'ns', 's8']


def read_cosmology(sim):
    """Read one sim's cosmology from its Cosmo_params_<sim>.dat file."""
    df = pd.read_csv(COSMO_TMPL.format(sim=sim), sep=r'\s+', header=None,
                     names=COSMO_COLS)
    return dict(zip(COSMO_COLS, df.values[0]))

box_length = 1000.0
# NOTE: load_halos pads by subsize/2 beyond each sub-box face if grid_pos holds
# CENTRES.  box/4 = 250 -> 125 padding < r_max, which truncates pairs.  box/2
# gives 250 and only 8 sub-boxes, hence TASKS_PER_SUB_BOX = 8 below.
subsize = box_length / 2
r_max = 140.0

N_PART = 1024                     # particles per side in the HR sims
N_LO, N_HI = 300, 400             # the mass bin, in particles

r_perp = np.linspace(5, 140, 135+1)
r_los = np.linspace(5, 140, 135+1)

r_p = 0.5*(r_perp[1:]+r_perp[:-1])
r_l = 0.5*(r_los[1:]+r_los[:-1])

R_edges = r_perp                  # noqa: F821  same grid as Abacus
rlos_edges = r_los                # noqa: F821
V_LO, V_HI, N_V = -4000.0, 4000.0, 400
# Rockstar hlist velocities are already km/s, unlike the Abacus catalogue.
# Kept explicit (rather than dropped) so the output records what was applied;
# VEL_CHECK below fails loudly if this is wrong.
CONV = 1.0
VEL_CHECK = (50.0, 2000.0)        # plausible 1-D halo velocity dispersion, km/s

RESUME = True
KEEP_SUBBOXES = False
N_JOBS = int(os.environ.get('PAIRHIST_NJOBS', 64))
# 8 sub-boxes * 8 = 64 tasks; keep 8*TASKS_PER_SUB_BOX a multiple of N_JOBS
TASKS_PER_SUB_BOX = int(os.environ.get('PAIRHIST_TASKS_PER_SB', 8))

_log_path = None


def log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    if _log_path:
        with open(_log_path, 'a') as f:
            f.write(line + '\n')


def particle_mass(om0):
    """Quijote particle mass [Msun/h] for a box of box_length with N_PART^3."""
    return 27.75e10 * om0 * box_length**3 / N_PART**3


def run_sim(sim, tag=''):
    global _log_path
    _log_path = LOG_TMPL.format(sim=sim)
    os.makedirs(os.path.dirname(_log_path), exist_ok=True)

    halo_path = HALO_TMPL.format(sim=sim)
    sb_path = SB_TMPL.format(sim=sim)
    out_dir = OUT_DIR.format(sim=sim)
    out_path = os.path.join(out_dir, OUT_NAME)

    if not os.path.exists(halo_path):
        raise FileNotFoundError(halo_path)

    cosmo = read_cosmology(sim)
    m_p = particle_mass(cosmo['Om'])
    M_LO, M_HI = N_LO * m_p, N_HI * m_p
    log(f'=== sim {sim} {tag} ===')
    log(f'{sim}: Om {cosmo["Om"]:.4f} Ob {cosmo["Ob"]:.4f} h {cosmo["h"]:.4f} '
        f'ns {cosmo["ns"]:.4f} s8 {cosmo["s8"]:.4f}')
    log(f'{sim}: m_p {m_p:.3e}, bin [{M_LO:.3e}, {M_HI:.3e}] Msun/h')

    log(f'{sim}: reading catalogue')
    with h5.File(halo_path, 'r') as hdf:      # flat datasets, no /halos group
        Mvir = hdf['Mvir'][()]
        pos = np.array([hdf['X'][()], hdf['Y'][()], hdf['Z'][()]]).T % box_length
        vel = np.array([hdf['VX'][()], hdf['VY'][()], hdf['VZ'][()]]).T
        hid = hdf['ID'][()]
    gc.collect()

    # Guard against a units mismatch: with CONV wrong by orders of magnitude
    # every pair lands in one velocity bin and the run is silently useless.
    sig1d = float(np.std(CONV * vel[:, 2]))
    log(f'{sim}: 1-D halo velocity dispersion {sig1d:.1f} km/s '
        f'(CONV = {CONV:g})')
    if not (VEL_CHECK[0] <= sig1d <= VEL_CHECK[1]):
        raise ValueError(
            f'sim {sim}: sigma_1D = {sig1d:.3e} km/s is outside {VEL_CHECK} '
            f'-- CONV is probably wrong for this catalogue')

    # one sample: the mass bin is both the seed set and the tracer set
    nb = (Mvir >= M_LO) & (Mvir <= M_HI)
    n_seeds = int(nb.sum())
    if n_seeds == 0:
        raise ValueError(f'sim {sim}: mass bin is empty')
    n_bar = n_seeds / box_length**3
    log(f'{sim}: {n_seeds:,} halos in bin, n_bar {n_bar:.3e}, '
        f'{n_bar * np.pi * r_max**2 * 2 * r_max:,.0f} pairs/halo')

    shutil.rmtree(sb_path, ignore_errors=True)     # append mode -> start clean
    os.makedirs(sb_path, exist_ok=True)
    chunksize = max(int(nb.sum()) // 1000, 1)
    generate_sub_box_ids(pos[nb], box_length, subsize, chunksize,
                         sb_path, name='halo')
    split_simulation_into_sub_boxes(
        positions=pos[nb], velocities=vel[nb], ids=hid[nb], upids=hid[nb],
        boxsize=box_length, subsize=subsize, chunksize=chunksize,
        dtypes=[pos.dtype, vel.dtype], path=sb_path, name='halo')

    seed_mbin = np.where(nb, 0, -1).astype(np.int32)
    logM_eff = float(np.mean(np.log10(Mvir[nb])))
    log(f'{sim}: <log10 M> = {logM_eff:.4f}')

    h = get_pairwise_hist(
        pos, vel, seed_mbin,
        r_max=r_max, boxsize=box_length, subsize=subsize, path=sb_path,
        n_M=1, R_edges=R_edges, rlos_edges=rlos_edges,
        v_lo=V_LO, v_hi=V_HI, n_v=N_V,
        conv=CONV, n_jobs=N_JOBS, tasks_per_sub_box=TASKS_PER_SUB_BOX,
    )

    off = int((h.n - h.counts.sum(-1)).sum())
    mu, sd = h.mean_std(0)
    n_min_cell = int(h.n[0][h.n[0] > 0].min()) if (h.n[0] > 0).any() else 0
    log(f'{sim}: {h.n.sum():,} pairs ({h.n.sum() / n_seeds:,.0f}/halo), '
        f'off-grid {off}, cells {(h.n[0] > 0).sum()}/{h.n[0].size}, '
        f'min/median cell n {n_min_cell}/'
        f'{np.median(h.n[0][h.n[0] > 0]):.0f}, '
        f'median sigma {np.nanmedian(sd):.1f} km/s')
    if off:
        log(f'{sim}: WARNING {off} pairs fell off the velocity grid')

    os.makedirs(out_dir, exist_ok=True)
    h.save(out_path)
    with h5.File(out_path, 'a') as hdf:        # what the accumulator forgets
        hdf.attrs.update(sim=str(sim), suite='Quijote_latin_hypercube_HR',
                         boxsize=box_length, subsize=subsize, r_max=r_max,
                         sample='auto', conv=CONV,
                         m_part=m_p, **{k: float(v) for k, v in cosmo.items()},
                         n_part_lo=N_LO, n_part_hi=N_HI,
                         M_lo=M_LO, M_hi=M_HI, logM_eff=logM_eff,
                         n_seeds=n_seeds, n_tracers=int(nb.sum()))
    log(f'{sim}: saved {out_path}')

    if not KEEP_SUBBOXES:
        shutil.rmtree(sb_path, ignore_errors=True)

    del pos, vel, hid, Mvir, nb, seed_mbin, h
    gc.collect()


def main():
    sims = [int(sys.argv[1])] if len(sys.argv) > 1 else SIMS

    failed, t0 = [], time.time()
    for n, sim in enumerate(sims, 1):
        out_path = os.path.join(OUT_DIR.format(sim=sim), OUT_NAME)
        if RESUME and os.path.exists(out_path):
            print(f'{sim}: exists, skipping ({n}/{len(sims)})', flush=True)
            continue

        t = time.time()
        try:
            run_sim(sim, tag=f'({n}/{len(sims)})')
            el = (time.time() - t0) / 60
            done = n - len(failed)
            eta = (el / done * (len(sims) - n)) if done else float('nan')
            log(f'{sim}: done in {(time.time() - t) / 60:.1f} min  '
                f'[{n}/{len(sims)}, elapsed {el:.0f} min, eta {eta:.0f} min]')
        except Exception:
            failed.append(sim)
            log(f'{sim}: FAILED\n{traceback.format_exc()}')

    if len(sims) > 1:
        log(f'batch done in {(time.time() - t0) / 3600:.1f} h; '
            f'{len(failed)} failed: {failed}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())