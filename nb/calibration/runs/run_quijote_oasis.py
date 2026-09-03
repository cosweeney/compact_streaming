"""PairHist for the Oasis halo catalogues: 10 Quijote LH sims x several z.

    python run_quijote_oasis.py                  # every (sim, z)
    python run_quijote_oasis.py 1997             # one sim, every z
    python run_quijote_oasis.py 1997 1.0         # one sim, one z

Same single-sample auto-correlation setup as run_quijote.py, but:
  * catalogues are Oasis, with 'pos' (N,3), 'vel' (N,3), 'Norb' (N,)
  * the mass bin is on Norb -- ORBITING particles, a different mass definition
    from Rockstar's, so the same 300-400 cut does NOT select the same halos
  * m_p is z-independent, so the mass edges are the same at every redshift;
    the ABUNDANCE is not, and falls off a cliff above z ~ 1 (see N_MIN_HALOS)

CHECK BEFORE RUNNING: CAT_TMPL below is a guess at the directory separators --
the path fragment as given concatenated without them.
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

from compact.velocities.pairhist import PairHist
from compact.velocities.pairhist_direct import get_pairwise_hist
from accumulate import generate_sub_box_ids, split_simulation_into_sub_boxes

# ---------------------------------------------------------------- config ----
SIMS = [1997, 1996, 1995, 1992, 1988, 1993, 1990, 1994, 1962, 1976]
REDSHIFTS = [0.5, 1, 2, 3]        # ordered low -> high; the sparse ones last

# i_z is the INDEX into the catalogue's own redshift ordering.  Confirm both
# the separators and the index convention against a real path on disk.
Z_INDEX = {3: 0, 2: 1, 1: 2, 0.5: 3}

CAT_TMPL = ('/spiffball/baylihayes/Oasis/Quijote/'
            'test_latinhypercube{sim}_{i_z}/'
            'run_{sim}/final_catalogue/catalogue.hdf5')
COSMO_TMPL = ('/spiffball/edgarmsc/simulations/Quijote/Snapshots/'
              'latin_hypercube_HR_cosmo/Cosmo_params_{sim}.dat')
OUT_DIR = ('/spiffball/cosweeney/simulations/Quijote/halos/'
           'oasis_latin_hypercube_HR/{sim}/z{z}/data/')
SB_TMPL = ('/spiffball/cosweeney/simulations/Quijote/halos/'
           'oasis_latin_hypercube_HR/{sim}/z{z}/SBp/')
LOG_TMPL = ('/spiffball/cosweeney/simulations/Quijote/logs/'
            'oasis/pairhist_{sim}_z{z}.log')
OUT_NAME = 'pairhist_massbin.h5'

COSMO_COLS = ['Om', 'Ob', 'h', 'ns', 's8']

box_length = 1000.0
subsize = box_length / 2          # padding subsize/2 = 250 > r_max
r_max = 140.0

N_PART = 1024
N_LO, N_HI = 300, 400             # the mass bin, in ORBITING particles
N_MIN_HALOS = 5_000               # below this the run is not worth doing

R_edges = r_perp                  # noqa: F821
rlos_edges = r_los                # noqa: F821
V_LO, V_HI, N_V = -4000.0, 4000.0, 400
CONV = 1.0
VEL_CHECK = (50.0, 3000.0)        # wider than the z=0 script: sigma grows with z

RESUME = True
KEEP_SUBBOXES = False
N_JOBS = int(os.environ.get('PAIRHIST_NJOBS', 64))
TASKS_PER_SUB_BOX = int(os.environ.get('PAIRHIST_TASKS_PER_SB', 8))

_log_path = None


def log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    if _log_path:
        with open(_log_path, 'a') as f:
            f.write(line + '\n')


def zlabel(z):
    return f'{z:g}'.replace('.', 'p')      # 0.5 -> '0p5', 1 -> '1'


def read_cosmology(sim):
    df = pd.read_csv(COSMO_TMPL.format(sim=sim), sep=r'\s+', header=None,
                     names=COSMO_COLS)
    return dict(zip(COSMO_COLS, df.values[0]))


def particle_mass(om0):
    """Quijote particle mass [Msun/h]; independent of redshift."""
    return 27.75e10 * om0 * box_length**3 / N_PART**3


def run_one(sim, z, tag=''):
    global _log_path
    zl = zlabel(z)
    _log_path = LOG_TMPL.format(sim=sim, z=zl)
    os.makedirs(os.path.dirname(_log_path), exist_ok=True)

    cat_path = CAT_TMPL.format(sim=sim, i_z=Z_INDEX[z])
    sb_path = SB_TMPL.format(sim=sim, z=zl)
    out_dir = OUT_DIR.format(sim=sim, z=zl)
    out_path = os.path.join(out_dir, OUT_NAME)

    if not os.path.exists(cat_path):
        raise FileNotFoundError(cat_path)

    cosmo = read_cosmology(sim)
    m_p = particle_mass(cosmo['Om'])
    M_LO, M_HI = N_LO * m_p, N_HI * m_p

    log(f'=== sim {sim} z={z} {tag} ===')
    log(f'{sim} z{zl}: Om {cosmo["Om"]:.4f} Ob {cosmo["Ob"]:.4f} '
        f'h {cosmo["h"]:.4f} ns {cosmo["ns"]:.4f} s8 {cosmo["s8"]:.4f}')
    log(f'{sim} z{zl}: m_p {m_p:.3e}, M_orb bin '
        f'[{M_LO:.3e}, {M_HI:.3e}] Msun/h')

    log(f'{sim} z{zl}: reading {cat_path}')
    with h5.File(cat_path, 'r') as hdf:
        pos = np.asarray(hdf['pos'][()], dtype=np.float64) % box_length
        vel = np.asarray(hdf['vel'][()], dtype=np.float64)
        norb = hdf['Norb'][()]
    if pos.shape[1] != 3 or vel.shape != pos.shape:
        raise ValueError(f'unexpected shapes: pos {pos.shape}, vel {vel.shape}')
    hid = np.arange(len(pos), dtype=np.int64)      # no ID column in Oasis
    gc.collect()

    sig1d = float(np.std(CONV * vel[:, 2]))
    log(f'{sim} z{zl}: 1-D velocity dispersion {sig1d:.1f} km/s '
        f'(CONV = {CONV:g})')
    if not (VEL_CHECK[0] <= sig1d <= VEL_CHECK[1]):
        raise ValueError(
            f'sim {sim} z{z}: sigma_1D = {sig1d:.3e} km/s outside {VEL_CHECK} '
            f'-- check CONV and the velocity convention at this redshift')

    nb = (norb >= N_LO) & (norb <= N_HI)
    n_halos = int(nb.sum())
    n_bar = n_halos / box_length**3
    log(f'{sim} z{zl}: {n_halos:,} halos in bin, n_bar {n_bar:.3e}, '
        f'{n_bar * np.pi * r_max**2 * 2 * r_max:,.0f} pairs/halo')
    if n_halos < N_MIN_HALOS:
        raise ValueError(
            f'sim {sim} z{z}: only {n_halos:,} halos in the bin '
            f'(< N_MIN_HALOS = {N_MIN_HALOS:,}) -- too sparse to be useful')

    shutil.rmtree(sb_path, ignore_errors=True)
    os.makedirs(sb_path, exist_ok=True)
    chunksize = max(n_halos // 1000, 1)
    generate_sub_box_ids(pos[nb], box_length, subsize, chunksize,
                         sb_path, name='halo')
    split_simulation_into_sub_boxes(
        positions=pos[nb], velocities=vel[nb], ids=hid[nb], upids=hid[nb],
        boxsize=box_length, subsize=subsize, chunksize=chunksize,
        dtypes=[pos.dtype, vel.dtype], path=sb_path, name='halo')

    seed_mbin = np.where(nb, 0, -1).astype(np.int32)
    Norb_eff = float(np.mean(norb[nb]))
    log(f'{sim} z{zl}: <Norb> = {Norb_eff:.1f}, '
        f'<log10 M_orb> = {np.log10(Norb_eff * m_p):.4f}')

    h = get_pairwise_hist(
        pos, vel, seed_mbin,
        r_max=r_max, boxsize=box_length, subsize=subsize, path=sb_path,
        n_M=1, R_edges=R_edges, rlos_edges=rlos_edges,
        v_lo=V_LO, v_hi=V_HI, n_v=N_V,
        conv=CONV, n_jobs=N_JOBS, tasks_per_sub_box=TASKS_PER_SUB_BOX,
    )

    off = int((h.n - h.counts.sum(-1)).sum())
    _, sd = h.mean_std(0)
    occ = h.n[0][h.n[0] > 0]
    log(f'{sim} z{zl}: {h.n.sum():,} pairs ({h.n.sum() / n_halos:,.0f}/halo), '
        f'off-grid {off} ({off / max(h.n.sum(), 1):.1e}), '
        f'cells {(h.n[0] > 0).sum()}/{h.n[0].size}, '
        f'min/median cell n {occ.min() if occ.size else 0}/'
        f'{np.median(occ) if occ.size else 0:.0f}, '
        f'median sigma {np.nanmedian(sd):.1f} km/s')

    os.makedirs(out_dir, exist_ok=True)
    h.save(out_path)
    with h5.File(out_path, 'a') as hdf:
        hdf.attrs.update(sim=str(sim), redshift=float(z),
                         suite='Quijote_latin_hypercube_HR', finder='Oasis',
                         mass_def='Morb', sample='auto',
                         boxsize=box_length, subsize=subsize, r_max=r_max,
                         conv=CONV, m_part=m_p,
                         n_part_lo=N_LO, n_part_hi=N_HI,
                         M_lo=M_LO, M_hi=M_HI, Norb_eff=Norb_eff,
                         n_halos=n_halos, sigma1d_cat=sig1d,
                         **{k: float(v) for k, v in cosmo.items()})
    log(f'{sim} z{zl}: saved {out_path}')

    if not KEEP_SUBBOXES:
        shutil.rmtree(sb_path, ignore_errors=True)

    del pos, vel, hid, norb, nb, seed_mbin, h
    gc.collect()


def main():
    sims = [int(sys.argv[1])] if len(sys.argv) > 1 else SIMS
    zs = [float(sys.argv[2])] if len(sys.argv) > 2 else REDSHIFTS
    jobs = [(s, z) for s in sims for z in zs]

    failed, t0 = [], time.time()
    for n, (sim, z) in enumerate(jobs, 1):
        out_path = os.path.join(OUT_DIR.format(sim=sim, z=zlabel(z)), OUT_NAME)
        if RESUME and os.path.exists(out_path):
            print(f'{sim} z{z}: exists, skipping ({n}/{len(jobs)})', flush=True)
            continue

        t = time.time()
        try:
            run_one(sim, z, tag=f'({n}/{len(jobs)})')
            log(f'{sim} z{z}: done in {(time.time() - t) / 60:.1f} min '
                f'[{n}/{len(jobs)}, elapsed {(time.time() - t0) / 60:.0f} min]')
        except Exception:
            failed.append((sim, z))
            log(f'{sim} z{z}: FAILED\n{traceback.format_exc()}')

    if len(jobs) > 1:
        log(f'batch done in {(time.time() - t0) / 60:.0f} min; '
            f'{len(failed)} failed: {failed}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())