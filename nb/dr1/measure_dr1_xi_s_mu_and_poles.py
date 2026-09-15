#Compute Redshift-space correlation function \xi(s, \mu) for each DESI DR1
#pre-recon tracer/redshift bin, downloading the LSS clustering catalogs as
#needed, then convert to multipoles saved in a plain-numpy (pycorr-agnostic)
#format. Uses the tracer-specific random counts from DESI 2024 II Sec. 8.1
#(BGS: 1, LRG: 8, ELG: 10, QSO: 4) rather than a flat 4.

import numpy as np
import time
import os
import requests
import fitsio
from pathlib import Path

from pycorr import TwoPointCorrelationFunction, setup_logging
from cosmoprimo.fiducial import DESI

import psutil
psutil.Process().nice(10)

REMOTE_ROOT = 'https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5'
LOCAL_ROOT = Path('/spiffball/cosweeney/DESI/DR1/LSS/v1.5')

NRAN_BY_TRACER = {'BGS_BRIGHT-21.5': 1, 'LRG': 8, 'ELG_LOPnotqso': 10, 'QSO': 4}

TRACERS = ['LRG', 'BGS_BRIGHT-21.5', 'ELG_LOPnotqso', 'QSO']

EDGES = (np.linspace(0., 200., 201), np.linspace(-1., 1., 201))

ZBINS = {
    'BGS_BRIGHT-21.5': [(0.1, 0.4)],
    'LRG':             [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)],
    'ELG_LOPnotqso':   [(0.8, 1.1), (1.1, 1.6)],
    'QSO':             [(0.8, 2.1)],
}

SRANGE = (20., 200.)
ELLS = (0, 2, 4)

setup_logging()
cosmo = DESI(engine='camb')

COLS = ['RA', 'DEC', 'Z', 'WEIGHT', 'WEIGHT_FKP']

NTHREADS = 45 #os.cpu_count()


def ensure_downloaded(tracer, cap, nran):
    """Download the data file and nran random files for one tracer/cap, if not already local."""
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    fnames = [f'{tracer}_{cap}_clustering.dat.fits']
    fnames += [f'{tracer}_{cap}_{n}_clustering.ran.fits' for n in range(nran)]

    for fn in fnames:
        local_path = LOCAL_ROOT / fn
        if local_path.exists():
            continue
        url = f'{REMOTE_ROOT}/{fn}'
        r = requests.get(url)
        if r.status_code != 200:
            raise RuntimeError(
                f'Download failed ({r.status_code}) for {url}. '
                'Check the filename convention against the live directory listing.'
            )
        local_path.write_bytes(r.content)


def load_cat(path, zmin, zmax):
    cat = fitsio.read(path, columns=COLS)
    cat = cat[(cat['Z'] > zmin) & (cat['Z'] < zmax)]
    pos = [cat['RA'], cat['DEC'], cosmo.comoving_radial_distance(cat['Z'])]
    return pos, cat['WEIGHT'] * cat['WEIGHT_FKP']


def measure_cap(tracer, cap, zmin, zmax, nran):
    ensure_downloaded(tracer, cap, nran)
    data_pos, data_w = load_cat(LOCAL_ROOT / f'{tracer}_{cap}_clustering.dat.fits', zmin, zmax)
    rans = [load_cat(LOCAL_ROOT / f'{tracer}_{cap}_{n}_clustering.ran.fits', zmin, zmax)
            for n in range(nran)]
    ran_pos = [np.concatenate([r[0][k] for r in rans]) for k in range(3)]
    ran_w = np.concatenate([r[1] for r in rans])
    return TwoPointCorrelationFunction(
        'smu', EDGES,
        data_positions1=data_pos, data_weights1=data_w,
        randoms_positions1=ran_pos, randoms_weights1=ran_w,
        position_type='rdd', estimator='landyszalay', nthreads=NTHREADS,
    )


tasks = [(t, zmin, zmax) for t in TRACERS for zmin, zmax in ZBINS[t]]

setup_logging(level='warning')
t_start = time.time()

for k, (tracer, zmin, zmax) in enumerate(tasks):
    xismu_out = LOCAL_ROOT / f'{tracer}_GCcomb_z{zmin}-{zmax}_xismu.npy'
    poles_out = LOCAL_ROOT / f'{tracer}_z{zmin}-{zmax}_poles.npz'
    if poles_out.exists():
        continue

    t0 = time.time()
    try:
        if xismu_out.exists():
            result = TwoPointCorrelationFunction.load(xismu_out)
        else:
            nran = NRAN_BY_TRACER[tracer]
            result = sum(measure_cap(tracer, cap, zmin, zmax, nran) for cap in ('NGC', 'SGC'))
            result.save(xismu_out)
    except Exception as e:
        print(f'FAILED {tracer} z{zmin}-{zmax}: {e}', flush=True)
        continue

    # pycorr needed only up to here; the .npz below (s, poles, ells as plain
    # arrays) is readable with plain numpy, no pycorr install required.
    s, poles = result.select(SRANGE)(ells=ELLS, return_sep=True)
    np.savez(poles_out, s=s, poles=np.array(poles), ells=np.array(ELLS))

    dt = time.time() - t0
    print(f'[{k + 1}/{len(tasks)}] {tracer} z{zmin}-{zmax}  '
          f'{dt / 60:.1f} min | elapsed {(time.time() - t_start) / 60:.1f} min', flush=True)