#Compute Redshift space correlation function \xi(s, \mu) for each mock realization for each DESI tracer. 
#Uses 4 random catalogs for DR, RR, except BGS (1) and QSO (2) -- see random/data ratio check.


import numpy as np
import time
import os
import fitsio
from pathlib import Path

from pycorr import TwoPointCorrelationFunction, TwoPointEstimator, NaturalTwoPointEstimator, utils, setup_logging
from cosmoprimo.fiducial import DESI

import psutil
psutil.Process().nice(10)

ROOT = 'https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/mocks/AbacusSummit'
LOCAL_ROOT = Path('/spiffball/cosweeney/mocks/AbacusSummit')

NRAN, NMOCK = 4, 25

NRAN_BY_TRACER = {'BGS_BRIGHT-21.5': 1, 'QSO': 2}

TRACERS = {
    'LRG':             ('dark', 'v4.2'), # compute these first, more useful atm
    'BGS_BRIGHT-21.5': ('bright', 'v1'),
    'ELG_LOPnotqso':   ('dark', 'v4.2'),
    'QSO':             ('dark', 'v4.2'),
}

EDGES = (np.linspace(0., 200., 201), np.linspace(-1., 1., 201))

ZBINS = {
    'BGS_BRIGHT-21.5': [(0.1, 0.4)],
    'LRG':             [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)],
    'ELG_LOPnotqso':   [(0.8, 1.1), (1.1, 1.6)],
    'QSO':             [(0.8, 2.1)],
}

setup_logging()
cosmo = DESI(engine='camb')

COLS = ['RA', 'DEC', 'Z', 'WEIGHT', 'WEIGHT_FKP']

NTHREADS = 45 #os.cpu_count()

def load_cat(path, zmin, zmax):
    cat = fitsio.read(path, columns=COLS)
    cat = cat[(cat['Z'] > zmin) & (cat['Z'] < zmax)]
    pos = [cat['RA'], cat['DEC'], cosmo.comoving_radial_distance(cat['Z'])]
    return pos, cat['WEIGHT'] * cat['WEIGHT_FKP']

def measure_cap(d, tracer, cap, zmin, zmax, nran):
    data_pos, data_w = load_cat(d / f'{tracer}_{cap}_clustering.dat.fits', zmin, zmax)
    rans = [load_cat(d / f'{tracer}_{cap}_{n}_clustering.ran.fits', zmin, zmax)
            for n in range(nran)]
    ran_pos = [np.concatenate([r[0][k] for r in rans]) for k in range(3)]
    ran_w = np.concatenate([r[1] for r in rans])
    return TwoPointCorrelationFunction(
        'smu', EDGES,
        data_positions1=data_pos, data_weights1=data_w,
        randoms_positions1=ran_pos, randoms_weights1=ran_w,
        position_type='rdd', estimator='landyszalay', nthreads=NTHREADS,
    )

tasks = [(t, prog, i, zmin, zmax)
         for t, (prog, _) in TRACERS.items()
         for zmin, zmax in ZBINS[t]
         for i in range(NMOCK)]

setup_logging(level='warning')
durations = {}
t_start = time.time()

for k, (tracer, prog, i, zmin, zmax) in enumerate(tasks):
    d = LOCAL_ROOT / prog / f'altmtl{i}'
    out = d / f'{tracer}_GCcomb_z{zmin}-{zmax}_xismu.npy'
    if out.exists():
        continue

    t0 = time.time()
    try:
        nran = NRAN_BY_TRACER.get(tracer, NRAN)
        result = sum(measure_cap(d, tracer, cap, zmin, zmax, nran) for cap in ('NGC', 'SGC'))
        result.save(out)
    except Exception as e:
        print(f'FAILED {tracer} z{zmin}-{zmax} mock{i}: {e}', flush=True)
        continue

    dt = time.time() - t0
    durations.setdefault(tracer, []).append(dt)

    remaining = sum(np.mean(durations.get(t, [dt])) for t, _, _, _, _ in tasks[k + 1:])
    print(f'[{k + 1}/{len(tasks)}] {tracer} z{zmin}-{zmax} mock{i}  '
          f'{dt / 60:.1f} min | elapsed {(time.time() - t_start) / 3600:.1f} h | '
          f'ETA {remaining / 3600:.1f} h', flush=True)