#%%
import numpy as np
from pathlib import Path

from pycorr import TwoPointCorrelationFunction

LOCAL_ROOT = Path('/spiffball/cosweeney/mocks/AbacusSummit')
NMOCK = 25
SRANGE = (20., 200.)
ELLS = (0, 2, 4)

TRACERS = {
    'LRG':             'dark',
    'BGS_BRIGHT-21.5': 'bright',
    'ELG_LOPnotqso':   'dark',
    'QSO':             'dark',
}

ZBINS = {
    'BGS_BRIGHT-21.5': [(0.1, 0.4)],
    'LRG':             [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)],
    'ELG_LOPnotqso':   [(0.8, 1.1), (1.1, 1.6)],
    'QSO':             [(0.8, 2.1)],
}

#%%
tasks = [(t, prog, zmin, zmax)
         for t, prog in TRACERS.items()
         for zmin, zmax in ZBINS[t]]

for tracer, prog, zmin, zmax in tasks:
    out = LOCAL_ROOT / prog / f'{tracer}_z{zmin}-{zmax}_poles.npz'
    if out.exists():
        continue

    s, poles = None, []
    for i in range(NMOCK):
        p = (LOCAL_ROOT / prog / f'altmtl{i}' /
             f'{tracer}_GCcomb_z{zmin}-{zmax}_xismu.npy')
        if not p.exists():
            continue
        r = TwoPointCorrelationFunction.load(p).select(SRANGE)
        s, pole = r(ells=ELLS, return_sep=True)
        poles.append(pole)

    if not poles:
        print(f'no mocks found for {tracer} z{zmin}-{zmax}', flush=True)
        continue

    np.savez(out, s=s, poles=np.array(poles), ells=np.array(ELLS))
    print(f'{tracer} z{zmin}-{zmax}: {len(poles)} mocks -> {out}', flush=True)
# %%
