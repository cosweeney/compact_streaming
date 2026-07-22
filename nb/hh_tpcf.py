# imports
import h5py as h5
import numpy as np
from fasttpcf.tpcf import *
from typing import List, Tuple, Union

import psutil
psutil.Process().nice(19)

def generate_bins(
    bmin: float,
    bmax: float,
    nbins: int,
    logspaced: bool = True,
    soft: float = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates equally spaced bins in linear or logarithmic (base 10) space.

    Parameters
    ----------
    bmin : float
        Minimum or lower bound (inclusive)
    bmax : float
        Maximum or upper bound (inclusive)
    nbins : int
        Number of bins to generate
    logspaced : bool, optional
        Generate log-spaced bins, by default True
    soft : float, optional
        If there is softening value, then bmin >= soft, by default 0

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Bins and edges in units of bmin and bmax
    """
    if bmin < soft:
        bmin = soft
    # Equally log-spaced bins
    if bmin == 0:
        logspaced = False

    if logspaced:
        r_edges = np.logspace(
            start=np.log10(bmin),
            stop=np.log10(bmax),
            num=nbins + 1,
            base=10,
        )
    else:
        r_edges = np.linspace(
            start=bmin,
            stop=bmax,
            num=nbins + 1,
        )
    # Bin middle point
    rbins = 0.5 * (r_edges[1:] + r_edges[:-1])
    return rbins, r_edges

# define radial bins 
r_cens, r_edges = generate_bins(0.01, 150, 70+1, logspaced=True) 

# path to MDPL2 halos and for data
halo_path = '/spiff/cosweeney/simulations/MDPL2/hlists/hlist_0.83760_update.hdf5'
save_path = '/spiffball/cosweeney/simulations/MDPL2/data/CorrFuncs/xi_hh_150.hdf5'

# sim quantities
L_box = 1_000 # h^-1 Mpc
g_box = L_box // 5 # grid size

with h5.File(halo_path, 'r') as hdf:
    hm_cut = hdf['mvir'][()] >= 1e13 # halo mass, h^-1 M_sun

    hpos = np.array([
        hdf['x'][()][hm_cut],
        hdf['y'][()][hm_cut], 
        hdf['z'][()][hm_cut],
    ], dtype=np.float64).T 

    hpos = np.mod(hpos, L_box)

    print('Computing gg correlation function...')
    xi, xi_samples, xi_mean, xi_cov = cross_tpcf_jk(hpos, hpos, r_edges, boxsize=L_box, gridsize=g_box, nthreads=84)

    print('Saving results to: '+save_path)
    with h5.File(save_path, 'w') as f:
        f.create_dataset(name=f'xi', data=xi, dtype=np.float64)
        f.create_dataset(name=f'xi_samples', data=xi_samples, dtype=np.float64)
        f.create_dataset(name=f'xi_mean', data=xi_mean, dtype=np.float64)
        f.create_dataset(name=f'xi_cov', data=xi_cov, dtype=np.float64)
        f.create_dataset(name=f'r_cens', data=r_cens, dtype=np.float64)
        f.create_dataset(name=f'r_edges', data=r_edges, dtype=np.float64)

print('Done.')