import numpy as np 
import h5py as h5 
import matplotlib.pyplot as plt

# plotting config
import matplotlib
matplotlib.rcParams.update({'text.usetex': True, 
                            'font.family': 'Computer Modern Roman'})

from nb.pairwise_collection.bin_utils import *

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

# plotting config
import matplotlib
matplotlib.rcParams.update({'text.usetex': True, 
                            'font.family': 'Computer Modern Roman'})
plt.rcParams['axes.labelsize'] = 20


from colossus.cosmology import cosmology
import pandas as pd

from nb.pairwise_collection.bin_utils import *

import psutil
psutil.Process().nice(10)

import os

# define LH selected cosmologies 
cosmo_nums = np.array([1992, 1994, 1927, 1986, 1852])

available_sim_nums = np.array([                                                    # only this subset of the 2000 sims are available on spiffball
    1402, 1655, 1740, 1791, 1807, 1820, 1839, 1861, 1889, 1902, 1911, 1915, 1923,
    1930, 1936, 1945, 1952, 1962, 1966, 1974, 1979, 1984, 1988, 1992, 1996,
    1625, 1656, 1757, 1793, 1810, 1824, 1852, 1871, 1894, 1906, 1912, 1916, 1927,
    1933, 1937, 1948, 1955, 1963, 1967, 1975, 1980, 1985, 1989, 1993, 1997,
    1630, 1659, 1764, 1795, 1818, 1826, 1857, 1877, 1897, 1907, 1913, 1919, 1928,
    1934, 1943, 1949, 1956, 1964, 1970, 1976, 1982, 1986, 1990, 1994, 1998,
    1640, 1704, 1769, 1797, 1819, 1835, 1859, 1887, 1898, 1909, 1914, 1921, 1929,
    1935, 1944, 1950, 1960, 1965, 1972, 1977, 1983, 1987, 1991, 1995, 590,
])

sim_nums = [i for i in range(2_000)] # all cosmologies

q_cosmos = np.zeros((len(sim_nums), 5))

for num in sim_nums:
    dat_path = f'/spiffball/edgarmsc/simulations/Quijote/Snapshots/latin_hypercube_HR_cosmo/Cosmo_params_{num}.dat'

    df = pd.read_csv(dat_path, sep=r'\s+', header=None, names=['Om', 'Ob', 'h', 'ns', 's8'])

    q_cosmos[num] = df.values[0]

LH_cosmos = q_cosmos[cosmo_nums]

# mass bins
dex_width = 0.1  # bin width in log10(mass); narrower = 0.1, wider = 0.3
log_edges = np.arange(np.log10(1e13), np.log10(1e15) + dex_width, dex_width)
massbins = np.column_stack([10**log_edges[:-1], 10**log_edges[1:]])[::6] # select 4 bins from the full set of mass bins

box_length = 1000 #Mpc/h, Quijote box side length

print('Starting pair selection for 5 cosmos, 4 mass bins...')
for i in range(len(cosmo_nums)):

    sim_num = cosmo_nums[i]
    sim_cosmo = LH_cosmos[i]

    params = {'flat': True, 'H0': 100 * sim_cosmo[2], 'Om0': sim_cosmo[0], 'Ob0': sim_cosmo[1], 'sigma8': sim_cosmo[4], 'ns': sim_cosmo[3]}

    cosmo = cosmology.setCosmology('myCosmo', params)

    # compile halo-halo pairs 
    halo_path = f'/spiffball/edgarmsc/simulations/Quijote/Halos/Rockstar/latin_hypercube_HR/{sim_num}/hlist_4.hdf5'

    hsb_path = f'/spiffball/cosweeney/simulations/Quijote/halos/{sim_num}/SBp/'

    vel_path = f'/spiffball/cosweeney/simulations/Quijote/halos/{sim_num}/data/pairwise_velocity_selections/halo_halo_pairs_no_vlos_cut_1e13_mass_cut_SB250.hdf5'

    os.makedirs(os.path.dirname(hsb_path), exist_ok=True)
    os.makedirs(os.path.dirname(vel_path), exist_ok=True)

    # split into subboxes
    with h5.File(halo_path, 'r') as hcat:
        hm_cut = hcat['Mvir'][()] >= 1e13

        hpos = np.array([
            hcat['X'][()][hm_cut],
            hcat['Y'][()][hm_cut], 
            hcat['Z'][()][hm_cut],
        ]).T 

        hvel = np.array([
            hcat['VX'][()][hm_cut],
            hcat['VY'][()][hm_cut], 
            hcat['VZ'][()][hm_cut],
        ]).T

        hid = hcat['ID'][()][hm_cut]
        upid = hcat['PID'][()][hm_cut]

        generate_sub_box_ids(hpos%box_length, box_length, box_length / 4, hid.shape[0] // 1000, hsb_path, name='halo')

        split_simulation_into_sub_boxes(
            positions=hpos%box_length, 
            velocities=hvel, 
            ids=hid, 
            upids=upid,
            boxsize=box_length, 
            subsize=box_length / 4,
            chunksize=hid.shape[0] // 1000, 
            dtypes=[hpos[0].dtype, hvel[0].dtype], 
            path=hsb_path, 
            name='halo'
        )
    
    print('Iterating over mass bins for sim_num:', sim_num)
    with h5.File(vel_path, 'w') as hdf:
        for k in range(len(massbins)):
                

            with h5.File(halo_path, 'r') as hcat:

                mcut = (hcat['Mvir'][()] >= massbins[k][0]) & (hcat['Mvir'][()] <= massbins[k][1]) # h^-1 M_sun

                hid = hcat['ID'][()][mcut]

                rel_r, rel_R, rel_rlos, vpeclos, vphyslos, rel_vr, rel_vt, hupid, hsbid = select_halos_around_haloes_parallel( len(hid), 
                                                                                                                            140, 
                                                                                                                            box_length, 
                                                                                                                            box_length / 4, 
                                                                                                                            halo_path, 
                                                                                                                            hsb_path, 
                                                                                                                            mass_mask=mcut, 
                                                                                                                            H=cosmo.Hz(0), 
                                                                                                                            h=cosmo.h, 
                                                                                                                            n_jobs=84)  
                
                hdf.create_dataset(name=f'r/{k}', data=rel_r, dtype=np.float64) 
                hdf.create_dataset(name=f'R/{k}', data=rel_R, dtype=np.float64) 
                hdf.create_dataset(name=f'rlos/{k}', data=rel_rlos, dtype=np.float64) 
                hdf.create_dataset(name=f'vpeclos/{k}', data=vpeclos, dtype=np.float64) 
                hdf.create_dataset(name=f'vphyslos/{k}', data=vphyslos, dtype=np.float64) 
                hdf.create_dataset(name=f'vr/{k}', data=rel_vr, dtype=np.float64)
                hdf.create_dataset(name=f'vt/{k}', data=rel_vt, dtype=np.float64) 
                hdf.create_dataset(name=f'hupid/{k}', data=hupid, dtype=np.float64)
                hdf.create_dataset(name=f'hsbid/{k}', data=hsbid, dtype=np.float64)

print('Done.')