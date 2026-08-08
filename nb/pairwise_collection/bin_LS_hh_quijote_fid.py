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

# define fid sim boxes 
sim_nums = np.array([0, 41, 53, 55, 66])

# mass bins
log_edges = np.array([[13.2, 13.3], [13.5, 13.7], [14.1, 14.7]])
massbins = 10**log_edges

box_length = 1000 #Mpc/h, Quijote box side length

sim_cosmo = [0.3175, 0.049, 0.6711, 0.9624, 0.834]

params = {'flat': True, 'H0': 100 * sim_cosmo[2], 'Om0': sim_cosmo[0], 'Ob0': sim_cosmo[1], 'sigma8': sim_cosmo[4], 'ns': sim_cosmo[3]}

cosmo = cosmology.setCosmology('myCosmo', params)


print('Starting pair selection for 5 fiducial sims, 3 mass bins...')
for i in range(len(sim_nums)):

    sim_num = sim_nums[i]

    # compile halo-halo pairs 
    halo_path = f'/spiffball/edgarmsc/simulations/Quijote/Halos/Rockstar/fiducial_HR/{sim_num}/hlist_4.hdf5'

    hsb_path = f'/spiffball/cosweeney/simulations/Quijote/halos/fid_HR/{sim_num}/SBp/'

    vel_path = f'/spiffball/cosweeney/simulations/Quijote/halos/fid_HR/{sim_num}/data/pairwise_velocity_selections/'

    os.makedirs(os.path.dirname(hsb_path), exist_ok=True)
    os.makedirs(os.path.dirname(vel_path), exist_ok=True)

    vel_path = vel_path+'halo_halo_pairs_no_vlos_cut_3massbins_SB250.hdf5'

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