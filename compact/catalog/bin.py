import os
import numpy as np
import h5py as h5
import pandas as pd

def bin_pairwise_cat(raw_file_path, save_dir, R_bins, rlos_bins, M_bins):
    """
    Reads HDF5 galaxy catalog data, bins it by R and rlos, and saves the result.
    
    Parameters:
    - raw_file_path: str, path to the raw hdf5 file
    - save_dir: str, directory where the binned data should be saved
    - R_bins: array-like, the bin edges for R
    - rlos_bins: array-like, the bin edges for rlos
    
    Returns:
    - binned_dict: A nested dictionary structured as dict[R_bin_idx][rlos_bin_idx]['vpeclos']
    - master_df: The full Pandas DataFrame for advanced querying.
    """
    Mcens = (M_bins[:, 0]*M_bins[:, 1])**0.5  

    all_dfs = []
    
    # 1. Read and compile all data into a single DataFrame
    with h5.File(raw_file_path, 'r') as hdf:
        for k in range(len(Mcens)):
            # Load data for mass bin 'k'
            df = pd.DataFrame({
                'R': hdf[f'R/{k}'][()],
                'rlos': hdf[f'rlos/{k}'][()],
                'vpeclos': hdf[f'vpeclos/{k}'][()],
                'vr': hdf[f'vr/{k}'][()],
                'vphyslos': hdf[f'vphyslos/{k}'][()],
                'Mvir': Mcens[k]*np.ones(len(hdf[f'R/{k}'][()])) # assign same mass to all
            })

        
            # 2. Apply binning using pd.cut
            # labels=False returns integer indices for the bins
            df['M_bin'] = k #pd.cut(df['Mvir'], bins=M_bins, labels=False, include_lowest=True)
            df['R_bin'] = pd.cut(df['R'], bins=R_bins, labels=False, include_lowest=True)
            df['rlos_bin'] = pd.cut(df['rlos'], bins=rlos_bins, labels=False, include_lowest=True)
            
            # Drop data points that fall completely outside your defined bin edges
            df = df.dropna(subset=['M_bin', 'R_bin', 'rlos_bin']).copy()
            
            # Convert bin labels to integers
            df['M_bin'] = df['M_bin'].astype(int)
            df['R_bin'] = df['R_bin'].astype(int)
            df['rlos_bin'] = df['rlos_bin'].astype(int)
            
            all_dfs.append(df)
            
    # Combine everything
    master_df = pd.concat(all_dfs, ignore_index=True)
    
    # 3. Generate an indicative filename based on the bin inputs
    # Format: binned_R[len]_[min]-[max]_rlos[len]_[min]-[max].h5
    filename = (
        f"binned_"
        f"M{len(M_bins)-1}_{M_bins[0, 0]:.2e}-{M_bins[-1, -1]:.2e}_" 
        f"R{len(R_bins)-1}_{R_bins[0]:.1f}-{R_bins[-1]:.1f}_"
        f"rlos{len(rlos_bins)-1}_{rlos_bins[0]:.1f}-{rlos_bins[-1]:.1f}.h5"
    )
    save_path = os.path.join(save_dir, filename)
    
    # 4. Save the DataFrame to disk for easy retrieval later
    # Format 'table' allows for fast querying directly from the hard drive later if needed
    master_df.to_hdf(save_path, key='binned_data', mode='w', format='table')
    print(f"Successfully saved binned data to: {save_path}")
    
    # 5. Build a nested dictionary for immediate in-script analysis
    binned_dict = {}
    
    # Pandas groupby handles the 3D separation (R, rlos) in one fast sweep
    for (k, r_i, rlos_i), group in master_df.groupby(['M_bin', 'R_bin', 'rlos_bin']):
        if k not in binned_dict:
            binned_dict[k] = {}
        if r_i not in binned_dict[k]:
            binned_dict[k][r_i] = {}
            
        binned_dict[k][r_i][rlos_i] = {
            'vpeclos': group['vpeclos'].values,
            'vr': group['vr'].values,
            'vphyslos': group['vphyslos'].values,
            'rlos': group['rlos'].values,
            'R': group['R'].values,
            'Mvir': group['Mvir'].values
        }
        
    return binned_dict, master_df


def load_pairwise_cat(file_path):
    """
    Loads binned HDF5 pair catalog data saved by bin_pairwise_cat.
    
    Parameters:
    - file_path: str, the exact path to the saved .h5 file
    
    Returns:
    - binned_dict: The reconstructed nested dictionary dict[R_bin_idx][rlos_bin_idx]
    - master_df: The full loaded Pandas DataFrame
    """
    # 1. Load the master DataFrame using the key we defined during saving
    master_df = pd.read_hdf(file_path, key='binned_data')
    
    # 2. Reconstruct the nested dictionary using the exact same groupby logic
    binned_dict = {}
    
    for (k, r_i, rlos_i), group in master_df.groupby(['M_bin', 'R_bin', 'rlos_bin']):
        if k not in binned_dict:
            binned_dict[k] = {}
        if r_i not in binned_dict[k]:
            binned_dict[k][r_i] = {}
            
        binned_dict[k][r_i][rlos_i] = {
            'vpeclos': group['vpeclos'].values,
            'vphyslos': group['vphyslos'].values,
            'vr': group['vr'].values,
            'rlos': group['rlos'].values,
            'R': group['R'].values,
            'Mvir': group['Mvir'].values
        }
        
    return binned_dict, master_df