
import os
import numba
import h5py as h5
import numpy as np
from time import time
from typing import Tuple
from datetime import timedelta
from tqdm.notebook import tqdm
from typing import Callable, List
from dataclasses import dataclass


import warnings
warnings.filterwarnings('ignore')

from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib



@dataclass(frozen=True)
class COLS:
    """
    """
    HEADER: str = "\033[95m"
    OKBLUE: str = "\033[94m"
    OKCYAN: str = "\033[96m"
    OKGREEN: str = "\033[92m"
    WARNING: str = "\033[93m"
    FAIL: str = "\033[91m"
    ENDC: str = "\033[0m"
    BOLD: str = "\033[1m"
    UNDERLINE: str = "\033[4m"
    BULLET: str = "\u25CF"


OKGOOD = f"{COLS.OKGREEN}{COLS.BULLET}{COLS.ENDC} "
FAIL = f"{COLS.FAIL}{COLS.BULLET}{COLS.ENDC} "


def get_np_unit_dytpe(obj):
    np_unit_dtypes = np.array([np.uint16, np.uint32, np.uint64])
    loc = np.argmax(
        [obj < np.iinfo(item).max for item in np_unit_dtypes])
    return np_unit_dtypes[loc]


def timer(procedure: Callable) -> Callable:
    """Decorator that prints the procedure's execution time

    Parameters
    ----------
    procedure : Callable
        Any callable

    Returns
    -------
    Callable
        Returns callable object/return value
    """

    def wrapper(*args, **kwargs):
        start = time()
        return_value = procedure(*args, **kwargs)
        print(
            f"\t{COLS.BULLET}{COLS.BOLD}{COLS.WARNING} Elapsed time:{COLS.ENDC} "
            + f"{COLS.OKCYAN}{timedelta(seconds=time()-start)}{COLS.ENDC} "
            + f"{COLS.OKGREEN}{procedure.__name__}{COLS.ENDC}"
        )
        return return_value

    return wrapper


def relative_coordinates(
    x0: np.ndarray,
    x: np.ndarray,
    boxsize: float,
    periodic: bool = True
) -> float:
    """Returns the coordinates x relative to x0 accounting for periodic boundary
    conditions

    Parameters
    ----------
    x0 : np.ndarray
        Reference position in cartesian coordinates
    x : np.ndarray
        Position array (N, 3)
    boxsize : float
        Size of simulation box
    periodic : bool, optional
        Set to True if the simulation box is periodic, by default True

    Returns
    -------
    float
        Relative positions
    """
    if periodic:
        return (x - x0 + 0.5*boxsize) % boxsize - 0.5*boxsize
    return x - x0


def generate_sub_box_grid(
    boxsize: float,
    subsize: float,
) -> Tuple[np.ndarray]:
    """Generates a 3D grid of sub-boxes.

    Parameters
    ----------
    boxsize : float
        Size of simulation box
    subsize : float
        Size of sub-box

    Returns
    -------
    Tuple[np.ndarray]
        ID and centre coordinate for all sub-boxes
    """

    # Number of sub-boxes per side
    boxes_per_side = np.int_(np.ceil(boxsize / subsize))

    # Determine data type for integer arrays based on the maximum number of
    # elements
    uint_dtype = get_np_unit_dytpe(boxes_per_side)
    # Set of natural numbers from 0 to N-1
    n_range = np.arange(boxes_per_side, dtype=uint_dtype)

    # Shift in each dimension for numbering sub-boxes
    uint_dtype = get_np_unit_dytpe(boxes_per_side**2)
    shift = np.array(
        [1, boxes_per_side, boxes_per_side * boxes_per_side], dtype=uint_dtype)

    # Set of index vectors. Each vector points to the (i, j, k)-th sub-box
    n_pos = np.int_(cartesian_product([n_range, n_range, n_range]))

    # Set of all possible unique IDs for each sub-box
    ids = np.sum(n_pos * shift, axis=1)
    sort_order = np.argsort(ids)

    # Sort IDs so that the ID matches the row index.
    n_pos = n_pos[sort_order]
    ids = ids[sort_order]

    # Sub-box central coordinate. Populate each sub-box with one point at the
    # centre.
    centres = subsize * (n_pos + 0.5)
    return ids, centres


def get_sub_box_id(
    x: np.ndarray,
    boxsize: float,
    subsize: float,
) -> int:
    """Returns the sub-box ID to which the coordinates `x` fall into

    Parameters
    ----------
    x : np.ndarray
        Position in cartesian coordinates
    boxsize : float
        Size of simulation box
    subsize : float
        Size of sub-box

    Returns
    -------
    int
        ID of the sub-box
    """
    # Number of sub-boxes per side
    boxes_per_side = np.int_(np.ceil(boxsize / subsize))
    # Determine data type for integer arrays based on the maximum number of
    # elements
    uint_dtype = get_np_unit_dytpe(boxes_per_side**3)
    # Shift in each dimension for numbering sub-boxes
    shift = np.array(
        [1, boxes_per_side, boxes_per_side * boxes_per_side], dtype=uint_dtype)
    # In the rare case an object is located exactly at the edge of the box,
    # move it 'inwards' by a tiny amount so that the box id is correct.
    x[np.where(x==boxsize)] -= 1e-8
    x[np.where(x==0)] += 1e-8
    if x.ndim > 1:
        return np.int_(np.sum(shift * np.floor(x / subsize), axis=1))
    else:
        return np.int_(np.sum(shift * np.floor(x / subsize)))


def get_adjacent_sub_box_ids(
    sub_box_id: np.ndarray,
    sub_box_ids: np.ndarray,
    positions: np.ndarray,
    boxsize: float,
    subsize: float,
) -> np.ndarray:
    """Returns a list of all IDs that are adjacent to the specified sub-box ID.
    There are always 27 adjacent boxes in a 3D volume, including the specified ID.

    Parameters
    ----------
    sub_box_id : np.ndarray
        ID of the sub-box
    sub_box_ids : np.ndarray
        IDs of all sub-boxes
    positions : np.ndarray
        Positions of all the centres of the sub-boxes
    boxsize : float
        Size of simulation box
    subsize : float
        Size of sub-box

    Returns
    -------
    np.ndarray
        List of sub-box IDs adjacent to `id`

    Raises
    ------
    ValueError
        If `id` is not found in the allowed values in `ids`
    """
    if sub_box_id not in sub_box_ids:
        raise ValueError(f'ID {sub_box_id} is out of bounds')

    x0 = positions[sub_box_ids == sub_box_id]
    d = relative_coordinates(x0, positions, boxsize)
    d = np.sqrt(np.sum(np.square(d), axis=1))
    mask = d <= 1.01*np.sqrt(3)*subsize
    return sub_box_ids[mask]


@timer
def generate_sub_box_ids(
    positions: np.ndarray,
    boxsize: float,
    subsize: float,
    chunksize: float,
    path: str,
    name: str = None
) -> None:
    """Gets the sub-box ID for each position

    Parameters
    ----------
    positions : np.ndarray
        Cartesian coordinates
    boxsize : float
        Size of simulation box
    subsize : float
        Size of sub-box
    chunksize : float
        Number of items to process at a time in chunks
    path : str
        Where to save the IDs
    name : str, optional
        An additional name or identifier appended at the end of the file name, 
        by default None

    Returns
    -------
    None
    """
    n_items = positions.shape[0]
    n_iter = n_items // chunksize

    # Determine data type for integer arrays based on the maximum number of
    # elements
    boxes_per_side = np.int_(np.ceil(boxsize / subsize))
    uint_dtype = get_np_unit_dytpe(boxes_per_side**3)

    ids = np.zeros(n_items, dtype=uint_dtype)

    for chunk in tqdm(range(n_iter), desc='Chunk', ncols=100, colour='blue'):
        low = chunk * chunksize
        if chunk < n_iter - 1:
            upp = (chunk + 1) * chunksize
        else:
            upp = None
        ids[low:upp] = get_sub_box_id(positions[low:upp], boxsize, subsize)
        # if np.max(ids) > boxes_per_side**3:
            # print(chunk)

    if name:
        file_name = f'sub_box_id_{name}.hdf5'
    else:
        file_name = f'sub_box_id.hdf5'
    with h5.File(path + file_name, 'w') as hdf:
        hdf.create_dataset('SBID', data=ids, dtype=uint_dtype)

    return None


def get_zw13_vr_vt(
    pos: np.ndarray,
    vel: np.ndarray,
) -> Tuple[np.ndarray]:
    """Computes the radial and tangential velocites from cartesian/rectangular 
    coordinates following the definitions of Zu & Weinberg 2013.
    https://academic.oup.com/mnras/article/431/4/3319/1147861

    Parameters
    ----------
    pos : np.ndarray
        Cartesian coordinates
    vel : np.ndarray
        Cartesian velocities

    Returns
    -------
    Tuple[np.ndarray]
        Radial velocity and tangential velocity. 
    """
    #Compute projected and 3D radial separation
    rp = np.sqrt(np.sum(np.square(pos[:, :2]), axis=1))
    r3d = np.sqrt(np.square(rp)+np.square(pos[:, 2]))
    
    #Compute angle between projected and 3D radial separation 
    theta = np.arctan2(pos[:, 2], rp) 

    #Compute radial velocity using cartesian coordinates
    vr = np.sum(pos * vel, axis=1) / r3d 
    vt = ( vel[:, 2] - vr * np.sin(theta) ) / np.cos(theta)

    return vr, vt



def cartesian_product(arrays: List[np.ndarray]):
    """Generalized N-dimensional products
    Taken from https://stackoverflow.com/questions/11144513/
    Answer by Nico Schlömer
    Updated for numpy > 1.25

    Parameters
    ----------
    arrays : List[np.ndarray]
        _description_

    Returns
    -------
    _type_
        _description_
    """
    la = len(arrays)
    dtype = np.result_type(*[a.dtype for a in arrays])
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)



@timer
def split_simulation_into_sub_boxes(
    positions: np.ndarray,
    velocities: np.ndarray,
    ids: np.ndarray,
    upids: np.ndarray,
    boxsize: float,
    subsize: float,
    chunksize: float,
    dtypes: list,
    path: str,
    name: str = None,
) -> None:
    """Sorts all items into sub-boxes and saves them in disc.

    Parameters
    ----------
    positions : np.ndarray
        _description_
    velocities : np.ndarray
        _description_
    ids : np.ndarray
        Unique IDs for each position (e.g. PID, HID)
    boxsize : float
        Size of simulation box
    subsize : float
        Size of sub-box
    chunksize : float
        Number of items to process at a time in chunks
    dtypes : list
        Data types for positions and velocities
    path : str
        Where to save the IDs
    name : str, optional
        An additional name or identifier appended at the end of the file name, 
        by default None

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the chunksize is larger than the number of items
    """
    # Check chunksize is smaller than the number of items
    n_items = positions.shape[0]
    if chunksize > n_items:
        raise ValueError(
            f"The specified chunksize {chunksize} is larger than the number of items {n_items}")

    # Create directory if it does not exist
    save_path = path + 'sub_boxes/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(save_path):
        generate_sub_box_ids(positions, boxsize, subsize,
                             chunksize, path, name)

    if name:
        sub_box_ids_file = path + f'sub_box_id_{name}.hdf5'
    else:
        sub_box_ids_file = path + f'sub_box_id.hdf5'
    with h5.File(sub_box_ids_file, 'r') as hdf:
        sub_box_ids = hdf['SBID'][()]

    n_iter = n_items // chunksize

    uint_dtype_row = get_np_unit_dytpe(n_items)
    row_idx = np.arange(n_items, dtype=uint_dtype_row)

    uint_dtype_ids = get_np_unit_dytpe(np.max(ids))
    for chunk in tqdm(range(n_iter), desc='Chunk', ncols=100, colour='blue'):
        # Select chunk
        low = chunk * chunksize
        if chunk < n_iter - 2:
            upp = (chunk + 1) * chunksize
        else:
            upp = None
        sb_ids = sub_box_ids[low:upp]
        pos = positions[low:upp]
        vel = velocities[low:upp]
        pid = ids[low:upp]
        upid = upids[low:upp]
        row = row_idx[low:upp]

        sb_unique = np.unique(sb_ids)
        order = np.argsort(sb_ids)
        # Save all items at each unique sub-box ID
        for sub_box in sb_unique:
            left = np.searchsorted(sb_ids, sub_box, side="left", sorter=order)
            right = np.searchsorted(
                sb_ids, sub_box, side="right", sorter=order)

            pos_item = pos[order][left:right]
            vel_item = vel[order][left:right]
            pid_item = pid[order][left:right]
            upid_item = upid[order][left:right]

            row_item = row[order][left:right]

            with h5.File(save_path + f"{sub_box}.hdf5", "a") as hdf:
                if not name in hdf.keys():
                    hdf.create_group(name)

                # If it is the first time opening this file, create datasets
                if not 'upID' in hdf[name].keys():
                    hdf.create_dataset(
                        name=f'{name}/ID',
                        data=pid_item,
                        maxshape=(None, ),
                        dtype=uint_dtype_ids,
                    )
                    hdf.create_dataset(
                        name=f'{name}/upID',
                        data=upid_item,
                        maxshape=(None, ),
                        dtype=uint_dtype_ids,
                    )
                    hdf.create_dataset(
                        name=f'{name}/pos',
                        data=pos_item,
                        maxshape=(None, pos_item.shape[-1]),
                        dtype=dtypes[0],
                    )
                    hdf.create_dataset(
                        name=f'{name}/vel',
                        data=vel_item,
                        maxshape=(None, vel_item.shape[-1]),
                        dtype=dtypes[1],
                    )
                    hdf.create_dataset(
                        name=f'{name}/row_idx',
                        data=row_item,
                        maxshape=(None, ),
                        dtype=uint_dtype_row,
                    )
                # If it is not the first time opening the file, reshape the
                # datasets
                else:
                    last_item = pid_item.shape[0]
                    new_shape = hdf[f'{name}/ID'].shape[0] + last_item

                    hdf[f'{name}/ID'].resize((new_shape), axis=0)
                    hdf[f'{name}/ID'][-last_item:] = pid_item
                    hdf[f'{name}/upID'].resize((new_shape), axis=0)
                    hdf[f'{name}/upID'][-last_item:] = upid_item
                    hdf[f'{name}/pos'].resize((new_shape), axis=0)
                    hdf[f'{name}/pos'][-last_item:] = pos_item
                    hdf[f'{name}/vel'].resize((new_shape), axis=0)
                    hdf[f'{name}/vel'][-last_item:] = vel_item
                    hdf[f'{name}/row_idx'].resize((new_shape), axis=0)
                    hdf[f'{name}/row_idx'][-last_item:] = row_item

    return None


def _load_sub_box(
    sub_box_id: int,
    path: str,
    name: str = None,
) -> Tuple[np.ndarray]:
    """Load sub-box

    Parameters
    ----------
    sub_box_id : int
        Sub-box ID
    path : str
        Location from where to load the file
    name : str, optional
        Identifier within the file, by default None

    Returns
    -------
    Tuple[np.ndarray]
        Position, velocity, ID and row index
    """
    if name:
        prefix = f'{name}/'
    else:
        prefix = None
    try:
        with h5.File(path + f'sub_boxes/{sub_box_id}.hdf5', 'r') as hdf:
            pos = hdf[prefix + 'pos'][()]
            vel = hdf[prefix + 'vel'][()]
            pid = hdf[prefix + 'ID'][()]
            upid = hdf[prefix + 'upID'][()]
            row = hdf[prefix + 'row_idx'][()]
    except:
        pos, vel, pid, upid, row = None, None, None, None, None
    return pos, vel, pid, upid, row


def load_halos(
    sub_box_id: int,
    boxsize: float,
    subsize: float,
    path: str,
    padding: float = 5.0,
) -> Tuple[np.ndarray]:
    """Load halos from a sub-box

    Parameters
    ----------
    sub_box_id : int
        Sub-box ID
    path : str
        Location from where to load the file
    boxsize : float
        Size of simulation box
    subsize : float
        Size of sub-box
    padding : float
        Only particles up to this distance from the sub-box edge are considered 
        for classification. Defaults to 5

    Returns
    -------
    Tuple[np.ndarray]
        Position, velocity, ID and row index
    """
    # Generate the IDs and positions of the sub-box grid
    grid_ids, grid_pos = generate_sub_box_grid(boxsize, subsize)
    # Get the adjacent sub-box IDs
    adj_sub_box_ids = get_adjacent_sub_box_ids(
        sub_box_id=sub_box_id,
        sub_box_ids=grid_ids,
        positions=grid_pos,
        boxsize=boxsize,
        subsize=subsize
    )

    # Create empty lists (containers) to save the data from file for each ID
    pos, vel, hid, hupid, row = ([[] for _ in range(len(adj_sub_box_ids))]
                          for _ in range(5))

    # Load all adjacent boxes
    for i, sub_box in enumerate(adj_sub_box_ids):
        pos[i], vel[i], hid[i], hupid[i], row[i] = _load_sub_box(
            sub_box, path, name='halo')
    # Concatenate into a single array
    pos = np.concatenate(pos)
    vel = np.concatenate(vel)
    hid = np.concatenate(hid)
    hupid = np.concatenate(hupid)
    row = np.concatenate(row)

    # Mask particles within a padding distance of the edge of the box in each
    # direction
    loc_id = grid_ids == sub_box_id
    padded_distance = subsize #0.5 * subsize + padding
    rel_abs_position = np.abs(relative_coordinates(
        grid_pos[loc_id], pos, boxsize, periodic=True))
    # Probably a better way to create this mask
    mask_x = (rel_abs_position[:, 0] < padded_distance)
    mask_y = (rel_abs_position[:, 1] < padded_distance)
    mask_z = (rel_abs_position[:, 2] < padded_distance)
    mask = mask_x & mask_y & mask_z

    return pos[mask], vel[mask], hid[mask], hupid[mask], row[mask]


@timer
def get_pairwise_halo_pw_quants(
    n_seeds: int,
    r_max: float,
    boxsize: float,
    subsize: float,
    file_seeds: str,
    path: str,
    mass_mask: np.ndarray, 
    pos_vel_labels: list = ['x', 'y', 'z', 'vx', 'vy', 'vz'],
    id_label: str = 'hid', 
    vmax_label: str = 'vmax',
    Abacus: bool = False,
    n_jobs: int = -1,
) -> Tuple[np.ndarray]:
    """Locates for the largest `v_max` seeds and searches for all the halos
    around them up to a distance `r_max`, and returns pairwise quantitiies.

    Parameters
    ----------
    n_seeds : int
        Number of seeds to process
    r_max : float
        Maximum distance to consider
    boxsize : float
        Size of simulation box
    subsize : float
        Size of sub-box
    file_seeds : str
        File containing the seeds, including path
    path : str
        Path to the sub-boxes
    mass_mask : array
        mask of halos with mass mask
    pos_vel_labels: list
        list of strings specifying naming conventions of position & velocity columns 
    n_jobs : int
        Number of parallel workers for the sub-box loop (default: -1, all cores)

    Returns
    -------
    Tuple[np.ndarray]
        Radial distance, radial velocity, and log of the square of the velocity
    """
    if n_seeds == 0:
        empty = np.array([])
        return (empty,) * 6
    
    # Load seed data
    with h5.File(file_seeds, 'r') as hdf:
        if Abacus:
            hdf = hdf['halos'][:]

        vmax = hdf[vmax_label][()][mass_mask]
        order = np.argsort(vmax)[::-1]

        hid = hdf[id_label][()] 
        pos_seed = np.vstack(
            [
                hdf[pos_vel_labels[0]][()][mass_mask],
                hdf[pos_vel_labels[1]][()][mass_mask],
                hdf[pos_vel_labels[2]][()][mass_mask],
            ]
        ).T
        vel_seed = np.vstack(
            [
                hdf[pos_vel_labels[3]][()][mass_mask],
                hdf[pos_vel_labels[4]][()][mass_mask],
                hdf[pos_vel_labels[5]][()][mass_mask],
            ]
        ).T

    vmax = vmax[order][:]
    hid = hid[order][:]
    pos_seed = pos_seed[order][:]
    vel_seed = vel_seed[order][:]

    # Locate sub-box IDs for all seeds
    seed_sub_box_id = get_sub_box_id(pos_seed%boxsize, boxsize, subsize)

    # # Sort by sub-box ID
    order = np.argsort(seed_sub_box_id)
    seed_sub_box_id = seed_sub_box_id[order]
    vmax = vmax[order]
    hid = hid[order]
    pos_seed = pos_seed[order]
    vel_seed = vel_seed[order]

    # Get unique sub-box ids
    unique_sub_box_ids = np.unique(seed_sub_box_id)

    def _process_sub_box(sub_box_id):
        """Process all seeds belonging to a single sub-box; returns concatenated arrays."""
        pos, vel, _, _, _ = load_halos(sub_box_id, boxsize, subsize, path)

        r_sb, R_sb, rlos_sb, vpeclos_sb, vr_sb, vt_sb = (
            [] for _ in range(6)
        )

        mask_seeds_in_sub_box = seed_sub_box_id == sub_box_id
        for i in range(mask_seeds_in_sub_box.sum()):
            # Compute the relative positions of all particles in the box
            rel_pos = relative_coordinates(pos_seed[mask_seeds_in_sub_box][i]%boxsize, pos,
                                           boxsize)
            z_sep = pos[:, 2]%boxsize - pos_seed[mask_seeds_in_sub_box][i][2]%boxsize 

            # Only work with those close to the seed
            sb_max = r_max
            mask_x = np.abs(rel_pos[:, 0]) <= sb_max
            mask_y = np.abs(rel_pos[:, 1]) <= sb_max
            mask_z = np.abs(rel_pos[:, 2]) <= sb_max
            mask_close = mask_x * mask_y * mask_z

            rel_pos = rel_pos[mask_close]
            rel_vel = vel[mask_close] - vel_seed[mask_seeds_in_sub_box][i]
            z_sep = z_sep[mask_close]

            # Compute radial and tangential velocity
            vrp, vtp = get_zw13_vr_vt(rel_pos, rel_vel)
            vpeclosp = rel_vel[:, 2] #

            # select gals within R cut
            R_max = r_max # h^-1 Mpc

            # Compute the radial separation 
            rp = np.sqrt(np.sum(np.square(rel_pos), axis=1)) #/ r200
            Rp = np.sqrt( np.sum(np.square(rel_pos[:, :2]), axis=1) )
            rlosp = rel_pos[:, 2]

            Rmask = Rp <= R_max

            r_sb.append(rp[Rmask])
            R_sb.append(Rp[Rmask])
            rlos_sb.append(rlosp[Rmask])
            vpeclos_sb.append(vpeclosp[Rmask])
            vr_sb.append(vrp[Rmask])
            vt_sb.append(vtp[Rmask])

        return (
            np.concatenate(r_sb),
            np.concatenate(R_sb),
            np.concatenate(rlos_sb),
            np.concatenate(vpeclos_sb),
            np.concatenate(vr_sb),
            np.concatenate(vt_sb),

        )

    with tqdm_joblib(tqdm(unique_sub_box_ids, desc='Processing sub-box',
                          colour='blue', ncols=100)):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_sub_box)(sub_box_id) for sub_box_id in unique_sub_box_ids
        )

    # Concatenate results across sub-boxes
    r, R, rlos, vpeclos, vr, vt = (
        np.concatenate(arrs) for arrs in zip(*results)
    )

    return r, R, rlos, vpeclos, vr, vt