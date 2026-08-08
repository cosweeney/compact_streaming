""" 
For computing pairwise velocity moments.
Ported from https://github.com/florpi/PairVelocities.jl to cython.
"""

import numpy as np
from joblib import Parallel, delayed


def _build_cell_list(pos, boxsize, rmax):
    ncells_1d = max(1, int(boxsize / rmax))
    cell_size = boxsize / ncells_1d
    nc = ncells_1d

    idx = np.floor(pos / cell_size).astype(np.int64) % nc
    cell_id = idx[:, 0] * nc * nc + idx[:, 1] * nc + idx[:, 2]

    order = np.argsort(cell_id, kind="stable")
    sorted_cell_id = cell_id[order]

    n_cells = nc ** 3
    cell_start = np.zeros(n_cells, dtype=np.int64)
    cell_count = np.zeros(n_cells, dtype=np.int64)
    unique, first, cnts = np.unique(sorted_cell_id, return_index=True, return_counts=True)
    cell_start[unique] = first
    cell_count[unique] = cnts

    offsets = np.array(
        [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)],
        dtype=np.int64,
    )
    ci = np.arange(n_cells)
    cz = ci % nc
    cy = (ci // nc) % nc
    cx = ci // (nc * nc)
    coords = np.stack([cx, cy, cz], axis=1)  # (n_cells, 3)

    neighbor_coords = coords[:, None, :] + offsets[None, :, :]  # (n_cells, 27, 3)
    neighbor_coords_wrapped = neighbor_coords % nc
    neighbor_cells = (
        neighbor_coords_wrapped[:, :, 0] * nc * nc
        + neighbor_coords_wrapped[:, :, 1] * nc
        + neighbor_coords_wrapped[:, :, 2]
    ).astype(np.int64)

    sorted_pos = pos[order]
    sorted_vel = None  # returned separately

    return sorted_pos, order, cell_id, cell_start, cell_count, neighbor_cells


def _convert_to_moments(counts, s1, s2r, s3r, s4r, s2t, s3rt, s4t, s4rt):
    mask = counts > 0
    n = counts.astype(np.float64)

    m10 = np.where(mask, s1 / n, 0.0)
    c20 = np.where(mask, s2r / n - m10**2, 0.0)
    c02 = np.where(mask, s2t / n, 0.0)
    c30 = np.where(mask, s3r / n - 3.0 * m10 * c20 - m10**3, 0.0)
    c12 = np.where(mask, s3rt / n - m10 * c02, 0.0)
    c40 = np.where(mask, s4r / n - 4.0 * m10 * c30 - 6.0 * m10**2 * c20 - m10**4, 0.0)
    c04 = np.where(mask, s4t / n, 0.0)
    c22 = np.where(mask, s4rt / n - 2.0 * c12 * m10 - m10**2 * c02, 0.0)

    sig_r = np.sqrt(np.maximum(c20, 0.0))
    sig_t = np.sqrt(np.maximum(c02, 0.0))

    # standardized moments (matching PairVelocities.jl convention)
    c20_out = sig_r
    c02_out = sig_t
    c30_out = np.where(sig_r > 0, c30 / sig_r**3, 0.0)
    c40_out = np.where(sig_r > 0, c40 / sig_r**4, 0.0)
    c04_out = np.where(sig_t > 0, c04 / sig_t**4, 0.0)
    c12_out = np.where((sig_r > 0) & (sig_t > 0), c12 / (sig_r**2 * sig_t), 0.0)
    c22_out = np.where((sig_r > 0) & (sig_t > 0), c22 / (sig_r**2 * sig_t**2), 0.0)

    return m10, c20_out, c02_out, c30_out, c40_out, c04_out, c12_out, c22_out


def pairwise_velocity_moments(pos, vel, rbins, boxsize):
    """
    Compute pairwise velocity moments as a function of 3D separation.

    Parameters
    ----------
    pos     : (N, 3) float64, particle positions in [0, boxsize)
    vel     : (N, 3) float64, particle velocities
    rbins   : (M+1,) float64, bin edges
    boxsize : float, cubic box side length (periodic)

    Returns
    -------
    dict with keys: counts, m10, c20, c02, c30, c40, c04, c12, c22
    Each value is an (M,) array.
    Moments follow PairVelocities.jl convention: c20/c02 are sigma (not sigma^2),
    higher moments are standardized (divided by sigma^n).
    """
    from .pairvel import accumulate_moments  # compiled .so

    pos    = np.ascontiguousarray(pos,  dtype=np.float64)
    vel    = np.ascontiguousarray(vel,  dtype=np.float64)
    rbins  = np.ascontiguousarray(rbins, dtype=np.float64)
    rmax   = rbins[-1]
    n_bins = len(rbins) - 1

    sorted_pos, order, cell_id, cell_start, cell_count, neighbor_cells = \
        _build_cell_list(pos, boxsize, rmax)
    sorted_vel = np.ascontiguousarray(vel[order])

    counts, s1, s2r, s3r, s4r, s2t, s3rt, s4t, s4rt = accumulate_moments(
        sorted_pos, sorted_vel,
        cell_id, cell_start, cell_count, neighbor_cells,
        rbins, n_bins,
    )

    m10, c20, c02, c30, c40, c04, c12, c22 = _convert_to_moments(
        counts, s1, s2r, s3r, s4r, s2t, s3rt, s4t, s4rt
    )

    return dict(counts=counts, m10=m10, c20=c20, c02=c02,
                c30=c30, c40=c40, c04=c04, c12=c12, c22=c22)


def _assign_jackknife_labels(pos, boxsize, n_jk_1d):
    jk_size = boxsize / n_jk_1d
    idx = np.floor(pos / jk_size).astype(np.int64) % n_jk_1d
    return idx[:, 0] * n_jk_1d * n_jk_1d + idx[:, 1] * n_jk_1d + idx[:, 2]
 
 
def pairwise_velocity_moments_jackknife(pos, vel, rbins, boxsize, n_jk_1d):
    """
    Compute pairwise velocity moments with jackknife covariances.
 
    The box is divided into n_jk_1d^3 sub-volumes. Each jackknife sample
    excludes all particles in one sub-volume. The full-sample estimate is
    also returned.
 
    Parameters
    ----------
    pos      : (N, 3) float64, particle positions in [0, boxsize)
    vel      : (N, 3) float64, particle velocities
    rbins    : (M+1,) float64, bin edges
    boxsize  : float, cubic box side length (periodic)
    n_jk_1d  : int, number of jackknife regions per side (total = n_jk_1d^3)
 
    Returns
    -------
    full    : dict with keys: counts, m10, c20, c02, c30, c40, c04, c12, c22
              Full-sample moments, each an (M,) array.
    jk      : dict with the same keys, each an (n_jk, M) array of
              leave-one-out estimates.
    cov     : (M, n_moments, n_moments) jackknife covariance array,
              where n_moments=8 (order: m10, c20, c02, c30, c40, c04, c12, c22).
    """
    n_jk = n_jk_1d ** 3
    jk_labels = _assign_jackknife_labels(pos, boxsize, n_jk_1d)
 
    full = pairwise_velocity_moments(pos, vel, rbins, boxsize)
 
    moment_keys = ['m10', 'c20', 'c02', 'c30', 'c40', 'c04', 'c12', 'c22']
    n_bins = len(rbins) - 1
    jk_samples = {k: np.zeros((n_jk, n_bins)) for k in moment_keys}
 
    def _jk_sample(k):
        mask = jk_labels != k
        return pairwise_velocity_moments(pos[mask], vel[mask], rbins, boxsize)

    results = Parallel(n_jobs=-1)(delayed(_jk_sample)(k) for k in range(n_jk))
    for k, result in enumerate(results):
        for key in moment_keys:
            jk_samples[key][k] = result[key]
 
    # jackknife covariance: (n_bins, n_moments, n_moments)
    jk_matrix = np.stack([jk_samples[k] for k in moment_keys], axis=-1)  # (n_jk, n_bins, n_moments)
    mean_jk = jk_matrix.mean(axis=0)                                       # (n_bins, n_moments)
    delta = jk_matrix - mean_jk                                            # (n_jk, n_bins, n_moments)
    cov = (n_jk - 1) / n_jk * np.einsum('kbi,kbj->bij', delta, delta)    # (n_bins, n_moments, n_moments)
 
    return full, jk_samples, cov
