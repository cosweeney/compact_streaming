# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.math cimport sqrt

def accumulate_moments(
    double[:, ::1] pos,
    double[:, ::1] vel,
    long[::1]      cell_id,
    long[::1]      cell_start,
    long[::1]      cell_count,
    long[:, ::1]   neighbor_cells,
    double[::1]    rbins,
    int            n_bins,
):
    cdef int n_particles = pos.shape[0]
    cdef int n_cells     = cell_start.shape[0]

    counts  = np.zeros(n_bins, dtype=np.int64)
    s1      = np.zeros(n_bins, dtype=np.float64)
    s2r     = np.zeros(n_bins, dtype=np.float64)
    s3r     = np.zeros(n_bins, dtype=np.float64)
    s4r     = np.zeros(n_bins, dtype=np.float64)
    s2t     = np.zeros(n_bins, dtype=np.float64)
    s3rt    = np.zeros(n_bins, dtype=np.float64)
    s4t     = np.zeros(n_bins, dtype=np.float64)
    s4rt    = np.zeros(n_bins, dtype=np.float64)

    cdef long[::1]    c_counts  = counts
    cdef double[::1]  c_s1      = s1
    cdef double[::1]  c_s2r     = s2r
    cdef double[::1]  c_s3r     = s3r
    cdef double[::1]  c_s4r     = s4r
    cdef double[::1]  c_s2t     = s2t
    cdef double[::1]  c_s3rt    = s3rt
    cdef double[::1]  c_s4t     = s4t
    cdef double[::1]  c_s4rt    = s4rt

    cdef double rmax = rbins[n_bins]
    cdef double rmax2 = rmax * rmax

    cdef int    ci, ni, nc, i, j, b
    cdef long   ncell_j
    cdef double dx, dy, dz, r2, r, inv_r
    cdef double dvx, dvy, dvz
    cdef double vr, rperp, cos_phi, sin_phi, cos_th, sin_th, vt
    cdef double vr2, vr3, vr4, vt2, vt4

    for ci in range(n_cells):
        if cell_count[ci] == 0:
            continue
        nc = neighbor_cells.shape[1]
        for ni in range(nc):
            ncell_j = neighbor_cells[ci, ni]
            if ncell_j < 0 or cell_count[ncell_j] == 0:
                continue
            for i in range(cell_start[ci], cell_start[ci] + cell_count[ci]):
                for j in range(cell_start[ncell_j], cell_start[ncell_j] + cell_count[ncell_j]):
                    if ncell_j == ci and j <= i:
                        continue

                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dz = pos[i, 2] - pos[j, 2]
                    r2 = dx*dx + dy*dy + dz*dz
                    if r2 == 0.0 or r2 > rmax2:
                        continue

                    r = sqrt(r2)
                    b = _searchsorted(rbins, r, n_bins + 1) - 1
                    if b < 0 or b >= n_bins:
                        continue

                    inv_r = 1.0 / r
                    dvx = vel[i, 0] - vel[j, 0]
                    dvy = vel[i, 1] - vel[j, 1]
                    dvz = vel[i, 2] - vel[j, 2]

                    vr = (dvx*dx + dvy*dy + dvz*dz) * inv_r

                    rperp = sqrt(dx*dx + dy*dy)
                    if rperp < 1e-10:
                        cos_phi = 1.0
                        sin_phi = 0.0
                    else:
                        cos_phi = dx / rperp
                        sin_phi = dy / rperp
                    cos_th = dz * inv_r
                    sin_th = rperp * inv_r

                    vt = dvx*cos_th*cos_phi + dvy*cos_th*sin_phi - dvz*sin_th

                    vr2 = vr * vr
                    vr3 = vr2 * vr
                    vr4 = vr3 * vr
                    vt2 = vt * vt
                    vt4 = vt2 * vt2

                    c_counts[b] += 1
                    c_s1[b]   += vr
                    c_s2r[b]  += vr2
                    c_s3r[b]  += vr3
                    c_s4r[b]  += vr4
                    c_s2t[b]  += vt2
                    c_s3rt[b] += vr * vt2
                    c_s4t[b]  += vt4
                    c_s4rt[b] += vr2 * vt2

    return counts, s1, s2r, s3r, s4r, s2t, s3rt, s4t, s4rt


cdef int _searchsorted(double[::1] arr, double val, int n) nogil:
    cdef int lo = 0, hi = n, mid
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] <= val:
            lo = mid + 1
        else:
            hi = mid
    return lo