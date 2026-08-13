"""Streaming histogram accumulator for P(v_los | M, R, rlos).

Replaces the raw pair catalogue: pairs are binned as they are produced and
then discarded.  Accumulators are additive, so each sub-box / simulation
worker builds its own and the results are summed.
"""

import numpy as np
import h5py as h5


class PairHist:
    """4-D count histogram over (mass bin, R bin, rlos bin, v_los bin).

    Also tracks per-cell n, sum(v), sum(v^2) -- and, when v_r is supplied,
    sum(v_r), sum(v_r^2) -- over *all* pairs landing in the (M, R, rlos) cell,
    including those outside the velocity grid.  So ``n - counts.sum(-1)`` is
    the number of pairs that fell off the v grid.
    """

    def __init__(self, n_M, R_edges, rlos_edges, v_lo, v_hi, n_v):
        self.n_M = int(n_M)
        self.R_edges = np.asarray(R_edges, dtype=np.float64)
        self.rlos_edges = np.asarray(rlos_edges, dtype=np.float64)
        self.v_lo = float(v_lo)
        self.v_hi = float(v_hi)
        self.n_v = int(n_v)

        self.n_R = len(self.R_edges) - 1
        self.n_l = len(self.rlos_edges) - 1
        self._inv_dv = self.n_v / (self.v_hi - self.v_lo)

        shape = (self.n_M, self.n_R, self.n_l)
        self.counts = np.zeros(shape + (self.n_v,), dtype=np.int64)
        self.n = np.zeros(shape, dtype=np.int64)
        self.sum_v = np.zeros(shape, dtype=np.float64)
        self.sum_v2 = np.zeros(shape, dtype=np.float64)
        self.sum_vr = np.zeros(shape, dtype=np.float64)
        self.sum_vr2 = np.zeros(shape, dtype=np.float64)

    @property
    def v_edges(self):
        return np.linspace(self.v_lo, self.v_hi, self.n_v + 1)

    @property
    def v_centers(self):
        e = self.v_edges
        return 0.5 * (e[1:] + e[:-1])

    def add(self, m_bin, R, rlos, v, vr=None):
        """Bin a batch of pairs.  ``m_bin`` may be a scalar or a per-pair array."""
        R = np.asarray(R)
        rlos = np.asarray(rlos)
        v = np.asarray(v)

        iR = np.searchsorted(self.R_edges, R, side="right") - 1
        il = np.searchsorted(self.rlos_edges, rlos, side="right") - 1
        iv = np.floor((v - self.v_lo) * self._inv_dv).astype(np.int64)
        im = np.broadcast_to(np.asarray(m_bin, dtype=np.int64), R.shape)

        ok = ((iR >= 0) & (iR < self.n_R)
              & (il >= 0) & (il < self.n_l)
              & (im >= 0) & (im < self.n_M))

        cell = (im[ok] * self.n_R + iR[ok]) * self.n_l + il[ok]
        v_ok = v[ok]
        n_cell = self.n.size
        shp = self.n.shape
        self.n += np.bincount(cell, minlength=n_cell).reshape(shp)
        self.sum_v += np.bincount(cell, weights=v_ok, minlength=n_cell).reshape(shp)
        self.sum_v2 += np.bincount(
            cell, weights=v_ok * v_ok, minlength=n_cell).reshape(shp)

        if vr is not None:
            vr_ok = np.asarray(vr)[ok]
            self.sum_vr += np.bincount(
                cell, weights=vr_ok, minlength=n_cell).reshape(shp)
            self.sum_vr2 += np.bincount(
                cell, weights=vr_ok * vr_ok, minlength=n_cell).reshape(shp)

        in_v = ok & (iv >= 0) & (iv < self.n_v)
        flat = ((im[in_v] * self.n_R + iR[in_v]) * self.n_l
                + il[in_v]) * self.n_v + iv[in_v]
        self.counts += np.bincount(
            flat, minlength=self.counts.size).reshape(self.counts.shape)

    def mean_std(self, k, radial=False):
        """Per-cell (mean, std) of v_los, or of v_r when radial=True."""
        n = np.where(self.n[k] > 0, self.n[k], np.nan)
        s1 = (self.sum_vr if radial else self.sum_v)[k]
        s2 = (self.sum_vr2 if radial else self.sum_v2)[k]
        mu = s1 / n
        return mu, np.sqrt(np.maximum(s2 / n - mu ** 2, 0.0))

    def __iadd__(self, other):
        if (self.counts.shape != other.counts.shape
                or not np.array_equal(self.R_edges, other.R_edges)
                or not np.array_equal(self.rlos_edges, other.rlos_edges)
                or (self.v_lo, self.v_hi) != (other.v_lo, other.v_hi)):
            raise ValueError("accumulators were built on different grids")
        self.counts += other.counts
        self.n += other.n
        self.sum_v += other.sum_v
        self.sum_v2 += other.sum_v2
        self.sum_vr += other.sum_vr
        self.sum_vr2 += other.sum_vr2
        return self

    def save(self, path):
        with h5.File(path, "w") as hdf:
            hdf.create_dataset("counts", data=self.counts, compression="gzip")
            for name in ("n", "sum_v", "sum_v2", "sum_vr", "sum_vr2"):
                hdf.create_dataset(name, data=getattr(self, name))
            hdf.create_dataset("R_edges", data=self.R_edges)
            hdf.create_dataset("rlos_edges", data=self.rlos_edges)
            hdf.attrs.update(n_M=self.n_M, v_lo=self.v_lo,
                             v_hi=self.v_hi, n_v=self.n_v)

    @classmethod
    def load(cls, path):
        with h5.File(path, "r") as hdf:
            obj = cls(hdf.attrs["n_M"], hdf["R_edges"][()],
                      hdf["rlos_edges"][()], hdf.attrs["v_lo"],
                      hdf.attrs["v_hi"], hdf.attrs["n_v"])
            obj.counts = hdf["counts"][()]
            for name in ("n", "sum_v", "sum_v2", "sum_vr", "sum_vr2"):
                if name in hdf:            # files written before v_r was added
                    setattr(obj, name, hdf[name][()])
        return obj


def reduce_hists(hists):
    """Sum a sequence of PairHist into the first one."""
    total = hists[0]
    for h in hists[1:]:
        total += h
    return total