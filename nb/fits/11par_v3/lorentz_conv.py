"""
Line-of-sight (LOS) Lorentzian redshift-error convolution acting on redshift-space correlation-function multipoles.

Model.  Every object's observed LOS position is shifted by an iid draw eps from a per-object kernel K1(eps), a Lorentzian of
half-width w (km/s), truncated at |eps| < L1.  A pair's separation shifts by u = eps_i - eps_j, whose density is the
autocorrelation K2 = K1 (x) K1 (symmetric, normalised, support |u| < 2 L1).  Then

    xi_obs(s_perp, s_par) = int du K2(u) xi_s(s_perp, s_par - u)

(the '1' in 1 + xi drops out because K2 is normalised).  xi_s is built from the multipoles xi_0, xi_2, xi_4 (the emulator output
after the AP step, i.e. in fiducial coordinates), and xi_obs is projected back onto multipoles with the same mu quadrature as
common.ap_multipoles.

Implementation.  The geometry (s', mu', interpolation weights) does not depend on w, so it is precomputed once per output grid as a stack
G[m] of (3 ns) x (3 nF) matrices, one per quadrature node u_m.  For a given w only the node weights omega_m(w) change:
M(w) = sum_m omega_m(w) G[m],  xi_obs = M(w) @ xi_s.  Only numpy/scipy are needed (no camb / emulator imports), so this file is
testable on its own:   python lorentz_conv.py
"""
import time
import numpy as np
from scipy.signal import fftconvolve
from scipy.special import eval_legendre

ELLS = (0, 2, 4)
DU_FINE_KMS = 2.0           # km/s step of the fine grid on which K1 and K2 are built


# ------------------------------------------------------------------ kernel
def default_u_nodes_kms(L1=3000.0, core=100.0, core_step=5.0, ratio=1.05):
    """Symmetric quadrature nodes (km/s) for the pair kernel: uniform near 0, geometric out to 2*L1."""
    lin = np.arange(0.0, core + core_step, core_step)
    geo = [lin[-1]]
    while geo[-1] < 2 * L1:
        geo.append(geo[-1] * ratio)
    pos = np.concatenate([lin, np.array(geo[1:])])
    pos[-1] = 2 * L1
    pos = np.unique(pos)
    return np.concatenate([-pos[:0:-1], pos])       # ..., -u2, -u1, 0, u1, u2, ...


def cell_widths(u):
    """Widths of the cells around sorted nodes u (trapezoid-style: half-gaps to each neighbour)."""
    edges = np.concatenate([[u[0] - 0.5 * (u[1] - u[0])], 0.5 * (u[1:] + u[:-1]), [u[-1] + 0.5 * (u[-1] - u[-2])]])
    return np.diff(edges)


def pair_kernel_fine(w, L1, du=DU_FINE_KMS):
    """(u grid, K2) on a uniform fine grid. K1 = truncated Lorentzian of half-width w, K2 = K1 (x) K1 (autocorrelation)."""
    n = int(round(L1 / du))
    eps = np.arange(-n, n + 1) * du
    k1 = 1.0 / (1.0 + (eps / w) ** 2)
    k1 /= k1.sum() * du
    k2 = fftconvolve(k1, k1, mode='full') * du      # length 4n+1, nodes -2n..2n (times du)
    u = np.arange(-2 * n, 2 * n + 1) * du
    return u, k2


def pair_weights(w, L1, u_nodes, du=DU_FINE_KMS, widths=None):
    """Quadrature weights (sum to 1) of the pair kernel at the nodes u_nodes (km/s)."""
    u, k2 = pair_kernel_fine(w, L1, du)
    half = len(u) // 2
    K2 = np.interp(np.abs(u_nodes), u[half:], k2[half:], right=0.0)
    wts = K2 * (cell_widths(u_nodes) if widths is None else widths)
    return wts / wts.sum()


# ------------------------------------------------------------------ geometry
class ConvGeometry:
    """Precomputed w-independent geometry for outputs at s_out (h^-1 Mpc) from inputs on the uniform grid s_fine.

    s_out    : output separations (the data bins), shape (ns,)
    s_fine   : uniform input grid of xi_ell^AP (fiducial coordinates), shape (nF,), spacing ds
    mu, w_mu : Gauss-Legendre nodes / weights on (0, 1) (common.MU_REF, common.W_MU)
    kms_to_h : h^-1 Mpc per km/s at the tracer redshift, (1+z)/(100 E_ref(z))
    u_nodes  : pair-kernel quadrature nodes in km/s
    ells_out : output multipoles (subset of ELLS)

    Input outside [s_fine[0], s_fine[-1]]: below -> held at the first grid value (it is already extrapolated by the caller);
    above -> contributes zero.
    """

    def __init__(self, s_out, s_fine, mu, w_mu, kms_to_h, u_nodes, ells_out=ELLS):
        self.s_out, self.s_fine, self.u_nodes, self.ells_out = np.asarray(s_out), np.asarray(s_fine), np.asarray(u_nodes), tuple(ells_out)
        self.widths = cell_widths(self.u_nodes)
        ns, nF, nu, nmu = len(s_out), len(s_fine), len(u_nodes), len(mu)
        ds = s_fine[1] - s_fine[0]
        assert np.allclose(np.diff(s_fine), ds), 's_fine must be uniform'
        nl_out = len(self.ells_out)
        self.G = np.zeros((nu, nl_out * ns, len(ELLS) * nF))
        s_par = s_out[:, None] * mu[None, :]                         # (ns, nmu)
        s_perp = s_out[:, None] * np.sqrt(1.0 - mu[None, :] ** 2)
        rows = (np.arange(ns)[:, None] * np.ones((1, nmu), dtype=int))
        out_coef = [(2 * l + 1) * w_mu[None, :] * eval_legendre(l, mu)[None, :] for l in self.ells_out]
        for m, u in enumerate(self.u_nodes):
            sp = s_par - u * kms_to_h                                # s'_par
            s_pr = np.sqrt(s_perp ** 2 + sp ** 2)
            mu_pr = sp / s_pr
            pos = (s_pr - s_fine[0]) / ds
            valid = (pos <= nF - 1).astype(float)
            pos = np.clip(pos, 0.0, nF - 1 - 1e-9)
            j = np.floor(pos).astype(int)
            fr = pos - j
            for a, coef_a in enumerate(out_coef):
                for b, lb in enumerate(ELLS):
                    c = coef_a * eval_legendre(lb, mu_pr) * valid
                    np.add.at(self.G[m], (a * ns + rows, b * nF + j), c * (1 - fr))
                    np.add.at(self.G[m], (a * ns + rows, b * nF + j + 1), c * fr)
        self._G2 = self.G.reshape(nu, -1)
        self.shape = (nl_out * ns, len(ELLS) * nF)

    def matrix(self, weights):
        """M(w) = sum_m weights[m] G[m]."""
        return (weights @ self._G2).reshape(self.shape)

    def apply(self, xi_in, weights):
        """xi_in: (3, nF) multipoles (0, 2, 4) on s_fine -> (len(ells_out), ns)."""
        return (self.matrix(weights) @ np.asarray(xi_in).ravel()).reshape(len(self.ells_out), len(self.s_out))


# ------------------------------------------------------------------ filling the AP'd grid outside the emulator range
def fill_outside_emulator(s_fine, xi, mode='powerlaw', nfit=6):
    """xi: (3, nF) with NaN where common.ap_multipoles had no emulator coverage (CubicSpline extrapolate=False).
    Above the last valid point -> 0.  Below the first valid point -> mode:
      'zero'     : 0
      'constant' : value at the first valid point
      'powerlaw' : xi_l(s) = xi_l(s_v) (s/s_v)^n, n from a log-log fit to the first nfit valid points (needs a single sign and
                   non-zero values; n clipped to [-3, 1]); otherwise falls back to 'constant'
    """
    out = np.array(xi, dtype=float)
    for l in range(out.shape[0]):
        ok = np.isfinite(out[l])
        if not ok.any():
            raise ValueError('no finite emulator output on s_fine')
        idx = np.where(ok)[0]
        j0, j1 = idx[0], idx[-1]
        if not ok[j0:j1 + 1].all():
            raise ValueError('non-finite emulator output inside the valid range')
        out[l, j1 + 1:] = 0.0
        if j0 == 0:
            continue
        sv, xv = s_fine[j0], out[l, j0]
        if mode == 'zero':
            out[l, :j0] = 0.0
        elif mode == 'constant':
            out[l, :j0] = xv
        elif mode == 'powerlaw':
            sel = idx[:nfit]
            y = out[l, sel]
            if np.all(y > 0) or np.all(y < 0):
                n = np.clip(np.polyfit(np.log(s_fine[sel]), np.log(np.abs(y)), 1)[0], -3.0, 1.0)
                out[l, :j0] = xv * (s_fine[:j0] / sv) ** n
            else:
                out[l, :j0] = xv
        else:
            raise ValueError(mode)
    return out


# ------------------------------------------------------------------ self tests
def _gl_mu(n=40):
    x, wt = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * wt


def _toy_multipoles(s):
    """Smooth analytic xi_0, xi_2, xi_4 (no singularity at s -> 0)."""
    sc = 3.0
    f0 = (s ** 2 + sc ** 2) ** (-0.9) * 8.0 ** 1.8 * np.exp(-(s / 170.0) ** 2)
    f2 = -0.35 * (s ** 2 + sc ** 2) ** (-0.7) * 8.0 ** 1.4 * np.exp(-(s / 170.0) ** 2)
    f4 = 0.06 * (s ** 2 + sc ** 2) ** (-0.6) * 8.0 ** 1.2 * np.exp(-(s / 170.0) ** 2)
    return f0, f2, f4


def selftest():
    mu, w_mu = _gl_mu(40)
    L1, kms = 3000.0, 0.0105                  # kms: h^-1 Mpc per km/s (~ z = 1.5, flat LCDM)
    s_out = np.arange(25.0, 130.1, 4.0)
    s_fine = np.arange(1.0, 200.1, 1.0)
    u_nodes = default_u_nodes_kms(L1)
    t0 = time.time()
    geom = ConvGeometry(s_out, s_fine, mu, w_mu, kms, u_nodes)
    print(f'geometry: {len(u_nodes)} u-nodes, G {geom.G.shape} ({geom.G.nbytes / 1e6:.0f} MB), built in {time.time() - t0:.1f} s')
    xi_in = np.array(_toy_multipoles(s_fine))

    # 1. kernel: normalisation, additive variance (independent of the K2 code path), K2(0) = int K1^2
    for w in (20.0, 100.0, 300.0):
        n = int(round(L1 / DU_FINE_KMS)); eps = np.arange(-n, n + 1) * DU_FINE_KMS
        k1 = 1.0 / (1.0 + (eps / w) ** 2); k1 /= k1.sum() * DU_FINE_KMS
        u, k2 = pair_kernel_fine(w, L1)
        var1, var2 = np.sum(eps ** 2 * k1) * DU_FINE_KMS, np.sum(u ** 2 * k2) * DU_FINE_KMS
        k20 = k2[len(k2) // 2]
        wts = pair_weights(w, L1, u_nodes)
        print(f'w={w:5.0f}: int K2={k2.sum() * DU_FINE_KMS:.6f}  var2/var1={var2 / var1:.4f} (expect 2)  '
              f'K2(0)/int K1^2={k20 / (np.sum(k1 ** 2) * DU_FINE_KMS):.4f} (expect 1)  node-weight rms(u)/rms(K2)='
              f'{np.sqrt(np.sum(wts * u_nodes ** 2) / var2):.3f}')

    # 2. w -> 0 limit is the identity (up to interpolation onto s_out)
    wts0 = np.zeros(len(u_nodes)); wts0[len(u_nodes) // 2] = 1.0
    obs0 = geom.apply(xi_in, wts0)
    f = _toy_multipoles(s_out)
    err0 = max(np.max(np.abs(obs0[i] - f[i]) / np.max(np.abs(f[i]))) for i in range(3))
    print(f'delta-kernel limit: max |M xi - xi(s_out)| / max|xi_l| = {err0:.2e}')

    # 3. mass conservation: xi_0 = const, xi_2 = xi_4 = 0 stays put
    flat = np.zeros_like(xi_in); flat[0] = 1.0
    for w in (30.0, 100.0, 300.0):
        o = geom.apply(flat, pair_weights(w, L1, u_nodes))
        print(f'w={w:5.0f}: constant xi_0=1 -> xi_0 in [{o[0].min():.5f}, {o[0].max():.5f}], max|xi_2|,|xi_4| = '
              f'{np.abs(o[1]).max():.1e}, {np.abs(o[2]).max():.1e}')

    # 4. brute force: direct u integral of the ANALYTIC xi_s(s_perp, s_par - u) on a fine uniform u grid
    for w in (30.0, 100.0, 300.0):
        u, k2 = pair_kernel_fine(w, L1)
        keep = np.abs(u) < 2 * L1 - 1e-9
        u_s, k2 = u[keep] * kms, k2[keep] * DU_FINE_KMS
        ref = np.zeros((3, len(s_out)))
        for i, s in enumerate(s_out):
            spar, sperp = s * mu, s * np.sqrt(1 - mu ** 2)
            sp_all = spar[:, None] - u_s[None, :]
            spr = np.sqrt(sperp[:, None] ** 2 + sp_all ** 2)
            mup = sp_all / spr
            f0, f2, f4 = _toy_multipoles(spr)
            xi_s = f0 + f2 * eval_legendre(2, mup) + f4 * eval_legendre(4, mup)
            xi_s = np.where(spr > s_fine[-1], 0.0, xi_s)                 # same upper cut as the operator
            xi_mu = xi_s @ k2 / k2.sum()
            for a, l in enumerate(ELLS):
                ref[a, i] = (2 * l + 1) * np.sum(w_mu * xi_mu * eval_legendre(l, mu))
        obs = geom.apply(xi_in, pair_weights(w, L1, u_nodes))
        scale = np.max(np.abs(f[0]))
        rel = [np.max(np.abs(obs[a] - ref[a])) / np.max(np.abs(ref[a])) for a in range(3)]
        shift = [np.max(np.abs(ref[a] - np.array(f[a]))) / np.max(np.abs(f[a])) for a in range(3)]
        print(f'w={w:5.0f}: operator vs brute force  max rel diff (l=0,2,4) = {rel[0]:.1e}, {rel[1]:.1e}, {rel[2]:.1e}'
              f'   [effect of the convolution itself: {shift[0]:.1e}, {shift[1]:.1e}, {shift[2]:.1e}]')

    # 4b. sensitivity to what is assumed below the lowest tabulated s' (toy xi with core 3 h^-1 Mpc; real xi is more singular)
    sf4 = np.arange(4.0, 200.1, 1.0)
    g4 = ConvGeometry(s_out, sf4, mu, w_mu, kms, u_nodes)
    for w in (100.0, 300.0):
        wt = pair_weights(w, L1, u_nodes)
        d = np.abs(g4.apply(np.array(_toy_multipoles(sf4)), wt) - geom.apply(xi_in, wt))
        ref_scale = np.abs(np.array(_toy_multipoles(s_out))).max(axis=1)
        print(f'w={w:5.0f}: holding xi constant below s\'=4 instead of using the analytic xi changes the output by '
              f'{d[0].max() / ref_scale[0]:.1e}, {d[1].max() / ref_scale[1]:.1e}, {d[2].max() / ref_scale[2]:.1e} of max|xi_l|')

    # 5. speed of the per-call part
    t0 = time.time()
    for _ in range(50):
        geom.apply(xi_in, pair_weights(100.0, L1, u_nodes))
    print(f'per-call cost (kernel + M(w) + apply): {(time.time() - t0) / 50 * 1e3:.1f} ms')


if __name__ == '__main__':
    selftest()