"""
Compact pairwise velocity PDF model.

9-parameter model:
  sigma_v family (4):  sigma0, A_rho0, A_rho_p, sigma_rho
  Delta family   (5):  A_delta0, r_delta, sigma_delta, beta_delta, alpha

Parameter ordering (pars vector index):
  0  sigma0       -- asymptotic pairwise velocity dispersion [km/s]
  1  A_rho0       -- amplitude of the correlation function at R_piv
  2  A_rho_p      -- exponential decay length of A_rho with R [h^-1 Mpc]
  3  sigma_rho    -- exponential decay length in x [h^-1 Mpc]
  4  A_delta0     -- amplitude of the Delta offset at R=0 [km/s]
  5  r_delta      -- exponential decay length of A_delta with R [h^-1 Mpc]
  6  sigma_delta  -- turnover scale in x for the Delta shape [h^-1 Mpc]
  7  beta_delta   -- power-law steepness of the Delta turnover
  8  alpha        -- small-x power-law index of Delta

This replaces the earlier 6-parameter model (flat-core Delta with only
A_delta, sigma_delta, and a fixed/toggled sigma_delta). The functional
forms here match cosmo_grid_vel_fit_out.py's fitting functions exactly,
so a 9-vector fitted there means the same thing at predict time.
"""

import numpy as np
from scipy.stats import t
from scipy.special import gamma
from scipy.interpolate import interp1d


# ------------------------------------------------------------------ PDF
def skewed_t_pdf(x, xi, omega, alpha, nu):
    """Azzalini & Capitanio skewed t-distribution PDF."""
    z = (x - xi) / omega
    t_pdf = t.pdf(z, df=nu)
    t_cdf = t.cdf(alpha * z * np.sqrt((nu + 1) / (nu + z**2)), df=nu + 1)
    return (2 / omega) * t_pdf * t_cdf


# ------------------------------------------------------------------ sigma_v model
R_PIV_FIXED = 72.5   # fixed pivot, matches the fitting script's explicit
                      # R_piv=72.5 in fit_sigma_joint -- verified stable
                      # across sims/z/bins (std=0.000 everywhere except a
                      # mild drift at z>=0.8 bin 1, which the fixed pivot
                      # now overrides for consistency)


def sigma_mod(x, sigma0, A_rho, sigma_rho):
    """Pairwise velocity dispersion as a function of x = r - r_perp,
    at a single R (r_perp) slice whose A_rho has already been evaluated."""
    arg = 2.0 * sigma0**2 * (1.0 - A_rho * np.exp(-x / sigma_rho))
    return np.sqrt(np.clip(arg, 1e-8, None))


def A_rho_mod(R, A_piv, A_rho_p, R_piv=R_PIV_FIXED):
    """R-dependence of the sigma_v correlation amplitude: exponential
    decay centered on R_piv."""
    return A_piv * np.exp(-(R - R_piv) / A_rho_p)


# ------------------------------------------------------------------ Delta model
def A_delta_mod(R, A_delta0, r_delta):
    """R-dependence of the Delta offset amplitude: exponential decay."""
    return A_delta0 * np.exp(-R / r_delta)


def delta_mod_alpha(x, A, sigma_delta, beta_delta, alpha):
    """Delta offset as a function of x, at a single R slice whose
    amplitude A has already been evaluated via A_delta_mod."""
    return A * x**alpha / (1.0 + (x / sigma_delta)**beta_delta)


# ------------------------------------------------------------------ skew-t mapping
class SkewTMeanInterpolator:
    """Pre-computes the relationship between (Delta_mean / sigma_v) and
    (omega, alpha) for a skewed t-distribution using the algebraic mean."""

    def __init__(self, nu=6, alpha_bounds=(-15, 15), pts=1000):
        self.nu = nu
        alphas = np.linspace(alpha_bounds[0], alpha_bounds[1], pts)
        delta = alphas / np.sqrt(1 + alphas**2)

        gamma_factor = gamma((nu - 1) / 2) / gamma(nu / 2)
        mu_z = delta * np.sqrt(nu / np.pi) * gamma_factor

        var_z = (nu / (nu - 2)) - mu_z**2
        sig_z = np.sqrt(var_z)

        R_vals = -mu_z / sig_z

        if R_vals[-1] < R_vals[0]:
            valid_R = R_vals[::-1]
            valid_alphas = alphas[::-1]
        else:
            valid_R = R_vals
            valid_alphas = alphas

        self.R_to_alpha = interp1d(valid_R, valid_alphas, kind='cubic',
                                   bounds_error=False,
                                   fill_value=(valid_alphas[0], valid_alphas[-1]))

        self.alpha_to_sigZ = interp1d(alphas, sig_z, kind='cubic',
                                      bounds_error=False,
                                      fill_value=(sig_z[0], sig_z[-1]))

    def get_skew_params(self, delta_mean, sig_v):
        """Maps physical (Delta_mean, sigma_v) -> skewed t (omega, alpha)."""
        ratio = delta_mean / sig_v
        alpha = self.R_to_alpha(ratio)
        sig_z = self.alpha_to_sigZ(alpha)
        omega = sig_v / sig_z
        return omega, alpha


# instantiate globally so it isn't rebuilt on every function call
skew_interp = SkewTMeanInterpolator(nu=6)


# ------------------------------------------------------------------ Pv_compact
def Pv_compact(v, r_p, r_los, mean_vr, pars, unit_conversion=None):
    """Compact pairwise velocity PDF evaluated at line-of-sight velocity v,
    given perpendicular separation r_p and line-of-sight separation r_los.

    Parameters
    ----------
    v : float or array
        Line-of-sight pairwise velocity [km/s].
    r_p : float
        Perpendicular (transverse) separation [h^-1 Mpc].
    r_los : float
        Line-of-sight separation [h^-1 Mpc].
    mean_vr : callable
        Interpolator returning the mean radial pairwise velocity v_r(r)
        [km/s] as a function of 3D separation r.
    pars : array-like, length 9
        Compact-model parameters in PARAM_NAMES order:
        (sigma0, A_rho0, A_rho_p, sigma_rho,
         A_delta0, r_delta, sigma_delta, beta_delta, alpha)
    unit_conversion : float or None
        If not None, applies Mpc/h -> km/s conversion factor (a*H/h).

    Returns
    -------
    pdf : float or array
        P(v | r_p, r_los) evaluated at v.
    """
    r = np.sqrt(r_p**2 + r_los**2)
    x = r - r_p

    # unpack -- explicit names for clarity
    sigma0, A_rho0, A_rho_p, sigma_rho = pars[0], pars[1], pars[2], pars[3]
    A_delta0, r_delta, sigma_delta, beta_delta, alpha = pars[4], pars[5], pars[6], pars[7], pars[8]

    # defensive clips: z-interpolation or cosmology-extrapolation can push
    # a predicted parameter outside the range it was ever fit within, even
    # though the fit itself was bounded -- catch that here rather than let
    # it produce NaN/inf downstream
    r_delta = max(r_delta, 1e-3)
    sigma_delta = max(sigma_delta, 1e-3)
    beta_delta = max(beta_delta, 1e-3)
    alpha = max(alpha, 0.0)

    # mean infall projected onto LOS
    mu = mean_vr(r) * r_los / r

    # sigma_v: R-dependent via A_rho_mod
    A_rho = A_rho_mod(r_p, A_rho0, A_rho_p)
    sig_v = sigma_mod(x, sigma0, A_rho, sigma_rho)

    # Delta offset: R-dependent via A_delta_mod, x-dependent via delta_mod_alpha
    sign_rlos = np.where(r_los != 0, np.sign(r_los), 1.0)
    A_delta = A_delta_mod(r_p, A_delta0, r_delta)
    delta_val = delta_mod_alpha(x, A_delta, sigma_delta, beta_delta, alpha) * sign_rlos

    # map (sig_v, delta) -> skewed t parameters
    omega, skew_alpha = skew_interp.get_skew_params(delta_val, sig_v)
    xi = mu + delta_val
    nu = skew_interp.nu

    if unit_conversion:
        return unit_conversion * skewed_t_pdf(unit_conversion * v, xi, omega, skew_alpha, nu)
    return skewed_t_pdf(v, xi, omega, skew_alpha, nu)