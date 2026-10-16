"""
Compact pairwise velocity PDF model.

11-parameter model:
  sigma_v family (4):  sigma0, A_rho0, A_rho_p, sigma_rho
  Delta family   (5):  A_delta0, r_delta, sigma_delta, beta_delta, alpha
  nu(r) family   (2):  nu_0, r_c

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
  9  nu_0         -- saturation value of the tail shape nu(r) at large r
 10  r_c          -- turnover scale of nu(r) [h^-1 Mpc]

The tail shape is nu(r) = 2 + (nu_0 - 2) / (1 + (r_c/r)^2), floored at 2.05.
The skew-t scale omega and skewness alpha are not free: they are set in closed
form so the PDF has mean mu, standard deviation sigma_v and location
xi = mu + Delta (see skewt_params_from_delta). This replaces the fixed-nu = 6
lookup table (SkewTMeanInterpolator).
"""

import numpy as np
from scipy.stats import t
from scipy.special import gammaln


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


# ------------------------------------------------------------------ nu(r) and skew-t mapping
def nu_of_r(r, nu0, rc, nu_floor=2.05):
    """Tail shape nu(r) = 2 + (nu0 - 2) / (1 + (rc/r)^2), floored at nu_floor.
    Note nu(r -> 0) = 2."""
    return np.maximum(2.0 + (nu0 - 2.0) / (1.0 + (rc / r)**2), nu_floor)


def skewt_params_from_delta(delta, sig_v, nu):
    """(omega, alpha) of a nu-dof skew-t with standard deviation sig_v and
    location offset delta = xi - mean. Closed form:
        mean = xi + omega * d * b_nu,
        var  = omega^2 nu/(nu-2) - (omega * d * b_nu)^2,
    with d = alpha/sqrt(1+alpha^2), b_nu = sqrt(nu/pi) Gamma((nu-1)/2)/Gamma(nu/2).
    d is clipped to +-0.9999."""
    b = np.sqrt(nu / np.pi) * np.exp(gammaln((nu - 1) / 2) - gammaln(nu / 2))
    omega = np.sqrt((sig_v**2 + delta**2) * (nu - 2) / nu)
    d = np.clip(-delta / (omega * b), -0.9999, 0.9999)
    return omega, d / np.sqrt(1 - d**2)


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
    pars : array-like, length 11
        Compact-model parameters in the order:
        (sigma0, A_rho0, A_rho_p, sigma_rho,
         A_delta0, r_delta, sigma_delta, beta_delta, alpha,
         nu_0, r_c)
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
    nu0, r_c = pars[9], pars[10]

    # defensive clips: z-interpolation or cosmology-extrapolation can push
    # a predicted parameter outside the range it was ever fit within, even
    # though the fit itself was bounded -- catch that here rather than let
    # it produce NaN/inf downstream
    r_delta = max(r_delta, 1e-3)
    sigma_delta = max(sigma_delta, 1e-3)
    beta_delta = max(beta_delta, 1e-3)
    alpha = max(alpha, 0.0)
    nu0 = max(nu0, 2.01)
    r_c = max(r_c, 1e-3)

    # mean infall projected onto LOS
    mu = mean_vr(r) * r_los / r

    # sigma_v: R-dependent via A_rho_mod
    A_rho = A_rho_mod(r_p, A_rho0, A_rho_p)
    sig_v = sigma_mod(x, sigma0, A_rho, sigma_rho)

    # Delta offset: R-dependent via A_delta_mod, x-dependent via delta_mod_alpha
    sign_rlos = np.where(r_los != 0, np.sign(r_los), 1.0)
    A_delta = A_delta_mod(r_p, A_delta0, r_delta)
    delta_val = delta_mod_alpha(x, A_delta, sigma_delta, beta_delta, alpha) * sign_rlos

    # r-dependent tail shape; map (sig_v, delta, nu) -> skewed t parameters
    nu = nu_of_r(r, nu0, r_c)
    omega, skew_alpha = skewt_params_from_delta(delta_val, sig_v, nu)
    xi = mu + delta_val

    if unit_conversion:
        return unit_conversion * skewed_t_pdf(unit_conversion * v, xi, omega, skew_alpha, nu)
    return skewed_t_pdf(v, xi, omega, skew_alpha, nu)

def Pv_gaussian(v, r_p, r_los, mean_vr, pars, unit_conversion=None):
    """Gaussian pairwise velocity PDF using only the compact model's first two moments:
    the LPT mean mu = <v_r|r> r_los/r and the compact-model variance sigma_v^2
    (sigma_mod with A_rho_mod). The offset Delta and tail shape nu(r) are ignored,
    so this has the same mean and standard deviation as Pv_compact but no skewness
    or heavy tails. Same call signature as Pv_compact (drop-in replacement);
    only pars[0:4] = (sigma0, A_rho0, A_rho_p, sigma_rho) are used."""
    r = np.sqrt(r_p**2 + r_los**2)
    x = r - r_p

    sigma0, A_rho0, A_rho_p, sigma_rho = pars[0], pars[1], pars[2], pars[3]

    mu = mean_vr(r) * r_los / r
    sig_v = sigma_mod(x, sigma0, A_rho_mod(r_p, A_rho0, A_rho_p), sigma_rho)

    def gauss(vv):
        return np.exp(-0.5 * ((vv - mu) / sig_v) ** 2) / (np.sqrt(2.0 * np.pi) * sig_v)

    if unit_conversion:
        return unit_conversion * gauss(unit_conversion * v)
    return gauss(v)
