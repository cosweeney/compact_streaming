import numpy as np
from scipy.stats import t, gamma

from scipy.interpolate import interp1d

def skewed_t_pdf(x, xi, omega, alpha, nu):
    """
    Azzalini & Capitanio skewed t-distribution PDF.
    
    Parameters:
    xi: location parameter
    omega: scale parameter (> 0)
    alpha: skewness parameter
    nu: degrees of freedom (> 0)
    """
    z = (x - xi) / omega
    
    t_pdf = t.pdf(z, df=nu)
    
    t_cdf = t.cdf(alpha * z * np.sqrt((nu + 1) / (nu + z**2)), df=nu + 1)
    
    pdf = (2 / omega) * t_pdf * t_cdf
    
    return pdf

def sigma_mod(x, sigma0, A_rho, sigma_rho):
    return np.sqrt(2 * sigma0**2 * (1 - A_rho * np.exp(-x / sigma_rho)))

def delta_mod(x, A_delta, sigma_delta):
    return A_delta * x / (sigma_delta + x)

class SkewTMeanInterpolator:
    """
    Pre-computes the relationship between (Delta_mean / sigma_v) and (omega, alpha)
    for a skewed t-distribution using the algebraic mean. 
    """
    def __init__(self, nu=6, alpha_bounds=(-15, 15), pts=1000):
        self.nu = nu
        
        alphas = np.linspace(alpha_bounds[0], alpha_bounds[1], pts)
        
        delta = alphas / np.sqrt(1 + alphas**2)
        
        # mean of skewed t (xi=0, omega=1)
        # mu_z = delta * sqrt(nu / pi) * [Gamma((nu - 1) / 2) / Gamma(nu / 2)]
        gamma_factor = gamma((nu - 1) / 2) / gamma(nu / 2)
        mu_z = delta * np.sqrt(nu / np.pi) * gamma_factor
        
        # standard dev of skewed t
        var_z = (nu / (nu - 2)) - mu_z**2
        sig_z = np.sqrt(var_z)
        
        # ratio R = Delta_mean / sigma_v for distribution
        # assuming Delta_mean = xi - mean = 0 - mu_z = -mu_z
        R_vals = -mu_z / sig_z
        
        # interp1d requires the x-array (R_vals) to be strictly monotonically increasing.
        # as alpha increases, R_vals strictly decreases, so we reverse the arrays.
        if R_vals[-1] < R_vals[0]:
            valid_R = R_vals[::-1]
            valid_alphas = alphas[::-1]
        else:
            valid_R = R_vals
            valid_alphas = alphas
            
        # interpolator 1: ratio to alpha 
        self.R_to_alpha = interp1d(valid_R, valid_alphas, kind='cubic', 
                                   bounds_error=False, 
                                   fill_value=(valid_alphas[0], valid_alphas[-1]))
        
        # interpolator 2: alpha to standard dev
        self.alpha_to_sigZ = interp1d(alphas, sig_z, kind='cubic', 
                                      bounds_error=False, 
                                      fill_value=(sig_z[0], sig_z[-1]))

    def get_skew_params(self, delta_mean, sig_v):
        """Maps physical parameters (Delta_mean, sigma_v) to skewed t parameters (omega, alpha)"""
        ratio = delta_mean / sig_v
        alpha = self.R_to_alpha(ratio)
        sig_z = self.alpha_to_sigZ(alpha)
        omega = sig_v / sig_z
        return omega, alpha
    
# instantiate globally so it isn't rebuilt on every function call
skew_interp = SkewTMeanInterpolator(nu=6)

sig_pars = np.array([0.20281935253098268, 27.382880273166126]) # TODO: update to follow from config file or smthg later. defaults to hh BF vals for now. 
del_pars = np.array([47.766923483904314, 0.29581892896320244])

def Pv_compact(v, r_p, r_los, mean_vr, pars=[*sig_pars, *del_pars], s_h=355):
    """
    Evaluates P(v | r_p, r_los, M) modeled as a skewed t distribution using the compact model.
    """
    # geometery
    r = np.sqrt(r_p**2 + r_los**2)
    x = r - r_p
    
    # inputs 
    mu = mean_vr(r) * r_los/r  
    sig_v = sigma_mod(x, s_h, *pars[:2])
    
    # enforce symmetry about r_los
    sign_rlos = np.where(r_los != 0, np.sign(r_los), 1.0)
    delta_val = delta_mod(x, *pars[2:]) * sign_rlos
    
    # interpolate skewed t params
    omega, alpha = skew_interp.get_skew_params(delta_val, sig_v)
    
    xi = mu + delta_val
    
    # eval PDF
    nu = skew_interp.nu
    
    pdf = skewed_t_pdf(v, xi, omega, alpha, nu)
    
    return pdf

