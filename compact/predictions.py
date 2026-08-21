""" 
Puts all of the ingredients of the compact streaming model together. 
"""

from lpt import *
from model import *
from streaming import * 

from scipy.interpolate import UnivariateSpline
from halotools.mock_observables import tpcf_multipole


class CompactStreamingModel:

    def __init__(self, redshift, config):
        self.z = redshift 
        self.cfg = config
        self._lpt = None 

    def _lpt_inputs(self, cosmo, bias):
        lpt_pars = np.concatenate( (bias, np.zeros(8)) )
        gli = GetLPTInputs(self.z, cosmo, lpt_pars )

        self._lpt = ( gli.xi_real(self.cfg.r, lpt_pars), 
                     gli.pwv_mean(self.cfg.r, lpt_pars), 
                     gli.f, gli.H, 
                     gli.D_A )

        return self._lpt
    
    def _compact_params(self, cosmo, mass_bin=None):
        """
        mass_bin : None (default, uses the simultaneous fit) | 'low_M' | 'high_M'
                Passing 'low_M'/'high_M' evaluates that mass bin's individual
                fit instead -- for systematic-error checks, not production use.
        """
        if self.z not in self.cfg.compact_emulator:
            raise NotImplementedError(
                f"No compact-model emulator calibrated at z={self.z}; "
                f"available redshifts: {list(self.cfg.compact_emulator)}"
            )
        emu = self.cfg.compact_emulator[self.z]

        h, ombh2, omch2, As = cosmo
        cosmo_vals = np.array([ombh2, omch2, np.log(1e10 * As), emu['ns_fiducial']])
        delta_vals = self.cfg.fiducial_cosmo - cosmo_vals   # Delta_theta = theta_fid - theta

        fits = emu['fits'] if mass_bin is None else emu['fits_indiv'][mass_bin]

        params = np.array(emu['mean_params'], dtype=float)
        for row_idx, dependent_cols, beta in fits:
            params[row_idx] = beta[0] + sum(beta[1 + n] * delta_vals[col]
                                            for n, col in enumerate(dependent_cols))
        return params

    def _pdf(self, lpt, cosmo):
        mean = UnivariateSpline(self.cfg.r, lpt[1], s=0)
        params = self._compact_params(cosmo)

        return Pv_compact(self.cfg.vlos, self.cfg.r, self.cfg.r, mean, params,
                        unit_conversion=1/lpt.conv, fix_sig_Del=False)
    
    def _multipoles(self, ells=(0, 2)):

        xi_real = UnivariateSpline(self.cfg.r, self._lpt[1], s=0)
        pdf = UnivariateSpline(self.cfg.r, self._pdf, s=0)

        xi_s_mu = simps_integrate(s_c=self.cfg.r, 
                                  mu_c=self.cfg.mu_c, 
                                  twopcf_function=xi_real, 
                                  los_pdf_function=pdf, 
                                  limit=140, 
                                  epsilon=0.01, 
                                  n=100)

        multi = np.zeros((len(ells), len(self.cfg.r)))

        for ell in ells:
            multi[ell//2] = tpcf_multipole(xi_s_mu, self.cfg.mu_bins, order=ell)

        return multi





