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
    
    def _pdf(self, lpt, params):
        mean = UnivariateSpline(self.cfg.r, lpt[1], s=0)

        return Pv_compact(self.cfg.vlos, self.cfg.r, self.cfg.r, mean, params, unit_conversion=1/lpt.conv)

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
            multi[ell//2] = tpcf_multipole(xi_s_mu, self.cf.mu_bins, order=ell)

        return multi





