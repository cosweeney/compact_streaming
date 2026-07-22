""" 
For predicting tpcf, velocity moments using LPT as implemented with velocileptors. 
"""
import numpy as np
from colossus.cosmology import cosmology
from velocileptors.LPT.gaussian_streaming_model_fftw import GaussianStreamingModel


class GetLPTInputs:

    def __init__(self, redshift=1/0.8376-1, cosmoname='planck13', pars=[1, 0, 0, 0, 0, 0, 0, 0, 0]):
        # setup, init cosmology, gsm
        self.cosmo =cosmology.setCosmology(cosmoname)  # MDPL2 uses planck '13

        a = 1/(1+redshift) 
        H = self.cosmo.Hz(redshift)
        h = self.cosmo.h

        self.conv = a*H/h # unit conversion factor from Mpc/h to km/s

        self.f = self.cosmo.Om(redshift) ** 0.55 # growth rate

        def get_mdpl2_pklin(self, z, kmin = 1e-4, kmax=20.0, nk=1024):
            k = np.logspace(np.log10(kmin), np.log10(kmax), nk)  # h/Mpc
            pk = cosmology.Cosmology.matterPowerSpectrum(self.cosmo, k=k, z=z, model='camb')  # (Mpc/h)^3
            return k, pk

        self.klin, self.Plin = get_mdpl2_pklin(redshift)
        self.gsm = GaussianStreamingModel(self.klin, self.Plin)
        self.gsm.convert_sigma_bases()

        self.pars = pars
        self.redshift = redshift

    def xi_real(self, rad, pars): 
        self.gsm.compute_cumulants(*pars)
        xir = self.gsm.compute_xi_real(rad,*pars)
        return xir

    def pwv_mean(self, r, pars):
        self.gsm.compute_cumulants(*pars)

        xi_int = np.interp(r, self.gsm.rint, self.gsm.xieft)
        v_int  = np.interp(r, self.gsm.rint, self.gsm.veft)
        
        return self.conv*self.f*v_int/(1+xi_int)

    def pwv_variance(self, r, mu, pars):
        self.gsm.compute_cumulants(*pars)
        
        xi_int = np.interp(r, self.gsm.rint, self.gsm.xieft)
        s0 =  np.interp(r, self.gsm.rint, self.gsm.s0eft)
        s2 = np.interp(r, self.gsm.rint, self.gsm.s2eft) 

        v_int  = self.pwv_mean(r, pars)

        return self.conv**2 * self.f**2 * (s0 + 0.5 * (3*mu**2 - 1) * s2) / (1+xi_int) - v_int**2