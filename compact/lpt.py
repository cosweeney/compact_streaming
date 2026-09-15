""" 
For predicting tpcf, velocity moments using LPT as implemented with velocileptors. 
"""
import numpy as np
import camb
from colossus.cosmology import cosmology
from velocileptors.LPT.gaussian_streaming_model_fftw import GaussianStreamingModel


class GetLPTInputs:

    cosmo_fid = [0.6777, 0.02214, 0.11891, 2.2e-9, 0.9649] 

    def __init__(self, redshift=1/0.8376-1, cosmo=cosmo_fid, pars=[1, 0, 0, 0, 0, 0, 0, 0, 0]):
        h, ombh2, omch2, As, ns = cosmo

        self.h = h
        self.redshift = redshift
        self.kmax = 20.0 
        self.pars = pars


        camb_pars = camb.set_params(H0=100.0 * h, 
                                    ombh2=ombh2, 
                                    omch2=omch2, 
                                    As=As, 
                                    ns=ns,                                   
                                    )

        camb_pars.NonLinear = camb.model.NonLinear_none
        camb_pars.set_matter_power(redshifts=[redshift], kmax=2.0 * self.kmax * h)
        self.results = camb.get_results(camb_pars)

        self.a = 1/(1+self.redshift) 
        self.H = self.results.hubble_parameter(redshift)
        self.D_A = self.results.angular_diameter_distance(redshift)

        self.conv = self.a*self.H/h # unit conversion factor from Mpc/h to km/s

        Om0 = (ombh2 + omch2) / h**2
        Omz = Om0 * (1.0 + redshift)**3 / (self.H / (100.0 * h))**2
        self.f = Omz**0.55 # growth rate

        self.klin = None
        self.Plin = None
        self.gsm = None

    def get_pklin_gsm(self, kmin = 1e-4, kmax=20.0, nk=1024):
        # k in h/Mpc, P in (Mpc/h)^3, at self.redshift
        k, _, pk = self.results.get_matter_power_spectrum(
            minkh=kmin, maxkh=kmax, npoints=nk)
        
        self.klin, self.Plin = k, pk[0]
        self.gsm = GaussianStreamingModel(self.klin, self.Plin, kmax=10.0) # set kmax higher than default here; potential issues with extrapolation
        self.gsm.convert_sigma_bases()
    
    def _ensure_cumulants(self, pars):
        pars = np.asarray(pars)
        if not np.array_equal(getattr(self.gsm, '_last_cumulant_pars', None), pars):
            self.gsm.compute_cumulants(*pars)
            self.gsm._last_cumulant_pars = pars.copy()

    def xi_real(self, rad, pars):
        if self.gsm is None: self.get_pklin_gsm()
        self._ensure_cumulants(pars)
        return self.gsm.compute_xi_real(rad, *pars)

    def pwv_mean(self, r, pars):
        if self.gsm is None: self.get_pklin_gsm()
        self._ensure_cumulants(pars)
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

    def get_AP(self,):

        h_fid, ombh2_fid, omch2_fid, As_fid, ns_fid = self.cosmo_fid

        pars_fid = camb.set_params(H0=100.0 * h_fid, 
                                    ombh2=ombh2_fid, 
                                    omch2=omch2_fid, 
                                    As=As_fid, 
                                    ns=ns_fid,  # fix to planck '19                                  
                                    )

        pars_fid.NonLinear = camb.model.NonLinear_none
        pars_fid.set_matter_power(redshifts=[self.redshift], kmax=2.0 * self.kmax * h_fid)
        results_fid = camb.get_results(pars_fid)

        H_fid = results_fid.hubble_parameter(self.redshift)
        D_A_fid = results_fid.angular_diameter_distance(self.redshift)

        q_perp = self.D_A / D_A_fid 
        q_par  = H_fid / self.H

        return q_perp, q_par
    

def apply_AP(s_fid, mu_fid, q_perp, q_par):
    
    q_a = q_par**(1/3) * q_perp**(2/3)
    q_e = ( q_par / q_perp )**(1/3)

    s_p  = s_fid*q_a*np.sqrt( q_e**4 * mu_fid**2 + (1-mu_fid**2)/q_e**2 )
    mu_p = 1/np.sqrt( 1+ (1/mu_fid**2 - 1)/q_e**6 )

    return s_p, mu_p