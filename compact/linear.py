"""
Linear-theory ingredients that tie sigma_0 to sigma_{v,lin} (the 11par_v3 model):

    sigma_0^2 = A_sigma * sigma_{v,lin}^2,
    sigma_{v,lin}^2 = (f a H / h)^2 / (6 pi^2) * int dk P_cb(k, z)      [km/s]^2

with P_cb the CDM + baryon linear power spectrum, f = f sigma_8 / sigma_8 from CAMB, and a H / h in km/s per
h^-1 Mpc. The same two functions are used by the model (compact.lpt.GetLPTInputs) and by the calibration, so
A_sigma means the same thing in both. CAMB is called exactly as GetLPTInputs calls it: only (H0, ombh2, omch2,
As, ns) are set; everything else is a CAMB default.
"""
import numpy as np
import camb
from scipy.integrate import simpson

KMIN, KMAX, NK = 1e-4, 20.0, 1024      # h/Mpc; the P(k) grid of GetLPTInputs.get_pklin_gsm


def camb_results(h, ombh2, omch2, As, ns, redshifts, kmax=KMAX):
    """CAMB results with the linear matter power spectrum at `redshifts` (same call as GetLPTInputs)."""
    pars = camb.set_params(H0=100.0 * h, ombh2=ombh2, omch2=omch2, As=As, ns=ns)
    pars.NonLinear = camb.model.NonLinear_none
    pars.set_matter_power(redshifts=list(redshifts), kmax=2.0 * kmax * h)
    return camb.get_results(pars)


def growth_rate(results, z):
    """f(z) = f sigma_8 / sigma_8 from CAMB (get_fsigma8 and get_sigma8 use the order of the transfer redshifts)."""
    tr = results.Params.Transfer
    zs = np.asarray(tr.PK_redshifts[:tr.PK_num_redshifts])
    i = int(np.argmin(np.abs(zs - z)))
    return float(results.get_fsigma8()[i] / results.get_sigma8()[i])


def sigma_v_lin(results, z, h):
    """1-D linear velocity dispersion [km/s] at redshift z (which must be one of the redshifts given to CAMB)."""
    k, zs, pk = results.get_matter_power_spectrum(minkh=KMIN, maxkh=KMAX, npoints=NK,
                                                  var1='delta_nonu', var2='delta_nonu')
    pk = pk[int(np.argmin(np.abs(np.asarray(zs) - z)))]          # P(k) is returned in ascending z
    aH_over_h = results.hubble_parameter(z) / (1.0 + z) / h
    return growth_rate(results, z) * aH_over_h * np.sqrt(simpson(pk, x=k) / (6.0 * np.pi**2))
