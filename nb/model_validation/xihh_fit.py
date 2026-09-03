""" 
For fitting LPT predictions of the tpcf, 
credit to Cheng Heng-Wei.  
"""


#%%
import numpy as np 
import h5py as h5 
import camb
from scipy.linalg import eigvals
import matplotlib.pyplot as plt
from velocileptors.LPT.gaussian_streaming_model_fftw import GaussianStreamingModel
from scipy.optimize import minimize

# plotting config
import matplotlib
matplotlib.rcParams.update({'text.usetex': True, 
                            'font.family': 'Computer Modern Roman'})

from colossus.cosmology import cosmology

#%%
cosmo=cosmology.setCosmology('planck13')  # MDPL2 uses Planck 2013

f = cosmo.Om(1/0.8376-1) ** 0.55  # growth rate at z=0.8376

def get_mdpl2_pklin(z, kmin = 1e-4, kmax=20.0, nk=1024):
    k = np.logspace(np.log10(kmin), np.log10(kmax), nk)  # h/Mpc
    pk = cosmology.Cosmology.matterPowerSpectrum(cosmo, k=k, z=z, model='camb')  # (Mpc/h)^3
    return k, pk

# Quick test
klin, Plin = get_mdpl2_pklin(z=1/0.8376-1)
gsm = GaussianStreamingModel(klin, Plin)
gsm.convert_sigma_bases()



#%%

def plot_pklin() : 
    plt.plot(klin, klin*Plin)
    plt.xscale('log')
    plt.xlabel(r'$k$ [h/Mpc]')
    plt.ylabel(r'$kP(k)$ [h$^2$/Mpc$^2$]')
    plt.grid(True)
    plt.show()
    #plt.savefig('plin.png')

def xihh_real(rad, pars) : 
    gsm.compute_cumulants(*pars)
    xir = gsm.compute_xi_real(rad,*pars)
    return xir

def xihh_fit():
    with h5.File("/spiffball/cosweeney/simulations/MDPL2/data/CorrFuncs/xi_hh_150.hdf5", "r") as hdf :
        xihh  = np.array(hdf['xi'][()])
        r      = np.array(hdf['r_cens'][()])
        sigma2 = np.diag(hdf['xi_cov'][()])
    
    rmins = np.array([0.1, 1, 5, 16])
    xihh_bests = np.zeros((len(r), len(rmins)))
    par_bests = np.zeros((len(rmins), 9))

    for idx, rmin in enumerate(rmins) :
        
        mask = ((rmin < r) & (r < 120))
        def xihh_loss(b):
            pars = b[:-1]
            xihh_in = xihh_real(r, pars)
            sigma2_soft = sigma2/np.sqrt(999) + b[-1]**2*xihh**2 
            return sum((xihh_in[mask]-xihh[mask])**2/(sigma2_soft[mask])) + sum(np.log(sigma2_soft[mask]))

        # velocipletors have 9 free parameters for hh auto corrfunc : 
        # 4 biases [b_1, b_2, b_s, b_3] (delta, delta^2, s_ij^2, delta^3)
        # 4 counter-terms : [alpha, alpha_v, alpha_s0, alpha_s2] (k^2, velocity ct, monopole and quadrupole pair-wise velocity ct)
        # 1 stochastic term [s2FoG]
        # But alpha_s0 and alpha_s2 are degenerate with others -> set to zero
        x0 = (1, -0.5, 0.3, 0, 0, 0, 0, 0, 0, 1e-3)
        #x0 = [0.5805860842963466, -1.6984844371740406, 0.5909806855955604, -0.37674584003944256, -3.830412439255513, 5.680829903647382, 0.0, 0.0, 0.0, 0.23419746609515735]

        # bounds = (
        #     (-np.inf, np.inf), # b_1
        #     (-np.inf, np.inf), # b_2
        #     (-np.inf, np.inf), # b_s
        #     (-np.inf, np.inf), # b_3
        #     (-np.inf, np.inf), # alpha for counter-term
        #     (-np.inf, np.inf), # alpha_v for velocity counter-term
        #     (0, 0),            # alpha_s0, set 0   
        #     (0, 0),            # alpha_s2, set 0
        #     (-np.inf, np.inf), # stochastic term
        #     (0, 1)             # Loss function Softening Scale
        # ) 

        bounds = [
            (0.0, 5.0),      # b_1: positive, O(1)
            (-5.0, 5.0),     # b_2
            (-5.0, 5.0),     # b_s
            (-5.0, 5.0),     # b_3
            (-50.0, 50.0),   # alpha
            (-50.0, 50.0),   # alpha_v: tight enough to prevent blow-up
            (0.0, 0.0),      # alpha_s0
            (0.0, 0.0),      # alpha_s2
            (0.0, 50.0),     # s2FoG: must be non-negative physically
            (0.0, 1.0),      # eps_soft
        ]

        pars_best = minimize(xihh_loss, x0 = x0, bounds = bounds, method = 'L-BFGS-B', options={'ftol': 1e-8, 'gtol': 1e-8}).x  #Nelder-Mead
        
        print(
            f'Best-fit with {rmin:f}: \n', 
            f'  b_1 = {pars_best[0]:.3e} \t',
            f'  b_2 = {pars_best[1]:.3e} \t',
            f'  b_s = {pars_best[2]:.3e} \t',
            f'  b_3 = {pars_best[3]:.3e} \n',
            f'  alpha    = {pars_best[4]:.3e} \t',
            f'  alpha_v  = {pars_best[5]:.3e} \t',
            f'  alpha_s0 = {pars_best[6]:.3e} \t',
            f'  alpha_s2 = {pars_best[7]:.3e} \n', 
            f'  s2FoG    = {pars_best[8]:.3e} \n'
            f'  eps_soft = {pars_best[9]:.3e} \n'
        )

        print(pars_best.tolist())

        xihh_bests[:, idx] = xihh_real(r, pars_best[:-1])
        par_bests[idx] = pars_best[:-1]
   
    fig, axs = plt.subplots(2, 1, figsize=(7,6), sharex = True, gridspec_kw={'height_ratios': [5, 3], 'hspace' : 0 })
    ##############################################
    plt.subplot(211)
    plt.scatter(r, r**2*xihh, color = 'k', linewidth = 3, label ='Simulation')
    plt.plot(r, r**2*(xihh + np.sqrt(sigma2)), color = 'gray', linewidth = 2, alpha = 1, linestyle = '--')
    plt.plot(r, r**2*(xihh - np.sqrt(sigma2)), color = 'gray', linewidth = 2, alpha = 1, linestyle = '--', label = '1$\sigma$-Error, JK')

    ## Fig 4 
    for idx, rmin in enumerate(rmins) :
        plt.plot(r, r**2*xihh_bests[:, idx], linewidth = 2, color = 'C' + str(1+idx), label = r'$r_{\rm min}=$' + f'{rmin:.1f}')
        plt.axvline(rmin, color = 'C' + str(1+idx), linewidth = 2, alpha = 0.5, linestyle = '--')

    plt.legend(loc = 'lower left', fontsize = 10, ncol = 2, framealpha = 0.95)
    plt.grid(True)
    #plt.ylim(0, 230)
    plt.ylim(0, max((r**2*xihh)[r > 10])*1.2)
    plt.yticks(fontsize = 20)
    plt.ylabel(r'$r^2\xi_{\rm hh}(r)$ [$h^{-2}$ Mpc$^2$]', fontsize = 20)
    plt.tight_layout()

    ##############################################
    plt.subplot(212)
    plt.plot(r, (xihh + np.sqrt(sigma2))/xihh, color = 'gray', linewidth = 2, alpha = 1, linestyle = '--')
    plt.plot(r, (xihh - np.sqrt(sigma2))/xihh, color = 'gray', linewidth = 2, alpha = 1, linestyle = '--')

    for idx, rmin in enumerate(rmins) :
        plt.plot(r, xihh_bests[:, idx]/xihh, linewidth = 2, color = 'C' + str(1+idx))
        plt.axvline(rmin, color = 'C' + str(1+idx), linewidth = 2, alpha = 0.5, linestyle = '--')

    plt.ylabel('Model/Sim', fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.ylim(0.942, 1.058)

    plt.xscale('log')
    plt.xlabel(r'$r$ [$h^{-1}$ Mpc]', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.xlim(1e-1, 1.4e2)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return r, xihh_bests, rmins, par_bests

#%%
# save best fits 

save_path = '/spiffball/cosweeney/simulations/MDPL2/data/CorrFuncs/xi_hh_LPT_fits.hdf5'

with h5.File(save_path, 'w') as f:
    for idx, rmin in enumerate(rmins) :
        f.create_dataset(name=f'xi_hh_rmin_{rmin:.1f}', data=xihh_bests[:, idx], dtype=np.float64)
        f.create_dataset(name=f'pars_rmin_{rmin:.1f}', data=par_bests[idx], dtype=np.float64)
        
    f.create_dataset(name=f'r', data=r, dtype=np.float64)

#%%
# predict, plot mean radial velocity from velocileptors

gsm.compute_cumulants(*par_bests[1]) # use rmin = 1 fit
v12 = gsm.veft

#%%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def interactive_xihh(r, xihh, sigma2, gsm):
    """
    Sliders for all 9 GSM parameters, live-updating against measured xihh.
    r, xihh, sigma2: arrays from your HDF5 file
    gsm: your GaussianStreamingModel instance (already initialized)
    """
    def model(pars):
        gsm.compute_cumulants(*pars)
        return gsm.compute_xi_real(r, *pars)

    init = (1, 0, 0, 0, 0, 0, 0, 0, 0) #[0.73, 0.0, 0.0, 0.0, -20.0, 0.0, 0.0, 0.0, 0.0]
    param_names = ['b_1', 'b_2', 'b_s', 'b_3', 'alpha', 'alpha_v', 'alpha_s0', 'alpha_s2', 's2FoG']
    slider_ranges = [
            (0.0, 5.0),      # b_1: positive, O(1)
            (-5.0, 5.0),     # b_2
            (-5.0, 5.0),     # b_s
            (-5.0, 5.0),     # b_3
            (-50.0, 50.0),   # alpha
            (-50.0, 50.0),   # alpha_v: tight enough to prevent blow-up
            (0.0, 0.0),      # alpha_s0
            (0.0, 0.0),      # alpha_s2
            (0.0, 50.0),     # s2FoG: must be non-negative physically
        ]
    # [
    #     (0.1, 3.0),    # b_1
    #     (-2.0, 2.0),   # b_2
    #     (-2.0, 2.0),   # b_s
    #     (-2.0, 2.0),   # b_3
    #     (-50.0, 50.0), # alpha
    #     (-10.0, 10.0), # alpha_v
    #     (0.0, 0.0),    # alpha_s0 (fixed)
    #     (0.0, 0.0),    # alpha_s2 (fixed)
    #     (0.0, 20.0),   # s2FoG
    # ]

    fig = plt.figure(figsize=(10, 8))
    ax_top = fig.add_axes([0.10, 0.52, 0.85, 0.43])
    ax_bot = fig.add_axes([0.10, 0.32, 0.85, 0.18], sharex=ax_top)

    # Draw simulation data (static)
    ax_top.scatter(r, r**2 * xihh, color='k', s=15, label='Simulation', zorder=3)
    ax_top.fill_between(r, r**2*(xihh - np.sqrt(sigma2)), r**2*(xihh + np.sqrt(sigma2)),
                        color='gray', alpha=0.3, label=r'1$\sigma$ JK')
    ax_bot.fill_between(r, 1 - np.sqrt(sigma2)/np.abs(xihh), 1 + np.sqrt(sigma2)/np.abs(xihh),
                        color='gray', alpha=0.3)
    ax_bot.axhline(1.0, color='gray', lw=1)

    xi_model = model(init)
    line_top, = ax_top.plot(r, r**2 * xi_model, color='C1', lw=2, label='Model')
    line_bot, = ax_bot.plot(r, xi_model / xihh, color='C1', lw=2)

    ax_top.set_xscale('log'); ax_top.set_xlim(8e-1, 1.2e2)
    ax_top.set_ylabel(r'$r^2\xi_{\rm hh}(r)$', fontsize=13)
    ax_top.legend(fontsize=11); ax_top.grid(True)
    ax_bot.set_xscale('log'); ax_bot.set_xlabel(r'$r$ [h$^{-1}$ Mpc]', fontsize=13)
    ax_bot.set_ylabel('Model/Sim', fontsize=11); ax_bot.set_ylim(0.94, 1.06); ax_bot.grid(True)

    # Build sliders (skip fixed params)
    sliders = []
    slider_bottom = 0.28
    for i, (name, (lo, hi)) in enumerate(zip(param_names, slider_ranges)):
        ax_s = fig.add_axes([0.15, slider_bottom - i*0.028, 0.70, 0.018])
        if lo == hi:  # fixed parameter
            s = Slider(ax_s, name, -1, 1, valinit=0, color='lightgray')
            s.set_active(False)
        else:
            s = Slider(ax_s, name, lo, hi, valinit=init[i])
        sliders.append(s)

    def update(_):
        pars = [s.val for s in sliders]
        try:
            xi_new = model(pars)
            line_top.set_ydata(r**2 * xi_new)
            line_bot.set_ydata(xi_new / xihh)
            ax_top.set_ylim(0, max((r**2 * xihh)[r > 10]) * 1.2)
        except Exception:
            pass
        fig.canvas.draw_idle()

    for s in sliders:
        s.on_changed(update)

    plt.show()

    

#%%
# if __name__ == "__main__":
#     plot_pklin()
#     xihh_fit()