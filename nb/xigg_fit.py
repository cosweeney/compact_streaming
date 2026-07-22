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

#%%
def get_mdpl2_pklin(z, kmax=20.0, nk=1024):
    h = 0.6777
    p = camb.CAMBparams()
    p.set_cosmology(H0=h*100, ombh2=0.048206*h**2, omch2=(0.307115-0.048206)*h**2, mnu=0)
    p.InitPower.set_params(ns=0.96, As=2e-9)
    p.set_matter_power(redshifts=[0.0, z], kmax=kmax*h) # kmax converted to Mpc^-1
    
    # Calibrate As to exactly match sigma8 = 0.8228
    p.InitPower.set_params(ns=0.96, As=2e-9 * (0.8228 / camb.get_results(p).get_sigma8_0())**2)
    
    # Get arrays (outputs are in Mpc^-1 and Mpc^3)
    k, _, pk = camb.get_results(p).get_matter_power_spectrum(minkh=1e-4*h, maxkh=kmax*h, npoints=nk)
    
    # Convert arrays back to h/Mpc and (Mpc/h)^3 to match MDPL2 units
    return k / h, pk[0 if z == 0.0 else 1, :] * (h**3)

# Quick test
klin, Plin = get_mdpl2_pklin(z=1/0.8376-1)
gsm = GaussianStreamingModel(klin, Plin)
gsm.convert_sigma_bases()

#%%

# TODO: can't seem to get this to fit very well; consult heng-wei?

def plot_pklin() : 
    plt.plot(klin, klin*Plin)
    plt.xscale('log')
    plt.xlabel(r'$k$ [h/Mpc]')
    plt.ylabel(r'$kP(k)$ [h$^2$/Mpc$^2$]')
    plt.grid(True)
    plt.show()
    #plt.savefig('plin.png')

def xigg_real(rad, pars) : 
    gsm.compute_cumulants(*pars)
    xir = gsm.compute_xi_real(rad,*pars)
    return xir

def xigg_fit():
    with h5.File("/spiff/cosweeney/simulations/MDPL2/data/CorrFuncs/xi_gg_150.hdf5", "r") as hdf :
        xigg  = np.array(hdf['xi'][()])
        r      = np.array(hdf['r_cens'][()])
        sigma2 = np.diag(hdf['xi_cov'][()])
    
    rmins = np.array([0.1, 1, 5, 16, 24])
    xigg_bests = np.zeros((len(r), len(rmins)))

    for idx, rmin in enumerate(rmins) :
        
        mask = ((rmin < r) & (r < 120))
        def xigg_loss(b):
            pars = b[:-1]
            xigg_in = xigg_real(r, pars)
            sigma2_soft = sigma2/np.sqrt(999) + b[-1]**2*xigg**2 
            return sum((xigg_in[mask]-xigg[mask])**2/(sigma2_soft[mask])) + sum(np.log(sigma2_soft[mask]))

        # velocipletors have 9 free parameters for gg auto corrfunc : 
        # 4 biases [b_1, b_2, b_s, b_3] (delta, delta^2, s_ij^2, delta^3)
        # 4 counter-terms : [alpha, alpha_v, alpha_s0, alpha_s2] (k^2, velocity ct, monopole and quadrupole pair-wise velocity ct)
        # 1 stochastic term [s2FoG]
        # But alpha_s0 and alpha_s2 are degenerate with others -> set to zero
        #x0 = (1.5, 0, 0, 0, 0, 0, 0, 0, 0, 1e-3)
        x0 = [0.9782324894058021, 3.428871129657945, -1.096201519273676, -10.55425430671276, -4.222233258768265, -8.569917137030155, 0.0, 0.0, -9.696646503069779, 0.10531072529680936]


        bounds = (
            (-np.inf, np.inf), # b_1
            (-np.inf, np.inf), # b_2
            (-np.inf, np.inf), # b_s
            (-np.inf, np.inf), # b_3
            (-np.inf, np.inf), # alpha for counter-term
            (-np.inf, np.inf), # alpha_v for velocity counter-term
            (0, 0),            # alpha_s0, set 0   
            (0, 0),            # alpha_s2, set 0
            (-np.inf, np.inf), # stochastic term
            (0, 1)             # Loss function Softening Scale
        ) 
        pars_best = minimize(xigg_loss, x0 = x0, bounds = bounds, method = 'Nelder-Mead', options={'xatol': 1e-18, 'fatol': 1e-18}).x  #Nelder-Mead
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

        target_rms = 0.05  # fractional RMS residual threshold, e.g. 1%
        max_restarts = 10

        x_current = np.array(x0)
        best_result = None

        for restart in range(max_restarts):
            result = minimize(
                xigg_loss, x0=x_current, bounds=bounds,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 50000}
            )

            if best_result is None or result.fun < best_result.fun:
                best_result = result

            # Check accuracy criterion on the masked region
            xigg_model = xigg_real(r, best_result.x[:-1])
            frac_resid = (xigg_model[mask] - xigg[mask]) / xigg[mask]
            rms = np.sqrt(np.mean(frac_resid**2))

            print(f'  restart {restart+1}: loss={best_result.fun:.4e}, frac RMS={rms:.4e}')

            if rms < target_rms:
                print(f'  Converged at restart {restart+1}')
                break

            # Perturb the best point slightly to escape local minima
            x_current = best_result.x * (1 + 0.05 * np.random.randn(len(best_result.x)))
            x_current[6] = 0.0  # keep alpha_s0 = 0
            x_current[7] = 0.0  # keep alpha_s2 = 0

        pars_best = best_result.x
        print(pars_best.tolist())


        xigg_bests[:, idx] = xigg_real(r, pars_best[:-1])
   
    fig, axs = plt.subplots(2, 1, figsize=(7,6), sharex = True, gridspec_kw={'height_ratios': [5, 3], 'hspace' : 0 })
    ##############################################
    plt.subplot(211)
    plt.scatter(r, r**2*xigg, color = 'k', linewidth = 3, label ='Simulation')
    plt.plot(r, r**2*(xigg + np.sqrt(sigma2)), color = 'gray', linewidth = 2, alpha = 1, linestyle = '--')
    plt.plot(r, r**2*(xigg - np.sqrt(sigma2)), color = 'gray', linewidth = 2, alpha = 1, linestyle = '--', label = '1$\sigma$-Error, JK')

    ## Fig 4 
    for idx, rmin in enumerate(rmins) :
        plt.plot(r, r**2*xigg_bests[:, idx], linewidth = 2, color = 'C' + str(1+idx), label = r'$r_{\rm min}=$' + f'{rmin:.1f}')

    plt.legend(loc = 'lower left', fontsize = 15, ncol = 2, framealpha = 0.95)
    plt.grid(True)
    #plt.ylim(0, 230)
    plt.ylim(0, max((r**2*xigg)[r > 10])*1.2)
    plt.yticks(fontsize = 20)
    plt.ylabel(r'$r^2\xi_{\rm gg}(r)$ [h$^{-2}$ Mpc$^2$]', fontsize = 20)
    plt.tight_layout()

    ##############################################
    plt.subplot(212)
    plt.plot(r, (xigg + np.sqrt(sigma2))/xigg, color = 'gray', linewidth = 2, alpha = 1, linestyle = '--')
    plt.plot(r, (xigg - np.sqrt(sigma2))/xigg, color = 'gray', linewidth = 2, alpha = 1, linestyle = '--')

    for idx, rmin in enumerate(rmins) :
        plt.plot(r, xigg_bests[:, idx]/xigg, linewidth = 2, color = 'C' + str(1+idx))

    plt.ylabel('Model/Sim', fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.ylim(0.942, 1.058)

    plt.xscale('log')
    plt.xlabel('r [h$^{-1}$ Mpc]', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.xlim(8e-1, 1.2e2)

    plt.grid(True)
    plt.tight_layout()
    plt.show()
    #plt.savefig('plots/Fig_xigg_MDPL2.png')

    #return best

#%%
# if __name__ == "__main__":
#     plot_pklin()
#     xigg_fit()