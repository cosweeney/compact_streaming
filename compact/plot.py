""" 
For plotting utilities.
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_multipoles_n_residuals(r, truths, preds, errs=None, label='Predicted', eps=1e-8, plot_hex=True, color='C2'):

    n_plots = 2

    if plot_hex:
        n_plots += 1
   
    fig, axes = plt.subplots(2, n_plots, figsize=(n_plots*5, 5), layout='constrained', sharey='row', sharex='col', gridspec_kw={'height_ratios': [3, 1]})

    mins = np.array([np.min(r**2*truth) for truth in truths])
    maxs = np.array([np.max(r**2*truth) for truth in truths])

    for i, ax in enumerate(axes[0]):
        if errs is not None:
            ax.errorbar(r, r**2*truths[i], r**2*errs[i], fmt='o', label='Measured', zorder=0, markersize=5, color='k')
        else:
            ax.scatter(r, r**2*truths[i], label='Measured', s=10, color='k')

        ax.axhline(0, color='k', ls=':')

        ax.plot(r, r**2*preds[i], label=label, c=color)

        ax.set_ylabel(rf'$s^2\xi_{2*i}(s)$')
        ax.margins(x=0)
        

    for i, ax in enumerate(axes[1]):
        if errs is not None:
            ax.plot(r, (preds[i] - truths[i])/errs[i], c=color)
            ax.fill_between(r, -1, 1, color='g', alpha=0.3)
            ax.fill_between(r, -2, 2, color='orange', alpha=0.15)
        else:
            ax.plot(r, (preds[i] - truths[i])/(np.abs(truths[i])+eps), c=color)
            ax.fill_between(r, -0.05, 0.05, color='g', alpha=0.3)
            ax.fill_between(r, -0.1, 0.1, color='orange', alpha=0.15)
        ax.axhline(0, color='k', ls=':')


        ax.set_xlabel(r'$s [h^{-1}$ Mpc]')
        ax.set_ylim((-0.25, 0.25))
        ax.margins(x=0)

    axes[0, 0].legend()
    #axes[0, 0].set_ylim((mins.min()*(1-0.2), maxs.max()*(1+0.2)))
       
    if errs is not None:
        axes[1, 0].set_ylabel(r'$\Delta \xi_i / \sigma_{\rm JK}$')
    else:   
        axes[1, 0].set_ylabel(r'Residual')


    return fig, axes


def plot_lpt_fits(r, xihh, v12_sim, sigma2, sigma_v2, rmins, pars_best, GLI):

    xihh_bests = np.array([GLI.xi_real(r, p[:-2]) for p in pars_best]).T
    v12_bests = np.array([GLI.pwv_mean(r, p[:-2]) for p in pars_best]).T

    fig, axs = plt.subplots(2, 2, figsize=(14, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1],})

    plt.subplot(221)
    plt.scatter(r, r**2*xihh, color = 'k', linewidth = 3, label ='Simulation')
    plt.plot(r, r**2*(xihh + np.sqrt(sigma2)), color = 'gray', linewidth = 2, alpha = 1, linestyle = '--')
    plt.plot(r, r**2*(xihh - np.sqrt(sigma2)), color = 'gray', linewidth = 2, alpha = 1, linestyle = '--', label = '1$\sigma$-Error, JK')

    for idx, rmin in enumerate(rmins) :
        
        style = '-'
        if idx > 0:
            style = '--'

        plt.plot(r, r**2*xihh_bests[:, idx], linewidth = 2, color = 'C' + str(1+idx), label = r'$r_{\rm min}=$' + f'{rmin:.1f}')
        plt.axvline(rmin, color = 'C' + str(1+idx), linewidth = 2, alpha = 0.5, linestyle = '--')

    plt.legend(loc = 'upper left', fontsize = 10, ncol = 2, framealpha = 0.95)
    plt.grid(True)
    #plt.ylim(0, 230)
    plt.ylim(0, max((r**2*xihh)[r > 10])*1.2)
    plt.yticks(fontsize = 20)
    plt.ylabel(r'$r^2\xi_{\rm hh}(r)$ [$h^{-2}$ Mpc$^2$]', fontsize = 20)
    plt.tight_layout()

    ##############################################
    plt.subplot(223)
    plt.plot(r, (xihh + np.sqrt(sigma2))/xihh, color = 'gray', linewidth = 2, alpha = 1, linestyle = '--')
    plt.plot(r, (xihh - np.sqrt(sigma2))/xihh, color = 'gray', linewidth = 2, alpha = 1, linestyle = '--')

    for idx, rmin in enumerate(rmins) :
        plt.plot(r, xihh_bests[:, idx]/xihh, linewidth = 2, color = 'C' + str(1+idx))
        plt.axvline(rmin, color = 'C' + str(1+idx), linewidth = 2, alpha = 0.5, linestyle = '--')

    plt.ylabel('Model/Sim', fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.ylim(0.9, 1.1)

    plt.xscale('log')
    plt.xlabel(r'$r$ [$h^{-1}$ Mpc]', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.xlim(8e-1, 1.4e2)

    plt.grid(True)
    plt.tight_layout()

    plt.subplot(222)
    plt.scatter(r, v12_sim, color = 'k', linewidth = 3, label ='Simulation')
    plt.plot(r, (v12_sim + np.sqrt(sigma_v2)), color = 'gray', linewidth = 2, alpha = 1, linestyle = '--')
    plt.plot(r, (v12_sim - np.sqrt(sigma_v2)), color = 'gray', linewidth = 2, alpha = 1, linestyle = '--', label = '1$\sigma$-Error, JK')

    
    ## Fig 4 
    for idx, rmin in enumerate(rmins) :
        plt.plot(r, v12_bests[:, idx], linewidth = 2, color = 'C' + str(1+idx), label = r'$r_{\rm min}=$' + f'{rmin:.1f}')
        plt.axvline(rmin, color = 'C' + str(1+idx), linewidth = 2, alpha = 0.5, linestyle = '--')

    plt.legend(loc = 'upper left', fontsize = 10, ncol = 2, framealpha = 0.95)
    plt.grid(True)
    #plt.ylim(0, 230)
    plt.ylim(-500, 100)
    plt.yticks(fontsize = 20)
    plt.ylabel(r'$\langle v_r|r\rangle$ [km/s]', fontsize = 20)
    plt.tight_layout()

    ##############################################
    plt.subplot(224)
    plt.plot(r, (v12_sim + np.sqrt(sigma_v2))/v12_sim, color = 'gray', linewidth = 2, alpha = 1, linestyle = '--')
    plt.plot(r, (v12_sim - np.sqrt(sigma_v2))/v12_sim, color = 'gray', linewidth = 2, alpha = 1, linestyle = '--', label = '1$\sigma$-Error, JK')


    for idx, rmin in enumerate(rmins) :
        plt.plot(r, v12_bests[:, idx]/v12_sim, linewidth = 2, color = 'C' + str(1+idx))
        plt.axvline(rmin, color = 'C' + str(1+idx), linewidth = 2, alpha = 0.5, linestyle = '--')

    plt.ylabel('Model/Sim', fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.ylim(0.85, 1.15)

    plt.xscale('log')
    plt.xlabel(r'$r$ [$h^{-1}$ Mpc]', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.xlim(8e-1, 1.4e2)

    plt.grid(True)
    plt.tight_layout()

    return fig, axs