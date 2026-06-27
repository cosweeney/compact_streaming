""" 
For plotting utilities.
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_multipoles_n_residuals(r, truths, preds, errs=None, label='Predicted'):

   
    fig, axes = plt.subplots(2, 3, figsize=(3*5, 5), layout='constrained', sharey='row', sharex='col', gridspec_kw={'height_ratios': [3, 1]})

    mins = np.array([np.min(r**2*truth) for truth in truths])
    maxs = np.array([np.max(r**2*truth) for truth in truths])

    for i, ax in enumerate(axes[0]):
        if errs is not None:
            ax.errorbar(r, r**2*truths[i], r**2*errs[i], fmt='o', label='Measured', zorder=0, markersize=5, color='k')
        else:
            ax.scatter(r, r**2*truths[i], label='Measured', s=10, color='k')

        ax.plot(r, r**2*preds[i], label=label)

        ax.set_ylabel(rf'$s^2\xi_{2*i}(s)$')
        ax.margins(x=0)
        

    for i, ax in enumerate(axes[1]):
        if errs is not None:
            ax.plot(r, (truths[i] - preds[i])/errs[i])
        else:
            ax.plot(r, (truths[i] - preds[i])/preds[i])
        ax.axhline(0, color='k', ls=':')
        ax.fill_between(r, -0.05, 0.05, color='k', alpha=0.3)
        ax.fill_between(r, -0.1, 0.1, color='k', alpha=0.15)

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
