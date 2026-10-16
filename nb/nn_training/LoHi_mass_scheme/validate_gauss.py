"""
Percentile-error plots for xi_0, xi_2, xi_4, at each trained redshift,
for the schemes in SCHEMES (avg_M only; add others to overlay them) using DESI per-tracer
colors and scheme-specific linestyles. Validates the Gaussian-model emulators
(generate_multiz_gauss.py / train_gauss.py).

Uses a softened percent-error metric that floors out near truth's zero
crossings instead of diverging, and the DESI per-tracer color scheme
(DESI 2024 II, arXiv:2411.12020, Appendix D).
"""

import os
import sys
sys.path.insert(0, '/home/cosweeney/code/EmulateLSS')
from emulator import Emulator as NPEmulator   # numpy inference class

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

USE_TEX = os.environ.get('USE_TEX', '1') == '1'
if USE_TEX:
    try:
        matplotlib.rcParams.update({'text.usetex': True,
                                    'font.family': 'Computer Modern Roman'})
    except Exception:
        USE_TEX = False

plt.rcParams['axes.labelsize'] = 20

# ---- configuration --------------------------------------------------------

REDSHIFTS = [0.295, 0.510, 0.706, 0.919, 1.317, 1.491]
ELLS = (0, 2, 4)

# DESI 2024 II (arXiv:2411.12020), Appendix D -- official per-tracer colors.
TRACER_COLOR = {
    0.295: ('BGS', 'yellowgreen'),
    0.510: ('LRG1', 'orange'),
    0.706: ('LRG2', 'orangered'),
    0.919: ('LRG3', 'firebrick'),
    1.317: ('ELG2', 'steelblue'),
    1.491: ('QSO',  'seagreen'),
}

# Scheme display: linestyle and label
SCHEME_STYLE = {
    'low_M':  {'ls': '--',  'label': r'low\_M ($12.5 < \log_{10}M < 12.6$)'},
    'high_M': {'ls': ':',   'label': r'high\_M ($13.5 < \log_{10}M < 13.6$)'},
    'avg_M':  {'ls': '-',   'label': r'avg\_M'},
}

SCHEMES = ['avg_M']
PCTS = [68, 95]
SHADE = {68: 0.35, 95: 1.0}   # alpha per percentile

# Data and model paths -- same suffixed convention for all three schemes.
BASE_DIR_TEMPLATE = '/spiffball/cosweeney/data/training/compact_multipole_training_set_gauss_{scheme}/'
MODEL_DIR_TEMPLATE = '/spiffball/cosweeney/data/models_gauss_{scheme}/'

OUT_DIR = '/spiffball/cosweeney/data/plots_gauss_mass_scheme/'
os.makedirs(OUT_DIR, exist_ok=True)

# Splits to overlay: linestyle not used here (we use scheme linestyle instead)
SPLITS = ['train', 'val', 'test']


# ---- helpers ---------------------------------------------------------------

def zdir(z):
    return f"z{str(z).replace('.', 'p')}"


def to_params(thetas):
    p = thetas.copy()
    p[:, 5] = np.log(1e10 * thetas[:, 5])
    return p


def flag(xi, rough_ref=None):
    xi0 = xi[:, 0]
    finite = np.isfinite(xi).all(axis=(1, 2))
    collapsed = (np.median(xi0, axis=1) < -0.5) | (np.abs(xi[:, 1]).max(axis=1) < 1e-8)
    rough = np.abs(np.diff(xi0, n=2, axis=1)).mean(axis=1) / np.abs(xi0).mean(axis=1)
    if rough_ref is None:
        rough_ref = np.median(rough[finite & ~collapsed])
    noisy = rough > 5 * rough_ref
    amp = np.abs(xi).max(axis=(1, 2))
    bad = ~finite | (amp > 10) | collapsed | noisy
    return bad, rough_ref


def soft_frac_pct(pred, truth, sigma_s, floor=0.3):
    """Percent-like error that floors out near truth's zero crossings instead
    of diverging."""
    denom = np.sqrt(truth**2 + (floor * sigma_s)**2)
    return np.abs(pred - truth) / denom


# ---- main ------------------------------------------------------------------

def main():
    for z in REDSHIFTS:
        tracer_name, base_color = TRACER_COLOR[z]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)

        legend_handles = []

        for scheme in SCHEMES:
            base_dir = BASE_DIR_TEMPLATE.format(scheme=scheme)
            model_dir = MODEL_DIR_TEMPLATE.format(scheme=scheme)
            z_sub = os.path.join(base_dir, zdir(z))

            if not os.path.isdir(z_sub):
                print(f"SKIP {scheme} z={z}: {z_sub} not found")
                continue

            s = np.load(os.path.join(z_sub, 's.npy'))
            ls = SCHEME_STYLE[scheme]['ls']

            # Load all splits
            data = {}
            for split in SPLITS:
                thetas_path = os.path.join(z_sub, f'thetas_{split}.npy')
                xi_path = os.path.join(z_sub, f'xi_{split}.npy')
                if not os.path.exists(thetas_path):
                    continue
                data[split] = dict(
                    thetas=np.load(thetas_path),
                    xi=np.load(xi_path),
                )

            if not data:
                print(f"SKIP {scheme} z={z}: no split data found")
                continue

            # Flag bad samples using train set as reference
            _, rough_ref = flag(data['train']['xi'])
            for split in data:
                bad, _ = flag(data[split]['xi'], rough_ref=rough_ref)
                data[split]['keep'] = ~bad
                print(f"{scheme} z={z} {split}: excluding {bad.sum()}/{len(bad)}")

            # Compute sigma_s from train set (shared across splits)
            sigma_s_train = {}
            for i, ell in enumerate(ELLS):
                sigma_s_train[ell] = np.std(
                    data['train']['xi'][data['train']['keep']][:, i], axis=0
                )

            # Predict + compute errors per (ell, split)
            for ax, ell in zip(axes, ELLS):
                model_path = os.path.join(model_dir,
                                          f'compact_xi{ell}_{zdir(z)}_gauss')
                if not os.path.exists(model_path + '.json'):
                    print(f"SKIP {scheme} z={z} ell={ell}: model not found")
                    continue
                emu = NPEmulator(model_path)

                # Use test split for the main validation curve
                for split in data:
                    th = data[split]['thetas'][data[split]['keep']]
                    xi = data[split]['xi'][data[split]['keep']][:, ELLS.index(ell)]
                    params = to_params(th)
                    _, pred = emu(params)
                    frac = soft_frac_pct(pred, xi, sigma_s_train[ell], 0.3)

                    for p in PCTS:
                        c = np.percentile(frac, p, axis=0)
                        ax.plot(s, c, color=base_color, ls=ls,
                                alpha=SHADE[p], lw=1.8)

                ax.axhline(1e-2, color='k', ls='--', lw=1.0, alpha=0.6)
                ax.set_yscale('log')
                ell_label = rf'$\xi_{ell}$' if USE_TEX else f'xi_{ell}'
                ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$' if USE_TEX
                              else 's [h^-1 Mpc]')
                ax.set_title(ell_label, fontsize=12)
                ax.margins(x=0)

            # One legend entry per (scheme × percentile)
            for p in PCTS:
                legend_handles.append(
                    Line2D([0], [0], color=base_color, ls=ls, alpha=SHADE[p],
                           lw=1.8, label=f'{scheme} ({p}th)')
                )

        axes[0].set_ylabel('softened fractional error')
        axes[0].legend(handles=legend_handles, frameon=False, fontsize=7.5,
                       ncol=2, loc='upper left')
        fig.suptitle(f'{tracer_name}  (z = {z})', y=1.03)
        fig.tight_layout()

        out_path = os.path.join(OUT_DIR, f'validation_pct_z{zdir(z)}.png')
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"saved {out_path}")


if __name__ == '__main__':
    main()