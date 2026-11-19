"""
Validation of the 8-input sigma_err emulators (train_sigerr_11par_v3.py): QSO, z = 1.491, v3 no-slope, avg_M.  theta = [b1, b2, omega_b, omega_cdm, h, As, ns, sigma_err].

Three figures (same softened fractional-error metric and flagging as validate_11par.py), written to OUT_DIR:
  validation_sigerr_pct_<z>.png          68th / 95th percentile error vs s for the train, val and test splits, one panel per multipole
  validation_sigerr_by_sigma_err_<z>.png test split only, 95th percentile (solid) and 68th (dashed) error vs s in bins of sigma_err
  validation_sigerr_zero_<z>.png         sigma_err = 0: this emulator vs the existing 7-input v3 no-slope emulator on the test thetas
                                         (a difference between two emulators, not an error against truth); skipped if the 7-input models are not found
and prints, per multipole and sigma_err bin, the median-over-s of the 68th / 95th percentile and the maximum over s of the 95th.

    python validate_sigerr_11par_v3.py
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
        matplotlib.rcParams.update({'text.usetex': True, 'font.family': 'Computer Modern Roman'})
    except Exception:
        USE_TEX = False

plt.rcParams['axes.labelsize'] = 20

# ---- configuration --------------------------------------------------------
Z = 1.491
ELLS = (0, 2, 4)
BASE_COLOR = 'seagreen'                                  # QSO, DESI 2024 II Appendix D
SPLITS = ['train', 'val', 'test']
SPLIT_LS = {'train': ':', 'val': '--', 'test': '-'}
PCTS = [68, 95]
SHADE = {68: 0.35, 95: 1.0}
SIGMA_ERR_BINS = [0, 250, 500, 750, 1000]                # km/s

BASE_DIR = '/spiffball/cosweeney/data/training/compact_multipole_training_set_11par_v3_noslope_sigerr_avg_M/'
MODEL_DIR = '/spiffball/cosweeney/data/models_11par_v3_noslope_sigerr_avg_M/'
SUFFIX = '_11par_v3_noslope_sigerr'
REF_MODEL_DIR = '/spiffball/cosweeney/data/models_11par_v3_noslope_avg_M/'     # existing 7-input emulators (the fits use these)
REF_SUFFIX = '_11par_v3_noslope'
OUT_DIR = '/spiffball/cosweeney/data/plots_11par_v3_noslope_sigerr/'


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
    """Percent-like error that floors out near truth's zero crossings instead of diverging."""
    denom = np.sqrt(truth**2 + (floor * sigma_s)**2)
    return np.abs(pred - truth) / denom


def load_data(z_sub):
    data = {}
    for split in SPLITS:
        tp, xp = os.path.join(z_sub, f'thetas_{split}.npy'), os.path.join(z_sub, f'xi_{split}.npy')
        if os.path.exists(tp):
            data[split] = dict(thetas=np.load(tp), xi=np.load(xp))
    _, rough_ref = flag(data['train']['xi'])
    for split in data:
        bad, _ = flag(data[split]['xi'], rough_ref=rough_ref)
        data[split]['keep'] = ~bad
        print(f"{split}: excluding {bad.sum()}/{len(bad)}")
    return data


def axes_style(ax, ell):
    ax.axhline(1e-2, color='k', ls='--', lw=1.0, alpha=0.6)
    ax.set_yscale('log')
    ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$' if USE_TEX else 's [h^-1 Mpc]')
    ax.set_title(rf'$\xi_{ell}$' if USE_TEX else f'xi_{ell}', fontsize=12)
    ax.margins(x=0)


# ---- main ------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    z_sub = os.path.join(BASE_DIR, zdir(Z))
    if not os.path.isdir(z_sub):
        sys.exit(f"{z_sub} not found")
    s = np.load(os.path.join(z_sub, 's.npy'))
    data = load_data(z_sub)
    sigma_s = {ell: np.std(data['train']['xi'][data['train']['keep']][:, i], axis=0) for i, ell in enumerate(ELLS)}

    emus = {}
    for ell in ELLS:
        path = os.path.join(MODEL_DIR, f'compact_xi{ell}_{zdir(Z)}{SUFFIX}')
        if not os.path.exists(path + '.json'):
            sys.exit(f"model not found: {path}")
        emus[ell] = NPEmulator(path)

    def predict(ell, th):
        return emus[ell](to_params(th))[1]

    # 1. percentile curves, per split
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    for ax, ell in zip(axes, ELLS):
        for split in data:
            th = data[split]['thetas'][data[split]['keep']]
            xi = data[split]['xi'][data[split]['keep']][:, ELLS.index(ell)]
            frac = soft_frac_pct(predict(ell, th), xi, sigma_s[ell], 0.3)
            for p in PCTS:
                ax.plot(s, np.percentile(frac, p, axis=0), color=BASE_COLOR, ls=SPLIT_LS[split], alpha=SHADE[p], lw=1.8)
        axes_style(ax, ell)
    axes[0].set_ylabel('softened fractional error')
    axes[0].legend(handles=[Line2D([0], [0], color=BASE_COLOR, ls=SPLIT_LS[sp], alpha=SHADE[p], lw=1.8, label=f'{sp} ({p}th)')
                            for sp in data for p in PCTS], frameon=False, fontsize=7.5, ncol=2, loc='upper left')
    fig.suptitle(f'QSO  (z = {Z}), sigma_err emulators', y=1.03)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'validation_sigerr_pct_{zdir(Z)}.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {out}")

    # 2. test split in bins of sigma_err
    th_t = data['test']['thetas'][data['test']['keep']]
    xi_t = data['test']['xi'][data['test']['keep']]
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(SIGMA_ERR_BINS) - 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    print(f"\n{'ell':>3s} {'sigma_err bin':>14s} {'n':>6s} {'med_s 68th':>11s} {'med_s 95th':>11s} {'max_s 95th':>11s}")
    for ax, ell in zip(axes, ELLS):
        frac = soft_frac_pct(predict(ell, th_t), xi_t[:, ELLS.index(ell)], sigma_s[ell], 0.3)
        for (lo, hi), c in zip(zip(SIGMA_ERR_BINS[:-1], SIGMA_ERR_BINS[1:]), colors):
            m = (th_t[:, 7] >= lo) & (th_t[:, 7] < hi)
            if m.sum() == 0:
                continue
            p68, p95 = np.percentile(frac[m], 68, axis=0), np.percentile(frac[m], 95, axis=0)
            ax.plot(s, p95, color=c, lw=1.8, label=f'{lo}-{hi} km/s')
            ax.plot(s, p68, color=c, lw=1.2, ls='--')
            print(f"{ell:>3d} {f'{lo}-{hi}':>14s} {m.sum():>6d} {np.median(p68):>11.2e} {np.median(p95):>11.2e} {p95.max():>11.2e}")
        axes_style(ax, ell)
    axes[0].set_ylabel('softened fractional error (test; solid 95th, dashed 68th)')
    axes[0].legend(frameon=False, fontsize=8, loc='upper left', title=r'$\sigma_{\rm err}$' if USE_TEX else 'sigma_err')
    fig.suptitle(f'QSO  (z = {Z}), error vs sigma_err', y=1.03)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'validation_sigerr_by_sigma_err_{zdir(Z)}.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {out}")

    # 3. sigma_err = 0 against the existing 7-input emulators
    ref = {}
    for ell in ELLS:
        path = os.path.join(REF_MODEL_DIR, f'compact_xi{ell}_{zdir(Z)}{REF_SUFFIX}')
        if os.path.exists(path + '.json'):
            ref[ell] = NPEmulator(path)
    if len(ref) < len(ELLS):
        print(f"\nSKIP sigma_err = 0 comparison: 7-input models not found in {REF_MODEL_DIR}")
        return
    th0 = th_t.copy()
    th0[:, 7] = 0.0
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    print(f"\nsigma_err = 0, this emulator vs the 7-input one (test thetas): median over s of the 68th / 95th percentile of the softened difference")
    for ax, ell in zip(axes, ELLS):
        new = predict(ell, th0)
        old = ref[ell](to_params(th0[:, :7]))[1]
        diff = soft_frac_pct(new, old, sigma_s[ell], 0.3)
        for p in PCTS:
            ax.plot(s, np.percentile(diff, p, axis=0), color=BASE_COLOR, alpha=SHADE[p], lw=1.8, label=f'{p}th')
        print(f"  ell={ell}: {np.median(np.percentile(diff, 68, axis=0)):.2e} / {np.median(np.percentile(diff, 95, axis=0)):.2e}")
        axes_style(ax, ell)
    axes[0].set_ylabel('softened fractional difference')
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    fig.suptitle(f'QSO  (z = {Z}), sigma_err = 0: 8-input vs 7-input emulator', y=1.03)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'validation_sigerr_zero_{zdir(Z)}.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {out}")


if __name__ == '__main__':
    main()