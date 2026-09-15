import numpy as np
import camb
from pathlib import Path
from scipy.special import eval_legendre
from scipy.interpolate import CubicSpline

import sys
sys.path.insert(0, '/home/cosweeney/code/EmulateLSS')
from train_nn_emu import train_emu 
from emulator import Emulator

import psutil
psutil.Process().nice(10)

LOCAL_ROOT = Path('/spiffball/cosweeney/mocks/AbacusSummit')
COV_DIR = Path('/spiffball/cosweeney/mocks/AbacusSummit/covariances')  # + 'dark'/'bright' + per tracer/z txt
CHAIN_DIR = Path('/spiffball/cosweeney/data/chains/mocks')
output_path = '/spiffball/cosweeney/data/models_free_ns/'

TRACERS = {   # tracer -> prog;  BGS excluded (no z=0.2 emulator node yet)
    'LRG':           'dark',
    'ELG_LOPnotqso': 'dark',
    'QSO':           'dark',
}
ZBINS = {
    'LRG':             [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)],
    'ELG_LOPnotqso':   [(1.1, 1.6)],
    'QSO':             [(0.8, 2.1)],
}
Z_EFF = {
    ('LRG', 0.4, 0.6): 0.510,
    ('LRG', 0.6, 0.8): 0.706,
    ('LRG', 0.8, 1.1): 0.919,
    # TODO: confirm exact effective redshifts -- bin midpoints used as placeholders
    # ('ELG_LOPnotqso', 0.8, 1.1): 0.95, # not used in DR1, no overlap
    ('ELG_LOPnotqso', 1.1, 1.6): 1.317,
    ('QSO', 0.8, 2.1): 1.491,
}

REBIN = 4
SMAX_FIT = 130.
NBINS_COV = 45
ELLS_FIT = (0, 2)
ELL_ALL = (0, 2, 4)
S_MODEL = np.arange(16.0, 169.0 + 4.0, 1.0)
 
N_MU = 40
_nodes, _wts = np.polynomial.legendre.leggauss(N_MU)
MU_REF = 0.5 * (_nodes + 1.0)
W_MU = 0.5 * _wts
 
COSMO_FID = [0.6777, 0.02214, 0.11891, 2.2e-9]
OMBH2_MEAN, OMBH2_SIG = 0.02237, 0.00037
LNAS_FID = np.log(1e10 * COSMO_FID[3])
NS_FIXED = 0.9649   # fixed, matches GetLPTInputs / Planck '19; inert for background-only AP calls
 
# getdist chain saving -- placeholders, confirm against real chains before trusting saved output
BURNIN = 1000
THIN = 15
 
NS_MEAN, NS_SIG = 0.9649, 0.0422   # DESI DR1 prior
 
PARAM_NAMES_7 = ['h', 'ombh2', 'omch2', 'logA', 'b1', 'b2', 'ns']
PARAM_LABELS_7 = ['h', r'\omega_b', r'\omega_{cdm}', r'\ln(10^{10}A_s)', 'b_1', 'b_2', 'n_s']
PARAM_NAMES_9 = PARAM_NAMES_7 + ['Delta0', 'Delta2']
PARAM_LABELS_9 = PARAM_LABELS_7 + [r'\Delta_0', r'\Delta_2']
 
 
def get_background(h, ombh2, omch2, ns):
    pars = camb.set_params(H0=100. * h, ombh2=ombh2, omch2=omch2, ns=ns)
    return camb.get_background(pars)
 
RESULTS_FID = get_background(*COSMO_FID[:3], NS_FIXED)
 
 
def load_emulators(z_eff):
    tag = f'{z_eff:.3f}'.replace('.', 'p')
    return (Emulator(output_path + f'compact_xi0_z{tag}'+'_free_ns'),
            Emulator(output_path + f'compact_xi2_z{tag}'+'_free_ns'),
            Emulator(output_path + f'compact_xi4_z{tag}'+'_free_ns'))
 
_emu_cache = {}
def get_emulators_cached(z_eff):
    """Load emulators once per worker process and reuse across tasks routed to it."""
    if z_eff not in _emu_cache:
        _emu_cache[z_eff] = load_emulators(z_eff)
    return _emu_cache[z_eff]
 
 
def ap_params(h, ombh2, omch2, ns, z_eff):
    results = get_background(h, ombh2, omch2, ns)
    q_perp = results.angular_diameter_distance(z_eff) / RESULTS_FID.angular_diameter_distance(z_eff)
    q_par = RESULTS_FID.hubble_parameter(z_eff) / results.hubble_parameter(z_eff)
    return q_perp, q_par
 
def apply_AP(s_fid, mu_fid, q_perp, q_par):
    q_a = q_par**(1/3) * q_perp**(2/3)
    q_e = (q_par / q_perp)**(1/3)
    s_p = s_fid * q_a * np.sqrt(q_e**4 * mu_fid**2 + (1 - mu_fid**2) / q_e**2)
    mu_p = 1 / np.sqrt(1 + (1 / mu_fid**2 - 1) / q_e**6)
    return s_p, mu_p
 
def ap_multipoles(s_data, ells_out, xi0_model, xi2_model, xi4_model, q_perp, q_par):
    spline0 = CubicSpline(S_MODEL, xi0_model, extrapolate=False)
    spline2 = CubicSpline(S_MODEL, xi2_model, extrapolate=False)
    spline4 = CubicSpline(S_MODEL, xi4_model, extrapolate=False)
    out = np.empty((len(ells_out), len(s_data)))
    for j, s_ref in enumerate(s_data):
        s_p, mu_p = apply_AP(s_ref, MU_REF, q_perp, q_par)
        xi_smu = (spline0(s_p) * eval_legendre(0, mu_p)
                  + spline2(s_p) * eval_legendre(2, mu_p)
                  + spline4(s_p) * eval_legendre(4, mu_p))
        for i, ell in enumerate(ells_out):
            out[i, j] = (2 * ell + 1) * np.sum(W_MU * xi_smu * eval_legendre(ell, mu_p))
    return out
 
 
def rebin_mean(arr, rebin, axis=-1):
    n = arr.shape[axis]
    n_keep = (n // rebin) * rebin
    arr = np.take(arr, np.arange(n_keep), axis=axis)
    new_shape = arr.shape[:-1] + (n_keep // rebin, rebin)
    return arr.reshape(new_shape).mean(axis=-1)
 
 
def cov_path_for(tracer, prog, zmin, zmax):
    return COV_DIR / prog / (f'xi024_{tracer}_GCcomb_{zmin}_{zmax}_default_FKP_lin4_'
                              f's20-200_cov_RascalC_rescaled_mocks.txt')
 
def poles_path_for(tracer, prog, zmin, zmax):
    return LOCAL_ROOT / prog / f'{tracer}_z{zmin}-{zmax}_poles.npz'
 
 
def load_tracer_arrays(tracer, prog, zmin, zmax):
    d = np.load(poles_path_for(tracer, prog, zmin, zmax))
    s_fine, poles_fine = d['s'], d['poles']
    s = rebin_mean(s_fine, REBIN)
    poles = rebin_mean(poles_fine, REBIN, axis=-1)   # (nmock_avail, 3, 45)
 
    cov_full = np.loadtxt(cov_path_for(tracer, prog, zmin, zmax))
    mask = s <= SMAX_FIT
    idx = np.concatenate([ELL_ALL.index(ell) * NBINS_COV + np.where(mask)[0] for ell in ELLS_FIT])
    cov = cov_full[np.ix_(idx, idx)]
    s_data = s[mask]
    return s_data, poles, mask, cov
 
def data_vec_from(poles, mask, mock_idx=None):
    target = poles[mock_idx] if mock_idx is not None else poles.mean(axis=0)
    return np.concatenate([target[ELL_ALL.index(ell)][mask] for ell in ELLS_FIT])
 
 
def model_multipoles(theta, emus, z_eff, s_data):
    h, ombh2, omch2, lnAs, b1, b2, ns = theta[:7]
    emu_xi0, emu_xi2, emu_xi4 = emus
    emu_params = [b1, b2, ombh2, omch2, h, lnAs, ns]
    _, xi0 = emu_xi0(emu_params)
    _, xi2 = emu_xi2(emu_params)
    _, xi4 = emu_xi4(emu_params)
    q_perp, q_par = ap_params(h, ombh2, omch2, ns, z_eff)
    return ap_multipoles(s_data, ELLS_FIT, xi0, xi2, xi4, q_perp, q_par).ravel()
 
 
def log_prior7(theta):
    h, ombh2, omch2, lnAs, b1, b2, ns = theta
    if not (0.55 < h < 0.91):     return -np.inf
    if not (0.08 < omch2 < 0.16): return -np.inf
    if not (2.0 < lnAs < 4.0):    return -np.inf
    if not (0.0 < b1 < 2.0):      return -np.inf
    if not (-5.0 < b2 < 10.0):    return -np.inf
    lp = -0.5 * ((ombh2 - OMBH2_MEAN) / OMBH2_SIG) ** 2
    lp += -0.5 * ((ns - NS_MEAN) / NS_SIG) ** 2
    return lp
 
 
def save_getdist_chain(root, sampler, discard=BURNIN, thin=THIN,
                        param_names=PARAM_NAMES_7, param_labels=PARAM_LABELS_7):
    """Write <root>.txt (weight, -logpost, params...) and <root>.paramnames for getdist."""
    root = Path(root)
    root.parent.mkdir(parents=True, exist_ok=True)
 
    flat = sampler.get_chain(discard=discard, thin=thin, flat=True)
    log_prob = sampler.get_log_prob(discard=discard, thin=thin, flat=True)
 
    weights = np.ones(len(flat))
    minuslogpost = -log_prob
    data = np.column_stack([weights, minuslogpost, flat])
    np.savetxt(f'{root}.txt', data)
 
    with open(f'{root}.paramnames', 'w') as f:
        for name, label in zip(param_names, param_labels):
            f.write(f'{name}\t{label}\n')