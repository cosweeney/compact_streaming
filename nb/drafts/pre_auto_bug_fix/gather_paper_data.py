"""
Gathers everything the Figs 0-7 plotting notebook needs into one small cache
file: paper_figs_cache.h5.

RUN THIS ON YOUR CLUSTER, inside an environment where `compact` is importable
(e.g. `pip install -e .` from the repo root, or run from nb/calibration/ with
the repo on your PYTHONPATH) and where /spiffball, /spiff are mounted.
I (Claude) cannot run this myself -- no access to your cluster or its data --
so treat this as a first draft to check over before running, not a
verified-working script.

WHAT THIS DOES NOT DO: recompute anything already sitting in
pairhist_allbins.h5 / pdf_fits.h5 / pairhist_2bins.h5 / compact_pars_cosmo_z0p500.h5.
Those are read as-is. The only genuinely new measurement is xi_hh(r) and
<v_r|r> per mass bin (Fig 1) -- see compute_xihh_avg()/run_fig1() below.
Everything else here is either a direct read or a cheap re-fit (scipy
curve_fit) of already-reduced arrays.

COMPUTE COST: Fig 1 now averages both xi_hh and <v_r|r> over all 25 c000
phases (not just ph000), so this reads 25 halo catalogs x 6 mass bins worth
of pycorr calls, plus 25 PairHist files -- a real (25x) increase over the
single-phase version, though still modest in absolute terms (pycorr/Corrfunc
is fast; the PairHist reads are just HDF5, not raw pair catalogs).

STORAGE: still tiny. The one new file this script writes on top of the cache
is the xi_hh measurement itself, folded into the same cache -- an r-array, a
xi(r) array, and a covariance matrix (from phase-to-phase scatter, not a
single-box jackknife) per mass bin. For 6 mass bins x ~70 r-bins that's still
tens of KB, not GB. Nothing else new is written to spiffball.
"""

import os
import sys
from pathlib import Path

import numpy as np
import h5py as h5

# --- path setup, anchored to this script's own location, before any
# `compact...` import that depends on it -------------------------------
# assumes this file lives at <repo_root>/nb/<some_folder>/gather_paper_data.py,
# i.e. two levels under the repo root, as a sibling of nb/calibration/.
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))                        # for `import compact...`
sys.path.insert(0, str(REPO_ROOT / 'nb' / 'calibration'))  # for abacus_cosmo_core

from compact.velocities.pairhist import PairHist
from pycorr import TwoPointCorrelationFunction
from concurrent.futures import ProcessPoolExecutor, as_completed

# Workaround for a bug in compact/velocities/pairhist_direct.py: it does a
# bare `from pairhist import PairHist` (line 28) rather than a
# package-relative import, which only resolves if compact/velocities/ itself
# is on sys.path. I'm not editing that file in your repo -- instead,
# register the module under the bare name it's looking for before anything
# (here: abacus_cosmo_core, which imports pairhist_direct) triggers that import.
import compact.velocities.pairhist as _pairhist_mod
sys.modules.setdefault('pairhist', _pairhist_mod)

import abacus_cosmo_core as cosmo_grid  # for read_params() on the c130-c181 grid

import psutil
psutil.Process().nice(19)   # same convention as abacus_cosmo_core.py / run_abacus_batch.py --
                            # this script's jackknife loop makes a lot of multi-threaded
                            # tpcf() calls, worth keeping off other users' toes on a shared node


# ============================================================ configuration
# -- fiducial (base) suite: Figs 0, 1, 2, 3, 4, 5, 7 --
FID_SIMS = [f'c000_ph{i:03d}' for i in range(0, 16)]   # ph000-ph015 (16 phases) actually available
FID_REDSHIFT = 'z0.200'
FID_HALO_TMPL = ('/spiff/hengweichang/AbacusSummit_base/snapshots/'
                  'AbacusSummit_base_{sim}/halos/{z}/halo_combined.h5')
FID_PAIRHIST_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base/halos/'
                      'AbacusSummit_base_{sim}/halos/{z}/data/pairhist_allbins.h5')
FID_PDF_FITS_PATH = '/spiffball/cosweeney/simulations/AbacusSummit_base/data/pdf_fits.h5'

BOX_LENGTH = 2000.0            # Mpc/h
M_PART = 2.1e9                 # h^-1 Msun -- AbacusSummit base particle mass (arXiv:2110.11413)
LOG_MASS_EDGES = np.array([[12, 12.01], [12.5, 12.6], [13, 13.1],
                            [13.5, 13.6], [14.0, 14.2], [14.4, 14.7]])
MASSBINS = 10 ** LOG_MASS_EDGES
M_CENS = np.sqrt(MASSBINS[:, 0] * MASSBINS[:, 1])
N_MASS = len(MASSBINS)

# single mass bin used for Figs 2, 3, 4, 7 -- ASSUMPTION, matches the
# mass bin (k=3, [13.5, 13.6]) used as the "test" bin throughout the existing
# calibration notebooks. Change here if you want a different one.
K_SINGLE = 3

# Fig 0 uses its own bin, independent of K_SINGLE above -- lowest mass bin.
K_FIG0 = 0

# low / high bins for Fig 5 -- ASSUMPTION: the two extremes of the 6-bin set.
K_LOW, K_HIGH = 0, 5

r_perp = np.linspace(5, 140, 135 + 1)
r_los = np.linspace(5, 140, 135 + 1)
r_p = 0.5 * (r_perp[1:] + r_perp[:-1])
r_l = 0.5 * (r_los[1:] + r_los[:-1])

# -- cosmology grid: Fig 6 --
GRID_SIMS = list(range(130, 182))
GRID_REDSHIFT = 0.5
GRID_PARS_PATH = ('/spiffball/cosweeney/simulations/AbacusSummit_base/halos/'
                   'cosmo_grid/data/compact_pars_cosmo_z0p500.h5')

OUT_PATH = './paper_figs_cache.h5'


# =================================================== shared fit machinery
# Lifted as-is from nb/calibration/ASbase_compact_calibration_multiple.ipynb
# (cell 16) -- duplicated here rather than imported since it currently only
# lives inline in that notebook, not in the `compact` package itself.

from scipy.optimize import curve_fit

SIG_FLOOR = 0.1
DEL_FLOOR = -1000.0


def sigma_mod(x, sigma0, A_rho, sigma_rho):
    arg = 2.0 * sigma0 ** 2 * (1.0 - A_rho * np.exp(-x / sigma_rho))
    return np.sqrt(np.clip(arg, 1e-8, None))


def A_rho_mod(R, A_piv, b, R_piv):
    return A_piv * np.exp(-(R - R_piv) / b)


def sigma_mod_2d(coords, sigma0, A_piv, b, sigma_rho, R_piv):
    x, R = coords
    return sigma_mod(x, sigma0, A_rho_mod(R, A_piv, b, R_piv), sigma_rho)


def delta_mod(x, A_delta, sigma_delta):
    return A_delta * x / (sigma_delta + x)


def _seed(x, R, sig, R_piv):
    sigma0 = sig.max() / np.sqrt(2.0) * 1.02
    y = 2.0 * sigma0 ** 2 - sig ** 2
    m = y > 0
    G = np.column_stack([np.ones(m.sum()), -(R[m] - R_piv), -x[m]])
    c = np.linalg.lstsq(G, np.log(y[m]), rcond=None)[0]
    b = 1.0 / c[1] if c[1] > 0 else np.ptp(R) + 1.0
    sigma_rho = 1.0 / c[2] if c[2] > 0 else 10.0
    A_piv = np.clip(np.exp(c[0]) / (2.0 * sigma0 ** 2), 1e-3, 0.99)
    return [sigma0, A_piv, b, sigma_rho]


def fit_sigma_joint(x, r_p_arr, sig, R_piv=None):
    R = np.broadcast_to(r_p_arr[:, None], x.shape)
    m = np.isfinite(x) & np.isfinite(sig) & (sig > SIG_FLOOR)
    X, RR, S = x[m], R[m], sig[m]
    if R_piv is None:
        R_piv = np.median(RR)
    popt, pcov = curve_fit(
        lambda c, s0, Ap, b, sr: sigma_mod_2d(c, s0, Ap, b, sr, R_piv),
        (X, RR), S, p0=_seed(X, RR, S, R_piv),
        bounds=([0.0, 1e-4, 1e-3, 1e-3], [np.inf, 1.0, np.inf, np.inf]),
        maxfev=20_000,
    )
    perr = np.sqrt(np.diag(pcov))
    return popt, perr, R_piv


def fit_A_per_R(x, sig, sigma0, sigma_rho):
    A = np.full(x.shape[0], np.nan)
    Aerr = np.full(x.shape[0], np.nan)
    for j in range(x.shape[0]):
        m = np.isfinite(x[j]) & np.isfinite(sig[j]) & (sig[j] > SIG_FLOOR)
        if m.sum() < 3:
            continue
        p, c = curve_fit(lambda xx, a: sigma_mod(xx, sigma0, a, sigma_rho),
                          x[j][m], sig[j][m], p0=[0.3],
                          bounds=(1e-4, 1.0), maxfev=10_000)
        A[j], Aerr[j] = p[0], np.sqrt(c[0, 0])
    return A, Aerr


def fit_delta(x, D):
    m = np.isfinite(x) & np.isfinite(D) & (D > DEL_FLOOR)
    popt, pcov = curve_fit(delta_mod, x[m], D[m], p0=[100.0, 1.0], maxfev=10_000)
    return popt, np.sqrt(np.diag(pcov))


def vr_profile(h, k, r_edges=None, r_max_trust=None):
    """Pair-weighted <v_r>(r) and sigma_vr(r) for mass bin k. From
    ASbase_compact_calibration.ipynb -- unchanged."""
    Rc = 0.5 * (h.R_edges[1:] + h.R_edges[:-1])
    lc = 0.5 * (h.rlos_edges[1:] + h.rlos_edges[:-1])
    r3 = np.sqrt(Rc[:, None] ** 2 + lc[None, :] ** 2).ravel()
    n = h.n[k].ravel().astype(float)
    s1 = h.sum_vr[k].ravel()
    s2 = h.sum_vr2[k].ravel()
    if r_edges is None:
        top = r_max_trust if r_max_trust is not None else h.R_edges[-1]
        r_edges = np.arange(0.0, top + 2.0, 2.0)
    nb_bins = len(r_edges) - 1
    idx = np.searchsorted(r_edges, r3, side='right') - 1
    ok = (idx >= 0) & (idx < nb_bins) & (n > 0)

    def acc(w):
        return np.bincount(idx[ok], weights=w[ok], minlength=nb_bins)

    N = acc(n)
    r_eff = np.where(N > 0, acc(n * r3) / np.where(N > 0, N, 1), np.nan)
    mu = np.where(N > 0, acc(s1) / np.where(N > 0, N, 1), np.nan)
    m2 = np.where(N > 0, acc(s2) / np.where(N > 0, N, 1), np.nan)
    sd = np.sqrt(np.maximum(m2 - mu ** 2, 0.0))
    return r_eff, mu, sd, N.astype(np.int64)


def vr_profile_combined(h, mass_bins_k, r_edges):
    """Pair-weighted <v_r>(r) over the UNION of several mass bins, computed
    correctly: sums the raw additive accumulator arrays (n, sum_vr, sum_vr2)
    across mass_bins_k FIRST, then does the same bincount reduction as
    vr_profile. This is the exact combined-sample mean -- not an average of
    each mass bin's own already-divided mean, which silently reweights every
    bin equally regardless of how many pairs it actually contributed.
    """
    Rc = 0.5 * (h.R_edges[1:] + h.R_edges[:-1])
    lc = 0.5 * (h.rlos_edges[1:] + h.rlos_edges[:-1])
    r3 = np.sqrt(Rc[:, None] ** 2 + lc[None, :] ** 2).ravel()

    n = np.sum([h.n[k] for k in mass_bins_k], axis=0).ravel().astype(float)
    s1 = np.sum([h.sum_vr[k] for k in mass_bins_k], axis=0).ravel()
    s2 = np.sum([h.sum_vr2[k] for k in mass_bins_k], axis=0).ravel()

    nb_bins = len(r_edges) - 1
    idx = np.searchsorted(r_edges, r3, side='right') - 1
    ok = (idx >= 0) & (idx < nb_bins) & (n > 0)

    def acc(w):
        return np.bincount(idx[ok], weights=w[ok], minlength=nb_bins)

    N = acc(n)
    r_eff = np.where(N > 0, acc(n * r3) / np.where(N > 0, N, 1), np.nan)
    mu = np.where(N > 0, acc(s1) / np.where(N > 0, N, 1), np.nan)
    m2 = np.where(N > 0, acc(s2) / np.where(N > 0, N, 1), np.nan)
    sd = np.sqrt(np.maximum(m2 - mu ** 2, 0.0))
    return r_eff, mu, sd, N.astype(np.int64)


def jackknife_mean_err(x):
    """Leave-one-sim-out jackknife over the first axis."""
    n = x.shape[0]
    total = x.sum(axis=0)
    mean_vals = total / n
    loo = (total - x) / (n - 1)
    err = np.sqrt((n - 1) / n * np.sum((loo - mean_vals) ** 2, axis=0))
    return mean_vals, err


# ============================================================ Fig 0: stack
def gather_fig0(mass_bin=K_FIG0, sim=FID_SIMS[0], r_p_max=60.0, r_los_max=60.0):
    """Sum PairHist.counts over r_p < r_p_max, r_los < r_los_max for one mass
    bin, at r_los > 0 -- which is automatically true since this accumulator's
    rlos_edges only ever cover [5, 140] (see abacus_cosmo_core.py docstring:
    "r_los is one-sided"). No standardization -- raw v_los,pec counts,
    summed as-is, since every cell already shares the same v grid.
    """
    path = FID_PAIRHIST_TMPL.format(sim=sim, z=FID_REDSHIFT)
    h = PairHist.load(path)

    Rc = 0.5 * (h.R_edges[1:] + h.R_edges[:-1])
    lc = 0.5 * (h.rlos_edges[1:] + h.rlos_edges[:-1])
    R_mask = Rc < r_p_max
    l_mask = lc < r_los_max

    cells = h.counts[mass_bin][R_mask][:, l_mask]   # (n_R_sel, n_l_sel, n_v)
    stacked_counts = cells.sum(axis=(0, 1))
    v_centers = h.v_centers
    n_pairs = int(h.n[mass_bin][R_mask][:, l_mask].sum())

    return dict(v_centers=v_centers, counts=stacked_counts,
                n_pairs=n_pairs, mass_bin=mass_bin, sim=sim,
                r_p_max=r_p_max, r_los_max=r_los_max,
                logM_range=LOG_MASS_EDGES[mass_bin])


# ============================================================ Fig 1: xi_hh
N_XI_WORKERS = 8   # phases run concurrently; tune down if calvinball is busy with others' jobs

def compute_xihh_all_bins_one_phase(r_edges, sim, nthreads=None, n_jk_1d=4):
    """Real-space halo-halo xi(r) AND its within-box jackknife covariance,
    for EVERY mass bin, ONE phase -- via pycorr (Corrfunc engine), matching
    nb/model_validation/hh_rscf_pycorr.ipynb's convention. Loads the halo
    catalog once and reuses it across all 6 mass bins, rather than reloading
    the same file once per bin.

    Jackknife covariance is computed per phase via pycorr's native
    data_samples1 mechanism -- this is a genuine, independent alternative to
    the phase-to-phase covariance computed in compute_xihh_avg below (which
    scatter-averages this same xi across phases instead). Averaging this
    per-phase jackknife covariance across phases and dividing by n_phases
    gives a second, independent estimate of the covariance of the
    phase-averaged mean -- useful as a cross-check, since phase-to-phase
    scatter with only ~16-25 phases is itself a noisy covariance estimate.

    nthreads defaults to every core (os.cpu_count()) when called on its own,
    but compute_xihh_avg below calls this with a smaller explicit nthreads
    so several phases can run at once without oversubscribing the machine.
    """
    halo_path = FID_HALO_TMPL.format(sim=sim, z=FID_REDSHIFT)
    with h5.File(halo_path, 'r') as hdf:
        hcat = hdf['halos'][:]
        Mvir = M_PART * hcat['N_total']
        pos = np.array([hcat['x'], hcat['y'], hcat['z']]).T % BOX_LENGTH

    n_r = len(r_edges) - 1
    xi_per_bin = np.zeros((N_MASS, n_r))
    cov_jk_per_bin = np.zeros((N_MASS, n_r, n_r))
    n_halos_per_bin = np.zeros(N_MASS, dtype=int)
    for k in range(N_MASS):
        lo, hi = MASSBINS[k]
        sel = (Mvir >= lo) & (Mvir <= hi)
        pos_bin = pos[sel]

        grid_idx = np.clip((pos_bin / BOX_LENGTH * n_jk_1d).astype(int), 0, n_jk_1d - 1)
        samples = grid_idx[:, 0] + n_jk_1d * grid_idx[:, 1] + (n_jk_1d ** 2) * grid_idx[:, 2]

        result = TwoPointCorrelationFunction(
            mode='s', edges=r_edges, data_positions1=pos_bin, data_samples1=samples,
            engine='corrfunc', boxsize=BOX_LENGTH, position_type='pos',
            nthreads=nthreads or os.cpu_count(), compute_sepsavg=False,
        )
        xi_per_bin[k] = result.corr
        cov_jk_per_bin[k] = result.cov()   # this phase's own within-box jackknife covariance
        n_halos_per_bin[k] = int(sel.sum())

    return xi_per_bin, cov_jk_per_bin, n_halos_per_bin


def compute_xihh_avg(r_edges, sims=FID_SIMS, n_workers=N_XI_WORKERS, n_jk_1d=4):
    """Mean xi_hh(r) for ALL mass bins at once, with TWO independent
    covariance estimates:
      - cov: ordinary sample covariance of the mean across n INDEPENDENT
        phases (phase-to-phase scatter) -- the one used everywhere so far.
      - cov_jk: each phase's own within-box jackknife covariance, averaged
        across phases and divided by n_phases (variance-of-the-mean of n
        iid single-box estimates) -- an independent cross-check, since a
        phase-to-phase covariance from only ~16-25 realizations is itself a
        noisy estimate.
    This is the one genuinely new data product in this whole pipeline;
    everything else in the script is a read of existing files. Output size:
    O(N_MASS * len(r_edges)^2) per covariance, still tiny.

    Parallelized across phases with a process pool: n_workers phases run
    concurrently, each getting os.cpu_count()//n_workers threads for its own
    pycorr calls (one per mass bin, from a single catalog load), rather than
    n_workers processes each trying to grab every core, or the catalog being
    re-read once per mass bin. Correlation-function pair counting typically
    saturates well before ~100 threads anyway, so splitting cores across
    concurrent phases should beat the single-phase, all-cores version on
    wall time, not just parallelize it -- worth checking against your own
    timing once you try it, not something I can verify without running it.

    Returns a list of N_MASS dicts (one per mass bin).
    """
    threads_per_worker = max(1, os.cpu_count() // n_workers)
    r_cens = 0.5 * (r_edges[1:] + r_edges[:-1])
    n = len(sims)
    n_r = len(r_cens)
    xi_by_phase = np.zeros((n, N_MASS, n_r))
    cov_jk_by_phase = np.zeros((n, N_MASS, n_r, n_r))
    n_halos_by_phase = np.zeros((n, N_MASS), dtype=int)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(compute_xihh_all_bins_one_phase, r_edges, sim, threads_per_worker, n_jk_1d): i
            for i, sim in enumerate(sims)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            xi_by_phase[i], cov_jk_by_phase[i], n_halos_by_phase[i] = fut.result()

    xi_mean = xi_by_phase.mean(axis=0)   # (N_MASS, n_r)
    cov_jk_mean = cov_jk_by_phase.mean(axis=0) / n   # (N_MASS, n_r, n_r): variance-of-the-mean

    out = []
    for k in range(N_MASS):
        delta = xi_by_phase[:, k, :] - xi_mean[k]
        # ordinary sample covariance of the mean across n INDEPENDENT phases --
        # not the jackknife formula (that's a different, larger normalization
        # meant for correlated delete-one subsamples of a single box, wrong here).
        cov = (delta.T @ delta) / (n * (n - 1))
        out.append(dict(r_cens=r_cens, xi=xi_mean[k], cov=cov, cov_jk=cov_jk_mean[k],
                         n_halos=int(n_halos_by_phase[0, k]), mass_bin=k, n_phases=n))
    return out


# mass bins combined into a single "wide threshold" sample for the diagnostic
# fit -- same four bins Fig 1's plot uses (mass_bins_available[:-2]), NOT a
# contiguous mass cut, since the actual bin edges have gaps between them
# (see LOG_MASS_EDGES) and we want the same halos as before, just combined
# into one real joint measurement instead of an average of four separate ones.
WIDE_MASS_BINS = [0, 1, 2, 3]


def compute_xihh_wide_one_phase(r_edges, mass_bins_k, sim, nthreads=None):
    """Real-space halo-halo xi(r) for the UNION of mass_bins_k, ONE phase --
    a single pycorr call on the combined halo positions. This is NOT the
    same as averaging each mass bin's own xi(r): cross-pairs between a halo
    in one bin and a halo in another are only captured by correlating the
    combined position array directly, since bin auto-correlations alone
    never see those pairs.
    """
    halo_path = FID_HALO_TMPL.format(sim=sim, z=FID_REDSHIFT)
    with h5.File(halo_path, 'r') as hdf:
        hcat = hdf['halos'][:]
        Mvir = M_PART * hcat['N_total']
        pos = np.array([hcat['x'], hcat['y'], hcat['z']]).T % BOX_LENGTH

    sel = np.zeros(len(Mvir), dtype=bool)
    for k in mass_bins_k:
        lo, hi = MASSBINS[k]
        sel |= (Mvir >= lo) & (Mvir <= hi)
    pos_sel = pos[sel]

    result = TwoPointCorrelationFunction(
        mode='s', edges=r_edges, data_positions1=pos_sel,
        engine='corrfunc', boxsize=BOX_LENGTH, position_type='pos',
        nthreads=nthreads or os.cpu_count(), compute_sepsavg=False,
    )
    return result.corr, int(sel.sum())


def compute_xihh_wide_avg(r_edges, mass_bins_k=WIDE_MASS_BINS, sims=FID_SIMS, n_workers=N_XI_WORKERS):
    """Mean xi_hh(r) (and covariance from phase-to-phase scatter) for the
    combined mass_bins_k sample, parallelized across phases exactly like
    compute_xihh_avg -- just one selection per phase instead of N_MASS."""
    threads_per_worker = max(1, os.cpu_count() // n_workers)
    r_cens = 0.5 * (r_edges[1:] + r_edges[:-1])
    n = len(sims)
    xi_by_phase = np.zeros((n, len(r_cens)))
    n_halos_by_phase = np.zeros(n, dtype=int)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(compute_xihh_wide_one_phase, r_edges, mass_bins_k, sim, threads_per_worker): i
            for i, sim in enumerate(sims)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            xi_by_phase[i], n_halos_by_phase[i] = fut.result()

    xi_mean = xi_by_phase.mean(axis=0)
    delta = xi_by_phase - xi_mean
    cov = (delta.T @ delta) / (n * (n - 1))

    return dict(r_cens=r_cens, xi=xi_mean, cov=cov,
                n_halos=int(n_halos_by_phase[0]), n_phases=n)


def run_fig1_wide(out, mass_bins_k=WIDE_MASS_BINS):
    """Combined-mass diagnostic measurement: genuinely joint xi_hh(r) and
    <v_r|r> over the union of mass_bins_k, as one real sample rather than an
    average of Fig 1's separately-computed per-mass-bin curves. Writes its
    own top-level group, 'fig1_wide', alongside (not replacing) 'fig1'.
    """
    print(f'Fig 1 wide-mass diagnostic: measuring xi_hh(r) and <v_r|r> over '
          f'mass bins {mass_bins_k} combined, averaged over all {len(FID_SIMS)} phases...')
    if 'fig1_wide' in out:
        del out['fig1_wide']
    r_edges_shared = np.arange(2.0, 142.0, 2.0)   # same shared grid as run_fig1
    g = out.create_group('fig1_wide')

    xihh = compute_xihh_wide_avg(r_edges_shared, mass_bins_k, sims=FID_SIMS)
    g.create_dataset('r_cens', data=xihh['r_cens'])
    g.create_dataset('xi', data=xihh['xi'])
    g.create_dataset('cov', data=xihh['cov'])
    g.attrs['n_halos'] = xihh['n_halos']
    g.attrs['n_phases'] = xihh['n_phases']
    g.attrs['mass_bins_k'] = mass_bins_k

    pairhists = [PairHist.load(FID_PAIRHIST_TMPL.format(sim=sim, z=FID_REDSHIFT)) for sim in FID_SIMS]
    n_ph = len(FID_SIMS)
    n_r = len(r_edges_shared) - 1
    mean_vr_by_phase = np.full((n_ph, n_r), np.nan)
    r_eff_by_phase = np.full((n_ph, n_r), np.nan)
    npairs_total = np.zeros(n_r, dtype=np.int64)
    for i, h in enumerate(pairhists):
        r_eff, mean_vr, sd_vr, npairs = vr_profile_combined(h, mass_bins_k, r_edges_shared)
        r_eff_by_phase[i] = r_eff
        mean_vr_by_phase[i] = mean_vr
        npairs_total += npairs

    r_eff_avg = np.nanmean(r_eff_by_phase, axis=0)
    vr_avg = np.nanmean(mean_vr_by_phase, axis=0)
    delta_vr = mean_vr_by_phase - vr_avg
    n_good = np.isfinite(delta_vr).sum(axis=0)
    vr_var_of_mean = np.where(n_good > 1,
                               np.nansum(delta_vr**2, axis=0) / np.maximum(n_good * (n_good - 1), 1),
                               np.nan)

    full_mask = n_good == n_ph
    delta_vr_filled = np.where(np.isfinite(delta_vr), delta_vr, 0.0)
    vr_cov = (delta_vr_filled.T @ delta_vr_filled) / (n_ph * (n_ph - 1))
    vr_cov[~full_mask, :] = 0.0
    vr_cov[:, ~full_mask] = 0.0

    g.create_dataset('vr_r', data=r_eff_avg)
    g.create_dataset('vr_mean', data=vr_avg)
    g.create_dataset('vr_sem', data=np.sqrt(vr_var_of_mean))
    g.create_dataset('vr_cov', data=vr_cov)
    g.create_dataset('vr_npairs', data=npairs_total)


# =================================================== Fig 2-5: sigma_v, Delta
def gather_sigma_delta(sims=FID_SIMS):
    """Re-fits sigma_mod/delta_mod/A_rho_mod per (sim, mass bin) from the
    ALREADY-CACHED pdf_fits.h5 (ST_pars, meas_sigv) -- no new histogramming,
    just cheap curve_fit calls on arrays that already exist on disk.
    Reproduces cell 16 of ASbase_compact_calibration_multiple.ipynb, but for
    every mass bin (needed for Fig 5) instead of just the k=3 test bin.
    """
    with h5.File(FID_PDF_FITS_PATH, 'r') as hdf:
        ST_pars = hdf['ST_pars'][:]          # (n_sim, n_M, n_R, n_l, 3): xi, omega, alpha
        meas_sigv = hdf['meas_sigv'][:]      # (n_sim, n_M, n_R, n_l)
        cached_sims = [s.decode() if isinstance(s, bytes) else s
                       for s in hdf.attrs['sims']]

    sim_idx = [cached_sims.index(s) for s in sims]
    n_sim, n_M, n_R = len(sims), N_MASS, len(r_p)

    r_p_2d, r_l_2d = np.meshgrid(r_p, r_l, indexing='ij')
    r3 = np.sqrt(r_p_2d ** 2 + r_l_2d ** 2)
    x = r3 - r_p_2d

    interps_by_sim = []
    for sim in sims:
        h = PairHist.load(FID_PAIRHIST_TMPL.format(sim=sim, z=FID_REDSHIFT))
        interps_by_sim.append(h)   # keep the PairHist itself; vr_profile called per-k below

    sig_opts = np.full((n_sim, n_M, 4), np.nan)
    sig_errs = np.full_like(sig_opts, np.nan)
    del_opts = np.full((n_sim, n_M, 2), np.nan)
    del_errs = np.full_like(del_opts, np.nan)
    A_rho_R = np.full((n_sim, n_M, n_R), np.nan)
    A_rho_R_err = np.full_like(A_rho_R, np.nan)
    R_pivs = np.full((n_sim, n_M), np.nan)

    for si, si_cache in enumerate(sim_idx):
        for k in range(n_M):
            r_eff, mean_vr, _, _ = vr_profile(interps_by_sim[si], k)
            good = np.isfinite(mean_vr)
            from scipy.interpolate import InterpolatedUnivariateSpline
            mean_vr_interp = InterpolatedUnivariateSpline(
                r_eff[good], mean_vr[good], k=3, ext=3)

            sig = meas_sigv[si_cache, k]
            Delta = ST_pars[si_cache, k, :, :, 0] - mean_vr_interp(r3) * r_l_2d / r3

            popt, perr, R_piv = fit_sigma_joint(x, r_p, sig)
            sig_opts[si, k], sig_errs[si, k], R_pivs[si, k] = popt, perr, R_piv
            del_opts[si, k], del_errs[si, k] = fit_delta(x, Delta)
            A_rho_R[si, k], A_rho_R_err[si, k] = fit_A_per_R(x, sig, popt[0], popt[3])

    return dict(sig_opts=sig_opts, sig_errs=sig_errs, del_opts=del_opts,
                del_errs=del_errs, A_rho_R=A_rho_R, A_rho_R_err=A_rho_R_err,
                R_pivs=R_pivs, r_p=r_p, sims=sims)


# ============================================================ Fig 6: cosmo
def gather_cosmo_grid():
    """Pure read + a bit of bookkeeping -- sig_opts/del_opts/A_rho_R for the
    c130-c181 grid were already fit and saved by
    nb/calibration/checks/check_abacus_cosmo_runs.ipynb; this just also pulls
    each sim's cosmology params via abacus_cosmo_core.read_params (cheap,
    reads small header files, no new heavy data)."""
    with h5.File(GRID_PARS_PATH, 'r') as hdf:
        out = {k: hdf[k][()] for k in
               ('sig_opts', 'sig_errs', 'del_opts', 'del_errs',
                'A_rho_R', 'A_rho_R_err', 'R_pivs')}

    sim_pars = [cosmo_grid.read_params(n, GRID_REDSHIFT) for n in GRID_SIMS]
    out['sim_pars'] = sim_pars
    out['sim_nums'] = GRID_SIMS
    return out


# ============================================================ Fig 7: P(v|r,rp)
def gather_fig7(r_p_target, r_los_targets, mass_bin=K_SINGLE, sim=FID_SIMS[0]):
    """Raw counts histograms for a fixed r_p and several r_los -- read only,
    from the same pairhist_allbins.h5 used everywhere else above."""
    h = PairHist.load(FID_PAIRHIST_TMPL.format(sim=sim, z=FID_REDSHIFT))
    j = np.argmin(np.abs(r_p - r_p_target))

    out = {'r_p': r_p[j], 'v_centers': h.v_centers, 'cells': []}
    for r_los_t in r_los_targets:
        i = np.argmin(np.abs(r_l - r_los_t))
        n = h.n[mass_bin, j, i]
        mu = h.sum_v[mass_bin, j, i] / n if n > 0 else np.nan
        sd = np.sqrt(max(h.sum_v2[mass_bin, j, i] / n - mu ** 2, 0.0)) if n > 0 else np.nan
        out['cells'].append(dict(r_los=r_l[i], n=int(n), mean=mu, std=sd,
                                  counts=h.counts[mass_bin, j, i].copy()))
    return out


# ============================================================ per-figure runners
# Each takes an open h5py.File (mode 'w' or 'a') and (re)writes just its own
# top-level group -- so a single section can be re-run against an existing
# cache without redoing the rest of the pipeline. Safe to call repeatedly:
# each deletes its own group first if present.

def run_fig0(out):
    print('Fig 0: stacking raw v_los,pec histogram...')
    if 'fig0' in out:
        del out['fig0']
    f0 = gather_fig0(mass_bin=K_FIG0)
    g = out.create_group('fig0')
    g.create_dataset('v_centers', data=f0['v_centers'])
    g.create_dataset('counts', data=f0['counts'])
    g.attrs['n_pairs'] = f0['n_pairs']
    g.attrs['mass_bin'] = f0['mass_bin']
    g.attrs['r_p_max'] = f0['r_p_max']
    g.attrs['r_los_max'] = f0['r_los_max']


def get_vr_by_phase(mass_bin, r_edges=None, sims=FID_SIMS):
    """Raw per-phase <v_r|r> curves for one mass bin -- not saved to the
    cache (only the mean and covariance are), but cheap to recompute on
    demand for a direct sanity check: overlay these against the quoted
    vr_sem/vr_cov error band to see by eye whether it plausibly captures
    the actual phase-to-phase spread, rather than trusting the covariance
    math alone.
    """
    if r_edges is None:
        r_edges = np.arange(2.0, 142.0, 2.0)
    n_ph = len(sims)
    n_r = len(r_edges) - 1
    mean_vr_by_phase = np.full((n_ph, n_r), np.nan)
    r_eff_by_phase = np.full((n_ph, n_r), np.nan)
    npairs_by_phase = np.full((n_ph, n_r), np.nan)
    for i, sim in enumerate(sims):
        h = PairHist.load(FID_PAIRHIST_TMPL.format(sim=sim, z=FID_REDSHIFT))
        r_eff, mean_vr, sd_vr, npairs = vr_profile(h, mass_bin, r_edges=r_edges)
        r_eff_by_phase[i] = r_eff
        mean_vr_by_phase[i] = mean_vr
        npairs_by_phase[i] = npairs
    return r_eff_by_phase, mean_vr_by_phase, npairs_by_phase


def run_fig1(out):
    print(f'Fig 1: measuring xi_hh(r) and <v_r|r> per mass bin, averaged over all {len(FID_SIMS)} phases...')
    if 'fig1' in out:
        del out['fig1']
    r_edges_shared = np.arange(2.0, 142.0, 2.0)   # linear, shared by xi_hh and <v_r|r> below --
                                                   # starts at 2, not 0: halotools' tpcf() requires
                                                   # strictly positive rbins
    g = out.create_group('fig1')

    n_ph = len(FID_SIMS)
    n_r = len(r_edges_shared) - 1

    # one call for every mass bin at once -- each of the N_XI_WORKERS
    # concurrent phase-workers loads its halo catalog once and computes all
    # 6 bins from it, instead of the catalog being re-read once per bin.
    xihh_all = compute_xihh_avg(r_edges_shared, sims=FID_SIMS)

    for k in range(N_MASS):
        xihh = xihh_all[k]
        gk = g.create_group(f'mbin_{k}')
        gk.create_dataset('r_cens', data=xihh['r_cens'])
        gk.create_dataset('xi', data=xihh['xi'])
        gk.create_dataset('cov', data=xihh['cov'])
        gk.create_dataset('cov_jk', data=xihh['cov_jk'])   # independent cross-check: per-phase
                                                             # within-box jackknife, averaged across
                                                             # phases and /n_phases -- not phase-to-phase
                                                             # scatter like 'cov' above
        gk.attrs['n_halos'] = xihh['n_halos']
        gk.attrs['n_phases'] = xihh['n_phases']

        # <v_r|r>, averaged the same way -- mean across phases, error from
        # phase-to-phase scatter (covariance of the mean), not the previous
        # single-box pair-count-derived std/sqrt(N).
        r_eff_by_phase, mean_vr_by_phase, npairs_by_phase = get_vr_by_phase(
            k, r_edges=r_edges_shared, sims=FID_SIMS)
        npairs_total = np.nansum(npairs_by_phase, axis=0).astype(np.int64)

        r_eff_avg = np.nanmean(r_eff_by_phase, axis=0)
        vr_avg = np.nanmean(mean_vr_by_phase, axis=0)
        delta_vr = mean_vr_by_phase - vr_avg
        n_good = np.isfinite(delta_vr).sum(axis=0)
        vr_var_of_mean = np.where(n_good > 1,
                                   np.nansum(delta_vr**2, axis=0) / np.maximum(n_good * (n_good - 1), 1),
                                   np.nan)

        # full cross-r covariance of the mean, from the same phase-to-phase
        # scatter -- previously we only kept the diagonal (vr_sem below).
        # Only bins where EVERY phase contributed are trustworthy for a
        # cross-covariance (partial-overlap bins would need pairwise-complete
        # handling we're not doing here); those get zeroed out and are
        # already excluded downstream by the existing vr_npairs/vr_sem mask.
        full_mask = n_good == n_ph
        delta_vr_filled = np.where(np.isfinite(delta_vr), delta_vr, 0.0)
        vr_cov = (delta_vr_filled.T @ delta_vr_filled) / (n_ph * (n_ph - 1))
        vr_cov[~full_mask, :] = 0.0
        vr_cov[:, ~full_mask] = 0.0

        gk.create_dataset('vr_r', data=r_eff_avg)
        gk.create_dataset('vr_mean', data=vr_avg)
        gk.create_dataset('vr_sem', data=np.sqrt(vr_var_of_mean))   # renamed from vr_std:
                                                                      # this is now the SEM of the
                                                                      # phase-averaged mean directly,
                                                                      # not a per-pair std needing
                                                                      # a further /sqrt(N) downstream
        gk.create_dataset('vr_cov', data=vr_cov)
        gk.create_dataset('vr_npairs', data=npairs_total)


def run_sigma_delta(out):
    print('Figs 2-5: re-fitting sigma_v/Delta/A_rho per mass bin, per phase...')
    if 'sigma_delta' in out:
        del out['sigma_delta']
    sd = gather_sigma_delta()
    g = out.create_group('sigma_delta')
    for key, val in sd.items():
        if key == 'sims':
            g.attrs['sims'] = np.array(val, dtype=h5.string_dtype())
        else:
            g.create_dataset(key, data=val)


def run_fig6(out):
    print('Fig 6: reading the already-fit cosmology grid...')
    if 'fig6' in out:
        del out['fig6']
    cg = gather_cosmo_grid()
    g = out.create_group('fig6')
    for key, val in cg.items():
        if key == 'sim_pars':
            continue   # dict per sim; save the handful of columns actually plotted instead
        g.create_dataset(key, data=val)
    cosmo_cols = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s']
    g.create_dataset('cosmo_arr', data=np.array(
        [[p[c] for c in cosmo_cols] for p in cg['sim_pars']]))
    g.attrs['cosmo_cols'] = cosmo_cols


def run_fig7(out):
    print('Fig 7: reading raw P(v_los|r_p,r_los) histograms...')
    if 'fig7' in out:
        del out['fig7']
    f7 = gather_fig7(r_p_target=30.0, r_los_targets=[10, 30, 60, 100])
    g = out.create_group('fig7')
    g.attrs['r_p'] = f7['r_p']
    g.create_dataset('v_centers', data=f7['v_centers'])
    for c, cell in enumerate(f7['cells']):
        gc_ = g.create_group(f'cell_{c}')
        gc_.attrs['r_los'] = cell['r_los']
        gc_.attrs['n'] = cell['n']
        gc_.attrs['mean'] = cell['mean']
        gc_.attrs['std'] = cell['std']
        gc_.create_dataset('counts', data=cell['counts'])


# ==================================================================== main
def main():
    with h5.File(OUT_PATH, 'w') as out:
        run_fig0(out)
        run_fig1(out)
        run_sigma_delta(out)
        run_fig6(out)
        run_fig7(out)

    print(f'done -- wrote {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()