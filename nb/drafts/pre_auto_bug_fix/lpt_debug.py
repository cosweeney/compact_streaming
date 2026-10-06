"""
lpt_debug.py -- systematic comparison of this project's own LPT wrapper
(compact.lpt.GetLPTInputs, as used by fit_sim_jk.py/fit_sim_debug.py's
model_xihh/model_vr) against the CONVENTIONS used in v12_joint.py and
modelerr_fit_v12.py, to isolate bugs or convention mismatches in xi_real
and v_r/v12.

REVISED against the actual compact/lpt.py source (cloned from
github.com/cosweeney/compact_streaming), which settles several things
that were only guessed at in the previous version of this file:

  - lpt.conv = a*H/h and lpt.f = Omz**0.55 -- algebraically the SAME
    formula v12_joint.py's velocity_factor computes (H0=100h makes
    100*E(z) == H(z)/h), and lpt.f uses the identical Om(z)^0.55 growth-
    index approximation v12_joint.py uses. This is UNRELATED to the
    catalog-unit "conv" documented elsewhere in this project's memory
    (that's a separate attribute on compact/velocities/pairhist_direct.py's
    own class) -- confirmed by grep, not assumed.
  - GaussianStreamingModel is built as
    GaussianStreamingModel(self.klin, self.Plin, kmax=10.0) inside
    get_pklin_gsm, followed by self.gsm.convert_sigma_bases(). The klin/
    Plin attributes live on the GetLPTInputs wrapper itself (self.klin,
    self.Plin), NOT on self.gsm -- Cell 4 below uses these directly
    rather than introspecting for them.
  - get_pklin_gsm's own `kmax` argument (default 20.0) only sets how far
    the CAMB linear P(k) table extends; it is NEVER forwarded to
    GaussianStreamingModel's kmax, which is hardcoded at 10.0 regardless
    of what's passed to get_pklin_gsm. Cell 6 checks this directly.
  - lpt.pwv_mean (windowed xi, same convention as v12_joint.py's model_v)
    is still called directly elsewhere in this repo (compact/plot.py,
    compact/predictions.py) -- so the windowed/unwindowed inconsistency
    fit_sim_jk.py's model_vr was written to avoid is NOT fully resolved
    project-wide; Cell 2 now compares model_vr against the real
    lpt.pwv_mean directly, not a hand-rolled reimplementation.

Checks, each its own independently-runnable cell after Setup:
  1. xi windowing: gsm.xieft (windowed) vs gsm.compute_xi_real
     (unwindowed), swept over a (b1,b2) grid.
  2. v_r: model_vr (fit_sim_jk.py's convention, unwindowed) vs the ACTUAL
     lpt.pwv_mean (windowed, still used in compact/plot.py and
     compact/predictions.py) -- both called on the same lpt object, so
     any difference is purely the windowing choice, isolated.
  3. Velocity prefactor: lpt.conv*lpt.f vs a manual velocity_factor built
     from the SAME formula v12_joint.py uses -- should now agree closely
     (see note above); a large disagreement here would mean something
     beyond the expected CAMB-radiation-density residual is off.
  4. kmax/nk sensitivity: lpt.gsm (kmax=10, nk=100 -- the latter is
     GaussianStreamingModel's own default, confirmed from source, since
     get_pklin_gsm never overrides nk) vs a second GaussianStreamingModel
     built on the same P(k) table with v12_joint.py's kmax=1.5, nk=130.
  5. Bias-expansion structural check: compute_xi_real should be exactly
     quadratic in (b1,b2) with bs=b3=alpha=...=0.
  6. get_pklin_gsm's kmax passthrough: confirms directly that changing
     the kmax argument to get_pklin_gsm does NOT change the resulting
     gsm's actual extrapolation kmax (hardcoded 10.0 in the source).

Cells 7-8 (added later) go further: rather than comparing compact.lpt
against ITSELF under different settings (Cells 1-6), or against
v12_joint.py's Quijote-specific precomputed templates (unavailable here),
they check compact.lpt against textbook linear-theory formulas, computed
via a Fourier-Bessel transform written from scratch (scipy.integrate.quad,
not velocileptors' own FFTLog machinery) -- a genuinely independent
cross-check of sign, units, and normalization, not just internal
self-consistency:
  7. xi_mm: an independently-transformed linear xi(r) from the SAME
     lpt.klin/lpt.Plin table, vs gsm.compute_xi_real at b1=b2=...=0
     (should converge at large r, where LPT reduces to linear theory).
  8. v_r: the classic pair-conservation Kaiser formula
     v_12(r) = -(2/3) conv*f*r*\bar{xi}(r)/(1+xi(r)), using the same
     independent xi(r) from Cell 7, vs model_vr at LAGRANGIAN b1=0 (the
     unbiased-matter case -- NOT b1=1; this project's own convention is
     Eulerian bias = 1+b1_Lagrangian, confirmed in this project's memory
     -- using b1=1 here gives a spurious, very clean-looking factor of ~2,
     which is how this got caught rather than assumed).

Entirely self-contained, never imports pycorr, per this project's
convention. Run Setup first; the eight check cells can then each run
independently, in any order.
"""
#%% Setup -- run first

import numpy as np
import matplotlib.pyplot as plt

from compact.lpt import GetLPTInputs
from velocileptors.LPT.gaussian_streaming_model_fftw import GaussianStreamingModel

FID_COSMO = [0.6736, 0.02237, 0.1200, 2.083e-9, 0.9649]   # h, ombh2, omch2, As, ns
FID_REDSHIFT_VAL = 0.3
R_TEST = np.linspace(15, 130, 60)

lpt = GetLPTInputs(redshift=FID_REDSHIFT_VAL, cosmo=FID_COSMO)
lpt.get_pklin_gsm()

# representative (b1, b2) grid, including corners known to stress the GSM
# (per this project's kmax=10 fix note, "extreme bias configurations")
B1_GRID = np.array([0.2, 0.8, 1.5, 3.0, 4.5])
B2_GRID = np.array([-8.0, -2.0, 0.0, 2.0, 8.0])

PARS0 = lambda b1, b2: np.array([b1, b2, 0.0, 0, 0, 0, 0, 0, 0])


#%% Cell 1 -- xi windowing convention, swept over (b1, b2)

max_frac_diff = np.zeros((len(B1_GRID), len(B2_GRID)))
fig, axes = plt.subplots(len(B1_GRID), len(B2_GRID),
                          figsize=(3.2*len(B2_GRID), 2.4*len(B1_GRID)),
                          sharex=True)

for i, b1 in enumerate(B1_GRID):
    for j, b2 in enumerate(B2_GRID):
        pars = PARS0(b1, b2)
        lpt._ensure_cumulants(pars)
        xi_windowed = np.interp(R_TEST, lpt.gsm.rint, lpt.gsm.xieft)
        xi_unwindowed = lpt.gsm.compute_xi_real(R_TEST, *pars)
        frac = xi_windowed / xi_unwindowed - 1.0
        max_frac_diff[i, j] = np.max(np.abs(frac))

        ax = axes[i, j]
        ax.plot(R_TEST, frac, 'k-', lw=1)
        ax.axhline(0, c='gray', lw=0.6)
        ax.set_title(f'b1={b1}, b2={b2}', fontsize=8)
        ax.tick_params(labelsize=7)

fig.suptitle('windowed/unwindowed - 1 for xi_real, swept over (b1,b2)')
fig.tight_layout()

print('max |windowed/unwindowed - 1| over the full grid:')
print(f'  overall max: {max_frac_diff.max():.4f}  '
      f'at b1={B1_GRID[np.unravel_index(max_frac_diff.argmax(), max_frac_diff.shape)[0]]}, '
      f'b2={B2_GRID[np.unravel_index(max_frac_diff.argmax(), max_frac_diff.shape)[1]]}')
print(f'  overall min: {max_frac_diff.min():.4f}')


#%% Cell 2 -- v_r: model_vr (unwindowed, fit_sim_jk.py's convention) vs the
# REAL lpt.pwv_mean (windowed, still used in compact/plot.py and
# compact/predictions.py) -- both on the same lpt object, so this isolates
# the windowing choice's actual effect on v_r itself, not a hand-rolled
# stand-in for it.

def model_vr(r, b1, b2, bs=0.0):
    """Copied from fit_sim_jk.py -- unwindowed xi in the denominator."""
    pars = PARS0(b1, b2)
    pars[2] = bs
    lpt._ensure_cumulants(pars)
    v_int = np.interp(r, lpt.gsm.rint, lpt.gsm.veft)
    xi_consistent = lpt.gsm.compute_xi_real(r, *pars)
    return lpt.conv * lpt.f * v_int / (1.0 + xi_consistent)


fig, axes = plt.subplots(len(B1_GRID), len(B2_GRID),
                          figsize=(3.2*len(B2_GRID), 2.4*len(B1_GRID)),
                          sharex=True)
max_frac_diff_v = np.zeros((len(B1_GRID), len(B2_GRID)))

for i, b1 in enumerate(B1_GRID):
    for j, b2 in enumerate(B2_GRID):
        pars = PARS0(b1, b2)
        v_unwindowed = model_vr(R_TEST, b1, b2)
        v_windowed = lpt.pwv_mean(R_TEST, pars)   # the ACTUAL project method
        frac = v_unwindowed / v_windowed - 1.0
        max_frac_diff_v[i, j] = np.nanmax(np.abs(frac))

        ax = axes[i, j]
        ax.plot(R_TEST, frac, 'k-', lw=1)
        ax.axhline(0, c='gray', lw=0.6)
        ax.set_title(f'b1={b1}, b2={b2}', fontsize=8)
        ax.tick_params(labelsize=7)

fig.suptitle('model_vr (unwindowed) / lpt.pwv_mean (windowed) - 1, swept over (b1,b2)\n'
             'this is the SAME live discrepancy between fit_sim_jk.py and '
             'compact/plot.py + compact/predictions.py')
fig.tight_layout()

print('max |model_vr/lpt.pwv_mean - 1| over the full grid:')
print(f'  overall max: {max_frac_diff_v.max():.4f}')
print(f'  overall min: {max_frac_diff_v.min():.4f}')
print('If this is not small, it means plot.py/predictions.py (via pwv_mean) and '
      'fit_sim_jk.py (via model_vr) will report meaningfully different v_r '
      'predictions for the SAME (b1,b2) -- worth deciding which convention should '
      'be the project standard and updating the other call sites to match.')


#%% Cell 3 -- velocity prefactor: lpt.conv*lpt.f vs a manual reconstruction
# of v12_joint.py's velocity_factor, using the SAME growth-index formula.
#
# From compact/lpt.py's actual source:
#   self.conv = self.a * self.H / h          # a*H/h, H from CAMB, km/s/Mpc
#   Omz = Om0*(1+z)**3 / (H/(100h))**2
#   self.f = Omz**0.55                       # SAME approximation v12_joint.py uses
#
# Algebraically, since H0 = 100h: conv = a*H/h = 100*E(z)/(1+z) where
# E(z) = H(z)/H0 -- exactly v12_joint.py's velocity_factor prefactor (minus
# the f(z) it also multiplies in). So conv*f should equal a manually
# computed velocity_factor almost exactly; any residual should come ONLY
# from CAMB's H(z) including radiation/neutrino density that the flat
# matter+DE-only formula below omits (negligible at z=0.3), not from any
# unit or definition mismatch. This is NOT the catalog-unit "conv" from
# compact/velocities/pairhist_direct.py -- confirmed a separate attribute
# on a separate class by grep, unrelated to this one.

print(f'lpt.conv         = {lpt.conv!r}')
print(f'lpt.f             = {lpt.f!r}')
print(f'lpt.conv * lpt.f  = {lpt.conv * lpt.f!r}')
print()

Om0 = (FID_COSMO[1] + FID_COSMO[2]) / FID_COSMO[0]**2
E_z = np.sqrt(Om0*(1.0 + FID_REDSHIFT_VAL)**3 + (1.0 - Om0))
f_growth_approx = (Om0*(1.0 + FID_REDSHIFT_VAL)**3 / E_z**2)**0.55
velocity_factor = 100.0*E_z/(1.0 + FID_REDSHIFT_VAL)*f_growth_approx

print(f'Om0 (ombh2+omch2)/h^2             = {Om0:.6f}')
print(f'E(z), flat matter+DE only          = {E_z:.6f}')
print(f'f_growth (Om(z)^0.55)              = {f_growth_approx:.6f}')
print(f'velocity_factor = 100*E(z)/(1+z)*f = {velocity_factor:.6f}  km/s/(Mpc/h)')
print()
print(f'lpt.f / f_growth_approx           = {lpt.f/f_growth_approx:.6f}  '
      f'(should be ~1: same Omz**0.55 formula, CAMB H(z) vs manual E(z))')
print(f'lpt.conv*lpt.f / velocity_factor  = {lpt.conv*lpt.f/velocity_factor:.6f}  '
      f'(should be ~1, residual only from CAMB radiation/neutrino density)')
print()
print('If both ratios are within ~0.1-1% of 1: everything here is behaving as')
print('expected -- conv*f genuinely IS this project\'s version of velocity_factor,')
print('via the identical formula, just sourced from CAMB\'s H(z) instead of a')
print('hand-computed flat-matter+DE-only E(z).')
print('If either ratio is off by an order-unity factor or a suspicious round number')
print('(e.g. ~h, ~(1+z), ~100): that IS a real bug, not the small residual expected')
print('from CAMB\'s more complete energy budget.')


#%% Cell 4 -- kmax/nk sensitivity: lpt.gsm (kmax=10, from get_pklin_gsm)
# vs a second GaussianStreamingModel on the SAME P(k) table with
# v12_joint.py's kmax=1.5, nk=130.
#
# klin/Plin live on the GetLPTInputs wrapper itself (lpt.klin, lpt.Plin),
# confirmed directly from compact/lpt.py's source -- not on lpt.gsm.
# get_pklin_gsm also calls self.gsm.convert_sigma_bases() right after
# construction; included here too so this is a kmax/nk-only comparison,
# not also a convert_sigma_bases difference.

gsm_v12style = GaussianStreamingModel(lpt.klin, lpt.Plin, kmax=1.5, nk=130)
gsm_v12style.convert_sigma_bases()

fig, axes = plt.subplots(len(B1_GRID), len(B2_GRID),
                          figsize=(3.2*len(B2_GRID), 2.4*len(B1_GRID)),
                          sharex=True)
n_failed = 0
for i, b1 in enumerate(B1_GRID):
    for j, b2 in enumerate(B2_GRID):
        pars = PARS0(b1, b2)
        xi_project = lpt.gsm.compute_xi_real(R_TEST, *pars)
        ax = axes[i, j]
        try:
            xi_v12style = gsm_v12style.compute_xi_real(R_TEST, *pars)
            frac = xi_v12style / xi_project - 1.0
            ax.plot(R_TEST, frac, 'k-', lw=1)
        except Exception as e:
            n_failed += 1
            ax.text(0.5, 0.5, f'FAILED:\n{type(e).__name__}', ha='center',
                    va='center', transform=ax.transAxes, fontsize=7, color='r')
        ax.axhline(0, c='gray', lw=0.6)
        ax.set_title(f'b1={b1}, b2={b2}', fontsize=8)
        ax.tick_params(labelsize=7)

fig.suptitle('xi_real: (v12_joint kmax=1.5,nk=130) / (this project\'s kmax=10) - 1')
fig.tight_layout()
print(f'{n_failed} of {len(B1_GRID)*len(B2_GRID)} (b1,b2) points raised an exception '
      f'at kmax=1.5 -- exactly the loginterp-blowup failures this project\'s kmax=10 '
      f'fix was written to avoid.')
print()
print('Note: velocileptors\' GaussianStreamingModel.__init__ default is nk=100 '
      '(confirmed directly from source: def __init__(self, *args, kmin=3e-3, '
      'kmax=0.5, nk=100, ...)). get_pklin_gsm\'s GaussianStreamingModel(self.klin, '
      'self.Plin, kmax=10.0) call never overrides nk, so this project\'s gsm runs '
      'at nk=100 vs v12_joint.py\'s explicit nk=130 -- a real, if modest (30%), '
      'grid-density difference on top of the much larger kmax=10 vs kmax=1.5 '
      'difference this cell is actually probing. Also note: the default kmin=3e-3 '
      'differs from get_pklin_gsm\'s own kmin=1e-4 (which only sets the CAMB '
      'table\'s range, same as the kmax passthrough issue in Cell 6) -- neither '
      'this project\'s code nor v12_joint.py overrides GaussianStreamingModel\'s '
      'kmin, so both share this same unexamined default.')


#%% Cell 5 -- bias-expansion structural check: is compute_xi_real actually
# quadratic in (b1, b2), as it should be with bs=b3=alpha=...=0?

B1_FINE = np.linspace(0.2, 3.0, 6)
B2_FINE = np.linspace(-6.0, 6.0, 6)
B1M, B2M = np.meshgrid(B1_FINE, B2_FINE, indexing='ij')
b1_flat, b2_flat = B1M.ravel(), B2M.ravel()

A = np.column_stack([np.ones_like(b1_flat), b1_flat, b2_flat,
                      b1_flat*b2_flat, b1_flat**2, b2_flat**2])

max_resid_frac = np.zeros_like(R_TEST)
for ir, r in enumerate(R_TEST):
    xi_vals = np.array([lpt.gsm.compute_xi_real(np.array([r]), b1, b2, 0, 0, 0, 0, 0, 0, 0)[0]
                         for b1, b2 in zip(b1_flat, b2_flat)])
    coeffs, *_ = np.linalg.lstsq(A, xi_vals, rcond=None)
    fit_vals = A @ coeffs
    resid = xi_vals - fit_vals
    max_resid_frac[ir] = np.max(np.abs(resid)) / np.max(np.abs(xi_vals))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(R_TEST, max_resid_frac, 'k-')
ax.set_yscale('log')
ax.set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
ax.set_ylabel('max |residual| / max |xi| over the (b1,b2) grid')
ax.set_title('Quadratic-surface fit residual for xi_real(b1,b2) at bs=b3=alpha=0')
fig.tight_layout()

print(f'max residual fraction over all r: {max_resid_frac.max():.2e}')
print('  ~1e-10 or smaller: xi_real is exactly quadratic in (b1,b2), as expected.')
print('  Meaningfully larger (>~1e-6): something -- windowing, kmax extrapolation, or')
print('  an implementation issue -- is introducing non-quadratic bias dependence.')


#%% Cell 6 -- get_pklin_gsm's kmax argument does NOT change the resulting
# gsm's actual extrapolation kmax (confirmed from source: kmax=10.0 is
# hardcoded in the GaussianStreamingModel(...) call inside get_pklin_gsm,
# independent of what's passed to get_pklin_gsm itself). Direct check.

lpt_default = GetLPTInputs(redshift=FID_REDSHIFT_VAL, cosmo=FID_COSMO)
lpt_default.get_pklin_gsm()   # default kmax=20.0 (only affects the CAMB table)

lpt_lowkmax = GetLPTInputs(redshift=FID_REDSHIFT_VAL, cosmo=FID_COSMO)
lpt_lowkmax.get_pklin_gsm(kmax=3.0)   # attempt to request a lower kmax

pars = PARS0(1.5, 0.0)
xi_default = lpt_default.gsm.compute_xi_real(R_TEST, *pars)
xi_lowkmax = lpt_lowkmax.gsm.compute_xi_real(R_TEST, *pars)

print('max |xi(get_pklin_gsm default) - xi(get_pklin_gsm(kmax=3.0))| / max|xi| = '
      f'{np.max(np.abs(xi_default - xi_lowkmax)) / np.max(np.abs(xi_default)):.2e}')
print('If this is ~0 (not exactly 0, since the input P(k) TABLE range still '
      'differs), that confirms GaussianStreamingModel\'s internal kmax stayed at '
      'the hardcoded 10.0 in both cases -- passing kmax=3.0 to get_pklin_gsm did '
      'NOT lower the GSM\'s own extrapolation range, only the input table\'s reach. '
      'Worth knowing if this project ever tries to use get_pklin_gsm\'s kmax '
      'argument to control GSM stability -- it currently can\'t.')


#%% Cell 7 -- independent linear-theory xi_mm(r), via a Fourier-Bessel
# transform written from scratch (scipy.integrate.quad, weight='sin'), on
# the SAME lpt.klin/lpt.Plin table -- vs gsm.compute_xi_real at
# b1=b2=bs=b3=alpha=...=0. This is NOT expected to match exactly at all r:
# compute_xi_real with all bias/counterterms zero is CLEFT's "zeroth
# order" prediction (1-loop-type corrections at small/intermediate scales,
# blended down to linear theory at low k via the weight function inside
# compute_cumulants -- see the peft/weight lines in
# gaussian_streaming_model_fftw.py), NOT literally pure linear theory
# everywhere. Convergence should improve toward large r, where nonlinear
# corrections should vanish and both should agree to within LPT's own
# few-percent accuracy.

from scipy.integrate import quad

def xi_lin_independent(r):
    """xi(r) = 1/(2 pi^2 r) * int_0^inf k P(k) sin(kr) dk, via a plain
    oscillatory quadrature -- deliberately NOT velocileptors' own FFTLog
    spherical-Bessel transform, so this is a genuinely separate
    implementation, not a repackaging of the same machinery."""
    integrand = lambda kk: np.interp(kk, lpt.klin, lpt.Plin) * kk
    val, _ = quad(integrand, lpt.klin[0], lpt.klin[-1],
                   weight='sin', wvar=r, limit=400)
    return val / (2*np.pi**2*r)


pars0 = PARS0(0.0, 0.0)
xi_lpt = lpt.gsm.compute_xi_real(R_TEST, *pars0)
xi_indep = np.array([xi_lin_independent(r) for r in R_TEST])

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(R_TEST, xi_lpt, 'k-', label='gsm.compute_xi_real (b1=b2=...=0)')
ax[0].plot(R_TEST, xi_indep, 'r--', label='independent Fourier-Bessel transform')
ax[0].set_yscale('log')
ax[0].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
ax[0].set_ylabel(r'$\xi(r)$')
ax[0].legend(fontsize=8)

ax[1].plot(R_TEST, xi_lpt/xi_indep - 1, 'k-')
ax[1].axhline(0, c='gray', lw=0.6)
ax[1].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
ax[1].set_ylabel('gsm.compute_xi_real / independent - 1')
fig.suptitle('Independent linear-theory xi_mm check')
fig.tight_layout()

print('gsm.compute_xi_real(b1=b2=..=0) vs independent Fourier-Bessel xi(r):')
for r, a, b in zip(R_TEST[::4], xi_lpt[::4], xi_indep[::4]):
    print(f'  r={r:7.2f}  lpt={a: .5f}  independent={b: .5f}  ratio={a/b:.4f}')
print('Expect agreement within a few percent at large r (>~40-60), with growing')
print('disagreement at small r (LPT nonlinear corrections genuinely differ from')
print('linear theory there -- not a bug) and noisy ratios right where xi crosses')
print('zero (~110-125, same ratio-near-zero artifact as Cells 1/4).')


#%% Cell 8 -- independent large-r check of model_vr itself, via the classic
# pair-conservation ("Kaiser") formula:
#   v_12(r) = -(2/3) * conv * f * r * xibar(r) / (1 + xi(r))
# using the SAME independent xi(r) from Cell 7 (not gsm's own xieft/
# compute_xi_real), compared against model_vr at LAGRANGIAN b1=0.
#
# IMPORTANT: b1=0 here, not b1=1 -- this project's own documented
# convention (see this project's memory: "velocileptors Lagrangian bias
# convention: b1=0.12 corresponds to Eulerian bias ~1.12") means the
# UNBIASED matter case is LAGRANGIAN b1=0, since Eulerian bias = 1 + b1.
# Using b1=1 here produces a very clean, consistent ~2x discrepancy (an
# easy trap -- this is exactly how it was caught while building this
# check) purely from that convention mismatch, not from any actual bug in
# model_vr.

from scipy.integrate import cumulative_trapezoid

r_fine = np.linspace(0.5, 140.0, 400)
xi_fine = np.array([xi_lin_independent(r) for r in r_fine])
xibar_fine = 3.0 / r_fine**3 * cumulative_trapezoid(xi_fine * r_fine**2, r_fine, initial=0)

R_CHECK = np.array([40.0, 60.0, 80.0, 100.0, 120.0])
xi_at = np.interp(R_CHECK, r_fine, xi_fine)
xibar_at = np.interp(R_CHECK, r_fine, xibar_fine)
v_kaiser = -(2.0/3.0) * lpt.conv * lpt.f * R_CHECK * xibar_at / (1.0 + xi_at)
v_model = model_vr(R_CHECK, 0.0, 0.0)   # b1=0 (Lagrangian) == unbiased matter

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(R_CHECK, v_kaiser, 'ko-', label='pair-conservation Kaiser formula')
ax.plot(R_CHECK, v_model, 'r^--', label='model_vr(b1=0, b2=0)')
ax.set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
ax.set_ylabel(r'$\langle v_r|r\rangle$ [km/s]')
ax.legend(fontsize=8)
fig.suptitle('Independent v_r check: Kaiser formula vs model_vr, unbiased matter')
fig.tight_layout()

print('Kaiser formula vs model_vr(b1=0, b2=0):')
for r, vk, vm in zip(R_CHECK, v_kaiser, v_model):
    print(f'  r={r:6.1f}  kaiser={vk:9.3f}  model_vr={vm:9.3f}  ratio={vm/vk:.4f}')
print('Expect ratio within a few percent of 1 for r>~40 -- confirms model_vr\'s')
print('overall sign, units, and conv*f normalization against a completely')
print('independent formula and transform, not just against lpt.pwv_mean (Cell 2)')
print('or against itself under different settings (Cells 1/4).')