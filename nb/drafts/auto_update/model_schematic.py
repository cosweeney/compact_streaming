"""
Schematic of the 11-parameter compact skew-t streaming model, for the paper.

All curves in the mini-panels come from compact.model (nu_of_r, sigma_mod, A_rho_mod,
A_delta_mod, delta_mod_alpha, Pv_compact, with its default pivot R_PIV_FIXED) evaluated at the
pooled-mean parameter vector below, so there is no cached data to load and nothing here
can drift from the model code. Output: PDF (vector) + PNG.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from compact.model import (sigma_mod, A_rho_mod, A_delta_mod, delta_mod_alpha, nu_of_r,
                           Pv_compact)

FIG_DIR = 'figures'
USETEX = False        # True needs a working LaTeX install; False uses matplotlib's Computer Modern mathtext

# pooled-mean compact-model parameters (config _MEAN_PARAMS): the "defaults" used for every mini-panel
PARS = np.array([330.631, 0.278847, 104.897, 29.3631, 169.618, 85.4652, 47.5656, 1.90638,
                 0.335382, 25.5585, 42.1358])
SIGMA0, A_RHO0, A_RHO_P, SIGMA_RHO, A_DELTA0, R_DELTA, SIGMA_DELTA, BETA_DELTA, ALPHA, NU0, R_C = PARS

plt.rcParams.update({
    'text.usetex': USETEX,
    'font.family': 'serif', 'font.serif': ['cmr10', 'Computer Modern Roman'],
    'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2, 'ytick.major.size': 2, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
    'axes.labelsize': 7,
})

W, H = 7.2, 6.6                                   # inches (full text width)
C_MEAN, C_WIDTH, C_OFFSET, C_TAIL, C_PDF, C_BUS = '#6b6b6b', '#2e6f9e', '#c8661a', '#3a8a58', '#7b4f9d', '#4a4a4a'

fig = plt.figure(figsize=(W, H))
bg = fig.add_axes([0, 0, 1, 1])
bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis('off')
fx, fy = 1 / W, 1 / H                              # inches -> figure fraction


def box(x0, y0, x1, y1, color, fill=0.10, lw=0.9):
    bg.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle='round,pad=0,rounding_size=0.008',
                                fc=color, ec=color, alpha=fill, lw=0, zorder=1))
    bg.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle='round,pad=0,rounding_size=0.008',
                                fc='none', ec=color, lw=lw, zorder=2))


def text(x, y, s, size=7, color='k', ha='left', va='top', **kw):
    bg.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, zorder=5, **kw)


def arrow(x0, y0, x1, y1, color='k', lw=1.0):
    bg.annotate('', xy=(x1, y1), xytext=(x0, y0), zorder=4,
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=7))


def line(xs, ys, color='k', lw=1.0, **kw):
    bg.plot(xs, ys, color=color, lw=lw, zorder=4, solid_capstyle='butt', **kw)


def mini_axes(x0, y0, x1, y1, xlabel, ylabel, color):
    ax = fig.add_axes([x0, y0, x1 - x0, y1 - y0])
    ax.margins(x=0)
    ax.set_xlabel(xlabel, labelpad=1); ax.set_ylabel(ylabel, labelpad=1)
    for sp in ax.spines.values():
        sp.set_color(color)
    ax.patch.set_alpha(0.0)
    return ax


# column geometry
XL0, XL1 = 0.035, 0.245                            # left (LPT / mean) column
XR1 = 0.972
XB = [(0.275, 0.505), (0.515, 0.745), (0.755, XR1)]     # width / offset / tail blocks
CX = [(XL0 + XL1) / 2] + [(a + b) / 2 for a, b in XB]   # block centres

# ------------------------------------------------------------------ tier 1: inputs
box(XL0, 0.943, XR1, 0.992, 'k', fill=0.05)
text(0.5035, 0.9795, r'Inputs: $\;b_1,\;b_2,\omega_b,\;\omega_{\rm cdm},\;h,\;A_s,\;n_s,\;z$', size=9.5, ha='center')
arrow(CX[0], 0.943, CX[0], 0.905)
arrow(0.63, 0.943, 0.63, 0.905)
text(CX[0] + 0.01, 0.9265, r'$b_1,\,b_2,\,z$', size=8.5)
text(0.64, 0.9265, r'$\omega_b,\,\omega_{\rm cdm},\,\ln A_s,\,n_s;\;z$', size=8.5)

# ------------------------------------------------------------------ tier 2: LPT and emulators
box(XL0, 0.795, XL1, 0.905, C_MEAN)
text(CX[0], 0.899, r'LPT (velocileptors)', size=7.5, ha='center', color=C_MEAN)
text(XL0 + 0.01, 0.874, r'$\xi(r)$: Real-space CF', size=6.8)
text(XL0 + 0.01, 0.848, r'$\langle v_r\,|\,r\rangle$ (or $v_{12}(r)$):', size=6.8)
text(XL0 + 0.01, 0.826, r'Mean pairwise velocity profile', size=6.8)

box(0.265, 0.795, XR1, 0.905, 'k', fill=0.04)
text((0.265 + XR1) / 2, 0.899, r'Cosmology emulators: regression in $\Delta\theta_{\Lambda{\rm CDM}}$, '
     r'interpolate in $z$', size=7.0, ha='center')
chips = [(C_WIDTH, r'$\sigma_0,\,A_{\rho,0},\,A_{\rho,p},\,\sigma_\rho$'),
         (C_OFFSET, r'$A_{\Delta,0},\,r_\Delta,\,\sigma_\Delta,\,\beta_\Delta,\,\alpha_\Delta$'),
         (C_TAIL, r'$\nu_0,\;r_c$')]
for (x0, x1), (c, s) in zip(XB, chips):
    box(x0 + 0.005, 0.803, x1 - 0.005, 0.862, c, fill=0.16, lw=0.8)
    text((x0 + x1) / 2, 0.832, s, size=7, ha='center', va='center')

# ------------------------------------------------------------------ tier 3: the four moments
Y3B, Y3T = 0.450, 0.755

# --- mean
box(XL0, Y3B, XL1, Y3T, C_MEAN)
arrow(CX[0], 0.795, CX[0], Y3T, color=C_MEAN)
text(CX[0], Y3T - 0.008, r'Mean', size=8, ha='center', color=C_MEAN)
text(CX[0], Y3T - 0.050, r'$\mu=v_{12}(r)\,\dfrac{r_{\rm los}}{r}$', size=8, ha='center')
text(XL0 + 0.01, Y3T - 0.125, r'LPT prediction ($b_1,b_2$);', size=6.2)
text(XL0 + 0.01, Y3T - 0.147, r'measured $\langle v_r|r\rangle$ at calibration', size=6.2)
# geometry sketch: pair separation r with its components r_perp and r_par
ox, oy = XL0 + 0.04, Y3B + 0.04
rp_x, rl_y = 0.82 * fx, 0.62 * fy                  # drawn in true proportions (inches)
line([ox, ox + rp_x], [oy, oy], lw=0.9, ls=(0, (2, 1.5)))
line([ox + rp_x, ox + rp_x], [oy, oy + rl_y], lw=0.9, ls=(0, (2, 1.5)))
line([ox, ox + rp_x], [oy, oy + rl_y], lw=0.9)
text(ox + rp_x / 2, oy - 0.006, r'$r_\perp$', size=7, ha='center')
text(ox + rp_x + 0.008, oy + rl_y / 2, r'$r_{\rm los}$', size=7, va='center')
text(ox + rp_x / 2 - 0.045, oy + rl_y / 2 + 0.012, r'$r$', size=7, ha='center')

# --- width
bx0, bx1 = XB[0]
box(bx0, Y3B, bx1, Y3T, C_WIDTH)
arrow(CX[1], 0.803, CX[1], Y3T, color=C_WIDTH)
text(CX[1], Y3T - 0.008, r'Width: $\sigma_v$', size=8, ha='center', color=C_WIDTH)
text(bx0 + 0.008, Y3T - 0.040, r'$\sigma_v^2=2\sigma_0^2\,\left[1-\rho(x)\right]$', size=7.0)
text(bx0 + 0.008, Y3T - 0.072, r'$\rho(x)=A_\rho(r_\perp)\,e^{-x/\sigma_\rho}$', size=6.5)
text(bx0 + 0.008, Y3T - 0.098, r'$A_\rho(r_\perp)=A_{\rho,0}\,e^{-(r_\perp-R_{\rm piv})/A_{\rho,p}}$', size=6.0)
ax = mini_axes(bx0 + 0.052, Y3B + 0.052, bx1 - 0.012, Y3B + 0.178, r'$x\;[h^{-1}{\rm Mpc}]$',
               r'$\sigma_v^2\;[10^4\,{\rm km^2\,s^{-2}}]$', C_WIDTH)
xx = np.linspace(0, 120, 200)
RP_CURVES, ALPHAS = (10, 40, 80, 120), (0.35, 0.55, 0.78, 1.0)   # light -> dark = r_perp increasing
var = {rp: sigma_mod(xx, SIGMA0, A_rho_mod(rp, A_RHO0, A_RHO_P), SIGMA_RHO) ** 2 / 1e4 for rp in RP_CURVES}
for rp, a_ in zip(RP_CURVES, ALPHAS):
    ax.plot(xx, var[rp], color=C_WIDTH, alpha=a_, lw=1.1)
ax.axhline(2 * SIGMA0 ** 2 / 1e4, color=C_WIDTH, lw=0.6, ls='--')
top = 2 * SIGMA0 ** 2 / 1e4
ax.set_ylim(var[10][0] - 3.0, top * 1.20)
ax.text(119, top + 0.3, r'$2\sigma_0^2$', fontsize=6, color=C_WIDTH, ha='right', va='bottom')
ax.text(8, top + 1.0, r'$r_\perp=120$', fontsize=5.6, color=C_WIDTH, ha='left', va='bottom')   # highest curve (leader line)
ax.plot([27, 27], [top + 0.9, var[120][45] + 0.15], color=C_WIDTH, lw=0.5)
ax.text(2, var[10][0] - 0.7, r'$r_\perp=10$', fontsize=5.6, color=C_WIDTH, ha='left', va='top')      # lowest curve

# --- offset
bx0, bx1 = XB[1]
box(bx0, Y3B, bx1, Y3T, C_OFFSET)
arrow(CX[2], 0.803, CX[2], Y3T, color=C_OFFSET)
text(CX[2], Y3T - 0.008, r'Offset: $\Delta=\xi_{\rm ST}-\mu$', size=8, ha='center', color=C_OFFSET)
text(bx0 + 0.008, Y3T - 0.044, r'$\Delta={\rm sgn}(r_{\rm los})\,A_\Delta(r_\perp)\,'
     r'\dfrac{x^{\alpha_\Delta}}{1+(x/\sigma_\Delta)^{\beta_\Delta}}$', size=6.5)
text(bx0 + 0.008, Y3T - 0.098, r'$A_\Delta(r_\perp)=A_{\Delta,0}\,e^{-r_\perp/r_\Delta}$', size=6.3)
ax = mini_axes(bx0 + 0.052, Y3B + 0.052, bx1 - 0.012, Y3B + 0.178, r'$x\;[h^{-1}{\rm Mpc}]$',
               r'$\Delta\;[{\rm km\,s^{-1}}]$', C_OFFSET)
dlt = {rp: delta_mod_alpha(xx, A_delta_mod(rp, A_DELTA0, R_DELTA), SIGMA_DELTA, BETA_DELTA, ALPHA) for rp in RP_CURVES}
for rp, a_ in zip(RP_CURVES, ALPHAS):
    ax.plot(xx, dlt[rp], color=C_OFFSET, alpha=a_, lw=1.1)
ax.set_ylim(-45, dlt[10].max() * 1.08)
ax.text(62, dlt[10][100] + 12, r'$r_\perp=10$', fontsize=5.6, color=C_OFFSET, ha='left', va='bottom')
ax.text(48, 33, r'$r_\perp=120$', fontsize=5.6, color=C_OFFSET, ha='left', va='top')

# --- tail
bx0, bx1 = XB[2]
box(bx0, Y3B, bx1, Y3T, C_TAIL)
arrow(CX[3], 0.803, CX[3], Y3T, color=C_TAIL)
text(CX[3], Y3T - 0.008, r'Tail weight: $\nu(r)$', size=8, ha='center', color=C_TAIL)
text(bx0 + 0.008, Y3T - 0.044, r'$\nu(r)=2+\dfrac{\nu_0-2}{1+(r_c/r)^2}$', size=6.8)
text(bx0 + 0.008, Y3T - 0.098, r'Enforce $\nu\geq2$ for defined width', size=6.2)
ax = mini_axes(bx0 + 0.052, Y3B + 0.052, bx1 - 0.012, Y3B + 0.178, r'$r\;[h^{-1}{\rm Mpc}]$', r'$\nu$', C_TAIL)
rr = np.linspace(1, 180, 300)
ax.plot(rr, nu_of_r(rr, NU0, R_C), color=C_TAIL, lw=1.3)
ax.axhline(NU0, color=C_TAIL, lw=0.6, ls='--'); ax.axvline(R_C, color=C_TAIL, lw=0.6, ls=':')
ax.text(176, NU0 + 0.6, r'$\nu_0$', fontsize=6, color=C_TAIL, ha='right', va='bottom')
ax.text(R_C + 3, 2.5, r'$r_c$', fontsize=6, color=C_TAIL, va='bottom')
ax.set_ylim(0, NU0 * 1.22)

# ------------------------------------------------------------------ bus: the four moments -> moment matching
Y4T, Y_BUS = 0.395, 0.425
for cx, c in zip(CX, [C_MEAN, C_WIDTH, C_OFFSET, C_TAIL]):
    line([cx, cx], [Y3B, Y_BUS], color=c, lw=1.0)
line([CX[0], CX[3]], [Y_BUS, Y_BUS], color=C_BUS, lw=1.2)
bg.plot([CX[0], CX[1], CX[2], CX[3]], [Y_BUS] * 4, 'o', ms=2.5, color=C_BUS, zorder=5)
arrow(0.32, Y_BUS, 0.32, Y4T, color=C_BUS)

# ------------------------------------------------------------------ tier 4: moment matching (lower left)
Y5B = 0.045
XM1 = 0.600
box(XL0, Y5B, XM1, Y4T, C_PDF)
text(XL0 + 0.012, Y4T - 0.010, r'Moment matching to a skew-$t$', size=8, color=C_PDF)
bul = r'$\bullet$ '
text(XL0 + 0.014, Y4T - 0.050, bul + r'Location: $\xi_{\rm ST}=\mu+\Delta$', size=7.2)
text(XL0 + 0.014, Y4T - 0.090, bul + r'Scale: $\omega=\sqrt{(\sigma_v^2+\Delta^2)\,(\nu-2)/\nu}$', size=7.2)
text(XL0 + 0.014, Y4T - 0.130, bul + r'Skewness: $\alpha=\delta/\sqrt{1-\delta^2}$, $\;\delta=-\Delta/(\omega\,b_\nu)$', size=7.2)
text(XL0 + 0.014, Y4T - 0.170, bul + r'with $b_\nu=\sqrt{\nu/\pi}\;\Gamma\!\left(\frac{\nu-1}{2}\right)/\Gamma\!\left(\frac{\nu}{2}\right)$', size=7.2)
line([XL0 + 0.012, XM1 - 0.012], [0.125, 0.125], color=C_PDF, lw=0.5)
text(XL0 + 0.014, 0.115, r'Result: $P(v_{\rm los}\,|\,r_\perp,r_{\rm los})={\rm ST}(v_{\rm los};\,\xi_{\rm ST},\omega,\alpha,\nu(r))$', size=7.2)
text(XL0 + 0.014, 0.080, r'with mean $\mu$ and standard deviation $\sigma_v$ exactly', size=7.0)
ax = mini_axes(0.405, 0.185, 0.588, 0.340, r'$v_{\rm los}\;[{\rm km\,s^{-1}}]$', r'$P(v_{\rm los}|r_\perp, r_{\rm los})$', C_PDF)
RP_EX, RL_EX = 15.0, 25.0
vv = np.linspace(-1500, 2000, 700)
pdf = Pv_compact(vv, RP_EX, RL_EX, lambda r: np.zeros_like(r), PARS)
x_ex = np.hypot(RP_EX, RL_EX) - RP_EX
delta_ex = delta_mod_alpha(x_ex, A_delta_mod(RP_EX, A_DELTA0, R_DELTA), SIGMA_DELTA, BETA_DELTA, ALPHA)
ax.plot(vv, pdf, color=C_PDF, lw=1.3)
ax.axvline(0, color=C_MEAN, lw=0.7, ls=':'); ax.axvline(delta_ex, color=C_OFFSET, lw=0.7, ls='--')
ymax = pdf.max() * 1.25
ax.set_ylim(0, ymax); ax.set_yticks([])
ax.text(-30, ymax * 0.97, r'$\mu$', fontsize=6.5, color=C_MEAN, ha='right', va='top')
ax.text(delta_ex + 30, ymax * 0.97, r'$\xi_{\rm ST}=\mu+\Delta$', fontsize=6.5, color=C_OFFSET, ha='left', va='top')
ax.text(0.97, 0.62, r'$r_\perp=%g,\;r_{\rm los}=%g$' % (RP_EX, RL_EX) + '\n' + r'($\mu$ set to 0)',
        transform=ax.transAxes, fontsize=5.3, ha='right', va='top')

# ------------------------------------------------------------------ tier 5: streaming integral over multipoles (lower right)
XS0 = 0.665
Y_S0, Y_S1 = 0.225, Y4T
box(XS0, Y_S0, XR1, Y_S1, 'k', fill=0.05)
text(XS0 + 0.010, Y_S1 - 0.010, r'Streaming integral', size=8)
text(XS0 + 0.012, Y_S1 - 0.047, r'$1+\xi_s(s_\perp,s_{\rm los})=\int\,dy\,\left[1+\xi(r)\right]$', size=6.8)
text(XS0 + 0.030, Y_S1 - 0.083, r'$\times\,P\!\left(v_{\rm los}=(s_{\rm los}-y)\,{\rm sgn}\,y\;\vert\;s_\perp,\,\vert s_{\rm los}\vert\right)$', size=6.8)
text(XS0 + 0.012, Y_S1 - 0.125, r'with $r=\sqrt{s_\perp^2+s_{\rm los}^2}$ and $\xi(r)$ from LPT', size=6.5)
Y_M0, Y_M1 = Y5B, 0.185
box(XS0, Y_M0, XR1, Y_M1, 'k', fill=0.05)
text(XS0 + 0.010, Y_M1 - 0.010, r'Multipoles', size=8)
text(XS0 + 0.012, Y_M1 - 0.050, r'$\xi_\ell(s)=(2\ell+1)\!\int_0^1\!d\mu\;\xi_s(s,\mu)\,\mathcal{L}_\ell(\mu)$', size=6.6)
text(XS0 + 0.012, Y_M1 - 0.100, r'$\ell=0,\,2,\,4$', size=6.8)

arrow(XM1, 0.31, XS0, 0.31, color=C_PDF)
#text((XM1 + XS0) / 2, 0.318, r'$P(v_{\rm los}|r_\perp, r_{\rm los})$', size=9, ha='center', va='bottom', color=C_PDF)
XM = (XS0 + XR1) / 2
arrow(XM, Y_S0, XM, Y_M1)
#text(XM + 0.010, (Y_S0 + Y_M1) / 2, r'$\xi_s(s,\mu)$', size=6.8, va='center')

# LPT -> streaming integral: xi(r) loops down the left margin, under the figure, up the right margin
X_L, X_R, Y_LOOP = 0.014, 0.992, 0.012
line([XL0, X_L], [0.845, 0.845], color=C_MEAN)
line([X_L, X_L, X_R, X_R], [0.845, Y_LOOP, Y_LOOP, 0.31], color=C_MEAN)
arrow(X_R, 0.31, XR1, 0.31, color=C_MEAN)
text(0.5, 0.0255, r'$\xi(r)$ from LPT', size=7, color=C_MEAN, ha='center', va='center')

os.makedirs(FIG_DIR, exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(FIG_DIR, f'model_schematic.{ext}'), dpi=300, bbox_inches='tight', pad_inches=0.03)