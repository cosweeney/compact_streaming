"""
QSO fits with a free Lorentzian redshift-error width, as a thin layer over the 11par v3 fit scripts (fits_apfix.py via fits_apfix_11par_v3.py).
Put this file and lorentz_conv.py next to fits_apfix_11par_v3.py.

Model: emulator xi_l(s) (l = 0, 2, 4) -> AP step (fits_apfix.ap_multi: c000 reference, h/h_fid factor) evaluated on a fine grid S_FINE in
fiducial coordinates -> LOS convolution with the pair kernel of a truncated Lorentzian per-object redshift error (lorentz_conv.py) -> theta-cut
window (DR1 only) -> compare with the data. The convolution is applied to the QSO bin only. One extra sampled parameter, log10(w / km s^-1), appended
LAST to the baseline parameter vector, flat prior (= log-uniform in w); w is the per-object Lorentzian half-width (HWHM) of the shift c dz / (1 + z);
the pair kernel has half-width ~ 2w. Everything else (data, covariance, window, emulators, the baseline parameters and their priors, the other
tracer-bins, emcee, convergence rule, chain format) is inherited from fits_apfix.FitProblem. No shift is applied.

Two modes: QSO alone (7 + 1 parameters) or, with --joint, all six tracer-bins sharing the cosmology (5 + 2 x 6 + 1 parameters).

Chains go to the v3 chain root, in the model folder next to the existing fits, with names
    <model>_<dr1|mean>_indiv_QSO_lorentz_smin<s>      <model>_<dr1|mean>_joint_lorentz_smin<s>       (this script)
    <model>_<dr1|mean>_indiv_QSO_0.8-2.1_smin<s>      <model>_<dr1|mean>_joint_smin<s>               (--baseline: the SAME fit without the convolution, run
                                                        through the unmodified fits_apfix problem; a no-op where that chain already exists and has converged)

    python fit_qso_lorentz.py mock_mean --smin 25 --check          # chi2 vs w at the c000 start point, extrapolation-mode sensitivity, timing (no sampling)
    python fit_qso_lorentz.py mock_mean --smin 25 --baseline       # raw QSO-only baseline at s_min = 25 (does not exist yet)
    python fit_qso_lorentz.py mock_mean --smin 25 --cores 4        # QSO-only convolution fit
    python fit_qso_lorentz.py mock_mean --smin 25 --joint --check  # the same for the joint fit (six tracer-bins, convolution on the QSO)
    python fit_qso_lorentz.py mock_mean --smin 25 --joint --cores 4
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fits_apfix_11par_v3 as v3          # first: pins the BLAS/OpenMP/TF threads before numpy loads, registers the v3 models, patches the chain root
fa = v3.fa

import numpy as np
from common import MU_REF, W_MU
import lorentz_conv as lc

QSO_KEY = ('QSO', 0.8, 2.1)
MODEL = 'compact_v3_noslope'
TAGS = {'dr1': 'dr1', 'mock_mean': 'mean'}
S_FINE = np.arange(2.0, 200.1, 1.0)       # fiducial-coordinate grid on which the AP'd multipoles are built before convolving
assert tuple(fa.ELLS_FIT) == tuple(fa.ELL_ALL[:len(fa.ELLS_FIT)]) and tuple(fa.ELL_ALL) == lc.ELLS


def make_spec(data, smin, model, ncores, lorentz, joint=False, **extra):
    tag = TAGS[data]
    if joint:
        name, keys, mode = f'{model}_{tag}_joint_' + ('lorentz_smin' if lorentz else 'smin') + f'{smin:g}', list(fa.FIT_KEYS), 'joint'
    else:
        name, keys, mode = f'{model}_{tag}_indiv_' + (f'QSO_lorentz_smin{smin:g}' if lorentz else f'{fa.kname(QSO_KEY)}_smin{smin:g}'), [QSO_KEY], 'indiv'
    name += extra.pop('tag', '')
    return dict(name=name, model=model, data=data, mode=mode, keys=keys, smin=smin, group='lorentz', priority=0, mock_idx=None, ncores=ncores, **extra)


class LorentzQSOProblem(fa.FitProblem):
    """theta = [baseline parameters (unchanged), log10_w]; indiv: [h, ombh2, omch2, logA, b1, b2, ns, log10_w],
    joint: [h, ombh2, omch2, logA, ns, b1_0, b2_0, ..., b1_5, b2_5, log10_w]. The convolution acts on the QSO bin only."""

    def __init__(self, spec, wlim=(20.0, 600.0), L1=2000.0, extrap='constant'):
        super().__init__(spec)
        assert not (self.shift_mode or self.w0wa or self.fix_ns), 'raw LCDM fits only'
        iq = [i for i, k in enumerate(spec['keys']) if tuple(k) == QSO_KEY]
        assert len(iq) == 1, 'the QSO bin must be in the fit exactly once'
        self.iq = iq[0]
        d = self.data[self.iq]
        self.wlim, self.L1, self.extrap = wlim, L1, extrap
        self.kms_to_h = (1.0 + d['z_eff']) * fa.AP_REF[0] / fa._H_REF[d['z_eff']]     # h_fid^-1 Mpc per km/s: (1+z) h_fid / H_ref(z), H_ref in km/s/Mpc
        self.u_nodes = lc.default_u_nodes_kms(L1)
        self.geom = lc.ConvGeometry(d['s_data'], S_FINE, MU_REF, W_MU, self.kms_to_h, self.u_nodes)
        self.ndim += 1
        self.theta0 = np.append(self.theta0, np.log10(100.0))
        self.scale = np.append(self.scale, 0.1)
        self.param_names = list(self.param_names) + ['log10w']
        self.param_labels = list(self.param_labels) + [r'\log_{10}(w/{\rm km\,s^{-1}})']

    def log_prior(self, theta):
        lp = super().log_prior(theta[:-1])
        if not (np.log10(self.wlim[0]) < theta[-1] < np.log10(self.wlim[1])):
            return -np.inf
        return lp                                          # flat in log10 w = log-uniform in w

    def _qso_model(self, b1, b2, h, ombh2, omch2, lnAs, ns, q_perp, q_par, lw, extrap=None):
        d = self.data[self.iq]
        xi = [emu([b1, b2, ombh2, omch2, h, lnAs, ns])[1] for emu in d['emus']]
        xi_ap = fa.ap_multipoles(S_FINE, fa.ELL_ALL, *xi, q_perp, q_par)          # (3, len(S_FINE)); NaN outside the emulator range
        xi_ap = lc.fill_outside_emulator(S_FINE, xi_ap, extrap or self.extrap)
        obs = self.geom.apply(xi_ap, lc.pair_weights(10.0 ** lw, self.L1, self.u_nodes))
        if d['W'] is None:
            return obs[:len(fa.ELLS_FIT)].ravel()
        return np.einsum('ots,ts->os', d['W'], obs).ravel()

    def _loglike(self, theta, extrap=None):
        base, lw = theta[:-1], theta[-1]
        biases = self.unpack(base)[5]
        h, ombh2, omch2, lnAs, ns = self.model_params(base)
        try:
            aps = fa.ap_multi(h, ombh2, omch2, ns, self.z_effs)                   # one CAMB call for all tracer-bins
            total = 0.0
            for i, (d, (b1, b2), (q_perp, q_par)) in enumerate(zip(self.data, biases, aps)):
                if i == self.iq:
                    model = self._qso_model(b1, b2, h, ombh2, omch2, lnAs, ns, q_perp, q_par, lw, extrap)
                else:
                    model = fa.model_one(d['emus'], [b1, b2, ombh2, omch2, h, lnAs, ns], d['s_data'], d['W'], q_perp, q_par)
                if not np.all(np.isfinite(model)):
                    return -np.inf
                resid = model - d['data_vec']
                total += -0.5 * (resid @ d['cov_inv'] @ resid + d['logdet'])
        except Exception:
            return -np.inf
        return total

    def log_likelihood(self, theta):
        return self._loglike(theta)

    def chi2(self, theta, extrap=None):
        return -2.0 * self._loglike(theta, extrap) - sum(d['logdet'] for d in self.data)


def check(problem):
    """No sampling: chi2 at the c000 start point vs w for the three below-emulator extrapolation modes, and timing."""
    th = problem.theta0.copy()
    print(f'start theta = {np.round(th[:-1], 4)}, {sum(len(d["data_vec"]) for d in problem.data)} data points, '
          f'kms_to_h = {problem.kms_to_h:.5f} h^-1 Mpc per km/s')
    for lw in np.log10([1.0, 20.0, 50.0, 100.0, 200.0, 400.0, 600.0]):
        th[-1] = lw
        print(f'  w = {10 ** lw:6.1f} km/s   chi2 -> ' + ', '.join(f'{m}: {problem.chi2(th, m):9.2f}' for m in ('powerlaw', 'constant', 'zero')))
    base = fa.FitProblem(problem.spec)                      # the unmodified baseline problem: its chi2 here should match the w = 1 line to ~0.1-0.2 (linear interpolation on the 1 h^-1 Mpc S_FINE grid)
    chi2_base = -2.0 * base.log_likelihood(th[:-1]) - sum(d['logdet'] for d in base.data)
    print(f'  baseline fits_apfix problem (no convolution): chi2 = {chi2_base:9.2f}')
    th[-1] = np.log10(100.0)
    t0 = time.time()
    for _ in range(20):
        problem.log_likelihood(th)
    print(f'time per log-likelihood call: {(time.time() - t0) / 20 * 1e3:.1f} ms')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data', choices=['mock_mean', 'dr1'])
    ap.add_argument('--smin', type=float, default=25)
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--joint', action='store_true', help='joint fit of all six tracer-bins, convolution on the QSO (default: QSO alone)')
    ap.add_argument('--baseline', action='store_true', help='run the no-convolution baseline fit instead')
    ap.add_argument('--wmin', type=float, default=20.0)
    ap.add_argument('--wmax', type=float, default=600.0)
    ap.add_argument('--L1', type=float, default=2000.0, help='per-object truncation of the Lorentzian, km/s (2000 = Yu et al. 2023 for QSOs)')
    ap.add_argument('--extrap', default='constant', choices=['powerlaw', 'constant', 'zero'], help='xi_l below the emulator range: hold the edge value (default)')
    ap.add_argument('--cores', type=int, default=4)
    ap.add_argument('--tag', default='')
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()

    fa.lower_priority()
    if args.baseline:
        spec = make_spec(args.data, args.smin, args.model, args.cores, False, joint=args.joint, tag=args.tag)
        fa.run_chain(fa.FitProblem(spec), fa.chain_root(spec), args.cores, tag=f"[{spec['name']}] ")
        return
    spec = make_spec(args.data, args.smin, args.model, args.cores, True, joint=args.joint, tag=args.tag, wmin=args.wmin, wmax=args.wmax, L1=args.L1, extrap=args.extrap)
    problem = LorentzQSOProblem(spec, wlim=(args.wmin, args.wmax), L1=args.L1, extrap=args.extrap)
    if args.check:
        check(problem)
        return
    fa.run_chain(problem, fa.chain_root(spec), args.cores, tag=f"[{spec['name']}] ")


if __name__ == '__main__':
    main()