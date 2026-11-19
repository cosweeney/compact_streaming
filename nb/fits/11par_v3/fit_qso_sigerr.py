"""
QSO fits with the sigma_err emulators (variance-addition treatment of the QSO redshift errors: sigma_v^2 -> sigma_v^2 + sigma_err^2), as a thin layer over
the 11par v3 fit scripts (fits_apfix.py via fits_apfix_11par_v3.py). Put this file next to fits_apfix_11par_v3.py.

The QSO bin is predicted by the 8-input emulators (train_sigerr_11par_v3.py), theta = [b1, b2, omega_b, omega_cdm, h, As, ns, sigma_err]; everything else
(the other tracer-bins in --joint, AP step, window, data, covariance, base-parameter priors, emcee, convergence rule, chain format) is inherited from
fits_apfix.FitProblem, exactly as in fit_qso_lorentz.py. One extra sampled parameter, log10(sigma_err / km s^-1), appended LAST to the baseline parameter
vector, flat prior between log10(errmin) and log10(errmax), i.e. log-uniform in sigma_err. sigma_err is a PAIR dispersion (sqrt(2) x a per-object RMS). The
upper bound cannot exceed the emulators' training range (1000 km/s); the lower cut must be > 0 for a log-uniform prior. No shift is applied.

Chains go to the v3 chain root, model folder compact_v3_noslope, next to the existing fits:
    <model>_<dr1|mean>_indiv_QSO_sigerr_smin<s>      <model>_<dr1|mean>_joint_sigerr_smin<s>
The raw comparison chains (<model>_<dr1|mean>_indiv_QSO_0.8-2.1_smin<s> and <model>_<dr1|mean>_joint_smin<s>) are the same ones the Lorentzian test used
(run `fit_qso_lorentz.py ... --baseline` if they do not exist yet).

    python fit_qso_sigerr.py mock_mean --smin 25 --check           # chi2 vs sigma_err at the c000 start point, timing (no sampling)
    python fit_qso_sigerr.py mock_mean --smin 25 --cores 4         # QSO alone
    python fit_qso_sigerr.py dr1       --smin 25 --cores 4
    python fit_qso_sigerr.py mock_mean --smin 25 --joint --check   # six tracer-bins sharing the cosmology, sigma_err on the QSO
    python fit_qso_sigerr.py mock_mean --smin 25 --joint --cores 4
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

QSO_KEY = ('QSO', 0.8, 2.1)
MODEL = 'compact_v3_noslope'
SIGERR_MODEL = 'compact_v3_noslope_sigerr'                      # registered in fa.MODELS below: the 8-input emulators
SIGERR_EMU_DIR = '/spiffball/cosweeney/data/models_11par_v3_noslope_sigerr_avg_M/'
SIGERR_EMU_SUFFIX = '_11par_v3_noslope_sigerr'
SIGERR_BOX = (0.0, 1000.0)                                      # training range of the emulators, km/s
TAGS = {'dr1': 'dr1', 'mock_mean': 'mean'}
START_SIGERR = 300.0


def make_spec(data, smin, model, ncores, joint=False, **extra):
    tag = TAGS[data]
    if joint:
        name, keys, mode = f'{model}_{tag}_joint_sigerr_smin{smin:g}', list(fa.FIT_KEYS), 'joint'
    else:
        name, keys, mode = f'{model}_{tag}_indiv_QSO_sigerr_smin{smin:g}', [QSO_KEY], 'indiv'
    name += extra.pop('tag', '')
    return dict(name=name, model=model, data=data, mode=mode, keys=keys, smin=smin, group='sigerr', priority=0, mock_idx=None, ncores=ncores, **extra)


class SigErrQSOProblem(fa.FitProblem):
    """theta = [baseline parameters (unchanged), log10_sigma_err]; indiv: [h, ombh2, omch2, logA, b1, b2, ns, log10_sigma_err],
    joint: [h, ombh2, omch2, logA, ns, b1_0, b2_0, ..., b1_5, b2_5, log10_sigma_err]. The QSO bin uses the 8-input emulators; the others are untouched."""

    def __init__(self, spec, errlim=(20.0, 1000.0)):
        super().__init__(spec)
        assert not (self.shift_mode or self.w0wa or self.fix_ns), 'raw LCDM fits only'
        assert SIGERR_BOX[0] <= errlim[0] < errlim[1] <= SIGERR_BOX[1] and errlim[0] > 0, f'errlim must be inside {SIGERR_BOX} with a positive lower cut'
        iq = [i for i, k in enumerate(spec['keys']) if tuple(k) == QSO_KEY]
        assert len(iq) == 1, 'the QSO bin must be in the fit exactly once'
        self.iq, self.errlim = iq[0], errlim
        self.sig_emus = fa.get_emus(SIGERR_MODEL, self.data[self.iq]['z_eff'])
        self.ndim += 1
        self.theta0 = np.append(self.theta0, np.log10(START_SIGERR))
        self.scale = np.append(self.scale, 0.1)
        self.param_names = list(self.param_names) + ['log10sigerr']
        self.param_labels = list(self.param_labels) + [r'\log_{10}(\sigma_{\rm err}/{\rm km\,s^{-1}})']

    def log_prior(self, theta):
        lp = super().log_prior(theta[:-1])
        if not (np.log10(self.errlim[0]) < theta[-1] < np.log10(self.errlim[1])):
            return -np.inf
        return lp                                          # flat in log10 sigma_err = log-uniform in sigma_err

    def _loglike(self, theta, sigma_err=None):
        base = theta[:-1]
        err = 10.0 ** theta[-1] if sigma_err is None else sigma_err
        biases = self.unpack(base)[5]
        h, ombh2, omch2, lnAs, ns = self.model_params(base)
        try:
            aps = fa.ap_multi(h, ombh2, omch2, ns, self.z_effs)                   # one CAMB call for all tracer-bins
            total = 0.0
            for i, (d, (b1, b2), (q_perp, q_par)) in enumerate(zip(self.data, biases, aps)):
                if i == self.iq:
                    model = fa.model_one(self.sig_emus, [b1, b2, ombh2, omch2, h, lnAs, ns, err], d['s_data'], d['W'], q_perp, q_par)
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

    def chi2(self, theta, sigma_err=None):
        return -2.0 * self._loglike(theta, sigma_err) - sum(d['logdet'] for d in self.data)


def check(problem):
    """No sampling: chi2 at the c000 start point vs sigma_err, against the unmodified baseline problem (7-input emulators), and timing."""
    th = problem.theta0.copy()
    print(f'start theta = {np.round(th[:-1], 4)}, {sum(len(d["data_vec"]) for d in problem.data)} data points')
    base = fa.FitProblem(problem.spec)
    chi2_base = -2.0 * base.log_likelihood(th[:-1]) - sum(d['logdet'] for d in base.data)
    print(f'  baseline fits_apfix problem (7-input emulators, no smear): chi2 = {chi2_base:9.2f}')
    for e in (0.0, 50.0, 100.0, 200.0, 300.0, 424.0, 600.0, 800.0, 1000.0):
        print(f'  sigma_err = {e:6.1f} km/s   chi2 = {problem.chi2(th, e):9.2f}')
    th[-1] = np.log10(START_SIGERR)
    t0 = time.time()
    for _ in range(20):
        problem.log_likelihood(th)
    print(f'time per log-likelihood call: {(time.time() - t0) / 20 * 1e3:.1f} ms')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data', choices=['mock_mean', 'dr1'])
    ap.add_argument('--smin', type=float, default=25)
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--joint', action='store_true', help='joint fit of all six tracer-bins, sigma_err on the QSO (default: QSO alone)')
    ap.add_argument('--errmin', type=float, default=20.0, help='lower cut of the log-uniform sigma_err prior, km/s (must be > 0)')
    ap.add_argument('--errmax', type=float, default=1000.0, help='upper edge, km/s (<= the emulators training range, 1000)')
    ap.add_argument('--emu-dir', default=SIGERR_EMU_DIR)
    ap.add_argument('--emu-suffix', default=SIGERR_EMU_SUFFIX)
    ap.add_argument('--cores', type=int, default=4)
    ap.add_argument('--tag', default='')
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()

    fa.MODELS[SIGERR_MODEL] = (args.emu_dir, args.emu_suffix)
    fa.lower_priority()
    spec = make_spec(args.data, args.smin, args.model, args.cores, joint=args.joint, tag=args.tag, errmin=args.errmin, errmax=args.errmax)
    problem = SigErrQSOProblem(spec, errlim=(args.errmin, args.errmax))
    if args.check:
        check(problem)
        return
    fa.run_chain(problem, fa.chain_root(spec), args.cores, tag=f"[{spec['name']}] ")


if __name__ == '__main__':
    main()