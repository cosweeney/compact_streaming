"""
Report whether compact/ is in the 11par_v3 source state, whichever way the changes got there (patch or by hand).
Read-only. Run from anywhere:

    python check_source_11par_v3.py [--repo /home/cosweeney/code/compact_streaming] [--no-run]

Checks (static, from the source text) that
  * compact/linear.py exists and matches the reference copy in source_files/linear.py (if present),
  * compact/lpt.py has GetLPTInputs.sigma_v_lin() (and still has your exact_growth option),
  * compact/predictions.py computes sigma_{v,lin} only when cfg.sigma0_from_linear is set, keeps conv as the LAST element of the _lpt tuple,
    and converts row 0 (A_sigma) to sigma_0 in _pdf,
and lists leftovers from the earlier hand-edit instructions (`use_camb_f`). Unless --no-run it then evaluates sigma_{v,lin}
for c000 at z = 0.2 with compact.linear (reference 310.33 km/s with CAMB 2.0.5 defaults; a different value means a different CAMB version/defaults).
"""
import argparse
import hashlib
import os
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--repo', default=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')))
ap.add_argument('--no-run', action='store_true')
args = ap.parse_args()
read = lambda rel: open(os.path.join(args.repo, rel)).read() if os.path.exists(os.path.join(args.repo, rel)) else None
lin, lpt, pred = read('compact/linear.py'), read('compact/lpt.py'), read('compact/predictions.py')
ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'source_files', 'linear.py')
ref = open(ref_path).read() if os.path.exists(ref_path) else None
print(f'repo: {args.repo}')
if lpt is None or pred is None:
    sys.exit('cannot find compact/lpt.py and compact/predictions.py under --repo')

required, stale = [], []
if lin is None:
    required.append(('compact/linear.py exists', False))
else:
    required.append(('compact/linear.py defines camb_results, growth_rate, sigma_v_lin',
                     all(re.search(rf'def {n}\(', lin) for n in ('camb_results', 'growth_rate', 'sigma_v_lin'))))
    if ref is not None:
        same = hashlib.sha256(lin.encode()).hexdigest() == hashlib.sha256(ref.encode()).hexdigest()
        required.append(('compact/linear.py is identical to the reference copy (source_files/linear.py)', same))
required += [
    ('lpt.py imports sigma_v_lin from compact.linear', bool(re.search(r'from compact\.linear import[^\n]*\bsigma_v_lin\b', lpt))),
    ('lpt.py defines GetLPTInputs.sigma_v_lin(self) using self.results, self.redshift, self.h',
     bool(re.search(r'def sigma_v_lin\(self\):[\s\S]{0,300}?self\.results[\s\S]{0,80}?self\.redshift[\s\S]{0,40}?self\.h', lpt))),
    ('lpt.py still has the exact_growth option', 'exact_growth' in lpt),
    ('predictions.py computes sigma_lin only if cfg.sigma0_from_linear',
     bool(re.search(r"sigma_lin\s*=\s*gli\.sigma_v_lin\(\)\s*if\s*getattr\(self\.cfg,\s*'sigma0_from_linear',\s*False\)\s*else\s*None", pred))),
    ('predictions.py _lpt tuple ends with "sigma_lin, gli.conv)" (conv last)', bool(re.search(r'gli\.D_A,\s*sigma_lin,\s*gli\.conv\)', pred))),
    ('predictions.py _pdf converts row 0: params[0] = sqrt(...) * lpt[-2] when sigma0_from_linear',
     bool(re.search(r"if getattr\(self\.cfg,\s*'sigma0_from_linear',\s*False\):[\s\S]{0,200}?params\[0\]\s*=\s*np\.sqrt\([^\n]*\*\s*lpt\[-2\]", pred))),
    ('predictions.py _pdf still reads conv = lpt[-1]', 'conv = lpt[-1]' in pred),
    ('predictions.py still passes exact_growth to GetLPTInputs', bool(re.search(r'GetLPTInputs\([^)]*exact_growth', pred))),
]
stale += [
    ('lpt.py mentions use_camb_f (earlier hand-edit instructions; redundant with exact_growth, remove it)', 'use_camb_f' in lpt),
    ('predictions.py mentions use_camb_f (same; use the config attribute exact_growth instead)', 'use_camb_f' in pred),
    ('lpt.py imports growth_rate without a leading underscore (leaks into `from compact.predictions import *`)',
     bool(re.search(r'from compact\.linear import[^\n]*(?<![\w_])growth_rate(?! as)', lpt))),
]
n_ok = sum(ok for _, ok in required)
for name, ok in required:
    print(f"  [{'OK' if ok else 'MISSING'}] {name}")
for name, hit in stale:
    if hit:
        print(f'  [STALE] {name}')
print(f'\n{n_ok}/{len(required)} required items present, {sum(h for _, h in stale)} stale item(s).')
print('VERDICT:', 'the source is already in the 11par_v3 state: do NOT apply the patch.' if n_ok == len(required) and not any(h for _, h in stale)
      else ('the source has the needed changes but with leftovers: remove the STALE items by hand; do NOT apply the patch.' if n_ok == len(required)
            else 'the source is not in the 11par_v3 state yet: send me the output (and `git diff a68c700 -- compact/lpt.py compact/predictions.py`), I will make a patch for exactly what you have.'))

if not args.no_run and lin is not None:
    sys.path.insert(0, args.repo)
    try:
        import warnings
        warnings.filterwarnings('ignore')
        import camb
        from compact.linear import camb_results, sigma_v_lin, growth_rate
        res = camb_results(0.6736, 0.02237, 0.12, 2.083e-9, 0.9649, [0.2])
        print(f'\nrun check: CAMB {camb.__version__}, c000 z = 0.2: f = {growth_rate(res, 0.2):.5f}, sigma_v,lin = {sigma_v_lin(res, 0.2, 0.6736):.2f} km/s '
              f'(reference with CAMB 2.0.5: f = 0.63919, 310.33 km/s)')
    except Exception as e:
        print(f'\nrun check skipped: {type(e).__name__}: {e}')
