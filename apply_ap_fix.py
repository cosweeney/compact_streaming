"""
Apply the two AP fixes to the repo, in place, by exact-string replacement (so local edits elsewhere in the files are untouched):

  nb/fits/common.py
    * new AP_REF = c000 (DESI fiducial) and H_REF; RESULTS_FID is built from AP_REF instead of COSMO_FID
    * ap_params: both q's carry the unit factor h / H_REF
  nb/fits/11par_fits/fit_lib_11par.py
    * imports H_REF; ap_multi: both q's carry h / H_REF  (_DA_FID / _H_FID follow RESULTS_FID, so they use c000 automatically)

Each replacement must match exactly once, otherwise the script stops before writing anything. Backups are written next to the files as *.orig
(not overwritten if they exist). Safe to re-run: a file that is already patched is skipped.

Usage:  python apply_ap_fix.py [path_to_repo_root]        (default: current directory)
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')

COMMON_EDITS = [
    ("COSMO_FID = [0.6777, 0.02214, 0.11891, 2.2e-9]\n",
     "COSMO_FID = [0.6777, 0.02214, 0.11891, 2.2e-9]\n"
     "# AP reference cosmology = the cosmology that turned the DESI redshifts into separations in h_ref^-1 Mpc: AbacusSummit c000 (the DESI fiducial).\n"
     "# Separate from COSMO_FID, which is only the sampler starting point (and sets LNAS_FID).\n"
     "AP_REF = (0.6736, 0.02237, 0.12)     # h, ombh2, omch2\n"
     "H_REF = AP_REF[0]\n"),
    ("RESULTS_FID = get_background(*COSMO_FID[:3], NS_FIXED)\n",
     "RESULTS_FID = get_background(*AP_REF, NS_FIXED)\n"),
    ("    q_perp = results.angular_diameter_distance(z_eff) / RESULTS_FID.angular_diameter_distance(z_eff)\n"
     "    q_par = RESULTS_FID.hubble_parameter(z_eff) / results.hubble_parameter(z_eff)\n"
     "    return q_perp, q_par\n",
     "    # data separations are in H_REF^-1 Mpc and the model is in h^-1 Mpc at the sampled h, so the physical-distance ratios carry a factor h / H_REF\n"
     "    f = h / H_REF\n"
     "    q_perp = f * results.angular_diameter_distance(z_eff) / RESULTS_FID.angular_diameter_distance(z_eff)\n"
     "    q_par = f * RESULTS_FID.hubble_parameter(z_eff) / results.hubble_parameter(z_eff)\n"
     "    return q_perp, q_par\n"),
]

FITLIB_EDITS = [
    ("get_background, RESULTS_FID, ap_multipoles, load_tracer_arrays, data_vec_from,",
     "get_background, RESULTS_FID, H_REF, ap_multipoles, load_tracer_arrays, data_vec_from,"),
    ("    r = get_background(h, ombh2, omch2, ns)\n"
     "    return [(r.angular_diameter_distance(z) / _DA_FID[z], _H_FID[z] / r.hubble_parameter(z)) for z in z_effs]\n",
     "    r = get_background(h, ombh2, omch2, ns)\n"
     "    f = h / H_REF     # data in H_REF^-1 Mpc, model in h^-1 Mpc at the sampled h (see common.ap_params)\n"
     "    return [(f * r.angular_diameter_distance(z) / _DA_FID[z], f * _H_FID[z] / r.hubble_parameter(z)) for z in z_effs]\n"),
]

TARGETS = [(ROOT / 'nb/fits/common.py', COMMON_EDITS, 'AP_REF'),
           (ROOT / 'nb/fits/11par_fits/fit_lib_11par.py', FITLIB_EDITS, 'H_REF')]


def main():
    new_text = {}
    for path, edits, marker in TARGETS:
        if not path.exists():
            sys.exit(f'{path} not found (pass the repo root as the first argument)')
        text = path.read_text()
        if marker in text and 'f = h / H_REF' in text:
            print(f'{path}: already patched, skipping')
            continue
        for old, new in edits:
            n = text.count(old)
            if n != 1:
                sys.exit(f'{path}: expected 1 match for\n{old!r}\nbut found {n}; nothing was written. Apply this edit by hand.')
            text = text.replace(old, new)
        new_text[path] = text
    for path, text in new_text.items():
        backup = path.with_name(path.name + '.orig')
        if not backup.exists():
            backup.write_text(path.read_text())
        path.write_text(text)
        print(f'{path}: patched (backup: {backup.name})')
    if new_text:
        print('\nCheck (should print (1.0, 1.0)):\n'
              '  cd nb/fits && python -c "from common import ap_params, AP_REF; print(ap_params(*AP_REF, 0.9649, 0.51))"')


if __name__ == '__main__':
    main()