"""
Pulls the AbacusSummit cosmology table and writes one .param file per
simulation (c130-c181), in the SAME key=value format read_params()/_parse_kv()
already expect -- so no changes are needed to abacus_cosmo_core.py's reading
logic, only to WHERE it looks.

Why this exists: not every sim folder under /spiffball/salcedo/... has a
cosmo_c<NNN>_ph000_z....param file, and the base_c<NNN> fallback is not
universal either. The canonical source is AbacusSummit's own published
table, not a per-simulation file that may or may not be present.

Source, verified directly rather than transcribed by hand:
  https://raw.githubusercontent.com/abacusorg/AbacusSummit/main/Cosmologies/cosmologies.csv
(mirrors the table at https://abacussummit.readthedocs.io/en/latest/cosmologies.html)
Columns: root, notes, omega_b, omega_cdm, h, A_s, n_s, alpha_s, N_ur, N_ncdm,
omega_ncdm, w0_fld, wa_fld, sigma8_m, sigma8_cb.  Confirmed 52 rows for
c130-c181, and c130's derived Om_for_mpart (0.3138) and m_p (2.109e9) match
the real run log from abacus_cosmo_core.py's own param-file-based reading
exactly -- this is the same underlying data, not a different source.

ONE file per sim, not per redshift -- saved at the sim level
(cosmo_grid/{sim}/base_c<NNN>.param), one directory above the {zdir}
structure, matching read_params()'s own PRIMARY candidate location (just
under cosweeney's tree instead of salcedo's, since this is new output, not
someone else's read-only data).

This is a ONE-TIME (or occasional re-run) preparation step, not part of
the per-job pipeline -- run it once before the long run, or any time you
want to refresh the fallback files.

Usage:
    python fetch_abacus_cosmologies.py
"""

import csv
import io
import os
import urllib.request

import nb.calibration.runs.auto_runs.abacus_cosmo_core_auto as core

CSV_URL = 'https://raw.githubusercontent.com/abacusorg/AbacusSummit/main/Cosmologies/cosmologies.csv'
LOCAL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cosmologies.csv')

OUT_TMPL = ('/spiffball/cosweeney/simulations/AbacusSummit_base/halos/'
            'cosmo_grid/{sim}/base_c{n}.param')

# a couple of KNOWN values from the run log you already have (c130, z0.5),
# read from that sim's own cosmo_c130_ph000_z0p500.param file -- if the
# pulled table doesn't reproduce these, something is wrong and nothing
# gets written.
KNOWN_GOOD = {
    130: {'h': 0.6736, 'om_for_mpart_ncdm_excl': 0.3138, 'sigma8_m': 0.711201},
}


def fetch_csv():
    """Network fetch with a local-cache fallback, since the compute
    cluster this eventually runs on may not have internet access at all --
    in that case, run this script once somewhere that does, then copy
    cosmologies.csv alongside it before running here."""
    try:
        print(f'fetching {CSV_URL}')
        with urllib.request.urlopen(CSV_URL, timeout=30) as resp:
            text = resp.read().decode('utf-8')
        with open(LOCAL_CACHE, 'w') as f:
            f.write(text)
        print(f'  saved a local copy to {LOCAL_CACHE}')
        return text
    except Exception as e:
        if os.path.exists(LOCAL_CACHE):
            print(f'  network fetch failed ({e}); using cached copy at {LOCAL_CACHE}')
            with open(LOCAL_CACHE) as f:
                return f.read()
        raise RuntimeError(
            f'could not fetch {CSV_URL} and no local cache at {LOCAL_CACHE} -- '
            f'run this once on a machine with internet access, then copy '
            f'cosmologies.csv next to this script and re-run here.') from e


def parse_rows(text):
    reader = csv.reader(io.StringIO(text))
    header = [h.strip() for h in next(reader)]
    rows = {}
    for r in reader:
        if not r or not r[0].strip():
            continue
        r = [c.strip().strip('"') for c in r]
        d = dict(zip(header, r))
        n = int(d['root'].replace('abacus_cosm', ''))
        rows[n] = d
    return rows


def write_param_file(n, d):
    """Same key=value format _parse_kv() already reads -- omega_b,
    omega_cdm, h, A_s, n_s, alpha_s, N_ur, N_ncdm, omega_ncdm, w0_fld,
    wa_fld, sigma8_m, sigma8_cb, plus the fields _omega_for_mpart() and the
    logging in run_one() actually use."""
    sim = core.sim_name(n)
    out_path = OUT_TMPL.format(sim=sim, n=n)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write("# pulled from AbacusSummit's published cosmology table\n")
        f.write(f"# source: {CSV_URL}\n")
        f.write(f"# root = abacus_cosm{n:03d}\n")
        for key in ('omega_b', 'omega_cdm', 'h', 'A_s', 'n_s', 'alpha_s',
                   'N_ur', 'N_ncdm', 'omega_ncdm', 'w0_fld', 'wa_fld',
                   'sigma8_m', 'sigma8_cb'):
            f.write(f'{key} = {d[key]}\n')
    return out_path


def main():
    text = fetch_csv()
    rows = parse_rows(text)
    print(f'parsed {len(rows)} cosmologies from the table\n')

    missing = [n for n in core.SIM_NUMS if n not in rows]
    if missing:
        raise RuntimeError(f'the table is missing cosmologies: {missing} -- '
                           f'check core.SIM_NUMS against the table before writing anything')

    # self-check against the one real (sim, z) job already run, so a bad
    # fetch or a column-order mixup is caught here, not 100 jobs into a
    # 20-hour run
    for n, known in KNOWN_GOOD.items():
        d = rows[n]
        h = float(d['h'])
        ob, oc = float(d['omega_b']), float(d['omega_cdm'])
        om_cb = (ob + oc) / h**2
        ok_h = abs(h - known['h']) < 1e-6
        ok_om = abs(om_cb - known['om_for_mpart_ncdm_excl']) < 1e-3
        ok_s8 = abs(float(d['sigma8_m']) - known['sigma8_m']) < 1e-5
        print(f'self-check c{n}: h {"OK" if ok_h else "MISMATCH"}, '
              f'Omega_cb {"OK" if ok_om else "MISMATCH"} ({om_cb:.4f} vs '
              f'{known["om_for_mpart_ncdm_excl"]}), '
              f'sigma8_m {"OK" if ok_s8 else "MISMATCH"}')
        if not (ok_h and ok_om and ok_s8):
            raise RuntimeError(
                f'self-check FAILED for c{n} -- the pulled table does not reproduce '
                f'the known-good values from the actual run log. Do not proceed; '
                f'check CSV_URL / column parsing before writing any param files.')
    print()

    written = []
    for n in core.SIM_NUMS:
        out_path = write_param_file(n, rows[n])
        written.append(out_path)
        print(f'wrote {out_path}')

    print(f'\n{len(written)} param files written under '
          f'/spiffball/cosweeney/simulations/AbacusSummit_base/halos/cosmo_grid/')
    print('read_params() finds these automatically once PARAM_DIR is updated -- '
          'see the accompanying diff to abacus_cosmo_core.py.')


if __name__ == '__main__':
    main()