"""
Puts all of the ingredients of the compact streaming model together.

Supports three emulation schemes via the `mass_bin` argument to `predict`:
  'low_M'   -- uses fits_indiv['low_M'] at each z-node (bin 0 only)
  'high_M'  -- uses fits_indiv['high_M'] at each z-node (bin 1 only)
  'avg_M'   -- uses fits (the bin-averaged emulator, default)
  None      -- same as 'avg_M' for backward compatibility

This does not imply anything about the actual mass of the tracers being
fit -- it's a choice of which mass-calibrated emulation scheme to use
for the compact-model parameters, so that the effect of mass-dependent
velocity modelling on the DESI fits can be tested by comparing runs
under each scheme.
"""

from compact.lpt import *
from compact.model import *
from compact.streaming import *

from scipy.interpolate import UnivariateSpline
from halotools.mock_observables import tpcf_multipole

import warnings
from scipy.interpolate import CubicSpline


# valid mass_bin values and how they map to the compact_emulator dict
_MASS_BIN_MAP = {
    None:      'fits',          # backward compat: default to bin-average
    'avg_M':   'fits',          # explicit bin-average
    'low_M':   'fits_indiv',    # per-bin: low mass
    'high_M':  'fits_indiv',    # per-bin: high mass
}


class CompactStreamingModel:

    def __init__(self, redshift, config, exact_growth=None):
        self.z = redshift
        self.cfg = config
        self._lpt = None
        # growth rate from CAMB instead of Omega_m(z)^0.55; default taken from config (False if unset)
        self.exact_growth = getattr(config, 'exact_growth', False) if exact_growth is None else exact_growth

    def _lpt_inputs(self, cosmo, bias):
        lpt_pars = np.concatenate((bias, np.zeros(7)))
        gli = GetLPTInputs(self.z, cosmo, lpt_pars, exact_growth=self.exact_growth)
        # sigma_{v,lin} is only needed when the config's row 0 is A_sigma (sigma_0^2 = A_sigma sigma_{v,lin}^2)
        sigma_lin = gli.sigma_v_lin() if getattr(self.cfg, 'sigma0_from_linear', False) else None

        # conv must stay last: _pdf and the Gaussian generators read it as lpt[-1]
        self._lpt = (gli.xi_real(self.cfg.r_lpt, lpt_pars),
                     gli.pwv_mean(self.cfg.r_lpt, lpt_pars),
                     gli.f, gli.H,
                     gli.D_A, sigma_lin, gli.conv)

        return self._lpt

    def _compact_params_at_node(self, z_node, cosmo, mass_bin=None):
        """Evaluate the compact-model parameter vector at a single z-node
        and a queried cosmology.

        mass_bin : None, 'avg_M', 'low_M', or 'high_M'
            Selects which set of ridge-fit coefficients to use at this node.
            None/'avg_M' -> emu['fits'] (bin-averaged);
            'low_M'/'high_M' -> emu['fits_indiv']['low_M' or 'high_M'].
        """
        emu = self.cfg.compact_emulator[z_node]
        if emu is None:
            raise NotImplementedError(f"z={z_node} node not yet populated in compact_emulator")

        h, ombh2, omch2, As, ns = cosmo[:5]
        w0, wa = (cosmo[5], cosmo[6]) if len(cosmo) == 7 else (-1.0, 0.0)
        n_cols = len(self.cfg.fiducial_cosmo)      # 4 (LCDM emulator) or 6 (adds w0, wa)
        if n_cols == 4 and (w0, wa) != (-1.0, 0.0):
            raise ValueError("this config's emulator has no w0, wa dependence; use compact.config_w0wa.ConfigW0wa")
        cosmo_vals = np.array([ombh2, omch2, np.log(1e10 * As), ns, w0, wa])[:n_cols]
        delta_vals = self.cfg.fiducial_cosmo - cosmo_vals

        # resolve mass_bin to the right fits list
        if mass_bin not in _MASS_BIN_MAP:
            raise ValueError(f"mass_bin must be one of {list(_MASS_BIN_MAP)}, got {mass_bin!r}")

        key = _MASS_BIN_MAP[mass_bin]
        if key == 'fits':
            fits = emu['fits']
        else:
            # key == 'fits_indiv' -- need the sub-key ('low_M' or 'high_M')
            fits = emu['fits_indiv'][mass_bin]

        params = np.array(emu['mean_params'], dtype=float)
        for row_idx, dependent_cols, beta in fits:
            params[row_idx] = beta[0] + sum(beta[1 + n] * delta_vals[col]
                                            for n, col in enumerate(dependent_cols))
        return params

    def _compact_params(self, cosmo, mass_bin=None, interp_scheme='spline_z'):
        """Interpolate the compact-model parameter vector to self.z from
        the populated z-nodes in compact_emulator.

        interp_scheme : 'spline_z' (default)
            Cubic spline in z, linearly extrapolated beyond the calibrated
            range using the slope at the nearest boundary node.
            'linear_lna' -- piecewise-linear interpolation in ln(a),
            extrapolated linearly from the nearest boundary segment's slope.
        mass_bin : None, 'avg_M', 'low_M', or 'high_M'
            Passed through to _compact_params_at_node at every z-node.
        """
        z_nodes = sorted(z for z, v in self.cfg.compact_emulator.items() if v is not None)

        if self.z in self.cfg.compact_emulator and self.cfg.compact_emulator[self.z] is not None:
            return self._compact_params_at_node(self.z, cosmo, mass_bin)

        if not z_nodes:
            raise NotImplementedError("No calibrated redshift nodes available")

        node_params = np.array([self._compact_params_at_node(zn, cosmo, mass_bin) for zn in z_nodes])

        out_of_range = self.z < z_nodes[0] or self.z > z_nodes[-1]
        if out_of_range:
            warnings.warn(
                f"z={self.z} is outside the calibrated range [{z_nodes[0]}, {z_nodes[-1]}]; "
                f"extrapolating linearly ('{interp_scheme}') -- treat results with caution.",
                stacklevel=2,
            )

        if interp_scheme == 'spline_z':
            spline = CubicSpline(z_nodes, node_params, axis=0, extrapolate=False)
            if not out_of_range:
                return spline(self.z)
            edge_z = z_nodes[0] if self.z < z_nodes[0] else z_nodes[-1]
            edge_params = spline(edge_z)
            edge_slope = spline(edge_z, 1)   # first derivative at the boundary
            return edge_params + edge_slope * (self.z - edge_z)

        elif interp_scheme == 'linear_lna':
            x_nodes = np.log(1.0 / (1.0 + np.asarray(z_nodes, dtype=float)))
            order = np.argsort(x_nodes)
            x_sorted = x_nodes[order]
            params_sorted = node_params[order]

            x_query = np.log(1.0 / (1.0 + self.z))
            i = np.clip(np.searchsorted(x_sorted, x_query) - 1, 0, len(x_sorted) - 2)
            slope = (params_sorted[i + 1] - params_sorted[i]) / (x_sorted[i + 1] - x_sorted[i])
            return params_sorted[i] + slope * (x_query - x_sorted[i])

        else:
            raise ValueError(f"Unknown interp_scheme: {interp_scheme!r}")

    def _pdf(self, lpt, cosmo, mass_bin=None, interp_scheme='spline_z', sigma_err=0.0):
        """Build the line-of-sight velocity PDF function for the streaming
        model integral, using compact-model parameters evaluated at the
        queried cosmology and mass_bin scheme. sigma_err [km/s] is an extra pairwise
        LOS dispersion added in quadrature to sigma_v (see Pv_compact); 0 = unchanged."""
        mean = UnivariateSpline(self.cfg.r_lpt, lpt[1], s=0)
        params = self._compact_params(cosmo, mass_bin=mass_bin, interp_scheme=interp_scheme)
        if getattr(self.cfg, 'sigma0_from_linear', False):
            params = np.array(params, dtype=float)
            params[0] = np.sqrt(max(params[0], 0.0)) * lpt[-2]       # row 0 is A_sigma: sigma_0 = sqrt(A_sigma) sigma_{v,lin}
        conv = lpt[-1]

        return lambda v, rp, rl: Pv_compact(v, rp, rl, mean, params,
                                            unit_conversion=conv, sigma_err=sigma_err)

    def _multipoles(self, los_pdf_function, ells=(0, 2)):
        """Compute redshift-space correlation function multipoles from
        the streaming model integral."""
        xi_real = UnivariateSpline(self.cfg.r_lpt, self._lpt[0], s=0)

        xi_s_mu = simps_integrate(s_c=self.cfg.r,
                                  mu_c=self.cfg.mu_c,
                                  twopcf_function=xi_real,
                                  los_pdf_function=los_pdf_function,
                                  limit=self.cfg.r.max() * 1.25,
                                  epsilon=0.01,
                                  n=300)

        multi = np.zeros((len(ells), len(self.cfg.r)))

        for ell in ells:
            multi[ell // 2] = tpcf_multipole(xi_s_mu, self.cfg.mu_edges, order=ell)

        return multi

    def predict(self, theta, ells=(0, 2), interp_scheme='spline_z', mass_bin=None):
        """Generate redshift-space multipole predictions.

        Parameters
        ----------
        theta : array-like, length 7 or 9
            [b1, b2, omega_b, omega_cdm, h, As, ns] or [..., ns, w0, wa]; w0, wa default to -1, 0
        ells : tuple of int
            Multipole orders to compute (default (0, 2)).
        interp_scheme : str
            Redshift interpolation method ('spline_z' or 'linear_lna').
        mass_bin : None, 'avg_M', 'low_M', or 'high_M'
            Which mass-calibrated emulation scheme to use for the compact-
            model parameters:
              'low_M'  -- low-mass bin calibration only (bin 0)
              'high_M' -- high-mass bin calibration only (bin 1)
              'avg_M'  -- bin-averaged calibration (default)
              None     -- same as 'avg_M' (backward compat)

        Returns
        -------
        multi : ndarray, shape (len(ells), len(cfg.r))
            Predicted multipoles xi_ell(s).
        """
        b1, b2, ombh2, omch2, h, As, ns, *de = theta
        if len(de) not in (0, 2):
            raise ValueError(f"theta must have 7 or 9 entries, got {len(theta)}")
        bias = np.array([b1, b2])
        cosmo = (h, ombh2, omch2, As, ns, *de)

        lpt = self._lpt_inputs(cosmo, bias)
        pdf = self._pdf(lpt, cosmo, mass_bin=mass_bin, interp_scheme=interp_scheme)
        return self._multipoles(pdf, ells=ells)