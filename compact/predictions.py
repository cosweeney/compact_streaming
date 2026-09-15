""" 
Puts all of the ingredients of the compact streaming model together. 
"""

from compact.lpt import *
from compact.model import *
from compact.streaming import * 

from scipy.interpolate import UnivariateSpline
from halotools.mock_observables import tpcf_multipole

import warnings
from scipy.interpolate import CubicSpline



class CompactStreamingModel:

    def __init__(self, redshift, config):
        self.z = redshift 
        self.cfg = config
        self._lpt = None 

    def _lpt_inputs(self, cosmo, bias):
        lpt_pars = np.concatenate( (bias, np.zeros(7)) )
        gli = GetLPTInputs(self.z, cosmo, lpt_pars )

        self._lpt = ( gli.xi_real(self.cfg.r_lpt, lpt_pars), 
                     gli.pwv_mean(self.cfg.r_lpt, lpt_pars), 
                     gli.f, gli.H, 
                     gli.D_A, gli.conv )

        return self._lpt
    
    def _compact_params_at_node(self, z_node, cosmo, mass_bin=None):
        emu = self.cfg.compact_emulator[z_node]
        if emu is None:
            raise NotImplementedError(f"z={z_node} node not yet populated in compact_emulator")

        h, ombh2, omch2, As, ns = cosmo
        cosmo_vals = np.array([ombh2, omch2, np.log(1e10 * As), ns])
        delta_vals = self.cfg.fiducial_cosmo - cosmo_vals

        fits = emu['fits'] if mass_bin is None else emu['fits_indiv'][mass_bin]

        params = np.array(emu['mean_params'], dtype=float)
        for row_idx, dependent_cols, beta in fits:
            params[row_idx] = beta[0] + sum(beta[1 + n] * delta_vals[col]
                                            for n, col in enumerate(dependent_cols))
        return params

    def _compact_params(self, cosmo, mass_bin=None, interp_scheme='spline_z'):
        """
        interp_scheme : 'spline_z' (default) -- cubic spline in z, linearly
                        extrapolated beyond the calibrated range using the
                        slope at the nearest boundary node.
                        'linear_lna' -- piecewise-linear interpolation in
                        ln(a), extrapolated linearly from the nearest
                        boundary segment's slope.
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

    def _pdf(self, lpt, cosmo, interp_scheme='spline_z'):
        mean = UnivariateSpline(self.cfg.r_lpt, lpt[1], s=0)
        params = self._compact_params(cosmo, interp_scheme=interp_scheme)
        conv = lpt[-1]

        return lambda v, rp, rl: Pv_compact(v, rp, rl, mean, params,
                        unit_conversion=conv, fix_sig_Del=False)
    
    def _multipoles(self, los_pdf_function, ells=(0, 2)):

        xi_real = UnivariateSpline(self.cfg.r_lpt, self._lpt[0], s=0)

        xi_s_mu = simps_integrate(s_c=self.cfg.r,
                                mu_c=self.cfg.mu_c,
                                twopcf_function=xi_real,
                                los_pdf_function=los_pdf_function,
                                limit=self.cfg.r.max()*1.25, # need to integrate out sufficiently far
                                epsilon=0.01,
                                n=300 
                                )

        multi = np.zeros((len(ells), len(self.cfg.r)))

        for ell in ells:
            multi[ell // 2] = tpcf_multipole(xi_s_mu, self.cfg.mu_edges, order=ell)

        return multi

    def predict(self, theta, ells=(0, 2), interp_scheme='spline_z'):
        """
        theta : array-like, [b1, b2, omega_b, omega_cdm, h, As, ns]
        """
        b1, b2, ombh2, omch2, h, As, ns = theta
        bias  = np.array([b1, b2])
        cosmo = (h, ombh2, omch2, As, ns)

        lpt = self._lpt_inputs(cosmo, bias)
        pdf = self._pdf(lpt, cosmo, interp_scheme=interp_scheme)
        return self._multipoles(pdf, ells=ells)
