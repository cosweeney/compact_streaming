""" 
For computing and projecting peculiar velocity moments. 
"""

import numpy as np

from typing import NamedTuple, Callable
from scipy.special import binom
from collections import namedtuple
from scipy.stats import norm

def get_moment(
    moments: NamedTuple, r: np.array, r_order: int, t_order: int, mode: str
) -> np.array:
    """
    Given a named tuple containing the radial and transverse moments, returns the ```r_order```
    radial moment and the ```t_order``` transverse moment.

    Args:
        moments: Named tuple containing the radial and transverse moments. 
        r:  pair separation.
        r_order: order of the radial moment
        t_order: order of the transverse moments
        mode: either ```c``` for central moments or ```m``` for moments.

    Returns:
        moment

    Example naming moments Tuple: ('m_10': Radial mean, 'c_20': Second order radial central moment)
    """
    if t_order % 2 != 0:
        # Due to isotropy all momens with t_order odd vanish
        return np.zeros_like(r)
    elif (r_order == 0) and (t_order == 0):
        # The PDF is normalised
        return np.ones_like(r)
    elif (mode == "c") and (r_order + t_order == 1):
        # The first order central moments are zero
        return np.zeros_like(r)
    else:
        return getattr(moments, f"{mode}_{r_order}{t_order}")(r)


def project_to_los(moments: NamedTuple, n: int, mode: str = "c") -> Callable:
    """ 
    Project the moments of the radial and tangential velocity field onto the line of sight moments.

    Args:
        moments: Named tuple containing the radial and transverse moments.
        n: order of the moment.
        mode: Type of moment. If central moments use c, if moments about the origin use m.
    Returns:
        2D function of r_parallel and r_perpendicular that returns the 
        n-th moment of the line of sight velocity PDF 
    """

    def los_moment(r_perpendicular, r_parallel):
        r_perpedicular = np.atleast_2d(r_perpendicular)
        r_parallel = np.atleast_2d(r_parallel)

        r = np.sqrt(r_parallel ** 2 + r_perpendicular ** 2)
        mu = r_parallel / r

        return np.sum(
            [
                binom(n, k)
                * mu ** k
                * np.sqrt(1 - mu ** 2) ** (n - k)
                * get_moment(moments, r, r_order=k, t_order=n - k, mode=mode)
                for k in range(n + 1)
            ],
            axis=0,
        )

    return los_moment

def losmoments2gaussian(mean: Callable, scale: Callable)->Callable:
    """
    Args:
        mean: function that takes r_parallel and r_perp as inputs and returns the mean 
        line of sight pairwise velocity
        std:  function that takes r_parallel and r_perp as inputs and returns the standard 
        deviation of the line of sight pairwise velocity


    Returns:
        pdf_los: line of sight pairwise velocity PDF 
    """
    def pdf_los(vlos: np.array, r_perp: np.array, r_parallel: np.array):
        return norm.pdf(
            vlos, loc=mean(r_perp, r_parallel), scale=scale(r_perp, r_parallel)
        )

    return pdf_los

def project_gaussian_moments(m_10: Callable, c_20: Callable, c_02: Callable)->Callable:
    """
    Args:
        m_10: function that takes pair separation (r) as input and returns the mean 
        radial pairwise velocity
        c_20:  function that takes pair separation (r) as input and returns the standard 
        deviation of the radial pairwise velocity
        c_02:  function that takes pair separation (r) as input and returns the standard 
        deviation of the radial pairwise velocity

    Returns:
        pdf_los: line of sight pairwise velocity PDF 
    """
    Moments = namedtuple('Moments', ['m_10', 'c_20', 'c_02'])
    moments = Moments(m_10, c_20, c_02)
    mean = project_to_los(moments, 1, mode='m')
    c_2 = project_to_los(moments, 2, mode='c')
    std = lambda r_perp, r_parallel: np.sqrt(c_2(r_perp, r_parallel))
    return mean, std

def moments2gaussian(m_10: Callable, c_20: Callable, c_02: Callable)->Callable:
    mean, std = project_gaussian_moments(m_10, c_20, c_02)
    return losmoments2gaussian(mean, std)


# skew t model

import numpy as np
from typing import Callable
from scipy.special import gamma
from scipy.stats import t
from scipy.optimize import fsolve
from scipy.special import gamma


def pdf(v, w, v_c, alpha, nu):
    """ Probability Density Function of a Skewed-Student-t distribution in one dimension.
    Args: 
	    v: random variable.
	    w: scale parameter.
	    v_c: location parameter.
	    alpha: skewness parameter.
	    nu: degrees of freedom.
    Returns:
	    Skewt PDF evaluated at v
    """
    rescaled_v = (v - v_c) / w
    cdf_arg = alpha * rescaled_v * ((nu + 1) / (rescaled_v ** 2 + nu)) ** 0.5
    values = (nu + 1) / (rescaled_v ** 2 + nu)
    return (
        2.0 / w * t.pdf(rescaled_v, scale=1, df=nu) * t.cdf(cdf_arg, df=nu + 1, scale=1)
    )


def losmoments2skewt(w: Callable, v_c: Callable, alpha: Callable, nu: Callable):
    def pdf_los(vlos, r_perp, r_parallel):
        # tricky hack, RectBivariateSpline sadly only takes sorted values, but r_parallel
        # won't be necessarily sorted
        sorted_r_perp = np.sort(r_perp[:, 0])
        idx_to_unsort_perp = r_perp[:, 0].argsort().argsort()

        sorted_r_parallel = np.sort(r_parallel[0, :])
        idx_to_unsort_parallel = r_parallel[0, :].argsort().argsort()
        return pdf(
            v=vlos,
            w=w(sorted_r_perp, sorted_r_parallel)[idx_to_unsort_perp, :][
                :, idx_to_unsort_parallel
            ],
            v_c=v_c(sorted_r_perp, sorted_r_parallel)[idx_to_unsort_perp, :][
                :, idx_to_unsort_parallel
            ],
            alpha=alpha(sorted_r_perp, sorted_r_parallel)[idx_to_unsort_perp, :][
                :, idx_to_unsort_parallel
            ],
            nu=nu(sorted_r_perp, sorted_r_parallel)[idx_to_unsort_perp, :][
                :, idx_to_unsort_parallel
            ],
        )

    return pdf_los

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Callable
from scipy.special import gamma
from scipy.optimize import fsolve, minimize, root
from scipy.interpolate import RectBivariateSpline

spl_path = Path(__file__).resolve().parents[0] / "gamma2params.csv"


def gamma1_constrain(alpha, dof, gamma1):
    b_dof = (dof / np.pi) ** 0.5 * gamma(0.5 * (dof - 1)) / gamma(0.5 * dof)
    delta = alpha / np.sqrt(1 + alpha ** 2)

    return gamma1 - delta * b_dof * (
        (dof * (3 - delta ** 2)) / (dof - 3)
        - 3 * dof / (dof - 2.0)
        + 2 * delta ** 2 * b_dof ** 2
    ) * (dof / (dof - 2) - delta ** 2 * b_dof ** 2) ** (-1.5)


def gamma2_constrain(alpha, dof, gamma2):
    b_dof = (dof / np.pi) ** 0.5 * gamma(0.5 * (dof - 1)) / gamma(0.5 * dof)
    delta = alpha / np.sqrt(1 + alpha ** 2)

    return (
        gamma2
        - (
            3 * dof ** 2 / ((dof - 2) * (dof - 4))
            - 4 * delta ** 2 * b_dof ** 2 * dof * (3 - delta ** 2) / (dof - 3)
            + 6 * delta ** 2 * b_dof ** 2 * dof / (dof - 2)
            - 3 * delta ** 4 * b_dof ** 4
        )
        * (dof / (dof - 2.0) - delta ** 2 * b_dof ** 2) ** (-2.0)
        + 3.0
    )


def constrains(x, gamma1, gamma2):
    alpha, nu = x
    return (gamma1_constrain(alpha, nu, gamma1), gamma2_constrain(alpha, nu, gamma2))


def moments2parameters_low_order(mean, std, alpha, nu):
    delta = alpha / np.sqrt(1 + alpha ** 2)
    b = (nu / np.pi) ** 0.5 * gamma((nu - 1) / 2.0) / gamma(nu / 2.0)
    w = std / np.sqrt(nu / (nu - 2) - delta ** 2 * b ** 2)
    v_c = mean - w * delta * b
    return w, v_c


def moments2parameters(mean, std, gamma1, gamma2, p0=(-0.7, 5)):
    alpha, nu = fsolve(constrains, p0, args=(gamma1, gamma2))
    w, v_c = moments2parameters_low_order(mean, std, alpha, nu)
    return w, v_c, alpha, nu


def interpolate_moments2parameters(
    r_perp: np.array,
    r_parallel: np.array,
    mean: Callable,
    std: Callable,
    gamma1: Callable,
    gamma2: Callable,
) -> List[Callable]:
    st_parameters = np.zeros((r_perp.shape[0], r_parallel.shape[0], 4))
    gamma1_values = []
    gamma2_values = []
    for i in range(len(r_perp)):
        for j in range(len(r_parallel)):
            gamma1_values.append(gamma1(r_perp[i], r_parallel[j])[0][0])
            gamma2_values.append(gamma2(r_perp[i], r_parallel[j])[0][0])
            st_parameters[i, j, :] = moments2parameters(
                mean(r_perp[i], r_parallel[j])[0][0],
                std(r_perp[i], r_parallel[j])[0][0],
                gamma1(r_perp[i], r_parallel[j])[0][0],
                gamma2(r_perp[i], r_parallel[j])[0][0],
            )
    callable_st_parameters = []
    for p in range(st_parameters.shape[-1]):
        callable_st_parameters.append(
            RectBivariateSpline(r_perp, r_parallel, st_parameters[:, :, p])
        )

    return callable_st_parameters


def direct_spline_moments2parameters(
    r_perp: np.array,
    r_parallel: np.array,
    mean: Callable,
    std: Callable,
    gamma1: Callable,
    gamma2: Callable,
) -> List[Callable]:

    df = pd.read_csv(spl_path)
    alpha_interp, nu_interp = get_interpolators(df)
    gamma1_values = gamma1(r_perp.reshape(-1, 1), r_parallel.reshape(1, -1))
    gamma2_values = gamma2(r_perp.reshape(-1, 1), r_parallel.reshape(1, -1))
    mean_values = mean(r_perp.reshape(-1, 1), r_parallel.reshape(1, -1))
    std_values = std(r_perp.reshape(-1, 1), r_parallel.reshape(1, -1))
    alpha = alpha_interp(gamma1_values, gamma2_values, grid=False)
    nu = nu_interp(gamma1_values, gamma2_values, grid=False)
    w, v_c = moments2parameters_low_order(mean_values, std_values, alpha, nu)
    st_parameters = [w, v_c, alpha, nu]
    callable_st_parameters = []
    for param in st_parameters:
        callable_st_parameters.append(RectBivariateSpline(r_perp, r_parallel, param))
    return callable_st_parameters


def generate_gamma_grid(min_gamma1, min_gamma2, max_gamma1, max_gamma2, n):
    p0 = (-0.7, 5)
    gamma1_values = np.linspace(min_gamma1, max_gamma1, n)
    gamma2_values = np.linspace(min_gamma2, max_gamma2, n)
    rows = []
    for i, gamma1 in enumerate(gamma1_values):
        for j, gamma2 in enumerate(gamma2_values):
            alpha, nu = fsolve(constrains, p0, args=(gamma1, gamma2))
            if alpha == -0.7 and nu == 5.0:
                continue
            rows.append([gamma1, gamma2, alpha, nu])
    return pd.DataFrame(rows, columns=["gamma1", "gamma2", "alpha", "nu"])


def get_interpolators(df):
    gamma1 = np.unique(df["gamma1"].values)
    gamma2 = np.unique(df["gamma2"].values)

    alpha_spline = RectBivariateSpline(
        gamma1, gamma2, df["alpha"].to_numpy().reshape((len(gamma1), len(gamma2))),
    )
    nu_spline = RectBivariateSpline(
        gamma1, gamma2, df["nu"].to_numpy().reshape((len(gamma1), len(gamma2))),
    )
    return alpha_spline, nu_spline


import numpy as np
from pathlib import Path



def project_moments(
    m_10: Callable,
    c_20: Callable,
    c_02: Callable,
    c_12: Callable,
    c_30: Callable,
    c_22: Callable,
    c_40: Callable,
    c_04: Callable,
) -> Callable:
    """
    Args:
        m_10: function that takes pair separation (r) as input and returns the mean 
        radial pairwise velocity
        c_20:  function that takes pair separation (r) as input and returns the standard 
        deviation of the radial pairwise velocity
        c_02:  function that takes pair separation (r) as input and returns the standard 
        deviation of the radial pairwise velocity

    Returns:
        pdf_los: line of sight pairwise velocity PDF 
    """

    Moments = namedtuple(
        "Moments", ["m_10", "c_20", "c_02", "c_12", "c_30", "c_22", "c_40", "c_04"]
    )
    moments = Moments(m_10, c_20, c_02, c_12, c_30, c_22, c_40, c_04)
    mean = project_to_los(moments, 1, mode="m")
    c_2 = project_to_los(moments, 2, mode="c")
    c_3 = project_to_los(moments, 3, mode="c")
    c_4 = project_to_los(moments, 4, mode="c")
    std = lambda r_perp, r_parallel: np.sqrt(c_2(r_perp, r_parallel))

    gamma1 = lambda r_perp, r_parallel: c_3(r_perp, r_parallel) / c_2(
        r_perp, r_parallel
    ) ** (3.0 / 2.0)
    gamma2 = (
        lambda r_perp, r_parallel: c_4(r_perp, r_parallel)
        / c_2(r_perp, r_parallel) ** 2
        - 3.0
    )
    return mean, std, gamma1, gamma2


def moments2skewt(
    m_10: Callable,
    c_20: Callable,
    c_02: Callable,
    c_12: Callable,
    c_30: Callable,
    c_40: Callable,
    c_04: Callable,
    c_22: Callable,
    r_max: float=70.,
    n_eval: int = 70,
    use_spl: bool = False,
) -> Callable:

    mean, std, gamma1, gamma2 = project_moments(
        m_10=m_10,
        c_20=c_20,
        c_02=c_02,
        c_12=c_12,
        c_30=c_30,
        c_40=c_40,
        c_04=c_04,
        c_22=c_22,
    )
    r_perp = np.geomspace(0.7, r_max, n_eval)
    r_parallel = np.geomspace(0.7, r_max, n_eval)
    if not use_spl:
        w, v_c, alpha, nu = interpolate_moments2parameters(
            r_perp, r_parallel, mean=mean, std=std, gamma1=gamma1, gamma2=gamma2
        )
    else:
        w, v_c, alpha, nu = direct_spline_moments2parameters(
            r_perp, r_parallel, mean=mean, std=std, gamma1=gamma1, gamma2=gamma2
        )
    return losmoments2skewt(w, v_c, alpha, nu)

if __name__ == "__main__":
    df = generate_gamma_grid(-1.0, 0.0, 1.5, 4.0, n=1000)
    df.to_csv(
        "gamma2params.csv",
        index=False,
    )

    alpha, nu = get_interpolators(df)
    print(alpha(-0.6, 1.4))
    print(nu(-0.6, 1.4))