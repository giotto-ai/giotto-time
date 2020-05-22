import numpy as np
from typing import Optional
from scipy.linalg import toeplitz
from scipy.stats import zscore


def durbin_levinson_recursion(x: np.array):
    """
    Durbin-Levinson algorithm to fix autocorrelation
    # TODO theoretical review required

    Parameters
    ----------
    x: np.array, input data, model parameters

    Returns
    -------
    x: np.array, transformed array

    """

    t = x.copy()
    x = x.copy()
    for i in range(1, len(x)):
        a = x[i]
        for j in range(i):
            t[j] += a * x[i - j - 1]
        x[:i] = t[:i]
    return x


def arma_polynomial_roots(params: np.array, len_p: int):
    """
    Checks stationarity and invertibility of ARMA model returning roots of its backshift operator polynomials.

    Parameters
    ----------
    params: np.array, ARMA model parameters: [mu, sigma, phi, theta]
    len_p: int, degree of AR, to parse the parameters

    Returns
    -------
    np.array, with polynomial roots and ``passing_values`` in place of mu and sigma to satisfy constraints

    """
    passing_value = 2.0
    phi = params[2 : 2 + len_p]
    theta = params[2 + len_p :]
    phi_roots = np.abs(np.roots(np.r_[-phi[::-1], 1.0]))
    theta_roots = np.abs(np.roots(np.r_[theta[::-1], 1.0]))
    return np.r_[
        passing_value, passing_value, phi_roots, theta_roots
    ]  # TODO refactor 2.0


def normalize(x: np.array) -> np.array:
    """
    Scales x to mean(x) == 0 and std(x) == 1

    Parameters
    ----------
    x: np.array, array of float to be scaled

    Returns
    -------
    np.array, scaled array

    """

    if len(x) <= 1 or x.std() == 0.0:
        return np.zeros(len(x))
    else:
        return zscore(x)


def autocorrelation(x: np.array) -> np.array:
    """
    Autocorrelation via np.correlate for a scaled array `x`

    Parameters
    ----------
    x: np.array, input array

    Returns
    -------
    np.array, autocorrelation for all lags

    """

    n = len(x)
    return np.correlate(x, x, mode="full")[-n:] / n


def solve_yw_equation(r: np.array) -> np.array:
    """
    Solution to Yule-Walker equations via Töplitz matrix

    Parameters
    ----------
    r: autocorrelation coefficients

    Returns
    -------
    np.array: partial autocorrelation function

    """

    R = toeplitz(r[:-1])
    try:
        return np.linalg.solve(R, r[1:])
    except np.linalg.LinAlgError:
        print("Solution is not defined for singular matrices")
        return [np.nan]


def yule_walker(x: np.array, order=1) -> np.array:

    """ Estimate ``order`` parameters from a sequence using the Yule-Walker equations.
    http://www-stat.wharton.upenn.edu/~steele/Courses/956/Resource/YWSourceFiles/YW-Eshel.pdf

    Parameters
    ----------
    x : np.array, input time series
    order : order of the autoregressive process
    unbiased : bool, debiasing correction, False by default

    Returns
    -------
    rho : np.array, autoregressive coefficients
    """

    if order == 0:
        return np.array([1.0])

    r = autocorrelation(x)
    rho = solve_yw_equation(r[: order + 1])
    return rho


def pacf(x: np.array, max_lags: Optional[int] = None) -> np.array:

    """Partial autocorrelation estimate based on Yule-Walker equations

    Parameters
    ----------
    x : np.array, a time series
    max_lags : int, maximum number of lags to be calculated

    Returns
    -------
    pacf : np.array, partial autocorrelations for min(max_lags, len(x)) lags, including lag 0
    """

    n = x.size
    if max_lags is None or max_lags > n:
        max_lags = n
    x = normalize(x)
    pacf = np.array([yule_walker(x, i)[-1] for i in range(max_lags)])
    return pacf


def acf(x: np.array, max_lags: Optional[int] = None) -> np.array:

    """ Autocorrelation estimate function

    Parameters
    ----------
    x : np.array, a time series
    max_lags : int, maximum number of lags to be calculated

    Returns
    -------
    acf : np.array, partial autocorrelations for min(max_lags, len(x)) lags, including lag 0
    """

    n = x.size
    if max_lags is None or max_lags > n:
        max_lags = n
    x = normalize(x)
    acf = autocorrelation(x)
    return acf[:max_lags]
