import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from .tools import durbin_levinson_recursion


def _run_css(params: np.array, X: np.array, len_p: int, errors: bool = False, transform: bool = True):
    """
    Conditional sum of squares estimate
    http://www.nuffield.ox.ac.uk/economics/papers/1997/w6/ma.pdf

    Parameters
    ----------
    params: np.array, ARMA model parameters
    X: np.array, training time series
    len_p: int, degree of AR, used to parse the params array
    errors: bool, whether or not return residuals. -Loglikelihood is returned otherwise
    transform: bool, whether to transform AR and MA parameters to impose stationarity and invertability

    Returns
    -------
    loglikelihood: float,  -ln(likelihood function)
    errors: np.array, residuals
    """

    mu = params[0]
    nobs = len(X) - len_p
    if transform:
        phi = np.r_[1, -_transform_params(params[2:len_p + 2])]
        theta = np.r_[1, _transform_params(params[len_p + 2:])]
    else:
        phi = np.r_[1, -params[2:len_p + 2]]
        theta = np.r_[1, params[len_p + 2:]]

    y = X - mu
    init = np.zeros((max(len(phi), len(theta)) - 1))
    for i in range(len_p):
        init[i] = sum(-phi[:i + 1][::-1] * y[:i + 1])
    eps = lfilter(phi, theta, y, zi=init)[0][len_p:]
    if errors:
        return eps
    else:
        ssr = np.dot(eps, eps)
        sigma2 = ssr / nobs
        loglikelihood = -nobs / 2. * (np.log(2 * np.pi * sigma2)) - ssr / (2. * sigma2)
        return -loglikelihood


def _transform_params(param: np.array):
    """Transforms parameters to impose stationarity and invertability
    Jones, R. H. (1980). Maximum likelihood fitting of ARMA models to time series with missing observations. Technometrics 22, 389-395

    Parameters
    ----------
    param: np.array, initial parameters

    Returns
    -------
    np.array, transformed parameters

    """

    param = np.tanh(param/2)
    return durbin_levinson_recursion(param)


def _np_shift(x: np.array, max_lag: int): # TODO possible merge somehow with Shift()
    """
    Lag matrix up to ``max_lags`` for a time series ``x``

    Parameters
    ----------
    x: np.array, input array
    max_lag: int,  number of lags

    Returns
    -------
    res: np.array, lag matrix
    """

    n = len(x)
    res = np.zeros((n, max_lag))
    for i in range(n-1):
        lim = min(i, max_lag)
        start = max(0, i - max_lag + 1)
        res[i+1, :lim+1] = x[start:i+1][::-1]
    return res


def _ols_arma_estimate(X: np.array, n_ar: int, n_ma: int):
    """
    Simple OLS estimate for ARMA(n_ar, n_ma) process fitted on ``X``
    Based on Box, G. E. P., G. M. Jenkins, and G. C. Reinsel. Time Series Analysis: Forecasting and Control.

    Parameters
    ----------
    X: np.array, input array
    n_ar: int, degree of autoregressive components
    n_ma: int, degree of moving average components

    Returns
    -------
    params: fitted parameters array

    """
    lags = _np_shift(X, n_ar)
    ar_coef = np.linalg.lstsq(lags, X, rcond=None)[0]
    residuals = X[n_ar:] - np.dot(lags[n_ar:], ar_coef)
    ols_lags = np.c_[lags[n_ar:], _np_shift(residuals, n_ma)]
    params = np.linalg.lstsq(ols_lags, X[n_ar:], rcond=None)[0]
    return params


class ARMAMLEModel:

    def __init__(self, order, method='css'):

        self.length = max(order[0], order[1] + 1)
        self.order = order
        self.method = method
        self.parameters = np.zeros(order[0] + order[1] + 2)

    def fit(self, X: np.array):
        """
        Fits MLE model, maximising likelihood function by selecting model parameters

        Parameters
        ----------
        X: np.array, training time series

        Returns
        -------
        self: ARMAMLEModel

        """
        mu = X.mean(keepdims=True)
        sigma = X.std(keepdims=True) / np.sqrt(len(X))
        self.parameters[0] = mu
        self.parameters[1] = sigma
        self.parameters[2:] = _ols_arma_estimate(X, self.order[0], self.order[1])


        Xmin = minimize(lambda phi: _run_css(phi, X, len_p=self.order[0]), x0=self.parameters, method='L-BFGS-B')

        fitted_params = Xmin['x']
        self.ml = Xmin['fun']
        self.mu = fitted_params[0]
        self.sigma = fitted_params[1]
        self.phi = _transform_params(fitted_params[2:self.order[0] + 2])
        self.theta = _transform_params(fitted_params[-self.order[1]:] if self.order[1] > 0 else np.array([]))
        self.parameters = np.r_[self.mu, self.sigma, self.phi, self.theta]
        return self

    def get_errors(self, X: np.array):
        """
        Returns the residuals of ``X`` given fitted model parameters

        Parameters
        ----------
        X: np.array, time series

        Returns
        -------
        errors: np.array, residuals

        """
        errors = _run_css(self.parameters, X, self.order[0], errors=True, transform=False)
        return errors
