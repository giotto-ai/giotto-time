import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from .tools import durbin_levinson_recursion


def _run_css(params, X, len_p, errors=False, transform=True):

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

    param = np.tanh(param/2)
    return durbin_levinson_recursion(param)


class MLEModel:

    def __init__(self, order, method='mle'):

        self.length = max(order[0], order[1] + 1)
        self.order = order
        self.method = method
        p0 = np.random.random(order[0]) #TODO can be better?
        q0 = np.random.random(order[1])
        self.parameters = np.r_[0.0, 0.0, p0, q0]

    def fit(self, X):

        mu = X.mean(keepdims=True)
        sigma = X.std(keepdims=True) / np.sqrt(len(X))
        self.parameters[0] = mu
        self.parameters[1] = sigma

        Xmin = minimize(lambda phi: _run_css(phi, X, len_p=self.order[0]), x0=self.parameters, method='L-BFGS-B')

        fitted_params = Xmin['x']
        self.ml = Xmin['fun']
        self.mu = fitted_params[0]
        self.sigma = fitted_params[1]
        self.phi = _transform_params(fitted_params[2:self.order[0] + 2])
        self.theta = _transform_params(fitted_params[-self.order[1]:] if self.order[1] > 0 else np.array([]))
        self.parameters = np.r_[self.mu, self.sigma, self.phi, self.theta]
        return self

    def get_errors(self, X):

        errors = _run_css(self.parameters, X, self.order[0], errors=True, transform=False)
        return errors
