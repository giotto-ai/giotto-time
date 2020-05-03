import numpy as np
import pandas as pd
from typing import Tuple
from gtime.forecasting.simple_models import SimpleForecaster
from gtime.stat_tools import MLEModel


def _arma_forecast(n: int, x0: np.array, eps0: np.array, mu: float, phi: np.array, theta: np.array) -> np.array:
    """
    Forecasts next ``n`` steps of ARIMA process.

    Parameters
    ----------
    n: int, number of steps to forecast
    x0: np.array, initial conditions, previous observations for the AR process
    eps0: np.array, initial conditions, previous residuals of the MA process
    mu: float, process mean
    phi: np.array, AR coefficients
    theta: np.array, MA coefficients

    Returns
    -------
    np.array, ``n``-step forecast

    """
    len_ar = len(phi)
    len_ma = len(theta)
    phi = phi[::-1]
    theta = theta[::-1]
    x = np.r_[x0, np.zeros(n)]
    eps = np.r_[eps0, np.zeros(n)]
    trend = mu * (1 - phi.sum())
    for i in range(n):
        x[i + len_ar] = trend + np.dot(phi, x[i:i + len_ar]) + np.dot(theta, eps[i:i + len_ma])
    return x[len_ar:]


def _arma_insample_errors(x: np.array, eps0: np.array, mu: float, phi: np.array, theta: np.array) -> np.array:

    """
    Forecasts next ``n`` steps of ARIMA process.

    Parameters
    ----------
    x: np.array, test time series
    eps0: np.array, initial conditions, previous residuals of the MA process
    mu: float, process mean
    phi: np.array, AR coefficients
    theta: np.array, MA coefficients

    Returns
    -------
    eps: np.array, in-sample errors

    """

    len_ar = len(phi)
    len_ma = len(theta)
    phi = phi[::-1]
    theta = theta[::-1]
    n = len(x) - len_ar
    x_f = np.zeros(n)
    eps = np.r_[eps0, np.zeros(n)]
    trend = mu * (1 - phi.sum())
    for i in range(n):
        x_f[i] = trend + np.dot(phi, x[i:i + len_ar]) + np.dot(theta, eps[i:i + len_ma])
        eps[i + len_ma] = x[i + len_ar] - x_f[i]
    return eps


class ARIMAForecaster(SimpleForecaster):
    """
    ARIMA forecaster

    Parameters
    ----------
    order: Tuple[int, int, int], model order of AR, I and MA
    method: str, estimation method

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gtime.model_selection import horizon_shift, FeatureSplitter
    >>> from gtime.forecasting import ARIMAForecaster
    >>> idx = pd.period_range(start='2011-01-01', end='2012-01-01')
    >>> np.random.seed(1)
    >>> df = pd.DataFrame(np.random.random((len(idx), 1)), index=idx, columns=['1'])
    >>> y = horizon_shift(df, horizon=5)
    >>> X_train, y_train, X_test, y_test = FeatureSplitter().transform(df, y)
    >>> m = ARIMAForecaster(order=(1, 0, 1), method='css')
    >>> m.fit(X_train, y_train).predict(X_test)
                         y_1       y_2       y_3       y_4       y_5
        2011-12-28  0.508831  0.508736  0.508830  0.508736  0.508829
        2011-12-29  0.871837  0.148431  0.866452  0.153777  0.861146
        2011-12-30  0.119179  0.895486  0.124959  0.889750  0.130652
        2011-12-31  0.476250  0.541073  0.476733  0.540594  0.477208
        2012-01-01  0.046294  0.967829  0.053154  0.961020  0.059913

    """

    def __init__(self, order: Tuple[int, int, int], method: str = 'css-mle'):
        self.order = order
        self.max_degree = max(order[0], order[2])
        self.n_ar = order[0]
        self.n_ma = order[2]
        self.method = method
        self.model = None

    def _deintegrate(self, X: np.array) -> np.array:
        """
        Desintegrates X returning its difference of ``self.order[1]`` order and recording initial values to ``self.diff_vals`` for invertability

        Parameters
        ----------
        X: np.array, input data

        Returns
        -------
        X: np.array, difference of ``self.order[1]`` order of X

        """
        n = len(X)
        i_order = self.order[1]
        target_lenth = n - i_order - self.n_ar
        self.diff_vals = np.zeros((target_lenth, i_order))
        for i in range(i_order):
            self.diff_vals[:, i] = np.diff(X, n=i)[self.n_ar + 1:self.n_ar + target_lenth + 1]
        X = np.diff(X, n=i_order)
        return X

    def _integrate(self, X: np.array) -> np.array:
        """
        Reverse transformation of ``self._desintegrate(X)``, restores initial order based on ``self.diff_vals``

        Parameters
        ----------
        X: np.array, input data

        Returns
        -------
        np.array, integrated time series

        """
        for i in range(self.order[1]):
            X = np.concatenate([self.diff_vals[:, [-i-1]], X], axis=1).cumsum(axis=1)
        return X

    def _set_params(self, model: MLEModel, x: np.array):
        """
        Extracts fitted model parameters for easier access

        Parameters
        ----------
        model: MLEModel, fitted model
        x: np.array, training series used to calculate residuals

        """
        self.errors_ = model.get_errors(x)
        self.mu_ = model.mu
        self.phi_ = model.phi
        self.theta_ = model.theta
        self.model = model

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Fit the estimator.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), train sample.

        y : pd.DataFrame, Used to store the predict feature names and prediction horizon.

        Returns
        -------
        self : ARIMAForecaster
            Returns self.

        """
        len_stored_values = self.n_ar + self.order[1]
        self.last_train_values_ = X.iloc[-len_stored_values:] if len_stored_values > 0 else X.iloc[:0]
        np_x = X.to_numpy().flatten()
        np_x = self._deintegrate(np_x)
        model = MLEModel((self.n_ar, self.n_ma), self.method)
        model.fit(np_x)
        self._set_params(model, np_x)
        super().fit(X, y)
        return self

    def _extend_x_test(self, X: pd.DataFrame) -> (pd.DataFrame, np.array):
        """
        If test time series directly follows the train one, adds last values of train observations and errors for ARIMA forecast.
        Otherwise assumes previous observations equal to the first one in test time series.
        Future errors are assumed to be 0.0.

        Parameters
        ----------
        X: pd.DataFrame, test time series

        Returns
        -------
        X: pd.DataFrame, extended time series required for predictions
        errors: np.array, error forecast required for predictions
        """
        n = len(X)
        train_test_diff = X.index.min().start_time - self.last_train_values_.index.max().end_time
        if train_test_diff.value == 1:
            X = pd.concat([self.last_train_values_, X])
            errors = self.errors_[-self.n_ma:]
        else:
            last_index = pd.period_range(periods=self.n_ar + self.order[1] + 1, end=X.index[0])[:-1]
            last_values = pd.DataFrame([X.iloc[0].values[0]] * len(last_index), index=last_index, columns=X.columns)
            X = pd.concat([last_values, X])
            errors = np.zeros(self.n_ma)
        return X, errors

    def _predict(self, X: pd.DataFrame) -> np.array:
        """
        Create a numpy array of predictions.

        Parameters
        ----------
        X: pd.DataFrame, shape (n_samples, 1), required
            The time series on which to predict.

        Returns
        -------
        np.array
        """
        n = len(X)
        X, errors = self._extend_x_test(X)
        np_x = X.values.flatten()
        np_x = self._deintegrate(np_x)
        errors = _arma_insample_errors(np_x, errors, self.mu_, self.phi_, self.theta_)

        res = [_arma_forecast(n=self.horizon_,
                              x0=np_x[i:i+self.n_ar],
                              eps0=errors[i:i+self.n_ma],
                              mu=self.model.mu,
                              phi=self.model.phi,
                              theta=self.model.theta
                              )
               for i in range(1, n+1)]
        y_pred = self._integrate(np.array(res))

        return y_pred[:, self.order[1]:]
