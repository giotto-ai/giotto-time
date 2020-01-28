from typing import Callable

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_is_fitted

from giottotime.utils.trends import TRENDS


class TrendForecaster(BaseEstimator, RegressorMixin):
    """Trend forecasting model

    This estimator optimizes a trend function on train data and will forecast using this trend function with optimized
    parameters.

    Parameters
    ----------
    trend : string,
        The kind of trend removal to apply.
        Supported trends: ['polynomial', 'exponential']

    trend_x0 : np.array,
        Initialisation parameters passed to the trend function

    loss : Callable,
        Loss function

    method : string,
        Loss function optimisation method

    """

    def __init__(
        self,
        trend: str,
        trend_x0: np.array,
        loss: Callable = mean_squared_error,
        method: str = "BFGS",
    ):
        self.trend = trend
        self.trend_x0 = trend_x0
        self.loss = loss
        self.method = method

    def fit(self, X, y=None):
        """Fit the estimator.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Input data.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """

        if self.trend not in TRENDS:
            raise ValueError(
                "The trend '%s' is not supported. Supported "
                "trends are %s." % (self.trend, list(sorted(TRENDS)))
            )

        self.best_trend_params_ = minimize(
            lambda opt: self.loss(
                X.values, [TRENDS[self.trend](t, opt) for t in range(0, X.shape[0])]
            ),
            self.trend_x0,
            method=self.method,
            options={"disp": False},
        )["x"]

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Using the fitted polynomial, predict the values starting from ``X``.

        Parameters
        ----------
        X: pd.DataFrame, shape (n_samples, 1), required
            The time series on which to predict.

        Returns
        -------
        predictions : pd.DataFrame, shape (n_samples, 1)
            The output predictions.

        Raises
        ------
        NotFittedError
            Raised if the model is not fitted yet.

        """
        check_is_fitted(self)

        predictions = TRENDS[self.trend](X.values, self.best_trend_params_)
        return predictions
