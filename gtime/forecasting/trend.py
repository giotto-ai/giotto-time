from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_is_fitted

from gtime.utils.trends import TRENDS


class TrendForecaster(BaseEstimator, RegressorMixin):
    """Trend forecasting model.

    This estimator optimizes a trend function on train data and will forecast using this trend function with optimized
    parameters.

    Parameters
    ----------
    trend : ``"polynomial"`` | ``"exponential"``, required
        The kind of trend removal to apply.

    trend_x0 : np.array, required
        Initialisation parameters passed to the trend function

    loss : Callable, optional, default: ``mean_squared_error``
        Loss function to minimize.

    method : str, optional, default: ``"BFGS"``
        Loss function optimisation method

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gtime.model_selection import horizon_shift, FeatureSplitter
    >>> from gtime.forecasting import TrendForecaster
    >>>
    >>> X = pd.DataFrame(np.random.random((10, 1)), index=pd.date_range("2020-01-01", "2020-01-10"))
    >>> y = horizon_shift(X, horizon=2)
    >>> X_train, y_train, X_test, y_test = FeatureSplitter().transform(X, y)
    >>>
    >>> tf = TrendForecaster(trend='polynomial', trend_x0=np.zeros(2))
    >>> tf.fit(X_train).predict(X_test)
    array([[0.39703029],
           [0.41734957]])

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

    def fit(self, X: pd.DataFrame, y=None) -> "TrendForecaster":
        """Fit the estimator.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), required
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
        print([TRENDS[self.trend](t, 111) for t in range(0, X.shape[0])])
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
