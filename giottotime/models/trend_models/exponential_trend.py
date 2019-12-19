from typing import Callable

from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from .base import TrendModel


class ExponentialTrend(TrendModel):
    """A model for fitting, predicting and removing an exponential trend from a time
    series.

    Parameters
    ----------
    loss : Callable, optional, default: ``mean_squared_error``
        The loss function to use when fitting the model. The loss function must accept
        y_true, y_pred and return a single real number.

    method : str, optional, default: ``'BFGS``
            The method to use in order to minimize the loss function.

    """

    def __init__(self, loss: Callable = mean_squared_error, method: str = "BFGS"):
        self.loss = loss
        self.method = method

    def fit(self, time_series: pd.DataFrame) -> TrendModel:
        """Fit the model on the ``time_series``, with respect to the provided ``loss``
        and using the provided ``method``. In order to see which methods are available,
        please check the 'scipy' `documentation
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.

        Parameters
        ----------
        time_series: pd.DataFrame, shape (n_samples, 1), required
            The time series on which to fit the model.

        Returns
        -------
        self : TrendModel
            The fitted object.

        """

        def prediction_error(exponent):
            predictions = [np.exp(t * exponent) for t in range(0, time_series.shape[0])]
            return self.loss(time_series.values, predictions)

        model_exponent = 0
        res = minimize(
            prediction_error,
            np.array([model_exponent]),
            method=self.method,
            options={"disp": False},
        )

        self.model_exponent_ = res["x"][0]

        self.t0_ = time_series.index[0]
        freq = time_series.index.freq
        if freq is not None:
            self.period_ = freq
        else:
            self.period_ = time_series.index[1] - time_series.index[0]

        return self

    def predict(self, time_series: pd.DataFrame) -> pd.DataFrame:
        """Using the fitted model, predict the value starting from ``time_series``.

        Parameters
        ----------
        time_series: pd.DataFrame, shape (n_samples, 1), required
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

        predictions = np.exp(time_series * self.model_exponent_)
        return predictions

    def transform(self, time_series: pd.DataFrame) -> pd.DataFrame:
        """Transform the ``time_series`` by removing the trend.

        Parameters
        ----------
        time_series : ``pd.DataFrame``, required.
            The time series to transform.

        Returns
        -------
        transformed_time_series : ``pd.DataFrame``
            The transformed time series, without the trend.

        """
        ts = (time_series.index - self.t0_) / self.period_

        predictions = pd.Series(
            index=time_series.index,
            data=[np.exp(t * self.model_exponent_) for t in ts],
        )

        return time_series.sub(predictions, axis=0)
