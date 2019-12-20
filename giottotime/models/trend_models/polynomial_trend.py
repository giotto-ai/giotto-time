from typing import Callable

from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from .base import TrendModel


class PolynomialTrend(TrendModel):
    """A model for fitting, predicting and removing an polynomial trend from a time
    series.

    Parameters
    ----------
    order : int, required
        The order of the polynomial.

    loss : Callable, optional, default: ``mean_squared_error``
        The loss function to use when fitting the model. The loss function must
         accept y_true, y_pred and return a single real number.

    method : str, optional, default: ``'BFGS``
        The method to use in order to minimize the loss function.

    """

    def __init__(
        self, order: int = 2, loss: Callable = mean_squared_error, method: str = "BFGS"
    ):
        self.order = order
        self.loss = loss
        self.method = method

    def fit(self, ts: pd.DataFrame,) -> "PolynomialTrend":
        """Fit the model on the ``time_series``, with respect to the provided ``loss``
        and using the provided ``method``. In order to see which methods are available,
        please check the 'scipy' `documentation
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.

        Parameters
        ----------
        ts: pd.DataFrame, shape (n_samples, n_features), required
            The time series on which to fit the model.

        Returns
        -------
        self : TrendModel
            The fitted object.

        """

        def prediction_loss(weights: np.ndarray) -> float:
            p = np.poly1d(weights)
            predictions = [p(t) for t in range(0, ts.shape[0])]
            return self.loss(ts.values, predictions)

        model_weights = np.zeros(self.order + 1)

        res = minimize(
            prediction_loss, model_weights, method=self.method, options={"disp": False}
        )

        self.model_weights_ = res["x"]
        self.t0_ = ts.index[0]
        freq = ts.index.freq

        if freq is not None:
            self.period_ = freq
        else:
            self.period_ = ts.index[1] - ts.index[0]

        return self

    def predict(self, ts: pd.DataFrame) -> pd.DataFrame:
        """Using the fitted polynomial, predict the values starting from ``X``.

        Parameters
        ----------
        ts: pd.DataFrame, shape (n_samples, 1), required
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

        p = np.poly1d(self.model_weights_)
        predictions = p(ts.values)
        return predictions

    def transform(self, ts: pd.DataFrame) -> pd.DataFrame:
        """Transform the ``time_series`` by removing the trend.

        Parameters
        ----------
        ts: pd.DataFrame, shape (n_samples, 1), required
            The time series to transform.

        Returns
        -------
        ts_t : pd.DataFrame, shape (n_samples, n_features)
            The transformed time series, without the trend.

        """
        check_is_fitted(self)

        p = np.poly1d(self.model_weights_)
        time_steps = (ts.index - self.t0_) / self.period_

        predictions = pd.Series(index=ts.index, data=[p(t) for t in time_steps])

        return ts.sub(predictions, axis=0)
