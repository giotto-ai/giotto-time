from typing import Callable

from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

import numpy as np
import pandas as pd

from ..utils import check_is_fitted
from ..trend_models.base import TrendModel


class PolynomialTrend(TrendModel):
    """
    A model for fitting, predicting and removing an polynomial trend from a
    time series.

    Parameters
    ----------
    order : ``int``, required.
        The order of the polynomial.

    loss : ``Callable``, optional, (default=``mean_squared_error``).
        The loss function to use when fitting the model. The loss function must
         accept y_true, y_pred and return a single real number.

    """

    def __init__(self, order: int, loss: Callable = mean_squared_error):
        self.order = order
        self.loss = loss

    def fit(self, time_series: pd.DataFrame, method: str = "BFGS") -> TrendModel:
        """Fit the model on the ``time_series``, with respect to the provided
        ``loss`` and using the provided ``method``. In order to see which
        methods are available, please check the 'scipy' `documentation
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.

        Parameters
        ----------
        time_series : ``pd.DataFrame``, required.
            The time series on which to fit the model.

        method : ``str``, optional, (default=``'BFGS``).
            The method to use in order to minimize the loss function.

        Returns
        -------
        self : ``TrendModel``
            The fitted object.

        """

        def prediction_loss(weights: np.ndarray) -> float:
            p = np.poly1d(weights)
            predictions = [p(t) for t in range(0, time_series.shape[0])]
            return self.loss(time_series.values, predictions)

        model_weights = np.zeros(self.order)

        res = minimize(
            prediction_loss, model_weights, method=method, options={"disp": False}
        )

        self.model_weights_ = res["x"]
        self.t0_ = time_series.index[0]
        freq = time_series.index.freq
        if freq is not None:
            self.period_ = freq
        else:
            self.period_ = time_series.index[1] - time_series.index[0]
            # raise warning

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Using the fitted polynomial, predict the values starting from ``X``.

        Parameters
        ----------
        X : ``pd.DataFrame``, required.
            The time series on which to predict.

        Returns
        -------
        predictions : ``pd.DataFrame``
            The output predictions.

        Raises
        ------
        ``NotFittedError``
            Raised if the model is not fitted yet.

        """
        check_is_fitted(self)

        p = np.poly1d(self.model_weights_)
        return p(X.values)

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
        p = np.poly1d(self.model_weights_)

        trans_freq = time_series.index.freq
        if trans_freq is not None:
            trans_freq = trans_freq
        else:
            trans_freq = time_series.index[1] - time_series.index[0]
            # raise warning

        ts = (time_series.index - self.t0_) / self.period_

        predictions = pd.Series(index=time_series.index, data=[p(t) for t in ts])

        return time_series.sub(predictions, axis=0)
