from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from ..trend_models.base import TrendModel


class ExponentialTrend(TrendModel):
    """A model for fitting, predicting and removing an exponential trend from a
     time series.

    Parameters
    ----------
    loss : ``Callable``, optional, (default=``mean_squared_error``).
        The loss function to use when fitting the model. The loss function must
        accept y_true, y_pred and return a single real number.

    """

    def __init__(self, loss=mean_squared_error):
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

        def prediction_error(exponent):
            predictions = [np.exp(t * exponent) for t in range(0, time_series.shape[0])]
            return self.loss(time_series.values, predictions)

        model_exponent = 0
        res = minimize(
            prediction_error,
            np.array([model_exponent]),
            method=method,
            options={"disp": False},
        )

        self.model_exponent_ = res["x"][0]
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Using the fitted model, predict the value starting from ``X``.

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
        check_is_fitted(self, attributes=["model_exponent_"])

        return np.exp(X * self.model_exponent_)

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
        predictions = pd.DataFrame(
            index=time_series.index,
            data=[
                np.exp(t * self.model_exponent_) for t in range(0, time_series.shape[0])
            ],
        )
        return time_series - predictions[0]
