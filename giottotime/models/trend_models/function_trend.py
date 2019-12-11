import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

from ..trend_models.base import TrendModel


class FunctionTrend(TrendModel):
    """A model for fitting, predicting and removing an custom functional trend
    from a time series. The transformed time series created will be trend
    stationary with respect to the specific function. To have more details,
    you can check this `link <https://en.wikipedia.org/wiki/Trend_stationary>`_.

    Parameters
    ----------
    loss : ``Callable``, optional, (default=``mean_squared_error``).
        The loss function to use when fitting the model. The loss function must
        accept y_true, y_pred and return a single real number.

    """

    def __init__(
        self, model_form, x0: np.ndarray, loss=mean_squared_error, method: str = "BFGS"
    ):
        self.x0 = x0
        self.model_form = model_form
        self.loss = loss
        self.method = method

    def fit(self, time_series: pd.DataFrame) -> TrendModel:
        """Fit the model on the ``time_series``, with respect to the provided
        ``loss`` and using the provided ``method``. In order to see which
        methods are available, please check the 'scipy' `documentation
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.

        Parameters
        ----------
        time_series : ``pd.DataFrame``, required.
            The time series on which to fit the model.

        x0 : ``list``.

        method : ``str``, optional, (default=``'BFGS``).
            The method to use in order to minimize the loss function.

        Returns
        -------
        self : ``TrendModel``
            The fitted object.

        """

        def prediction_error(model_weights):
            predictions = [
                self.model_form(t, model_weights)
                for t in range(0, time_series.shape[0])
            ]
            return self.loss(time_series.values, predictions)

        res = minimize(
            prediction_error, self.x0, method=self.method, options={"disp": False}
        )

        self.model_weights_ = res["x"]
        return self

    def predict(self, t):
        """Using the fitted model, predict the values starting from ``X``.

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
        # check fit run
        return self.model_form(t, self.model_weights_)

    def transform(self, time_series):
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
        # check fit run
        predictions = pd.Series(
            index=time_series.index,
            data=[
                self.model_form(t, self.model_weights_)
                for t in range(0, time_series.shape[0])
            ],
        )

        return time_series.sub(predictions, axis=0)
