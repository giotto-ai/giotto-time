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

    def __init__(self, model_form, loss=mean_squared_error):
        self.model_form = model_form
        self.loss = loss

    def fit(
        self, time_series: pd.DataFrame, x0: list, method: str = "BFGS"
    ) -> TrendModel:
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

        res = minimize(prediction_error, x0, method=method, options={"disp": False})

        self.model_weights_ = res["x"]

        self.t0_ = time_series.index[0]
        freq = time_series.index.freq
        if freq is not None:
            self.period_ = freq
        else:
            self.period_ = time_series.index[1] - time_series.index[0]
            # raise warning

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

        ts = (time_series.index - self.t0_) / self.period_

        predictions = pd.Series(
            index=time_series.index,
            data=[self.model_form(t, self.model_weights_) for t in ts],
        )

        return time_series.sub(predictions, axis=0)
