from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

import numpy as np
import pandas as pd

from giottotime.models.trend_models.base import TrendModel


# TODO: finish this class
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

    def fit(self, time_series, x0, method="BFGS"):
        def prediction_error(model_weights):
            predictions = [self.model_form(t, model_weights) for t in
                           range(0, time_series.shape[0])]
            return self.loss(time_series.values, predictions)

        res = minimize(prediction_error, x0, method=method,
                       options={'disp': False})

        self.model_weights_ = res['x']
        return self

    def predict(self, t):
        # check fit run
        return self.model_form(t, self.model_weights_)

    def transform(self, time_series):
        # check fit run
        predictions = pd.DataFrame(index=time_series.index, data=[
            self.model_form(t, self.model_weights_) for t in
            range(0, time_series.shape[0])])
        return time_series - predictions[0]
