from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

import numpy as np

from giottotime.models.trend_models.base import TrendModel


class ExponentialTrend(TrendModel):
    """
    A model for fitting, predicting and removing an exponential trend from a time series.

    Parameters
    ----------

    loss: Callable, default: mean_squared_error
    must accept y_true, y_pred and return a single real number.

    """
    def __init__(self, loss=mean_squared_error):
        self.loss = loss

    def fit(self, time_series, method="BFGS"):
        def prediction_error(model_exponent):
            predictions = [ np.exp(t*model_exponent) for t in range( 0, time_series.shape[0] ) ]
            return self.loss(time_series.values, predictions)

        model_exponent = 0
        res = minimize(prediction_error, np.array([model_exponent]), method=method, options={'disp': False})
        self.model_exponent = res['x'][0]
        return self

    def predict(self, t):
        #check fit run
        return np.exp(t*self.model_exponent)

    def transform(self, time_series):
        #check fit run
        predictions = pd.DataFrame( index=time_series.index, data=[ np.exp(t*self.model_exponent) for t in range( 0, time_series.shape[0] ) ] )
        return time_series - predictions[0]
