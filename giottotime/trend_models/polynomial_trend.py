from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

import numpy as np
import pandas as pd

class PolynomialTrend(TrendModel):
    """
    A model for fitting, predicting and removing an polynomial trend from a time series.

    Parameters
    ----------

    loss: Callable, default: mean_squared_error
    must accept y_true, y_pred and return a single real number.

    """
    def __init__(self, order, loss=mean_squared_error):
        self.order = order
        self.loss = loss

    def fit(self, time_series, method="BFGS"):
        def prediction_error(model_weights):
            p = np.poly1d(model_weights)
            predictions = [ p(t) for t in range( 0, time_series.shape[0] ) ]
            return self.loss(time_series.values, predictions)

        model_weights = np.zeros(self.order)
        res = minimize(prediction_error, model_weights, method=method, options={'disp': False})
        self.model_weights = res['x']
        return self

    def predict(self, t):
        p = np.poly1d(self.model_weights)
        return p(t)

    def transform(self, time_series):
        #check fit run
        p = np.poly1d( self.model_weights )
        predictions = pd.DataFrame( index=time_series.index, data=[ p(t) for t in range( 0, time_series.shape[0] ) ] )
        return time_series - predictions[0]


class LinearTrend(PolynomialTrend):

    def __init__(self, loss=mean_squared_error):
        super.__init__()

#
