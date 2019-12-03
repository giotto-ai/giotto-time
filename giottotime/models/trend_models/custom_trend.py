from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

import pandas.util.testing as testing

import numpy as np


class CustomTrend:
    def __init__(self, model_form, inital_params, loss=mean_squared_error):
        self.loss = loss
        self.model_form = model_form
        self.inital_params = inital_params

    def fit(self, time_series):
        def prediction_error(model_params):
            predictions = [ self.model_form( t, model_params ) for t in range( 0, time_series.shape[0] ) ]
            return self.loss(time_series.values, predictions)

        res = minimize(prediction_error, self.inital_params, method='Powell', options={'disp': False})
        self.model_params_ = res['x']
        return self

    def predict(self, t):
        #check fit run
        return self.model_form( t, self.model_params_ )
