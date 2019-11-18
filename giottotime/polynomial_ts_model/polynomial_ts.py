from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

import pandas.util.testing as testing

import numpy as np

class Polynomial_ts:
    def __init__(self, order, loss=mean_squared_error):
        self.order = order
        self.loss = loss

    def fit(self, time_series):
        def prediction_error(model_weights):
            index_len = time_series.shape[0]
            p = np.poly1d(model_weights)
            predictions = [ p(t) for t in range( 0, index_len ) ]
            return self.loss(time_series.values, predictions)

        model_weights = np.zeros(self.order)
        res = minimize(prediction_error, model_weights, method='BFGS', options={'disp': False})
        self.model_weights = res['x']
        return self

    def predict(self, X):
        predictions = pd.DataFrame(index=X.index)
        return predictions

if __name__ == "__main__":
    pts = Polynomial_ts(2)
    testing.N, testing.K = 200, 1

    ts = testing.makeTimeDataFrame(freq='MS')

    print(ts)

    pts.fit(ts)

    print(pts.model_weights)

#
