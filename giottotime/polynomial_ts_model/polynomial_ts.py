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
            p = np.poly1d(model_weights)
            predictions = [ p(t) for t in range( 0, time_series.shape[0] ) ]
            return self.loss(time_series.values, predictions)

        model_weights = np.zeros(self.order)
        res = minimize(prediction_error, model_weights, method='BFGS', options={'disp': False})
        self.model_weights = res['x']
        return self

    def predict(self, t):
        #check fit run
        p = np.poly1d(self.model_weights)
        #predictions = pd.DataFrame(index=X.index, data=[ p(t) for t in range( 0, X.shape[0] )   ])
        return p(t)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    pts = Polynomial_ts(5)
    testing.N, testing.K = 200, 1

    ts = testing.makeTimeDataFrame(freq='MS')

    ts['A'] *= 10000

    ts['B'] = 1
    ts['B'] = ts['B'].cumsum()
    ts['A'] = ts['A'] + 10*ts['B'] + 10*ts['B'].pow(2) + 0.001*np.exp(ts['B']/10)

    ts = ts.drop('B', axis=1)

    pts.fit(ts)

    print(pts.model_weights)

    #print(pts.predict(100))

    ts['preds'] = [ pts.predict(t) for t in range(len(ts)) ]

    ts.plot()
    plt.show()





#
