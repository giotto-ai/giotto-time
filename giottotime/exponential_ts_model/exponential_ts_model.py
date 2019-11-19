from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

import pandas.util.testing as testing

import numpy as np

class Exponential_ts:
    def __init__(self, loss=mean_squared_error):
        self.loss = loss

    def fit(self, time_series):
        def prediction_error(model_exponent):

            predictions = [ np.exp(t*model_exponent) for t in range( 0, time_series.shape[0] ) ]
            return self.loss(time_series.values, predictions)

        model_exponent = 0
        res = minimize(prediction_error, np.array([model_exponent]), method='BFGS', options={'disp': False})
        self.model_exponent = res['x'][0]
        return self

    def predict(self, t):
        #check fit run
        #predictions = pd.DataFrame(index=X.index, data=[ p(t) for t in range( 0, X.shape[0] )   ])
        return np.exp(t*self.model_exponent)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    pts = Exponential_ts()
    testing.N, testing.K = 200, 1

    ts = testing.makeTimeDataFrame(freq='MS')

    ts['A'] *= 10000

    ts['B'] = 1
    ts['B'] = ts['B'].cumsum()
    ts['A'] = ts['A'] + 10*ts['B'] + 10*ts['B'].pow(2) + 0.01*np.exp(ts['B']/10)

    ts = ts.drop('B', axis=1)

    pts.fit(ts)

    print(pts.model_exponent)

    #print(pts.predict(100))

    ts['preds'] = [ pts.predict(t) for t in range(len(ts)) ]

    ts.plot()
    plt.show()


#
