from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

import pandas.util.testing as testing

import numpy as np

class CustomTrendForm_ts:
    def __init__(self, model_form, inital_params, loss=mean_squared_error):
        self.loss = loss
        self.model_form = model_form
        self.inital_params = inital_params

    def fit(self, time_series):
        def prediction_error(model_params):
            predictions = [ self.model_form( t, model_params ) for t in range( 0, time_series.shape[0] ) ]
            return self.loss(time_series.values, predictions)

        res = minimize(prediction_error, self.inital_params, method='Powell', options={'disp': False})
        self.model_params = res['x']
        return self

    def predict(self, t):
        #check fit run
        #predictions = pd.DataFrame(index=X.index, data=[ p(t) for t in range( 0, X.shape[0] )   ])
        return self.model_form( t, self.model_params )

    def de_trend(self, time_series):
        #check fit run
        predictions = pd.DataFrame( index=time_series.index, data=blasdfasdfasdfa )
        return time_series - predictions[0]





if __name__ == "__main__":
    import matplotlib.pyplot as plt

    pts = CustomTrendForm_ts( lambda t, L : L[0] + L[1]*t + L[2]*np.sin(L[3]*t), [1, 1, 300, 0.25] )
    testing.N, testing.K = 200, 1

    ts = testing.makeTimeDataFrame(freq='MS')

    #ts['A'] *= 0

    ts['B'] = 1
    ts['B'] = ts['B'].cumsum()

    ts['A'] = 300*ts['A'] + 10*ts['B'] + ts['B'].apply( lambda t : 500*np.sin(0.25*t) )

    ts = ts.drop('B', axis=1)

    pts.fit(ts)

    print(pts.model_params)

    #print(pts.predict(100))

    ts['preds'] = [ pts.predict(t) for t in range(len(ts)) ]

    ts.plot()
    plt.show()


#
