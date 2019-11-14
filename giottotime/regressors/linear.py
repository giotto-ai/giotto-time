from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

#

import numpy as np

#

class LinearModel():
    def __init__(self):
        pass

    def fit(self, X, y, loss=mean_squared_error):
        print(X.shape, y.shape)
        coeffs = [0.0]*(X.shape[1] + 1)
        def pred_err(coeffs):
            loss_val = 0
            preds = []
            for row in X:
                preds.append(coeffs[0] + np.dot(coeffs[1:], row))
            l = loss(preds, y)
            return l

        res = minimize(pred_err, coeffs, method='BFGS', options={'disp': False})
        self._coeffs = res['x']
        return res

    def predict(self, q):
        return self._coeffs[0] + np.dot(self._coeffs[1:], q)

if __name__ == "__main__":
    X = [[100, 5], [100, 9]]
    y = [50, 100]

    lm = linearModel()
    q = lm.fit(X, y)

    print( lm.predict( [ 100, 10 ] ) )




#
