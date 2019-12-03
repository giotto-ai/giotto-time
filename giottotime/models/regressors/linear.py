from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import numpy as np


class LinearModel:
    def __init__(self, loss=mean_squared_error):
        self.loss = loss
        self.model_weights = None

    def fit(self, X, y):
        def prediction_error(model_weights):
            predictions = [model_weights[0] + np.dot(model_weights[1:], row) for row in X]
            return self.loss(y, predictions)

        model_weights = np.zeros(X.shape[1]+1)
        res = minimize(prediction_error, model_weights, method='BFGS', options={'disp': False})
        self.model_weights = res['x']
        return self

    def predict(self, X):
        return self.model_weights[0] + np.dot(X, self.model_weights[1:])



#
