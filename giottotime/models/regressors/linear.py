from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import numpy as np

from sklearn.linear_model import LinearRegression


class LinearRegressor:
    def __init__(self, loss=mean_squared_error): #weight_initialization_rule = lambda X, y: np.zeros(X.shape[1]) ):
        self.loss = loss
        self.model_weights = None

    def fit(self, X, y, disp=False, **kwargs):
        def prediction_error(model_weights):
            predictions = [model_weights[0] + np.dot(model_weights[1:], row) for row in X.values]
            return self.loss(y, predictions)

        if 'r2_seed' in kwargs:
            lm = LinearRegression(fit_intercept=True).fit(X, y)
            self.r2_seed = [lm.intercept_] + list(lm.coef_)
            kwargs['x0'] = self.r2_seed
            print(kwargs['x0'])
            del kwargs['r2_seed']

        if not 'x0' in kwargs:
            kwargs['x0'] = np.zeros(X.shape[1]+1) #weight_initialization_rule(X, y) np.zeros(X.shape[1]+1)
        else:
            #assert (len(kwargs['x0']) + ) == (X.shape[1] + 1))
            kwargs['x0'] = kwargs['x0'] + [0]*(X.shape[1]+1 - len(kwargs['x0']))

        res = minimize(prediction_error, **kwargs)

        self.model_weights = res['x']
        print(self.model_weights)

        return self

    def predict(self, X):
        return self.model_weights[0] + np.dot(X, self.model_weights[1:])