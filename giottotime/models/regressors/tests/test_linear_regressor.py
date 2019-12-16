import numpy as np
import pandas as pd

from hypothesis import given, strategies as st

from pandas.util import testing as testing

from giottotime.models.regressors.linear_regressor import LinearRegressor

from random import random

class TestLinearRegressor:
    def test_linear_regressor_linear_regressor(self):
        a, b = random()*10, 2*(1-random())

        testing.N, testing.K = 100, 1
        df = pd.DataFrame()

        df['x1'] = list(range(100))
        df['y'] = [ a + b*t for t in range(100) ]

        train = df[:90]
        test = df[90:]

        lr = LinearRegressor()

        lr.fit(train[['x1']], train['y'], x0=[0, 0])

        preds_y = lr.predict(test[['x1']])
        test_y = test['y'].values

        #print(preds_y)
        #print(test_y)

        np.testing.assert_array_almost_equal(preds_y, test_y, decimal=2)

        return False


#
