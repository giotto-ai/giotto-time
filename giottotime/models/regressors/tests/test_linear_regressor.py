import numpy as np
import pandas as pd

from hypothesis import given, strategies as st

from pandas.util import testing as testing
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from hypothesis import given, settings

from giottotime.models.regressors.linear_regressor import LinearRegressor

from random import random


class TestLinearRegressor:
    def test_linear_regressor_linear_regressor(self):
        a1, a2, b = random() * 10, random() * 100, 2 * (1 - random())

        testing.N, testing.K = 100, 1
        df = pd.DataFrame()

        df["x1"] = list(range(100))
        df["x2"] = [random() for k in range(100)]

        df["y"] = [b + a1 * t for t in range(100)]
        df["y"] = df["y"] + a2 * df["x2"]

        train = df[:90]
        test = df[90:]

        lr = LinearRegressor()

        lr.fit(train[["x1", "x2"]], train["y"], x0=[0, 0, 0])

        preds_y = lr.predict(test[["x1", "x2"]])
        test_y = test["y"].values

        np.testing.assert_array_almost_equal(preds_y, test_y, decimal=2)

    @settings(deadline=None)
    @given(arrays(dtype=float, shape=(100, 1), elements=floats(-10000000, 10000000)))
    def test_linear_regressor_linear_regressor_random_array(self, random_array):
        a1, a2, b = random() * 10, random() * 100, 2 * (1 - random())

        testing.N, testing.K = 100, 1
        df = pd.DataFrame()

        df["x1"] = list(range(100))
        df["x2"] = random_array

        df["y"] = [b + a1 * t for t in range(100)]
        df["y"] = df["y"] + a2 * df["x2"]

        train = df[:90]
        test = df[90:]

        lr = LinearRegressor()

        lr.fit(train[["x1", "x2"]], train["y"], x0=[0, 0, 0])

        preds_y = lr.predict(test[["x1", "x2"]])
        test_y = test["y"].values

        preds_y = preds_y / np.sum(preds_y)
        test_y = test_y / np.sum(test_y)

        print(preds_y)
        print(test_y)
        print("\n\n±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±\n\n")

        np.testing.assert_array_almost_equal(preds_y, test_y, decimal=0)


#
