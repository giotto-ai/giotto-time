from random import random

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats

from giottotime.models.regressors.linear_regressor import LinearRegressor


class TestLinearRegressor:
    def test_linear_regressor(self):
        train, test = train_test_dataframe()

        predictions = compute_predictions_for_train_test(train, test)
        expected = compute_expectation_from_test(test)

        np.testing.assert_array_almost_equal(predictions, expected, decimal=2)

    @settings(deadline=None)
    @given(arrays(dtype=float, shape=(100, 1), elements=floats(-10000000, 10000000)))
    def test_linear_regressor_random_array(self, random_array):
        train, test = train_test_dataframe(random_array)

        predictions = compute_predictions_for_train_test(train, test)
        expected = compute_expectation_from_test(test)

        np.testing.assert_array_almost_equal(predictions, expected, decimal=0)


def train_test_dataframe(
    random_array: np.ndarray = None,
) -> (pd.DataFrame, pd.DataFrame):
    random_array = (
        random_array if random_array is not None else [random() for _ in range(100)]
    )

    a1, a2, b = random() * 10, random() * 100, 2 * (1 - random())

    df = pd.DataFrame()
    df["x1"] = list(range(100))
    df["x2"] = random_array
    df["y"] = [b + a1 * t for t in range(100)]
    df["y"] = df["y"] + a2 * df["x2"]

    train = df[:90]
    test = df[90:]

    return train, test


def compute_predictions_for_train_test(
    train: pd.DataFrame, test: pd.DataFrame
) -> np.ndarray:
    lr = LinearRegressor()

    lr.fit(train[["x1", "x2"]], train["y"], x0=[0, 0, 0])

    preds_y = lr.predict(test[["x1", "x2"]])
    preds_y = preds_y / np.sum(preds_y)

    return preds_y


def compute_expectation_from_test(test: pd.DataFrame) -> np.ndarray:
    test_y = test["y"].values
    test_y = test_y / np.sum(test_y)
    return test_y
