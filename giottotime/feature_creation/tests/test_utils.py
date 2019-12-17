import numpy as np
import pandas.util.testing as testing

from giottotime.feature_creation.utils import _get_non_nan_values


def test_correctly_remove_nan_values():
    testing.N, testing.K = 500, 50

    df = testing.makeTimeDataFrame(freq="MS")
    X = df.iloc[:, :40]
    y = df.iloc[:, 40:]

    x_nan_mat = np.random.random(X.shape) < 0.3

    X_with_nans = X.mask(x_nan_mat)

    X_non_na, y_non_na = _get_non_nan_values(X_with_nans, y)

    assert not X_non_na.isnull().values.any()
    assert not y_non_na.isnull().values.any()

    assert X_non_na.shape[0] == y_non_na.shape[0]
    assert X_non_na.shape[1] == X.shape[1]
    assert y_non_na.shape[1] == y.shape[1]


def test_without_nan_values():
    testing.N, testing.K = 500, 50

    df = testing.makeTimeDataFrame(freq="MS")
    X = df.iloc[:, :40]
    y = df.iloc[:, 40:]

    X_non_na, y_non_na = _get_non_nan_values(X, y)

    assert not X_non_na.isnull().values.any()
    assert not y_non_na.isnull().values.any()

    assert X_non_na.shape[0] == y_non_na.shape[0]
    assert X_non_na.shape == X.shape
    assert y_non_na.shape == y.shape
