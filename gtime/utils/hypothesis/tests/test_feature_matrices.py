from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis.strategies import data
from sklearn.compose import make_column_selector

from gtime.compose import FeatureCreation
from gtime.feature_extraction import Shift, MovingAverage
from gtime.utils.hypothesis.feature_matrices import (
    X_y_matrices,
    X_matrices,
    y_matrices,
    numpy_X_y_matrices,
    numpy_X_matrix,
)
from gtime.utils.hypothesis.general_strategies import (
    shape_X_y_matrices,
    ordered_pair,
    shape_matrix,
)

df_transformer = FeatureCreation(
    [
        ("shift_0", Shift(0), make_column_selector(dtype_include=np.number)),
        ("shift_1", Shift(1), make_column_selector(dtype_include=np.number)),
        (
            "moving_average_3",
            MovingAverage(window_size=3),
            make_column_selector(dtype_include=np.number),
        ),
    ]
)


class TestXyMatrices:
    @given(X_y_matrices(horizon=3, df_transformer=df_transformer))
    def test_X_shape_correct(self, X_y: Tuple[pd.DataFrame, pd.DataFrame]):
        X, y = X_y
        assert X.shape[1] == len(df_transformer.transformers_)

    @given(X_y_matrices(horizon=3, df_transformer=df_transformer))
    def test_shape_consistent(self, X_y: Tuple[pd.DataFrame, pd.DataFrame]):
        X, y = X_y
        assert X.shape[0] == y.shape[0]

    @given(
        X_y_matrices(horizon=3, df_transformer=df_transformer, allow_nan_infinity=False)
    )
    def test_allow_nan_false(self, X_y: Tuple[pd.DataFrame, pd.DataFrame]):
        X, y = X_y
        assert X.dropna(axis=0, how="any").shape[0] == max(X.shape[0] - 2, 0)
        assert y.dropna(axis=0, how="any").shape[0] == max(y.shape[0] - 3, 0)


class TestXMatrices:
    @given(X_matrices(df_transformer=df_transformer))
    def test_X_shape_correct(self, X: pd.DataFrame):
        assert X.shape[1] == len(df_transformer.transformers_)

    @given(X_matrices(df_transformer=df_transformer))
    def test_X_column_names(self, X: pd.DataFrame):
        assert list(X.columns) == [
            "shift_0__time_series__Shift",
            "shift_1__time_series__Shift",
            "moving_average_3__time_series__MovingAverage",
        ]

    @given(X_matrices(df_transformer=df_transformer, allow_nan_infinity=False))
    def test_allow_nan_false(self, X: pd.DataFrame):
        assert X.dropna(axis=0, how="any").shape[0] == max(X.shape[0] - 2, 0)


class TestYMatrices:
    @given(y_matrices(horizon=1))
    def test_horizon_1(self, y: pd.DataFrame):
        assert y.shape[1] == 1

    @given(y_matrices(horizon=2))
    def test_horizon_2(self, y: pd.DataFrame):
        assert y.shape[1] == 2

    @given(y_matrices(horizon=3))
    def test_horizon_3(self, y: pd.DataFrame):
        assert y.shape[1] == 3

    @given(y_matrices(horizon=3))
    def test_y_column_names(self, y: pd.DataFrame):
        assert list(y.columns) == ["y_1", "y_2", "y_3"]

    @given(y_matrices(horizon=3, allow_nan_infinity=False))
    def test_allow_nan_false(self, y: pd.DataFrame):
        assert y.dropna(axis=0, how="any").shape[0] == max(y.shape[0] - 3, 0)


class TestNumpyXyMatrices:
    @given(data(), shape_X_y_matrices())
    def test_input_as_tuples(self, data, shape_X_y):
        X, y = data.draw(numpy_X_y_matrices(shape_X_y))
        assert X.shape == shape_X_y[0]
        assert y.shape == shape_X_y[1]

    @given(data())
    def test_input_as_strategy(self, data):
        data.draw(numpy_X_y_matrices(shape_X_y_matrices()))

    @given(data())
    def test_error_shape_0_smaller_shape_1(self, data):
        with pytest.raises(ValueError):
            data.draw(numpy_X_y_matrices([[10, 20], [10, 1]]))

    @given(data())
    def test_error_shape_0_different(self, data):
        with pytest.raises(ValueError):
            data.draw(numpy_X_y_matrices([[10, 5], [4, 1]]))

    @given(data(), shape_X_y_matrices(), ordered_pair(32, 47))
    def test_min_max_values(self, data, shape_X_y, min_max_values):
        min_value, max_value = min_max_values
        X, y = data.draw(
            numpy_X_y_matrices(shape_X_y, min_value=min_value, max_value=max_value)
        )
        assert X.min() >= min_value
        assert y.min() >= min_value
        assert X.max() <= max_value
        assert y.max() <= max_value

    @given(data(), shape_X_y_matrices())
    def test_no_nan(self, data, shape_X_y):
        X, y = data.draw(
            numpy_X_y_matrices(shape_X_y, allow_nan=False, allow_infinity=True)
        )
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()

    @given(data(), shape_X_y_matrices())
    def test_no_infinity(self, data, shape_X_y):
        X, y = data.draw(
            numpy_X_y_matrices(shape_X_y, allow_nan=True, allow_infinity=False)
        )
        assert not np.isinf(X).any()
        assert not np.isinf(y).any()


class TestNumpyXMatrix:
    @given(data(), shape_matrix())
    def test_input_as_tuples(self, data, shape):
        X = data.draw(numpy_X_matrix(shape))
        assert X.shape == shape

    @given(data())
    def test_input_as_strategy(self, data):
        data.draw(numpy_X_matrix(shape_matrix()))

    @given(data())
    def test_error_shape_0_smaller_shape_1(self, data):
        with pytest.raises(ValueError):
            data.draw(numpy_X_matrix([10, 20]))

    @given(data(), shape_matrix(), ordered_pair(32, 47))
    def test_min_max_values(self, data, shape, min_max_values):
        min_value, max_value = min_max_values
        X = data.draw(numpy_X_matrix(shape, min_value=min_value, max_value=max_value))
        assert X.min() >= min_value
        assert X.max() <= max_value

    @given(data(), shape_matrix())
    def test_no_nan(self, data, shape):
        X = data.draw(numpy_X_matrix(shape, allow_nan=False, allow_infinity=True))
        assert not np.isnan(X).any()

    @given(data(), shape_matrix())
    def test_no_infinity(self, data, shape):
        X = data.draw(numpy_X_matrix(shape, allow_nan=True, allow_infinity=False))
        assert not np.isinf(X).any()
