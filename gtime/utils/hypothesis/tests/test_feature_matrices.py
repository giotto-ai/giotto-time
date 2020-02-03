from typing import Tuple

import numpy as np
import pandas as pd
from hypothesis import given
from sklearn.compose import ColumnTransformer, make_column_selector

from gtime.compose import FeatureCreation
from gtime.feature_extraction import Shift, MovingAverage
from gtime.utils.hypothesis.feature_matrices import (
    X_y_matrices,
    X_matrices,
    y_matrices,
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
