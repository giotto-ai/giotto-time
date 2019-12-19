from typing import Tuple

import pandas as pd
from hypothesis import given

from giottotime.feature_creation import ShiftFeature, MovingAverageFeature
from giottotime.utils.hypothesis.feature_matrices import (
    X_y_matrices,
    X_matrices,
    y_matrices,
)

features = [
    ShiftFeature(0, "0"),
    ShiftFeature(1, "1"),
    MovingAverageFeature(3, "3"),
]


class TestXyMatrices:
    @given(X_y_matrices(horizon=1, time_series_features=features))
    def test_horizon_1(self, X_y: Tuple[pd.DataFrame, pd.DataFrame]):
        X, y = X_y
        assert y.shape[1] == 1

    @given(X_y_matrices(horizon=2, time_series_features=features))
    def test_horizon_2(self, X_y: Tuple[pd.DataFrame, pd.DataFrame]):
        X, y = X_y
        assert y.shape[1] == 2

    @given(X_y_matrices(horizon=3, time_series_features=features))
    def test_horizon_3(self, X_y: Tuple[pd.DataFrame, pd.DataFrame]):
        X, y = X_y
        assert y.shape[1] == 3

    @given(X_y_matrices(horizon=3, time_series_features=features))
    def test_X_column_names(self, X_y: Tuple[pd.DataFrame, pd.DataFrame]):
        X, y = X_y
        assert list(X.columns) == ["0", "1", "3"]

    @given(X_y_matrices(horizon=3, time_series_features=features))
    def test_X_shape_correct(self, X_y: Tuple[pd.DataFrame, pd.DataFrame]):
        X, y = X_y
        assert X.shape[1] == len(features)

    @given(X_y_matrices(horizon=3, time_series_features=features))
    def test_y_column_names(self, X_y: Tuple[pd.DataFrame, pd.DataFrame]):
        X, y = X_y
        assert list(y.columns) == ["y_1", "y_2", "y_3"]

    @given(X_y_matrices(horizon=3, time_series_features=features))
    def test_shape_consistent(self, X_y: Tuple[pd.DataFrame, pd.DataFrame]):
        X, y = X_y
        assert X.shape[0] == y.shape[0]

    @given(
        X_y_matrices(horizon=3, time_series_features=features, allow_nan_infinity=False)
    )
    def test_allow_nan_false(self, X_y: Tuple[pd.DataFrame, pd.DataFrame]):
        X, y = X_y
        assert X.dropna(axis=0, how="any").shape[0] == max(X.shape[0] - 2, 0)
        assert y.dropna(axis=0, how="any").shape[0] == max(y.shape[0] - 3, 0)


class TestXMatrices:
    @given(X_matrices(time_series_features=features))
    def test_X_shape_correct(self, X: pd.DataFrame):
        assert X.shape[1] == len(features)

    @given(X_matrices(time_series_features=features))
    def test_X_column_names(self, X: pd.DataFrame):
        assert list(X.columns) == ["0", "1", "3"]

    @given(X_matrices(time_series_features=features, allow_nan_infinity=False))
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
