import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings, HealthCheck
from sklearn.compose import make_column_selector

from gtime.compose import FeatureCreation
from gtime.feature_extraction import Shift, MovingAverage
from gtime.model_selection import horizon_shift
from gtime.model_selection.splitters import FeatureSplitter
from gtime.utils.hypothesis.feature_matrices import X_y_matrices

# TODO: refactor, make hypothesis generator instead of a full pipeline
from gtime.utils.hypothesis.time_indexes import giotto_time_series

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

horizon = 4


class TestFeatureSplitter:
    def test_constructor(self):
        FeatureSplitter()

    @given(st.text().filter(lambda x: x != "any"))
    def test_constructor_wrong_parameter(self, drop_na_mode: str):
        with pytest.raises(ValueError):
            FeatureSplitter(drop_na_mode)

    @settings(suppress_health_check=(HealthCheck.too_slow,))
    @given(
        X_y_matrices(
            horizon=horizon, df_transformer=df_transformer, allow_nan_infinity=False,
        )
    )
    def test_transform(self, X_y):
        X, y = X_y
        feature_splitter = FeatureSplitter()
        X_train, y_train, X_test, y_test = feature_splitter.transform(X, y)

        assert X_train.shape[0] == max(0, X.shape[0] - 2 - horizon)
        assert y_train.shape[0] == X_train.shape[0]
        assert X_test.shape[0] == min(max(0, X.shape[0] - 2), horizon)
        assert y_test.shape[0] == X_test.shape[0]


class TestHorizonShift:
    @given(
        giotto_time_series(min_length=10, allow_infinity=False, allow_nan=False),
        st.integers(1, 8),
    )
    def test_horizon_int(self, time_series, horizon):
        y_shifted = horizon_shift(time_series, horizon)
        assert y_shifted.shape[1] == horizon

        # Check first line of y_shifted
        for i in range(1, horizon + 1):
            assert time_series.iloc[i, 0] == y_shifted.iloc[0, i - 1]

    @given(
        giotto_time_series(min_length=10, allow_infinity=False, allow_nan=False),
        st.sets(elements=st.integers(1, 8), min_size=1, max_size=8),
    )
    def test_horizon_list(self, time_series, horizon):
        y_shifted = horizon_shift(time_series, horizon)
        assert y_shifted.shape[1] == len(horizon)

        # Check first line of y_shifted
        for i, elem in enumerate(horizon):
            assert time_series.iloc[elem, 0] == y_shifted.iloc[0, i]
