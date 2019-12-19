import pytest

import hypothesis.strategies as st
from hypothesis import given, settings, HealthCheck

from giottotime.feature_creation import ShiftFeature, MovingAverageFeature
from giottotime.model_selection.feature_splitters import FeatureSplitter
from giottotime.utils.hypothesis.feature_matrices import X_y_matrices


FEATURES = [
    ShiftFeature(0, "0"),
    ShiftFeature(1, "1"),
    MovingAverageFeature(3, "3"),
]

HORIZON = 4


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
            horizon=HORIZON, time_series_features=FEATURES, allow_nan_infinity=False,
        )
    )
    def test_transform(self, X_y):
        X, y = X_y
        feature_splitter = FeatureSplitter()
        X_train, y_train, X_test, y_test = feature_splitter.transform(X, y)

        assert X_train.shape[0] == max(0, X.shape[0] - 2 - HORIZON)
        assert y_train.shape[0] == X_train.shape[0]
        assert X_test.shape[0] == min(max(0, X.shape[0] - 2), HORIZON)
        assert y_test.shape[0] == X_test.shape[0]
