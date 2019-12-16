import pytest

import hypothesis.strategies as st
from hypothesis import given

from giottotime.feature_creation import ShiftFeature, MovingAverageFeature
from giottotime.model_selection.feature_splitters import FeatureSplitter
from giottotime.utils.hypothesis.feature_matrices import X_y_matrices


features = [
    ShiftFeature(1, "1"),
    ShiftFeature(2, "2"),
    MovingAverageFeature(3, "3"),
]


class TestFeatureSplitter:
    def test_constructor(self):
        FeatureSplitter()

    @given(st.text().filter(lambda x: x != "any"))
    def test_constructor_wrong_parameter(self, drop_na_mode: str):
        with pytest.raises(ValueError):
            FeatureSplitter(drop_na_mode)

    @given(
        X_y_matrices(
            horizon=4, time_series_features=features, allow_nan_infinity=False,
        )
    )
    def test_transform(self, X_y):
        X, y = X_y
        feature_splitter = FeatureSplitter()
        X_train, y_train, X_test, y_test = feature_splitter.transform(X, y)

        assert X_train.shape[0] == max(0, X.shape[0] - 3 - 3)
        assert y_train.shape[0] == max(0, X.shape[0] - 3 - 3)
        assert X_test.shape[0] == min(max(0, X.shape[0] - 3), 3)
        assert y_test.shape[0] == min(max(0, X.shape[0] - 3), 3)
