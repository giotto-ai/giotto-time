import numpy as np
import pytest

import hypothesis.strategies as st
from hypothesis import given, settings, HealthCheck
from sklearn.compose import make_column_selector

from gtime.compose import FeatureCreation
from gtime.feature_extraction import Shift, MovingAverage
from gtime.model_selection.splitters import FeatureSplitter
from gtime.utils.hypothesis.feature_matrices import X_y_matrices

# TODO: refactor, make hypothesis generator instead of a full pipeline
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
