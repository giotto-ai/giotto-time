import random

import numpy as np
import pandas.util.testing as testing
import pytest
from hypothesis import given, strategies as st
from sklearn.compose import make_column_selector
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

from gtime.compose import FeatureCreation
from gtime.feature_extraction import (
    MovingAverage,
    Shift,
)
from gtime.feature_generation import Constant
from gtime.model_selection import FeatureSplitter
from gtime.forecasting import GAR, GARFF
from gtime.utils.hypothesis.feature_matrices import X_y_matrices

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


@pytest.fixture
def time_series():
    testing.N, testing.K = 200, 1
    return testing.makeTimeDataFrame(freq="MS")


def arbitrary_features(feature_length):
    possible_features = [MovingAverage, Constant, Shift]
    random_features = []
    random_params = random.sample(range(1, 100), feature_length)

    feature_names = []
    for random_param in random_params:
        random_feature = random.sample(possible_features, 1)[0]
        output_name = f"{random_feature}_{random_params}"
        if output_name in feature_names:
            continue
        feature_names.append(output_name)
        feature = random_feature(random_param)
        random_features.append(feature)

    return random_features, feature_names


class TestFitPredict:
    def test_predict_with_no_fit(self, time_series):
        base_model = LinearRegression()

        gar_no_feedforward = GAR(estimator=base_model)

        with pytest.raises(NotFittedError):
            gar_no_feedforward.predict(time_series)

    @given(
        X_y_matrices(
            horizon=4,
            df_transformer=df_transformer,
            allow_nan_infinity=False,
            min_length=10,
        )
    )
    def test_correct_fit_date(self, X_y):
        base_model = LinearRegression()
        feature_splitter = FeatureSplitter()
        x, y = X_y[0], X_y[1]
        x_train, y_train, x_test, y_test = feature_splitter.transform(x, y)

        gar_no_feedforward = GAR(estimator=base_model)

        gar_no_feedforward.fit(x_train, y_train)

        predictions = gar_no_feedforward.predict(x_test)

        assert len(predictions) == len(x_test)
        np.testing.assert_array_equal(predictions.index, x_test.index)

        gar_with_feedforward = GARFF(estimator=base_model)

        gar_with_feedforward.fit(x_train, y_train)

        predictions = gar_with_feedforward.predict(x_test)

        assert len(predictions) == len(x_test)
        np.testing.assert_array_equal(predictions.index, x_test.index)
