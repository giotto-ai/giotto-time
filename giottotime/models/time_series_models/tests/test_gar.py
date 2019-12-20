import random

import numpy as np
import pandas.util.testing as testing
import pytest
from hypothesis import given, strategies as st
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

from giottotime.feature_creation import (
    MovingAverageFeature,
    ConstantFeature,
    ShiftFeature,
)
from giottotime.feature_creation.feature_creation import FeatureCreation
from giottotime.model_selection import FeatureSplitter
from giottotime.models.time_series_models.gar import GAR


class TestInputs:
    @pytest.mark.parametrize(
        "base_model", [None, FeatureCreation(horizon=1, time_series_features=[])]
    )
    def test_invalid_base_model(self, base_model):
        with pytest.raises(TypeError):
            GAR(base_model=base_model, feed_forward=False)

        with pytest.raises(TypeError):
            GAR(base_model=base_model, feed_forward=True)


@pytest.fixture
def time_series():
    testing.N, testing.K = 200, 1
    return testing.makeTimeDataFrame(freq="MS")


def arbitrary_features(feature_length):
    possible_features = [MovingAverageFeature, ConstantFeature, ShiftFeature]
    random_features = []
    random_params = random.sample(range(1, 100), feature_length)

    feature_names = []
    for random_param in random_params:
        random_feature = random.sample(possible_features, 1)[0]
        output_name = f"{random_feature}_{random_params}"
        if output_name in feature_names:
            continue
        feature_names.append(output_name)
        feature = random_feature(random_param, output_name=output_name)
        random_features.append(feature)

    return random_features


class TestFitPredict:
    def test_predict_with_no_fit(self, time_series):
        base_model = LinearRegression()

        gar_no_feedforward = GAR(base_model=base_model, feed_forward=False)

        with pytest.raises(NotFittedError):
            gar_no_feedforward.predict(time_series)

        gar_feedforward = GAR(base_model=base_model, feed_forward=True)

        with pytest.raises(NotFittedError):
            gar_feedforward.predict(time_series)

    @given(st.builds(arbitrary_features, st.integers(1, 50)))
    def test_correct_features_dimension(self, time_series, features):
        horizon = 4
        feature_creation = FeatureCreation(
            horizon=horizon, time_series_features=features
        )
        base_model = LinearRegression()

        x, y = feature_creation.fit_transform(time_series)

        feature_splitter = FeatureSplitter()
        x_train, y_train, x_test, y_test = feature_splitter.transform(x, y)

        gar_no_feedforward = GAR(base_model=base_model, feed_forward=False)

        gar_no_feedforward.fit(x_train, y_train)
        assert gar_no_feedforward.train_features_.shape[1] == len(features)

        gar_feedforward = GAR(base_model=base_model, feed_forward=True)

        gar_feedforward.fit(x_train, y_train)
        assert gar_feedforward.train_features_.shape[1] == len(features)

    @given(st.builds(arbitrary_features, st.integers(1, 50)))
    def test_correct_fit_date(self, time_series, features):
        horizon = 4
        feature_creation = FeatureCreation(
            horizon=horizon, time_series_features=features
        )
        base_model = LinearRegression()

        x, y = feature_creation.fit_transform(time_series)

        feature_splitter = FeatureSplitter()
        x_train, y_train, x_test, y_test = feature_splitter.transform(x, y)

        gar_no_feedforward = GAR(base_model=base_model, feed_forward=False)

        gar_no_feedforward.fit(x_train, y_train)

        predictions = gar_no_feedforward.predict(x_test)

        assert len(predictions) == len(x_test)
        np.testing.assert_array_equal(predictions.index, x_test.index)

        gar_with_feedforward = GAR(base_model=base_model, feed_forward=True)

        gar_with_feedforward.fit(x_train, y_train)

        predictions = gar_with_feedforward.predict(x_test)

        assert len(predictions) == len(x_test)
        np.testing.assert_array_equal(predictions.index, x_test.index)
