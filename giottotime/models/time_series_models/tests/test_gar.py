import random

import pandas.util.testing as testing
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis._settings import duration
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

from giottotime.feature_creation import (
    MovingAverageFeature,
    ConstantFeature,
    ShiftFeature,
)
from giottotime.feature_creation.feature_creation import FeatureCreation

from giottotime.feature_creation.utils import trim_feature_nans
from giottotime.models.time_series_models.gar import GAR


class TestInputs:
    @pytest.mark.parametrize("base_model", [None, FeatureCreation(1, [])])
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

    for random_param in random_params:
        random_feature = random.sample(possible_features, 1)[0]
        random_features.append(
            random_feature(random_param, output_name=str(random_feature))
        )

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

    @settings(deadline=duration(milliseconds=500))
    @given(st.builds(arbitrary_features, st.integers(1, 50)))
    def test_correct_features_dimension(self, time_series, features):
        horizon = 4
        feature_creation = FeatureCreation(horizon, features)
        base_model = LinearRegression()

        x, y = feature_creation.fit_transform(time_series)

        x_train, y_train, x_test, y_test = trim_feature_nans(x, y)

        gar_no_feedforward = GAR(base_model=base_model, feed_forward=False)

        gar_no_feedforward.fit(x_train, y_train)
        assert gar_no_feedforward.train_features_.shape[1] == len(features)

        gar_feedforward = GAR(base_model=base_model, feed_forward=True)

        gar_feedforward.fit(x_train, y_train)
        assert gar_feedforward.train_features_.shape[1] == len(features)

    @settings(deadline=duration(milliseconds=500))
    @given(st.builds(arbitrary_features, st.integers(1, 50)))
    def test_correct_fit_date(self, time_series, features):
        horizon = 4
        feature_creation = FeatureCreation(horizon, features)
        base_model = LinearRegression()

        x, y = feature_creation.fit_transform(time_series)

        x_train, y_train, x_test, y_test = trim_feature_nans(x, y)

        gar_no_feedforward = GAR(base_model=base_model, feed_forward=False)

        gar_no_feedforward.fit(x_train, y_train)

        predictions = gar_no_feedforward.predict(x_test)

        assert len(predictions) == len(x_test)
        assert (predictions.index == x_test.index).all()

        gar_with_feedforward = GAR(base_model=base_model, feed_forward=True)

        gar_with_feedforward.fit(x_train, y_train)

        predictions = gar_with_feedforward.predict(x_test)

        assert len(predictions) == len(x_test)
        assert (predictions.index == x_test.index).all()
