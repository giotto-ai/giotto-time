import random

import pandas.util.testing as testing
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

from giottotime.feature_creation.feature_creation import FeaturesCreation
from giottotime.feature_creation.time_series_features import \
    MovingAverageFeature, ConstantFeature, ShiftFeature
from giottotime.models.gar import GAR


class TestInputs:
    @pytest.mark.parametrize("base_model", [None, FeaturesCreation(1, [])])
    def test_invalid_base_model(self, base_model):
        with pytest.raises(TypeError):
            GAR(base_model=base_model)


@pytest.fixture
def time_series():
    testing.N, testing.K = 200, 1
    return testing.makeTimeDataFrame(freq='MS')


def arbitrary_features(feature_length):
    possible_features = [
        MovingAverageFeature,
        ConstantFeature,
        ShiftFeature
    ]
    random_features = []
    random_params = random.sample(range(1, 100), feature_length)

    for random_param in random_params:
        random_feature = random.sample(possible_features, 1)[0]
        random_features.append(random_feature(random_param))

    return random_features


class TestFitPredict:
    def test_predict_with_no_fit(self, time_series):
        base_model = LinearRegression()

        gar = GAR(base_model=base_model)

        with pytest.raises(NotFittedError):
            gar.predict(time_series)

    @settings(suppress_health_check=(HealthCheck.filter_too_much,))
    @given(st.builds(
        arbitrary_features,
        st.integers().filter(lambda x: 1 <= x <= 50)))
    def test_correct_features_dimension(self, time_series, features):
        horizon = 4
        feature_creation = FeaturesCreation(horizon, features)
        base_model = LinearRegression()

        x, y, _ = feature_creation.fit_transform(time_series)

        gar = GAR(base_model=base_model)

        gar.fit(x, y)
        assert gar.train_features_.shape[1] == len(features)

    @settings(suppress_health_check=(HealthCheck.filter_too_much,))
    @given(st.builds(
        arbitrary_features,
        st.integers().filter(lambda x: 1 <= x <= 50)))
    def test_correct_fit_date(self, time_series, features):
        horizon = 4
        feature_creation = FeaturesCreation(horizon, features)
        base_model = LinearRegression()

        x_train, y_train, x_test = feature_creation.fit_transform(time_series)

        gar = GAR(base_model=base_model)

        gar.fit(x_train, y_train)

        predictions = gar.predict(x_test)

        assert len(predictions) == len(x_test)
        assert (predictions.index == x_test.index).all()

