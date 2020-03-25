import numpy as np
import pytest
import sklearn
from hypothesis import given
from pytest import fixture
from sklearn.compose import make_column_selector
from sklearn.linear_model import LinearRegression, Ridge

from gtime.feature_extraction import MovingAverage, Shift
from gtime.forecasting import GAR
from gtime.time_series_models import TimeSeriesForecastingModel
from gtime.utils.hypothesis.time_indexes import giotto_time_series


@fixture(scope="function")
def features1():
    return [
        ("shift_0", Shift(0), make_column_selector(dtype_include=np.number)),
        ("shift_1", Shift(1), make_column_selector(dtype_include=np.number)),
        (
            "moving_average_3",
            MovingAverage(window_size=3),
            make_column_selector(dtype_include=np.number),
        ),
    ]


@fixture(scope="function")
def features2():
    return [
        ("shift_0", Shift(0), make_column_selector(dtype_include=np.number)),
        ("shift_1", Shift(1), make_column_selector(dtype_include=np.number)),
    ]


@fixture(scope="function")
def model1():
    lr = LinearRegression()
    return GAR(lr)


@fixture(scope="function")
def model2():
    lr = Ridge(alpha=0.1)
    return GAR(lr)


@fixture(scope="function")
def time_series_forecasting_model1_no_cache(features1, model1):
    return TimeSeriesForecastingModel(
        features=features1, horizon=2, model=model1, cache_features=False,
    )


@fixture(scope="function")
def time_series_forecasting_model1_cache(features1, model1):
    return TimeSeriesForecastingModel(
        features=features1, horizon=2, model=model1, cache_features=True,
    )


class TestTimeSeriesForecastingModel:
    def test_constructor(self, features1, model1):
        horizon, cache_features = 2, True
        time_series_forecasting_model = TimeSeriesForecastingModel(
            features=features1,
            horizon=horizon,
            model=model1,
            cache_features=cache_features,
        )
        assert time_series_forecasting_model.features == features1
        assert time_series_forecasting_model.horizon == horizon
        assert time_series_forecasting_model.model == model1
        assert time_series_forecasting_model.cache_features == cache_features

    @given(
        time_series=giotto_time_series(
            allow_infinity=False, allow_nan=False, min_length=5
        )
    )
    def test_fit_no_cache_stores_X_test_and_model(
        self, time_series, time_series_forecasting_model1_no_cache
    ):
        time_series_forecasting_model1_no_cache.fit(time_series)
        assert hasattr(time_series_forecasting_model1_no_cache, "model_")
        assert hasattr(time_series_forecasting_model1_no_cache, "X_test_")

    @given(
        time_series=giotto_time_series(
            allow_infinity=False, allow_nan=False, min_length=5
        )
    )
    def test_fit_no_cache_does_not_store_X_train_y_train(
        self, time_series, time_series_forecasting_model1_no_cache
    ):
        time_series_forecasting_model1_no_cache.fit(time_series)
        assert not hasattr(time_series_forecasting_model1_no_cache, "X_train_")
        assert not hasattr(time_series_forecasting_model1_no_cache, "y_train_")

    @given(
        time_series=giotto_time_series(
            allow_infinity=False, allow_nan=False, min_length=5
        )
    )
    def test_fit_cache_stores_all_training_params(
        self, time_series, time_series_forecasting_model1_cache
    ):
        time_series_forecasting_model1_cache.fit(time_series)
        assert hasattr(time_series_forecasting_model1_cache, "model_")
        assert hasattr(time_series_forecasting_model1_cache, "X_test_")
        assert hasattr(time_series_forecasting_model1_cache, "X_train_")
        assert hasattr(time_series_forecasting_model1_cache, "y_train_")

    @given(
        time_series=giotto_time_series(
            allow_infinity=False, allow_nan=False, min_length=5
        )
    )
    def test_predict_no_cache_fail_model_not_fitted(
        self, time_series, time_series_forecasting_model1_no_cache
    ):
        with pytest.raises(sklearn.exceptions.NotFittedError):
            time_series_forecasting_model1_no_cache.predict(time_series)

    @given(
        time_series=giotto_time_series(
            allow_infinity=False, allow_nan=False, min_length=5
        )
    )
    def test_predict_cache_fail_model_not_fitted(
        self, time_series, time_series_forecasting_model1_cache
    ):
        with pytest.raises(sklearn.exceptions.NotFittedError):
            time_series_forecasting_model1_cache.predict(time_series)

    @given(
        time_series=giotto_time_series(
            allow_infinity=False, allow_nan=False, min_length=5
        )
    )
    def test_predict_works_no_input(
        self, time_series, time_series_forecasting_model1_no_cache
    ):
        time_series_forecasting_model1_no_cache.fit(time_series).predict()

    @given(
        time_series=giotto_time_series(
            allow_infinity=False, allow_nan=False, min_length=5
        )
    )
    def test_predict_works_input(
        self, time_series, time_series_forecasting_model1_no_cache
    ):
        time_series_forecasting_model1_no_cache.fit(time_series).predict(time_series)

    @given(
        time_series=giotto_time_series(
            allow_infinity=False, allow_nan=False, min_length=5
        )
    )
    def test_error_fit_twice_no_cache_only_models(
        self, time_series, time_series_forecasting_model1_no_cache
    ):
        with pytest.raises(AttributeError):
            time_series_forecasting_model1_no_cache.fit(time_series).fit(
                time_series, only_model=True
            )

    @given(
        time_series=giotto_time_series(
            allow_infinity=False, allow_nan=False, min_length=5
        )
    )
    def test_error_fit_once_only_models(
        self, time_series, time_series_forecasting_model1_cache
    ):
        with pytest.raises(sklearn.exceptions.NotFittedError):
            time_series_forecasting_model1_cache.fit(time_series, only_model=True)

    @given(
        time_series=giotto_time_series(
            allow_infinity=False, allow_nan=False, min_length=5
        )
    )
    def test_fit_twice_only_models(
        self, time_series, time_series_forecasting_model1_cache
    ):
        time_series_forecasting_model1_cache.fit(time_series).fit(
            time_series, only_model=True
        )

    @given(
        time_series=giotto_time_series(
            allow_infinity=False, allow_nan=False, min_length=5
        )
    )
    def test_error_fit_twice_set_features_only_models(
        self, time_series, time_series_forecasting_model1_cache, features2
    ):
        time_series_forecasting_model1_cache.fit(time_series)
        time_series_forecasting_model1_cache.set_params(features=features2)
        with pytest.raises(sklearn.exceptions.NotFittedError):
            time_series_forecasting_model1_cache.fit(time_series, only_model=True)

    @given(
        time_series=giotto_time_series(
            allow_infinity=False, allow_nan=False, min_length=5
        )
    )
    def test_fit_twice_set_model_only_models(
        self, time_series, time_series_forecasting_model1_cache, model2
    ):
        time_series_forecasting_model1_cache.fit(time_series)
        time_series_forecasting_model1_cache.set_params(model=model2)
        time_series_forecasting_model1_cache.fit(time_series, only_model=True)
