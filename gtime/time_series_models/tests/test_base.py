import pytest
import sklearn
from hypothesis import given
import hypothesis.strategies as st

from gtime.utils.fixtures import (
    features1,
    features2,
    model1,
    model2,
    time_series_forecasting_model1_cache,
    time_series_forecasting_model1_no_cache,
)
from gtime.time_series_models import TimeSeriesForecastingModel, AR
from gtime.utils.hypothesis.time_indexes import giotto_time_series


class TestAR:
    @given(
        p=st.integers(min_value=1, max_value=5),
        horizon=st.integers(min_value=1, max_value=3),
    )
    def test_constructor(self, p, horizon):
        ar = AR(p, horizon)
        assert len(ar.features) == p
        assert ar.horizon == horizon

    @given(
        time_series=giotto_time_series(
            allow_nan=False, allow_infinity=False, min_length=7, max_length=200
        ),
        p=st.integers(min_value=1, max_value=5),
        horizon=st.integers(min_value=1, max_value=3),
    )
    def test_results(self, time_series, p, horizon):
        ar = AR(p, horizon)
        predictions = ar.fit(time_series).predict()
        assert predictions.shape[0] == horizon
        assert predictions.shape[1] == horizon


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
