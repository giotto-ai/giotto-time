import pandas as pd
import pytest
from pandas.util import testing as testing
from hypothesis import given, note
import hypothesis.strategies as st
from gtime.utils.hypothesis.time_indexes import giotto_time_series


from gtime.time_series_models import (
    NaiveForecastModel, SeasonalNaiveForecastModel, AverageForecastModel, DriftForecastModel
)


@st.composite
def forecast_input(draw, max_lenth):
    length = draw(st.integers(min_value=2, max_value=max_lenth))
    horizon = draw(st.integers(min_value=1, max_value=length - 1))
    window = draw(st.integers(min_value=1, max_value=length - horizon))
    df = draw(giotto_time_series(
                min_length=horizon + window,
                max_length=max_lenth,
                allow_nan=False,
                allow_infinity=False
            ))
    return df, horizon, window


class TestNaiveForecast:

    @given(x=forecast_input(50))
    def test_fit_predict(self, x):
        df, horizon, _ = x
        model = NaiveForecastModel(horizon=horizon)
        model.fit(df)
        y_pred = model.predict()
        assert len(y_pred) == horizon
        expected_df = pd.DataFrame(float(df.iloc[-(horizon+1)]), index=model.X_test_.index, columns=model.X_test_.columns)
        testing.assert_frame_equal(y_pred, expected_df)


class TestSeasonalNaiveForecast:

    @given(x=forecast_input(50))
    def test_fit_predict(self, x):
        df, horizon, seasonal_length = x
        model = SeasonalNaiveForecastModel(horizon=horizon, seasonal_length=seasonal_length)
        model.fit(df)
        y_pred = model.predict()
        note(y_pred)
        assert len(y_pred) == horizon
        if seasonal_length < horizon:
            assert all(y_pred.iloc[0] == y_pred.iloc[seasonal_length])


class TestAverageForecast:

    @given(x=forecast_input(50))
    def test_fit_predict(self, x):
        df, horizon, window = x
        model = AverageForecastModel(horizon=horizon, window_size=window)
        model.fit(df)
        y_pred = model.predict()
        note(model.model.last_value_)
        note(y_pred)
        assert len(y_pred) == horizon
        # note((window, horizon))
        mean = df.iloc[-(horizon+window):-(horizon)].mean()
        # note(mean)
        expected_df = pd.DataFrame(float(mean), index=model.X_test_.index, columns=model.X_test_.columns)
        # note(expected_df)
        # if any(expected_df.values - y_pred.values > 0.1):
        #     print('AAA')
        testing.assert_frame_equal(y_pred, expected_df)

class TestDriftForecast:

    @given(x=forecast_input(50))
    def test_fit_predict(self, x):
        df, horizon, seasonal_length = x
        model = DriftForecastModel(horizon=horizon)
        model.fit(df)
        y_pred = model.predict()
        note(y_pred)
        assert len(y_pred) == horizon
        assert pytest.approx(y_pred.diff().diff().sum(), 0)