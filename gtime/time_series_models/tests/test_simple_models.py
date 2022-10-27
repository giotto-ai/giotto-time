import pandas as pd
import numpy as np
import pytest
from pandas.util import testing as testing
from hypothesis import given, note
import hypothesis.strategies as st
from gtime.utils.hypothesis.time_indexes import giotto_time_series


from gtime.time_series_models import (
    Naive,
    SeasonalNaive,
    Average,
    Drift,
)


@st.composite
def forecast_input(draw, max_lenth):
    length = draw(st.integers(min_value=4, max_value=max_lenth))
    horizon = draw(st.integers(min_value=1, max_value=length - 1))
    window = draw(st.integers(min_value=1, max_value=length - horizon))
    df = draw(
        giotto_time_series(
            min_length=horizon + window,
            max_length=max_lenth,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return df, horizon, window


class TestNaiveForecast:
    @given(x=forecast_input(50))
    def test_fit_predict(self, x):
        df, horizon, _ = x
        model = Naive(horizon=horizon)
        model.fit(df)
        y_pred = model.predict()
        assert y_pred.shape == (horizon, horizon)
        res = np.broadcast_to(df.iloc[-horizon:], (horizon, horizon))
        y_cols = ["y_" + str(x + 1) for x in range(horizon)]
        expected_df = pd.DataFrame(res, index=model.X_test_.index, columns=y_cols)
        testing.assert_frame_equal(y_pred, expected_df)


class TestSeasonalNaiveForecast:
    @given(x=forecast_input(50))
    def test_fit_predict(self, x):
        df, horizon, seasonal_length = x
        model = SeasonalNaive(horizon=horizon, seasonal_length=seasonal_length)
        model.fit(df)
        y_pred = model.predict()
        note(y_pred)
        assert y_pred.shape[1] == horizon
        if seasonal_length < horizon:
            assert all(y_pred.iloc[:, 0] == y_pred.iloc[:, seasonal_length])


class TestAverageForecast:
    @given(x=forecast_input(50))
    def test_fit_predict(self, x):
        df, horizon, _ = x
        model = Average(horizon=horizon)
        model.fit(df)
        y_pred = model.predict()
        note(y_pred)
        assert y_pred.shape == (horizon, horizon)
        assert pytest.approx(y_pred.diff(axis=1).sum().sum()) == 0
        means = [df.mean()] + [df.iloc[:-i].mean() for i in range(1, horizon)]


class TestDriftForecast:
    @given(x=forecast_input(50))
    def test_fit_predict(self, x):
        df, horizon, _ = x
        model = Drift(horizon=horizon)
        model.fit(df)
        y_pred = model.predict()
        note(y_pred)
        assert len(y_pred) == horizon
        # assert pytest.approx(y_pred.diff().diff().sum().sum()) == 0
