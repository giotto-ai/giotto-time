import numpy as np
import pandas as pd
import re
import pytest
import hypothesis.strategies as st
from hypothesis import given, settings, example
from gtime.utils.hypothesis.time_indexes import giotto_time_series, period_indexes
from gtime.plotting.preprocessing import (
    seasonal_split,
    acf,
    pacf,
    _get_cycle_names,
    _get_season_names,
    _autocorrelation,
    _normalize,
    _solve_yw_equation,
    _week_of_year,
    yule_walker,
)


class TestSplits:
    @given(t=period_indexes(min_length=1, max_length=1))
    @example(t=pd.PeriodIndex(["1974-12-31"], freq="W"))
    @example(t=pd.PeriodIndex(["1972-01-01"], freq="W"))
    @settings(deadline=None)
    def test_week_of_year(self, t):
        period = t[0]
        week = _week_of_year(period)
        assert re.match(r"\d{4}_\d\d?$", week)

    @given(
        df=giotto_time_series(min_length=3, max_length=500),
        cycle=st.one_of(
            st.sampled_from(["year", "quarter", "month", "week"]),
            st.from_regex(r"[1-9][DWMQY]", fullmatch=True),
        ),
    )
    @settings(deadline=None)
    def test__get_cycle_names_size(self, df, cycle):
        cycle = _get_cycle_names(df, cycle)
        assert len(cycle) == len(df)

    @given(
        df=giotto_time_series(min_length=3, max_length=500),
        cycle=st.one_of(
            st.sampled_from(["year", "quarter", "month", "week"]),
            st.from_regex(r"[1-9][DWMQY]", fullmatch=True),
        ),
        freq=st.from_regex(r"[1-9]?[DWMQ]", fullmatch=True),
    )
    @settings(deadline=None)
    def test__get_season_names_size(self, df, cycle, freq):
        seasons = _get_season_names(df, cycle, freq)
        assert len(seasons) == len(df)

    @given(
        df=giotto_time_series(min_length=3, max_length=500),
        cycle=st.one_of(
            st.sampled_from(["year", "quarter", "month", "week"]),
            st.from_regex(r"[1-9][DWMQY]", fullmatch=True),
        ),
        freq=st.one_of(st.from_regex(r"[1-9]?[DWMQ]", fullmatch=True), st.none()),
        agg=st.sampled_from(["mean", "sum", "last"]),
    )
    @settings(deadline=None)
    def test_seasonal_split_shape_named(self, df, cycle, freq, agg):
        split = seasonal_split(df, cycle=cycle, freq=freq, agg=agg)
        if freq is None:
            freq = df.index.freqstr
        assert split.stack().shape == df.resample(freq).agg(agg).dropna().shape


class TestAcf:
    @given(x=st.lists(st.floats(allow_nan=False), min_size=1))
    def test_autocorrelation(self, x):
        autocorr = _autocorrelation(np.array(x))
        expected = np.correlate(x, x, mode="full")[-len(x) :] / len(x)
        np.testing.assert_array_equal(autocorr, expected)

    @given(
        x=st.lists(
            st.floats(
                allow_nan=False, allow_infinity=False, max_value=1e20, min_value=1e20
            ),
            min_size=1,
        )
    )
    def test_scale(self, x):
        scaled_x = _normalize(np.array(x))
        assert scaled_x.mean() == pytest.approx(0.0)
        assert scaled_x.std() == pytest.approx(1.0) or scaled_x.std() == pytest.approx(
            0.0
        )

    @given(x=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2))
    def test_solve_yw(self, x):
        rho = _solve_yw_equation(np.array(x))
        if not np.isnan(np.sum(rho)):
            assert len(rho) == len(x) - 1

    @given(
        x=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2),
        order=st.integers(min_value=1),
    )
    def test_yule_walker_abs(self, x, order):
        pacf = yule_walker(np.array(x), order)
        if not (np.isnan(np.sum(pacf)) or len(pacf) == 0):
            assert all(abs(pacf) <= 2)

    @given(
        df=giotto_time_series(min_length=1, allow_nan=False, allow_infinity=False),
        max_lag=st.one_of(st.integers(min_value=1, max_value=100), st.none()),
    )
    def test_acf_len(self, df, max_lag):
        df_array = np.ravel(df.values)
        res = acf(df_array, max_lag)
        if max_lag is None:
            max_lag = len(df)
        assert len(res) == min(max_lag, len(df))

    @given(
        df=giotto_time_series(
            min_length=1, allow_nan=False, allow_infinity=False, max_length=50
        ),
        max_lag=st.one_of(st.integers(min_value=1, max_value=100), st.none()),
    )
    def test_pacf_len(self, df, max_lag):
        df_array = np.ravel(df.values)
        res = pacf(df_array, max_lag)
        if max_lag is None:
            max_lag = len(df)
        assert len(res) == min(max_lag, len(df))
