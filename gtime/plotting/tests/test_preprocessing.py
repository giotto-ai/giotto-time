import numpy as np
import pandas as pd
import re
import pytest
import hypothesis.strategies as st
from hypothesis import given, settings, example
from gtime.utils.hypothesis.time_indexes import giotto_time_series, period_indexes
from gtime.plotting.preprocessing import (
    seasonal_split,
    _get_cycle_names,
    _get_season_names,
    _week_of_year,
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
