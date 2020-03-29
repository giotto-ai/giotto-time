import pytest
import numpy as np
from hypothesis import given, settings, note
import hypothesis.strategies as st
import matplotlib
from gtime.utils.hypothesis.time_indexes import giotto_time_series

from gtime.plotting import lag_plot, acf_plot, subplots, seasonal_plot
from gtime.plotting.preprocessing import seasonal_split


class TestLagplots:
    @given(
        df=giotto_time_series(min_length=1, allow_nan=False, allow_infinity=False),
        lags=st.lists(st.integers(min_value=1), min_size=1, max_size=10),
    )
    @settings(deadline=None)
    def test_subplots_number(self, df, lags):
        ax = lag_plot(df, lags)
        num_plots = sum(map(lambda x: x.has_data(), ax.flatten()))
        assert num_plots == len(lags)
        matplotlib.pyplot.close("all")

    @given(
        df=giotto_time_series(min_length=1, allow_nan=False, allow_infinity=False),
        lags=st.lists(st.integers(min_value=1), min_size=1, max_size=20),
        plots_per_row=st.integers(min_value=1, max_value=7),
    )
    @settings(deadline=None)
    def test_rows_and_cols(self, df, lags, plots_per_row):
        ax = lag_plot(df, lags, plots_per_row)
        assert ax.shape == (
            (len(lags) - 1) // plots_per_row + 1,
            min(len(lags), plots_per_row),
        )
        matplotlib.pyplot.close("all")


class TestACFplots:
    @given(
        df=giotto_time_series(min_length=2, allow_nan=False, allow_infinity=False),
        maxlags=st.integers(min_value=1),
        ci=st.floats(min_value=0.0, max_value=1.0),
        partial=st.booleans(),
    )
    @settings(deadline=None)
    def test_ci_lines(self, df, maxlags, ci, partial):
        if float(df.diff().sum()) > 0:
            ax = acf_plot(df, maxlags, ci, partial)
            assert len(ax.lines) == 3
            # num_plots = sum(map(lambda x: x.has_data(), ax.flatten()))
            # assert num_plots == len(lags)
            matplotlib.pyplot.close("all")

    @given(
        df=giotto_time_series(min_length=2, allow_nan=False, allow_infinity=False),
        maxlags=st.integers(min_value=1),
        ci=st.floats(min_value=0.0, max_value=1.0),
        partial=st.booleans(),
    )
    @settings(deadline=None)
    def test_num_bars(self, df, maxlags, ci, partial):
        if float(df.diff().sum()) > 0:
            ax = acf_plot(df, maxlags, ci, partial)
            assert len(ax.containers[0]) == min(len(df), maxlags)
            matplotlib.pyplot.close("all")


class TestSubplots:
    @given(
        df=giotto_time_series(min_length=3, max_length=50),
        cycle=st.sampled_from(["year", "quarter", "month"]),
        freq=st.from_regex(r"[1-9][WMQ]", fullmatch=True),
        agg=st.sampled_from(["mean", "sum", "last"]),
        box=st.booleans(),
    )
    @settings(deadline=None)
    def test_subplots_number(self, df, cycle, freq, agg, box):
        ax = subplots(df, cycle, freq, agg, box)
        split = seasonal_split(df, cycle, freq, agg)
        assert ax.size == split.shape[0]
        matplotlib.pyplot.close("all")


class TestSeasonalPlots:
    @given(
        df=giotto_time_series(min_length=3, max_length=50),
        cycle=st.sampled_from(["year", "quarter", "month", "week"]),
        freq=st.from_regex(r"[1-9][WMQ]", fullmatch=True),
        agg=st.sampled_from(["mean", "sum", "last"]),
        polar=st.booleans(),
    )
    @settings(deadline=None)
    def test_seasonal_num_lines(self, df, cycle, freq, agg, polar):
        ax = seasonal_plot(df, cycle, freq, agg, polar)
        split = seasonal_split(df, cycle, freq, agg)
        assert len(ax.lines) == split.shape[1]
        matplotlib.pyplot.close("all")
