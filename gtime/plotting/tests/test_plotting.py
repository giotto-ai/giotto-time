from hypothesis import given, settings
import hypothesis.strategies as st
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gtime.utils.hypothesis.time_indexes import giotto_time_series

from gtime.plotting import lag_plot, acf_plot, seasonal_subplots, seasonal_plot
from gtime.plotting.preprocessing import seasonal_split


@pytest.fixture()
def time_series():
    idx = pd.period_range(start="2000-01-01", end="2003-01-01")
    df = pd.DataFrame(np.random.random((len(idx), 1)), index=idx, columns=["ts"])
    return df


class TestLagplots:
    @pytest.mark.parametrize("lags", [1, 5, [1], [1, 3, 5, 100]])
    def test_subplots_number(self, time_series, lags):
        ax = lag_plot(time_series, lags=lags)
        num_plots = sum(map(lambda x: x.has_data(), ax.flatten()))
        if isinstance(lags, int):
            expected_num_plots = lags
        else:
            expected_num_plots = len(lags)
        assert num_plots == expected_num_plots
        plt.close("all")

    @pytest.mark.parametrize("lags", [1, 5, [1], [1, 3, 5, 100]])
    @pytest.mark.parametrize("plots_per_row", [1, 3, 10])
    def test_rows_and_cols(self, time_series, lags, plots_per_row):
        ax = lag_plot(time_series, lags=lags, plots_per_row=plots_per_row)
        if isinstance(lags, int):
            lag_length = lags
        else:
            lag_length = len(lags)
        assert ax.shape == (
            (lag_length - 1) // plots_per_row + 1,
            min(lag_length, plots_per_row),
        )
        plt.close("all")


class TestACFplots:
    @pytest.mark.parametrize("maxlags", [1, 5, 100])
    @pytest.mark.parametrize("ci", [0.0, 0.05])
    @pytest.mark.parametrize("partial", [True, False])
    def test_ci_lines(self, time_series, maxlags, ci, partial):
        ax = acf_plot(time_series, max_lags=maxlags, ci=ci, partial=partial)
        assert len(ax.lines) == 3
        plt.close("all")

    @pytest.mark.parametrize("maxlags", [1, 5, 100])
    @pytest.mark.parametrize("ci", [0.0, 0.05])
    @pytest.mark.parametrize("partial", [True, False])
    def test_num_bars(self, time_series, maxlags, ci, partial):
        ax = acf_plot(time_series, maxlags, ci, partial)
        assert len(ax.containers[0]) == min(len(time_series), maxlags)
        plt.close("all")


class TestSubplots:
    @pytest.mark.parametrize("cycle", ["year", "6M"])
    @pytest.mark.parametrize("freq", ["M"])
    @pytest.mark.parametrize("box", [True, False])
    def test_subplots_number(self, time_series, cycle, freq, box):
        ax = seasonal_subplots(time_series, cycle=cycle, freq=freq, box=box)
        split = seasonal_split(time_series, cycle=cycle, freq=freq)
        assert ax.size == split.shape[0]
        plt.close("all")


class TestSeasonalPlots:
    @pytest.mark.parametrize("cycle", ["year", "6M"])
    @pytest.mark.parametrize("freq", ["M", None])
    @pytest.mark.parametrize("polar", [True, False])
    @pytest.mark.parametrize("new_ax", [True, False])
    def test_seasonal_num_lines(self, time_series, cycle, freq, polar, new_ax):
        if new_ax:
            if polar:
                ax = plt.subplot(111, projection="polar")
            else:
                ax = plt.subplot(111)
        else:
            ax = None
        ax = seasonal_plot(time_series, cycle=cycle, freq=freq, polar=polar, ax=ax)
        split = seasonal_split(time_series, cycle=cycle, freq=freq)
        assert len(ax.lines) == split.shape[1]
        plt.close("all")
