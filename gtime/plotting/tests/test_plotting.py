import pytest
import numpy as np
from hypothesis import given, settings, note
import hypothesis.strategies as st
import matplotlib
from gtime.utils.hypothesis.time_indexes import giotto_time_series

from gtime.plotting import lagplot, acf_plot, subplots, seasonal_plot

class TestLagplots:
    @given(
        df=giotto_time_series(min_length=1, allow_nan=False, allow_infinity=False),
        lags=st.lists(st.integers(min_value=1), min_size=1, max_size=10),
    )
    @settings(deadline=None)
    def test_subplots_number(self, df, lags):
        ax = lagplot(df, lags)
        num_plots = sum(map(lambda x: x.has_data(), ax.flatten()))
        assert num_plots == len(lags)
        matplotlib.pyplot.close('all')


class TestACFplots:

    @given(
        df=giotto_time_series(min_length=2, allow_nan=False, allow_infinity=False),
        maxlags=st.integers(min_value=1),
        ci=st.floats(min_value=0.0, max_value=1.0),
        partial=st.booleans()
    )
    @settings(deadline=None)
    def test_ci_lines(self, df, maxlags, ci, partial):
        if float(df.diff().sum()) > 0:
            ax = acf_plot(df, maxlags, ci, partial)
            assert len(ax.lines) == 3
            # num_plots = sum(map(lambda x: x.has_data(), ax.flatten()))
            # assert num_plots == len(lags)
            matplotlib.pyplot.close('all')

    @given(
        df=giotto_time_series(min_length=2, allow_nan=False, allow_infinity=False),
        maxlags=st.integers(min_value=1),
        ci=st.floats(min_value=0.0, max_value=1.0),
        partial=st.booleans()
    )
    @settings(deadline=None)
    def test_num_bars(self, df, maxlags, ci, partial):
        if float(df.diff().sum()) > 0:
            ax = acf_plot(df, maxlags, ci, partial)
            assert len(ax.containers[0]) == min(len(df), maxlags)
            matplotlib.pyplot.close('all')
