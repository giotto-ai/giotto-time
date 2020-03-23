import pytest
from hypothesis import given, note
import hypothesis.strategies as st
import matplotlib
from gtime.utils.hypothesis.time_indexes import giotto_time_series

from gtime.plotting import lagplot, acf_plot, subplots, seasonal_plot


@given(
    df=giotto_time_series(min_length=1, allow_nan=False, allow_infinity=False),
    lags=st.lists(st.integers(min_value=1), min_size=1, max_size=10),
)
def test_lagplot(df, lags):
    ax = lagplot(df, lags)
    assert isinstance(ax, matplotlib.axes._axes.Axes)
    matplotlib.pyplot.close('all')


@given(
    df=giotto_time_series(min_length=1, allow_nan=False, allow_infinity=False),
    max_lags=st.integers(min_value=1),
)
def test_acf_plot(df, max_lags):
    ax = acf_plot(df, max_lags=max_lags)
    assert isinstance(ax, matplotlib.axes._axes.Axes)
    matplotlib.pyplot.close('all')

# test_lagplot()
