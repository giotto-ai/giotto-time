import pytest
from hypothesis import given, assume
import pandas as pd
from pandas.testing import assert_series_equal
import numpy as np

from giottotime.utils.hypothesis.time_indexes import (
    period_indexes,
    series_with_period_index,
    datetime_indexes,
    series_with_datetime_index,
    timedelta_indexes,
    series_with_timedelta_index,
    available_freqs,
    positive_bounded_integers,
    pair_of_ordered_timedeltas,
    pair_of_ordered_dates,
    samples_from,
)
from giottotime.utils.hypothesis.utils import freq_to_timedelta

NON_UNIFORM_FREQS = ["B", "Q", "A"]


class TestPeriodIndex:
    @given(period_indexes())
    def test_period_indexes_is_period(self, index):
        assert isinstance(index, pd.PeriodIndex)

    @given(period_indexes(min_length=10, max_length=1000))
    def test_period_indexes_size(self, index):
        assert 10 <= len(index) <= 1000

    @given(period_indexes(min_length=10, max_length=10))
    def test_period_indexes_size_fixed_value(self, index):
        assert len(index) == 10

    @given(period_indexes(min_length=0, max_length=0))
    def test_period_indexes_size_fixed_value_0(self, index):
        assert len(index) == 0

    @given(period_indexes())
    def test_period_indexes_boundaries(self, index):
        start_datetime = pd.Period("1979-12-31").to_timestamp()
        end_datetime = pd.Period("2020-01-01").to_timestamp()
        if len(index):
            assert index[0].to_timestamp() >= start_datetime
            assert index[-1].to_timestamp() <= end_datetime


class TestPeriodSeries:
    @given(series_with_period_index())
    def test_period_series_has_period_index(self, series):
        assert isinstance(series.index, pd.PeriodIndex)

    @given(series_with_period_index(min_length=10, max_length=1000))
    def test_period_series_size(self, series):
        assert 10 <= len(series.index) <= 1000

    @given(series_with_period_index(min_length=10, max_length=10))
    def test_period_series_size_fixed_value(self, series):
        assert len(series.index) == 10

    @given(series_with_period_index(min_length=0, max_length=0))
    def test_period_series_size_fixed_value_0(self, series):
        assert len(series.index) == 0

    @given(series_with_period_index())
    def test_period_series_boundaries(self, series):
        start_datetime = pd.Period("1979-12-31").to_timestamp()
        end_datetime = pd.Period("2020-01-01").to_timestamp()
        if len(series):
            assert series.index[0].to_timestamp() >= start_datetime
            assert series.index[-1].to_timestamp() <= end_datetime

    @given(series_with_period_index())
    def test_period_series_has_float_values(self, series: pd.Series):
        assert series.dtype == "float64"

    @given(series_with_period_index(allow_nan=False))
    def test_period_series_no_nan(self, series: pd.Series):
        assert_series_equal(series, series.dropna())

    @given(series_with_period_index(allow_infinity=False))
    def test_period_series_no_infinity(self, series: pd.Series):
        assert_series_equal(series, series.replace([np.inf, -np.inf], np.nan))


class TestDatetimeIndex:
    @given(datetime_indexes())
    def test_datetime_indexes_is_datetime(self, index):
        assert isinstance(index, pd.DatetimeIndex)

    @given(datetime_indexes(min_length=30, max_length=1000))
    def test_datetime_indexes_size(self, index):
        assert 30 <= len(index) <= 1000

    @given(datetime_indexes(min_length=30, max_length=30))
    def test_datetime_indexes_size_fixed_value(self, index):
        assert len(index) == 30

    @given(datetime_indexes(min_length=0, max_length=0))
    def test_datetime_indexes_size_fixed_value_0(self, index):
        assert len(index) == 0

    @given(datetime_indexes())
    def test_datetime_indexes_boundaries(self, index):
        start_datetime = pd.Timestamp("1979-12-31")
        end_datetime = pd.Timestamp("2020-01-02")
        if len(index):
            assert index[0] >= start_datetime
            assert index[-1] <= end_datetime


class TestDatetimeSeries:
    @given(series_with_datetime_index())
    def test_datetime_series_has_datetime_index(self, series):
        assert isinstance(series.index, pd.DatetimeIndex)

    @given(series_with_datetime_index(min_length=40, max_length=1000))
    def test_datetime_series_size(self, series):
        assert 40 <= len(series.index) <= 1000

    @given(series_with_datetime_index(min_length=40, max_length=40))
    def test_datetime_series_size_fixed_value(self, series):
        assert len(series.index) == 40

    @given(series_with_datetime_index(min_length=0, max_length=0))
    def test_datetime_series_size_fixed_value_0(self, series):
        assert len(series.index) == 0

    @given(series_with_datetime_index())
    def test_datetime_series_boundaries(self, series):
        start_datetime = pd.Timestamp("1979-12-31")
        end_datetime = pd.Timestamp("2020-01-02")
        if len(series):
            assert series.index[0] >= start_datetime
            assert series.index[-1] <= end_datetime

    @given(series_with_datetime_index())
    def test_datetime_series_has_float_values(self, series: pd.Series):
        assert series.dtype == "float64"

    @given(series_with_datetime_index(allow_nan=False))
    def test_datetime_series_no_nan(self, series: pd.Series):
        assert_series_equal(series, series.dropna())

    @given(series_with_datetime_index(allow_infinity=False))
    def test_datetime_series_no_infinity(self, series: pd.Series):
        assert_series_equal(series, series.replace([np.inf, -np.inf], np.nan))


class TestTimedeltaIndex:
    @given(timedelta_indexes())
    def test_timedelta_indexes_is_timedelta(self, index):
        assert isinstance(index, pd.TimedeltaIndex)

    @given(timedelta_indexes(min_length=30, max_length=1000))
    def test_timedelta_indexes_size(self, index):
        assert 30 <= len(index) <= 1000

    @given(timedelta_indexes(min_length=30, max_length=30))
    def test_timedelta_indexes_size_fixed_value(self, index):
        assert len(index) == 30

    @given(timedelta_indexes(min_length=0, max_length=0))
    def test_timedelta_indexes_size_fixed_value_0(self, index):
        assert len(index) == 0

    @given(timedelta_indexes())
    def test_timedelta_indexes_boundaries(self, index):
        start_timedelta, end_timedelta = pd.Timedelta(0), pd.Timedelta("40Y")
        if len(index):
            assert index[0] >= start_timedelta
            assert index[-1] <= end_timedelta


class TestTimedeltaSeries:
    @given(series_with_timedelta_index())
    def test_timedelta_series_has_timedelta_index(self, series):
        assert isinstance(series.index, pd.TimedeltaIndex)

    @given(series_with_timedelta_index(min_length=15, max_length=1000))
    def test_timedelta_series_size(self, series):
        assert 15 <= len(series.index) <= 1000

    @given(series_with_timedelta_index(min_length=15, max_length=15))
    def test_timedelta_series_size_fixed_value(self, series):
        assert len(series.index) == 15

    @given(series_with_timedelta_index(min_length=0, max_length=0))
    def test_timedelta_series_size_fixed_value_0(self, series):
        assert len(series.index) == 0

    @given(series_with_timedelta_index())
    def test_timedelta_series_boundaries(self, series):
        start_timedelta, end_timedelta = pd.Timedelta(0), pd.Timedelta("41y")
        if len(series):
            assert series.index[0] >= start_timedelta
            assert series.index[-1] <= end_timedelta

    @given(series_with_timedelta_index())
    def test_timedelta_series_has_float_values(self, series: pd.Series):
        assert series.dtype == "float64"

    @given(series_with_timedelta_index(allow_nan=False))
    def test_timedelta_series_no_nan(self, series: pd.Series):
        assert_series_equal(series, series.dropna())

    @given(series_with_timedelta_index(allow_infinity=False))
    def test_timedelta_series_no_infinity(self, series: pd.Series):
        assert_series_equal(series, series.replace([np.inf, -np.inf], np.nan))


class TestGeneric:
    @given(available_freqs())
    def test_available_freqs_is_timedelta(self, frequency):
        assert isinstance(frequency, pd.Timedelta)

    @given(positive_bounded_integers(100000))
    def test_positive_bounded_integers_is_positive(self, integer):
        assert integer >= 0

    @given(pair_of_ordered_timedeltas())
    def test_pair_of_ordered_timedeltas_is_ordered(self, pair):
        assert pair[0] < pair[1]

    @given(pair_of_ordered_dates())
    def test_pair_of_ordered_dates_is_ordered(self, pair):
        assert pair[0] < pair[1]

    @given(samples_from(NON_UNIFORM_FREQS))
    def test_freq_to_timedelta(self, freq: str):
        with pytest.raises(ValueError):
            freq_to_timedelta(freq, approximate_if_non_uniform=False)
