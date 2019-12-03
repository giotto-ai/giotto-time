from hypothesis import given

from ..time_indexes import *


class TestPeriodIndex:

    @given(period_indexes())
    def test_period_indexes_is_period(self, index):
        assert isinstance(index, pd.PeriodIndex)

    @given(period_indexes(max_length=1000))
    def test_period_indexes_size(self, index):
        assert len(index) <= 1000

    @given(period_indexes())
    def test_period_indexes_boundaries(self, index):
        start_datetime = pd.Period('1979-12-31').to_timestamp()
        end_datetime = pd.Period('2020-01-01').to_timestamp()
        if len(index):
            assert index[0].to_timestamp() >= start_datetime
            assert index[-1].to_timestamp() <= end_datetime


class TestPeriodSeries:

    @given(series_with_period_index())
    def test_period_series_has_period_index(self, series):
        assert isinstance(series.index, pd.PeriodIndex)

    @given(series_with_period_index(max_length=1000))
    def test_period_series_size(self, series):
        assert len(series.index) <= 1000

    @given(series_with_period_index())
    def test_period_series_boundaries(self, series):
        start_datetime = pd.Period('1979-12-31').to_timestamp()
        end_datetime = pd.Period('2020-01-01').to_timestamp()
        if len(series):
            assert series.index[0].to_timestamp() >= start_datetime
            assert series.index[-1].to_timestamp() <= end_datetime

    @given(series_with_period_index())
    def test_period_series_has_float_values(self, series: pd.Series):
        assert series.dtype == 'float64'


class TestDatetimeIndex:

    @given(datetime_indexes())
    def test_datetime_indexes_is_datetime(self, index):
        assert isinstance(index, pd.DatetimeIndex)

    @given(datetime_indexes(max_length=1000))
    def test_datetime_indexes_size(self, index):
        assert len(index) <= 1000

    @given(datetime_indexes())
    def test_datetime_indexes_boundaries(self, index):
        start_datetime = pd.Timestamp('1979-12-31')
        end_datetime = pd.Timestamp('2020-01-02')
        if len(index):
            assert index[0] >= start_datetime
            assert index[-1] <= end_datetime


class TestDatetimeSeries:

    @given(series_with_datetime_index())
    def test_datetime_series_has_datetime_index(self, series):
        assert isinstance(series.index, pd.DatetimeIndex)

    @given(series_with_datetime_index(max_length=1000))
    def test_datetime_series_size(self, series):
        assert len(series.index) <= 1000

    @given(series_with_datetime_index())
    def test_datetime_series_boundaries(self, series):
        start_datetime = pd.Timestamp('1979-12-31')
        end_datetime = pd.Timestamp('2020-01-02')
        if len(series):
            assert series.index[0] >= start_datetime
            assert series.index[-1] <= end_datetime

    @given(series_with_datetime_index())
    def test_datetime_series_has_float_values(self, series: pd.Series):
        assert series.dtype == 'float64'


class TestTimedeltaIndex:

    @given(timedelta_indexes())
    def test_timedelta_indexes_is_timedelta(self, index):
        assert isinstance(index, pd.TimedeltaIndex)

    @given(timedelta_indexes(max_length=1000))
    def test_timedelta_indexes_size(self, index):
        assert len(index) <= 1000

    @given(timedelta_indexes())
    def test_timedelta_indexes_boundaries(self, index):
        start_timedelta, end_timedelta = pd.Timedelta(0), pd.Timedelta('40Y')
        if len(index):
            assert index[0] >= start_timedelta
            assert index[-1] <= end_timedelta


class TestTimedeltaSeries:

    @given(series_with_timedelta_index())
    def test_timedelta_series_has_timedelta_index(self, series):
        assert isinstance(series.index, pd.TimedeltaIndex)

    @given(series_with_timedelta_index(max_length=1000))
    def test_timedelta_series_size(self, series):
        assert len(series.index) <= 1000

    @given(series_with_timedelta_index())
    def test_timedelta_series_boundaries(self, series):
        start_timedelta, end_timedelta = pd.Timedelta(0), pd.Timedelta('41y')
        if len(series):
            assert series.index[0] >= start_timedelta
            assert series.index[-1] <= end_timedelta

    @given(series_with_timedelta_index())
    def test_timedelta_series_has_float_values(self, series: pd.Series):
        assert series.dtype == 'float64'


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


