from typing import Iterable, Union

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
import pytest

from ..time_series_conversion_ import TimeSeriesToPandasSeriesConversion, \
    compute_index_from_length_start_date_end_date

LENGTH_START_DATE_END_DATE_FREQ_PARAMETRIZATION = [
    (20, '2018-01-01', '2019-01-01', None),
    (30, '2013-01-01', '2017-01-05', None),
    (10, '2018-02-01', '2018-07-01', None),
    (20, '2018-01-01', None, '1M'),
    (30, '2013-01-01', None, '1D'),
    (10, '2015-02-01', None, '1Y'),
    (20, None, '2019-01-01', '1M'),
    (30, None, '2013-01-01', '1D'),
    (10, None, '2015-02-01', '1Y'),
]

LENGTH_START_DATE_END_DATE_PARAMETRIZATION = [
    (20, '2018-01-01', '2019-01-01'),
    (30, '2013-01-01', '2017-01-05'),
    (29, '2013-01-01', '2017-01-05'),
    (10, '2018-02-01', '2018-07-01'),
]


class TestTimeSeriesConversion:

    @pytest.mark.parametrize('length, start_date, end_date',
                             LENGTH_START_DATE_END_DATE_PARAMETRIZATION)
    def test_compute_index_from_length_start_date_end_date(
            self,
            length,
            start_date,
            end_date
    ):
        computed_index = compute_index_from_length_start_date_end_date(
            length, start_date, end_date
        )
        assert computed_index.shape[0] == length

    @pytest.mark.parametrize('length, start_date, end_date, freq',
                             LENGTH_START_DATE_END_DATE_FREQ_PARAMETRIZATION)
    def test_np_array_as_input(
            self,
            length: int,
            start_date: str,
            end_date: str,
            freq: str
    ):
        input_array = self._random_array_of_length(length)
        computed_pandas_series = self._transform_into_time_series_given_start_date_end_date_freq(
            input_array, start_date, end_date, freq
        )
        expected_pandas_series = self._pandas_series_with_period_index(
            input_array, start_date, end_date, freq
        )
        assert_series_equal(computed_pandas_series, expected_pandas_series)

    @pytest.mark.parametrize('length, start_date, end_date, freq',
                             LENGTH_START_DATE_END_DATE_FREQ_PARAMETRIZATION)
    def test_list_as_input(
            self,
            length: int,
            start_date: str,
            end_date: str,
            freq: str
    ):
        input_list = self._random_list_of_length(length)
        computed_pandas_series = self._transform_into_time_series_given_start_date_end_date_freq(
            input_list, start_date, end_date, freq
        )
        expected_pandas_series = self._pandas_series_with_period_index(
            input_list, start_date, end_date, freq
        )
        assert_series_equal(computed_pandas_series, expected_pandas_series)

    @pytest.mark.parametrize('length, start_date, end_date, freq',
                             LENGTH_START_DATE_END_DATE_FREQ_PARAMETRIZATION)
    def test_pd_series_datetime_index_as_input(
            self,
            length: int,
            start_date: str,
            end_date: str,
            freq: str
    ):
        series_values = self._random_list_of_length(length)
        datetime_index_series = self._pandas_series_with_datetime_index(
            series_values, length, start_date, end_date, freq
        )
        freq = freq if freq is not None else self._infer_freq_from_datetime_index(datetime_index_series.index)
        expected_period_index_series = pd.Series(
            data=datetime_index_series.values,
            index=datetime_index_series.index.to_period(freq=freq)
        )
        computed_period_index_series = self._transform_into_time_series_given_start_date_end_date_freq(
            datetime_index_series
        )
        assert_series_equal(computed_period_index_series,
                            expected_period_index_series)

    @pytest.mark.parametrize('length, start_date, end_date, freq',
                             LENGTH_START_DATE_END_DATE_FREQ_PARAMETRIZATION)
    def test_pd_series_timedelta_index_as_input(
            self,
            length: int,
            start_date: str,
            end_date: str,
            freq: str,
    ):
        series_values = self._random_list_of_length(length)
        timedelta_index_series = self._pandas_series_with_timedelta_index(
            series_values, length, start_date, end_date, freq
        )
        expected_period_index_series = pd.Series(
            data=timedelta_index_series.values,
            index=timedelta_index_series.index.to_period(freq=freq)
        )
        computed_period_index_series = self._transform_into_time_series_given_start_date_end_date_freq(
            timedelta_index_series
        )
        assert_series_equal(computed_period_index_series,
                            expected_period_index_series)


    def _random_array_of_length(self, length: int):
        return np.random.random(length)


    def _random_list_of_length(self, length: int):
        return list(np.random.random(length))


    def _pandas_series_with_period_index(self, values: Iterable, start_date: str,
                                         end_date: str, freq: str):
        length = len(values)
        if freq is None:
            index = compute_index_from_length_start_date_end_date(
                length, start_date, end_date)
        else:
            index = pd.period_range(start=start_date, end=end_date, periods=length,
                                    freq=freq)
        return pd.Series(index=index, data=values)


    def _pandas_series_with_datetime_index(
            self,
            values: Iterable,
            length: int,
            start_date: str,
            end_date: str,
            freq: str,
    ):
        index = pd.date_range(start=start_date, end=end_date, periods=length,
                              freq=freq)
        return pd.Series(index=index, data=values)


    def _pandas_series_with_timedelta_index(
            self,
            values: Iterable,
            length: int,
            start_date: str,
            end_date: str,
            freq: str,
    ):
        index = pd.timedelta_range(start=start_date, end=end_date,
                                   periods=length, freq=freq)
        return pd.Series(index=index, data=values)

    def _transform_into_time_series_given_start_date_end_date_freq(
            self,
            array_like_object: Union[np.array, list, pd.Series],
            start_date: str = None,
            end_date: str = None,
            freq: str = None,
    ):
        time_series_conversion = TimeSeriesToPandasSeriesConversion(
            start_date, end_date, freq
        )
        return time_series_conversion.fit_transform(array_like_object)

    def _infer_freq_from_datetime_index(self, index: pd.DatetimeIndex):
        if pd.infer_freq(index) is not None:
            return pd.infer_freq(index)
        else:
            return '1D'
