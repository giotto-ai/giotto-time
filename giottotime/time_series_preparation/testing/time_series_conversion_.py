from typing import Iterable, Union, Tuple

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
import pytest
from hypothesis import given
from hypothesis.strategies import lists
from hypothesis.extra.numpy import arrays
from giottotime.core.hypothesis.time_indexes import available_freq, \
    samples_from, pair_of_ordered_dates, series_with_datetime_index

from ..time_series_conversion import SequenceToPandasTimeSeries, \
    PandasSeriesToPandasTimeSeries


class TestSequenceToPandasTimeSeries:

    @given(arrays(), pair_of_ordered_dates(), samples_from(available_freq))
    def test_np_array_to_pandas_time_series(
            self,
            array: np.ndarray,
            start_end_dates: Tuple[pd.Timestamp, pd.Timestamp],
            freq: str,
    ):
        start_date, end_date = start_end_dates
        computed_pandas_series = self._transform_into_time_series_given_start_date_end_date_freq(
            array, start_date, end_date, freq
        )
        expected_pandas_series = self._pandas_series_with_period_index(
            array, start_date, end_date, freq
        )
        assert_series_equal(computed_pandas_series, expected_pandas_series)

    @given(lists(), pair_of_ordered_dates(), samples_from(available_freq))
    def test_list_to_pandas_time_series(
            self,
            list_: np.ndarray,
            start_end_dates: Tuple[pd.Timestamp, pd.Timestamp],
            freq: str,
    ):
        start_date, end_date = start_end_dates
        computed_pandas_series = self._transform_into_time_series_given_start_date_end_date_freq(
            list_, start_date, end_date, freq
        )
        expected_pandas_series = self._pandas_series_with_period_index(
            list_, start_date, end_date, freq
        )
        assert_series_equal(computed_pandas_series, expected_pandas_series)

    @given(series_with_datetime_index())
    def test_pd_series_datetime_index_to_pandas_time_series(
            self,
            datetime_index_series: pd.Series,
    ):
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
            start_date: pd.Timestamp = None,
            end_date: pd.Timestamp = None,
            freq: str = None,
    ):
        time_series_conversion = SequenceToPandasTimeSeries(
            start_date, end_date, freq
        )
        return time_series_conversion.fit_transform(array_like_object)

    def _infer_freq_from_datetime_index(self, index: pd.DatetimeIndex):
        if pd.infer_freq(index) is not None:
            return pd.infer_freq(index)
        else:
            return '1D'
