from typing import Union, List, Tuple

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
import pytest
from hypothesis import given
from hypothesis.strategies import lists, datetimes, floats, integers
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import series as pd_series
from giottotime.core.hypothesis.time_indexes import available_freq, \
    samples_from, pair_of_ordered_dates, series_with_datetime_index, \
    available_freqs


from ..time_series_conversion import SequenceToPandasTimeSeries, \
    PandasSeriesToPandasTimeSeries

min_date = pd.Timestamp('1980-01-01')
max_date = pd.Timestamp('2020-01-01')


class TestListToPandasTimeSeries:

    @given(lists(floats()), datetimes(min_date, max_date), available_freqs())
    def test_list_start_date_freq_as_input(
            self,
            list_: List[float],
            start_date: pd.Timestamp,
            freq: str,
    ):
        compare_input_sequence_to_expected_one(
            list_, start_date=start_date, end_date=None, freq=freq
        )

    @given(lists(floats()), datetimes(min_date, max_date), available_freqs())
    def test_list_end_date_freq_as_input(
            self,
            list_: List[float],
            end_date: pd.Timestamp,
            freq: str,
    ):
        compare_input_sequence_to_expected_one(
            list_, start_date=None, end_date=end_date, freq=freq
        )

    @given(lists(floats()), pair_of_ordered_dates())
    def test_error_with_start_end_dates_as_input(
            self,
            list_: List[float],
            start_end_dates: Tuple[pd.Timestamp, pd.Timestamp],
    ):
        with pytest.raises(ValueError):
            compare_input_sequence_to_expected_one(
                list_, *start_end_dates, freq=None,
            )

    @given(lists(floats()), pair_of_ordered_dates(), available_freqs())
    def test_too_many_parameters(
            self,
            list_: List[float],
            start_end_dates: Tuple[pd.Timestamp, pd.Timestamp],
            freq: str,
    ):
        with pytest.raises(ValueError):
            compare_input_sequence_to_expected_one(
                list_, *start_end_dates, freq=freq,
            )


class TestArrayToPandasTimeSeries:

    @given(arrays(np.float64, integers(0, 1000)),
           datetimes(min_date, max_date),
           available_freqs())
    def test_array_start_date_freq_as_input(
            self,
            array: np.ndarray,
            start_date: pd.Timestamp,
            freq: str,
    ):
        compare_input_sequence_to_expected_one(
            array, start_date=start_date, end_date=None, freq=freq
        )

    @given(arrays(np.float64, integers(0, 1000)),
           datetimes(min_date, max_date),
           available_freqs())
    def test_array_end_date_freq_as_input(
            self,
            array: List[float],
            end_date: pd.Timestamp,
            freq: str,
    ):
        compare_input_sequence_to_expected_one(
            array, start_date=None, end_date=end_date, freq=freq
        )

    @given(arrays(np.float64, integers(0, 1000)), pair_of_ordered_dates())
    def test_error_with_start_end_dates_as_input(
            self,
            array: List[float],
            start_end_dates: Tuple[pd.Timestamp, pd.Timestamp],
    ):
        with pytest.raises(ValueError):
            compare_input_sequence_to_expected_one(
                array, *start_end_dates, freq=None,
            )

    @given(arrays(np.float64, integers(0, 1000)),
           pair_of_ordered_dates(),
           available_freqs())
    def test_too_many_parameters(
            self,
            array: List[float],
            start_end_dates: Tuple[pd.Timestamp, pd.Timestamp],
            freq: str,
    ):
        with pytest.raises(ValueError):
            compare_input_sequence_to_expected_one(
                array, *start_end_dates, freq=freq,
            )


class TestPandasSeriesToPandasTimeSeries:

    @given(pd_series(dtype=float),
           datetimes(min_date, max_date),
           available_freqs())
    def test_array_start_date_freq_as_input(
            self,
            series: pd.Series,
            start_date: pd.Timestamp,
            freq: str,
    ):
        compare_input_time_series_to_expected_one(
            series, start_date=start_date, end_date=None, freq=freq,
        )

    @given(pd_series(dtype=float),
           datetimes(min_date, max_date),
           available_freqs())
    def test_array_end_date_freq_as_input(
            self,
            series: pd.Series,
            end_date: pd.Timestamp,
            freq: str,
    ):
        compare_input_time_series_to_expected_one(
            series, start_date=None, end_date=end_date, freq=freq
        )

    @given(pd_series(dtype=float), pair_of_ordered_dates())
    def test_error_with_start_end_dates_as_input(
            self,
            series: pd.Series,
            start_end_dates: Tuple[pd.Timestamp, pd.Timestamp],
    ):
        with pytest.raises(ValueError):
            compare_input_time_series_to_expected_one(
                series, *start_end_dates, freq=None,
            )

    @given(pd_series(dtype=float), pair_of_ordered_dates(), available_freqs())
    def test_too_many_parameters(
            self,
            series: pd.Series,
            start_end_dates: Tuple[pd.Timestamp, pd.Timestamp],
            freq: str,
    ):
        with pytest.raises(ValueError):
            compare_input_time_series_to_expected_one(
                series, *start_end_dates, freq=freq,
            )

#
# class TestSequenceToPandasTimeSeries:
#
#     @pytest.mark.parametrize('length, start_date, end_date, freq',
#                              BAD_LENGTH_START_DATE_END_DATE_FREQ_PARAMETRIZATION)
#     def test_wrong_parameters_with_list_as_input(
#             self,
#             length: int,
#             start_date: str,
#             end_date: str,
#             freq: str
#     ):
#         input_list = random_list_of_length(length)
#         with pytest.raises(ValueError):
#             compare_input_sequence_to_expected_one(
#                 input_list, start_date, end_date, freq
#             )
#
#     @pytest.mark.parametrize('length, start_date, end_date, freq',
#                              LENGTH_START_DATE_END_DATE_FREQ_PARAMETRIZATION)
#     def test_np_array_as_input(
#             self,
#             length: int,
#             start_date: str,
#             end_date: str,
#             freq: str
#     ):
#         input_array = random_array_of_length(length)
#         compare_input_sequence_to_expected_one(
#             input_array, start_date, end_date, freq
#         )
#
#     @pytest.mark.parametrize('length, start_date, end_date, freq',
#                              BAD_LENGTH_START_DATE_END_DATE_FREQ_PARAMETRIZATION)
#     def test_wrong_parameters_with_np_array_as_input(
#             self,
#             length: int,
#             start_date: str,
#             end_date: str,
#             freq: str
#     ):
#         input_array = random_array_of_length(length)
#         with pytest.raises(ValueError):
#             compare_input_sequence_to_expected_one(
#                 input_array, start_date, end_date, freq
#             )
#
#
# class TestPandasToPandasTimeSeries:
#
#     @pytest.mark.parametrize('length, start_date, end_date, freq',
#                              LENGTH_START_DATE_END_DATE_FREQ_PARAMETRIZATION)
#     def test_non_time_index_as_input(
#             self,
#             length: int,
#             start_date: str,
#             end_date: str,
#             freq: str,
#     ):
#         time_series = non_time_index_series(length)
#         computed_pandas_series = transform_sequence_into_time_series(
#             time_series, start_date, end_date, freq
#         )
#         expected_pandas_series_index = pd.period_range(
#             start=start_date,
#             end=end_date,
#             freq=freq,
#             periods=time_series.shape[0]
#         )
#         expected_pandas_series = pd.Series(data=time_series.values,
#                                            index=expected_pandas_series_index)
#         assert_series_equal(computed_pandas_series, expected_pandas_series)
#
#     @pytest.mark.parametrize('length, start_date, end_date, freq',
#                              BAD_LENGTH_START_DATE_END_DATE_FREQ_PARAMETRIZATION)
#     def test_wrong_parameters_non_time_index_as_input(
#             self,
#             length: int,
#             start_date: str,
#             end_date: str,
#             freq: str,
#     ):
#         time_series = non_time_index_series(length)
#         with pytest.raises(ValueError):
#             transform_sequence_into_time_series(
#                 time_series, start_date, end_date, freq
#             )
#
#     @pytest.mark.parametrize('length, start_date, end_date, freq',
#                              LENGTH_START_DATE_END_DATE_FREQ_PARAMETRIZATION)
#     def test_datetime_index_series_as_input(
#             self,
#             length: int,
#             start_date: str,
#             end_date: str,
#             freq: str,
#     ):
#         time_series = datetime_index_series(length, start_date, end_date, freq)
#         computed_pandas_series = transform_pandas_series_into_pandas_time_series(
#             time_series
#         )
#         assert_series_equal(computed_pandas_series, time_series)
#
#     @pytest.mark.parametrize('length, start_date, end_date, freq',
#                              LENGTH_START_DATE_END_DATE_FREQ_PARAMETRIZATION)
#     def test_period_index_series_as_input(
#             self,
#             length: int,
#             start_date: str,
#             end_date: str,
#             freq: str,
#     ):
#         time_series = period_index_series(length, start_date, end_date, freq)
#         computed_pandas_series = transform_pandas_series_into_pandas_time_series(
#             time_series
#         )
#         assert_series_equal(computed_pandas_series, time_series)
#
#     def test_timedelta_index_series_as_input(self):
#         pass


def compare_input_sequence_to_expected_one(
        input_sequence,
        start_date,
        end_date,
        freq,
):
    computed_pandas_series = transform_sequence_into_time_series(
        input_sequence, start_date, end_date, freq
    )
    expected_pandas_series = pandas_series_with_period_index(
        input_sequence, start_date, end_date, freq
    )
    assert_series_equal(computed_pandas_series, expected_pandas_series)


def compare_input_time_series_to_expected_one(
        input_sequence,
        start_date,
        end_date,
        freq,
):
    computed_pandas_series = transform_pandas_series_into_pandas_time_series(
        input_sequence, start_date, end_date, freq
    )
    expected_pandas_series = pandas_series_with_period_index(
        input_sequence.values, start_date, end_date, freq
    )
    assert_series_equal(computed_pandas_series, expected_pandas_series)


def transform_sequence_into_time_series(
        array_like_object: Union[np.array, list, pd.Series],
        start_date: str = None,
        end_date: str = None,
        freq: str = None,
):
    time_series_conversion = SequenceToPandasTimeSeries(
        start_date, end_date, freq
    )
    return time_series_conversion.transform(array_like_object)


def transform_pandas_series_into_pandas_time_series(
        array_like_object: Union[np.array, list, pd.Series],
        start_date: str = None,
        end_date: str = None,
        freq: str = None,
):
    time_series_conversion = PandasSeriesToPandasTimeSeries(
        start_date, end_date, freq
    )
    return time_series_conversion.transform(array_like_object)


def pandas_series_with_period_index(
        values: Union[np.array, List[float]],
        start_date: str = None,
        end_date: str = None,
        freq: str = None,
):
    index = pd.period_range(
        start=start_date,
        end=end_date,
        periods=len(values),
        freq=freq,
    )
    return pd.Series(index=index, data=values)
