from typing import Union, List, Tuple

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
import pytest
from hypothesis import given
from hypothesis.strategies import lists, datetimes, floats, integers
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import series as pd_series
from giottotime.core.hypothesis.time_indexes import (
    pair_of_ordered_dates,
    series_with_datetime_index,
    series_with_period_index,
    available_freqs,
)


from ..time_series_conversion import (
    SequenceToPandasTimeSeries,
    PandasSeriesToPandasTimeSeries,
)

min_date = pd.Timestamp("1980-01-01")
max_date = pd.Timestamp("2020-01-01")


class TestListToPandasTimeSeries:
    @given(lists(floats()), datetimes(min_date, max_date), available_freqs())
    def test_list_start_date_freq_as_input(
        self, list_: List[float], start_date: pd.Timestamp, freq: str,
    ):
        compare_input_sequence_to_expected_one(
            list_, start_date=start_date, end_date=None, freq=freq
        )

    @given(lists(floats()), datetimes(min_date, max_date), available_freqs())
    def test_list_end_date_freq_as_input(
        self, list_: List[float], end_date: pd.Timestamp, freq: str,
    ):
        compare_input_sequence_to_expected_one(
            list_, start_date=None, end_date=end_date, freq=freq
        )

    @given(lists(floats()), pair_of_ordered_dates())
    def test_error_with_start_end_dates_as_input(
        self, list_: List[float], start_end_dates: Tuple[pd.Timestamp, pd.Timestamp],
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
    @given(
        arrays(np.float64, integers(0, 1000)),
        datetimes(min_date, max_date),
        available_freqs(),
    )
    def test_array_start_date_freq_as_input(
        self, array: np.ndarray, start_date: pd.Timestamp, freq: str,
    ):
        compare_input_sequence_to_expected_one(
            array, start_date=start_date, end_date=None, freq=freq
        )

    @given(
        arrays(np.float64, integers(0, 1000)),
        datetimes(min_date, max_date),
        available_freqs(),
    )
    def test_array_end_date_freq_as_input(
        self, array: List[float], end_date: pd.Timestamp, freq: str,
    ):
        compare_input_sequence_to_expected_one(
            array, start_date=None, end_date=end_date, freq=freq
        )

    @given(arrays(np.float64, integers(0, 1000)), pair_of_ordered_dates())
    def test_error_with_start_end_dates_as_input(
        self, array: List[float], start_end_dates: Tuple[pd.Timestamp, pd.Timestamp],
    ):
        with pytest.raises(ValueError):
            compare_input_sequence_to_expected_one(
                array, *start_end_dates, freq=None,
            )

    @given(
        arrays(np.float64, integers(0, 1000)),
        pair_of_ordered_dates(),
        available_freqs(),
    )
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
    @given(pd_series(dtype=float), datetimes(min_date, max_date), available_freqs())
    def test_series_start_date_freq_as_input(
        self, series: pd.Series, start_date: pd.Timestamp, freq: str,
    ):
        compare_input_time_series_to_expected_one(
            series, start_date=start_date, end_date=None, freq=freq,
        )

    @given(pd_series(dtype=float), datetimes(min_date, max_date), available_freqs())
    def test_series_end_date_freq_as_input(
        self, series: pd.Series, end_date: pd.Timestamp, freq: str,
    ):
        compare_input_time_series_to_expected_one(
            series, start_date=None, end_date=end_date, freq=freq
        )

    @given(pd_series(dtype=float), pair_of_ordered_dates())
    def test_error_with_start_end_dates_as_input(
        self, series: pd.Series, start_end_dates: Tuple[pd.Timestamp, pd.Timestamp],
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

    @given(series_with_datetime_index())
    def test_datetime_index_as_input(self, datetime_index_series: pd.Series):
        computed_series = transform_pandas_series_into_pandas_time_series(
            datetime_index_series
        )
        expected_series = datetime_index_series
        assert_series_equal(computed_series, expected_series)

    @given(series_with_period_index())
    def test_period_index_as_input(self, period_index_series: pd.Series):
        computed_series = transform_pandas_series_into_pandas_time_series(
            period_index_series
        )
        expected_series = period_index_series
        assert_series_equal(computed_series, expected_series)


def compare_input_sequence_to_expected_one(
    input_sequence, start_date, end_date, freq,
):
    computed_pandas_series = transform_sequence_into_time_series(
        input_sequence, start_date, end_date, freq
    )
    expected_pandas_series = pandas_series_with_period_index(
        input_sequence, start_date, end_date, freq
    )
    assert_series_equal(computed_pandas_series, expected_pandas_series)


def compare_input_time_series_to_expected_one(
    input_sequence, start_date, end_date, freq,
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
    time_series_conversion = SequenceToPandasTimeSeries(start_date, end_date, freq)
    return time_series_conversion.transform(array_like_object)


def transform_pandas_series_into_pandas_time_series(
    array_like_object: Union[np.array, list, pd.Series],
    start_date: str = None,
    end_date: str = None,
    freq: str = None,
):
    time_series_conversion = PandasSeriesToPandasTimeSeries(start_date, end_date, freq)
    return time_series_conversion.transform(array_like_object)


def pandas_series_with_period_index(
    values: Union[np.array, List[float]],
    start_date: str = None,
    end_date: str = None,
    freq: str = None,
):
    index = pd.period_range(
        start=start_date, end=end_date, periods=len(values), freq=freq,
    )
    return pd.Series(index=index, data=values)
