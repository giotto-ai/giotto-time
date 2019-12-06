from typing import Union, List, Tuple, Optional

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
import pytest
from hypothesis import given, assume
from hypothesis.strategies import lists, datetimes, floats, integers
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import series as pd_series
from giottotime.core.hypothesis.time_indexes import (
    pair_of_ordered_dates,
    series_with_datetime_index,
    series_with_period_index,
    available_freqs,
    series_with_timedelta_index,
)

from giottotime.time_series_preparation.time_series_conversion import (
    SequenceToTimeIndexSeries,
    PandasSeriesToTimeIndexSeries,
    TimeIndexSeriesToPeriodIndexSeries,
    count_not_none,
)
from giottotime.core.testing_constants import DEFAULT_START, DEFAULT_END, DEFAULT_FREQ

PandasDate = Union[pd.datetime, pd.Timestamp, str]


class TestListToTimeIndexSeries:
    @given(lists(floats()), datetimes(DEFAULT_START, DEFAULT_END), available_freqs())
    def test_list_start_freq_as_input(
        self, list_: List[float], start: pd.Timestamp, freq: pd.Timedelta,
    ):
        compare_output_of_input_sequence_to_expected_one(
            list_, start=start, end=None, freq=freq
        )

    @given(lists(floats()), datetimes(DEFAULT_START, DEFAULT_END), available_freqs())
    def test_list_end_freq_as_input(
        self, list_: List[float], end: pd.Timestamp, freq: pd.Timedelta,
    ):
        compare_output_of_input_sequence_to_expected_one(
            list_, start=None, end=end, freq=freq
        )

    @given(lists(floats()), pair_of_ordered_dates())
    def test_error_with_start_end_as_input(
        self, list_: List[float], start_end: Tuple[pd.Timestamp, pd.Timestamp],
    ):
        with pytest.raises(ValueError):
            compare_output_of_input_sequence_to_expected_one(
                list_, *start_end, freq=None,
            )

    @given(lists(floats()), pair_of_ordered_dates(), available_freqs())
    def test_too_many_parameters(
        self,
        list_: List[float],
        start_end: Tuple[pd.Timestamp, pd.Timestamp],
        freq: pd.Timedelta,
    ):
        with pytest.raises(ValueError):
            compare_output_of_input_sequence_to_expected_one(
                list_, *start_end, freq=freq,
            )

    @given(lists(floats()), datetimes(DEFAULT_START, DEFAULT_END))
    def test_list_and_only_start(
        self, list_: List[float], start: pd.Timestamp,
    ):
        compare_output_of_input_sequence_to_expected_one(
            list_, start=start, end=None, freq=None
        )

    @given(lists(floats()), datetimes(DEFAULT_START, DEFAULT_END))
    def test_list_and_only_end(
        self, list_: List[float], end: pd.Timestamp,
    ):
        compare_output_of_input_sequence_to_expected_one(
            list_, start=None, end=end, freq=None
        )

    @given(lists(floats()), available_freqs())
    def test_list_and_only_freq(
        self, list_: List[float], freq: pd.Timedelta,
    ):
        compare_output_of_input_sequence_to_expected_one(
            list_, start=None, end=None, freq=freq
        )

    @given(lists(floats()))
    def test_only_list(
        self, list_: List[float],
    ):
        compare_output_of_input_sequence_to_expected_one(
            list_, start=None, end=None, freq=None
        )


class TestArrayToTimeIndexSeries:
    @given(
        arrays(np.float64, integers(0, 1000)),
        datetimes(DEFAULT_START, DEFAULT_END),
        available_freqs(),
    )
    def test_array_start_freq_as_input(
        self, array: np.ndarray, start: pd.Timestamp, freq: str,
    ):
        compare_output_of_input_sequence_to_expected_one(
            array, start=start, end=None, freq=freq
        )

    @given(
        arrays(np.float64, integers(0, 1000)),
        datetimes(DEFAULT_START, DEFAULT_END),
        available_freqs(),
    )
    def test_array_end_freq_as_input(
        self, array: np.ndarray, end: pd.Timestamp, freq: str,
    ):
        compare_output_of_input_sequence_to_expected_one(
            array, start=None, end=end, freq=freq
        )

    @given(arrays(np.float64, integers(0, 1000)), pair_of_ordered_dates())
    def test_error_with_start_end_as_input(
        self, array: np.ndarray, start_end: Tuple[pd.Timestamp, pd.Timestamp],
    ):
        with pytest.raises(ValueError):
            compare_output_of_input_sequence_to_expected_one(
                array, *start_end, freq=None,
            )

    @given(
        arrays(np.float64, integers(0, 1000)),
        pair_of_ordered_dates(),
        available_freqs(),
    )
    def test_too_many_parameters(
        self,
        array: np.ndarray,
        start_end: Tuple[pd.Timestamp, pd.Timestamp],
        freq: str,
    ):
        with pytest.raises(ValueError):
            compare_output_of_input_sequence_to_expected_one(
                array, *start_end, freq=freq,
            )

    @given(arrays(np.float64, integers(0, 1000)), datetimes(DEFAULT_START, DEFAULT_END))
    def test_array_and_only_start(
        self, array: np.ndarray, start: pd.Timestamp,
    ):
        compare_output_of_input_sequence_to_expected_one(
            array, start=start, end=None, freq=None
        )

    @given(arrays(np.float64, integers(0, 1000)), datetimes(DEFAULT_START, DEFAULT_END))
    def test_list_and_only_end(
        self, array: np.ndarray, end: pd.Timestamp,
    ):
        compare_output_of_input_sequence_to_expected_one(
            array, start=None, end=end, freq=None
        )

    @given(arrays(np.float64, integers(0, 1000)), available_freqs())
    def test_list_and_only_freq(
        self, array: np.ndarray, freq: pd.Timedelta,
    ):
        compare_output_of_input_sequence_to_expected_one(
            array, start=None, end=None, freq=freq
        )

    @given(arrays(np.float64, integers(0, 1000)))
    def test_only_list(
        self, array: np.ndarray,
    ):
        compare_output_of_input_sequence_to_expected_one(
            array, start=None, end=None, freq=None
        )


class TestPandasSeriesToTimeIndexSeries:
    @given(
        pd_series(dtype=float), datetimes(DEFAULT_START, DEFAULT_END), available_freqs()
    )
    def test_series_start_freq_as_input(
        self, series: pd.Series, start: pd.Timestamp, freq: str,
    ):
        compare_output_of_input_series_to_expected_one(
            series, start=start, end=None, freq=freq,
        )

    @given(
        pd_series(dtype=float), datetimes(DEFAULT_START, DEFAULT_END), available_freqs()
    )
    def test_series_end_freq_as_input(
        self, series: pd.Series, end: pd.Timestamp, freq: str,
    ):
        compare_output_of_input_series_to_expected_one(
            series, start=None, end=end, freq=freq
        )

    @given(pd_series(dtype=float), pair_of_ordered_dates())
    def test_error_with_start_end_as_input(
        self, series: pd.Series, start_end: Tuple[pd.Timestamp, pd.Timestamp],
    ):
        with pytest.raises(ValueError):
            compare_output_of_input_series_to_expected_one(
                series, *start_end, freq=None,
            )

    @given(pd_series(dtype=float), pair_of_ordered_dates(), available_freqs())
    def test_too_many_parameters(
        self,
        series: pd.Series,
        start_end: Tuple[pd.Timestamp, pd.Timestamp],
        freq: str,
    ):
        with pytest.raises(ValueError):
            compare_output_of_input_series_to_expected_one(
                series, *start_end, freq=freq,
            )

    @given(pd_series(dtype=float), datetimes(DEFAULT_START, DEFAULT_END))
    def test_series_and_only_start(
        self, series: pd.Series, start: pd.Timestamp,
    ):
        compare_output_of_input_series_to_expected_one(
            series, start=start, end=None, freq=None
        )

    @given(pd_series(dtype=float), datetimes(DEFAULT_START, DEFAULT_END))
    def test_series_and_only_end(
        self, series: pd.Series, end: pd.Timestamp,
    ):
        compare_output_of_input_series_to_expected_one(
            series, start=None, end=end, freq=None
        )

    @given(pd_series(dtype=float), available_freqs())
    def test_series_and_only_freq(
        self, series: pd.Series, freq: pd.Timedelta,
    ):
        compare_output_of_input_series_to_expected_one(
            series, start=None, end=None, freq=freq
        )

    @given(pd_series(dtype=float))
    def test_only_series(
        self, series: pd.Series,
    ):
        compare_output_of_input_series_to_expected_one(
            series, start=None, end=None, freq=None
        )

    @given(series_with_timedelta_index())
    def test_timedelta_index_as_input(self, timedelta_index_series: pd.Series):
        computed_series = transform_series_into_time_index_series(
            timedelta_index_series
        )
        expected_series = timedelta_index_series
        assert_series_equal(computed_series, expected_series)

    @given(series_with_datetime_index())
    def test_datetime_index_as_input(self, datetime_index_series: pd.Series):
        computed_series = transform_series_into_time_index_series(datetime_index_series)
        expected_series = datetime_index_series
        assert_series_equal(computed_series, expected_series)

    @given(series_with_period_index())
    def test_period_index_as_input(self, period_index_series: pd.Series):
        computed_series = transform_series_into_time_index_series(period_index_series)
        expected_series = period_index_series
        assert_series_equal(computed_series, expected_series)


class TestTimeIndexSeriesToPeriodIndexSeries:
    @given(series_with_period_index())
    def test_only_period_index_as_input(self, period_index_series: pd.Series):
        computed_series = transform_time_index_series_into_period_index_series(
            period_index_series
        )
        expected_series = period_index_series
        assert_series_equal(computed_series, expected_series)

    @given(series_with_period_index(), available_freqs())
    def test_period_index_and_freq_as_input(
        self, period_index_series: pd.Series, freq: pd.Timedelta
    ):
        computed_series = transform_time_index_series_into_period_index_series(
            period_index_series, freq
        )
        expected_series = period_index_series
        assert_series_equal(computed_series, expected_series)

    @given(series_with_datetime_index())
    def test_only_datetime_index_as_input(self, datetime_index_series: pd.Series):
        computed_series = transform_time_index_series_into_period_index_series(
            datetime_index_series
        )
        expected_series = datetime_index_series_to_period_index_series(
            datetime_index_series
        )
        assert_series_equal(computed_series, expected_series)

    @given(series_with_datetime_index(), available_freqs())
    def test_datetime_index_and_freq_as_input(
        self, datetime_index_series: pd.Series, freq: pd.Timedelta
    ):
        computed_series = transform_time_index_series_into_period_index_series(
            datetime_index_series, freq=freq
        )
        expected_series = datetime_index_series_to_period_index_series(
            datetime_index_series, freq
        )
        assert_series_equal(computed_series, expected_series)

    @given(series_with_timedelta_index())
    def test_only_timedelta_index_as_input(self, timedelta_index_series: pd.Series):
        computed_series = transform_time_index_series_into_period_index_series(
            timedelta_index_series
        )
        expected_series = timedelta_index_series_to_period_index_series(
            timedelta_index_series
        )
        assert_series_equal(computed_series, expected_series)

    @given(series_with_timedelta_index(), available_freqs())
    def test_timedelta_index_and_freq_as_input(
        self, timedelta_index_series: pd.Series, freq: pd.Timedelta
    ):
        computed_series = transform_time_index_series_into_period_index_series(
            timedelta_index_series, freq=freq
        )
        expected_series = timedelta_index_series_to_period_index_series(
            timedelta_index_series, freq=freq
        )
        assert_series_equal(computed_series, expected_series)


def compare_output_of_input_sequence_to_expected_one(
    input_sequence, start, end, freq,
):
    computed_pandas_series = transform_sequence_into_time_index_series(
        input_sequence, start, end, freq
    )
    expected_pandas_series = pandas_series_with_period_index(
        input_sequence, start, end, freq
    )
    assert_series_equal(computed_pandas_series, expected_pandas_series)


def compare_output_of_input_series_to_expected_one(
    input_sequence, start, end, freq,
):
    computed_pandas_series = transform_series_into_time_index_series(
        input_sequence, start, end, freq
    )
    expected_pandas_series = pandas_series_with_period_index(
        input_sequence.values, start, end, freq
    )
    assert_series_equal(computed_pandas_series, expected_pandas_series)


def transform_sequence_into_time_index_series(
    array_like_object: Union[np.array, list, pd.Series],
    start: str = None,
    end: str = None,
    freq: str = None,
):
    time_series_conversion = SequenceToTimeIndexSeries(start, end, freq)
    return time_series_conversion.transform(array_like_object)


def transform_series_into_time_index_series(
    array_like_object: Union[np.array, list, pd.Series],
    start: str = None,
    end: str = None,
    freq: str = None,
):
    time_series_conversion = PandasSeriesToTimeIndexSeries(start, end, freq)
    return time_series_conversion.transform(array_like_object)


def transform_time_index_series_into_period_index_series(
    series: pd.Series, freq: pd.Timedelta = None,
):
    to_period_conversion = TimeIndexSeriesToPeriodIndexSeries(freq=freq)
    return to_period_conversion.transform(series)


def pandas_series_with_period_index(
    values: Union[np.array, List[float]],
    start: str = None,
    end: str = None,
    freq: str = None,
):
    start, end, freq = _initialize_start_end_freq(start, end, freq)
    index = pd.period_range(start=start, end=end, periods=len(values), freq=freq,)
    return pd.Series(index=index, data=values)


def _initialize_start_end_freq(start: PandasDate, end: PandasDate, freq: pd.Timedelta):
    not_none_params = count_not_none(start, end, freq)
    if not_none_params == 0:
        start, end, freq = _default_params_initialization()
    elif not_none_params == 1:
        start, end, freq = _one_not_none_param_initialization(start, end, freq)
    elif not_none_params == 2:
        start, end, freq = _two_not_none_params_initialization(start, end, freq)
    else:
        raise ValueError(
            "Of the three parameters: start, end, and "
            "freq, exactly two must be specified"
        )
    return start, end, freq


def _default_params_initialization():
    start = DEFAULT_START
    end = None
    freq = DEFAULT_FREQ
    return start, end, freq


def _one_not_none_param_initialization(start, end, freq):
    if start is not None:
        start = start
        end = None
        freq = DEFAULT_FREQ
    elif end is not None:
        start = None
        end = end
        freq = DEFAULT_FREQ
    else:
        start = DEFAULT_START
        end = None
        freq = freq
    return start, end, freq


def _two_not_none_params_initialization(start, end, freq):
    start = start
    end = end
    freq = freq
    return start, end, freq


def datetime_index_series_to_period_index_series(
    datetime_index_series: pd.Series, freq: Optional[pd.Timedelta] = None
):
    if datetime_index_series.index.freq is not None:
        try:
            return pd.Series(
                index=pd.PeriodIndex(datetime_index_series.index),
                data=datetime_index_series.values,
            )
        except Exception as e:
            print(freq, datetime_index_series.index.freq)
            raise e
    else:
        freq = "1D" if freq is None else freq
        return pd.Series(
            index=pd.PeriodIndex(datetime_index_series.index, freq=freq),
            data=datetime_index_series.values,
        )


def timedelta_index_series_to_period_index_series(
    timedelta_index_series: pd.Series, freq: Optional[pd.Timedelta] = None
):
    datetime_index = pd.to_datetime(timedelta_index_series.index)
    if datetime_index.freq is None:
        freq = "1D" if freq is None else freq
        period_index = pd.PeriodIndex(datetime_index, freq=freq)
    else:
        period_index = pd.PeriodIndex(datetime_index)
    return pd.Series(index=period_index, data=timedelta_index_series.values)
