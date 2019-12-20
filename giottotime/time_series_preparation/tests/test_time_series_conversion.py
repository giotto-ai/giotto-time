from datetime import timedelta
from typing import Union, List, Tuple

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import series as pd_series
from hypothesis.strategies import lists, datetimes, floats, integers
from pandas.testing import assert_series_equal

from giottotime.utils.hypothesis.time_indexes import (
    pair_of_ordered_dates,
    series_with_datetime_index,
    series_with_period_index,
    available_freqs,
    series_with_timedelta_index,
)
from giottotime.utils.testing_constants import DEFAULT_START, DEFAULT_END
from .utils import (
    compare_output_of_input_sequence_to_expected_one,
    compare_output_of_input_series_to_expected_one,
    transform_series_into_time_index_series,
    transform_time_index_series_into_period_index_series,
    timedelta_index_series_to_period_index_series,
    datetime_index_series_to_period_index_series,
)

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

    @settings(suppress_health_check=(HealthCheck.too_slow,))
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
    @settings(suppress_health_check=(HealthCheck.too_slow,))
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

    def test_basic_timedelta_index_as_input(self):
        timedelta_index_series = pd.Series(
            index=pd.timedelta_range(start=pd.Timedelta(days=1), freq="10D", periods=3),
            data=[1, 2, 3],
        )
        expected_series = pd.Series(
            index=pd.PeriodIndex(["1970-01-02", "1970-01-12", "1970-01-22"], freq="D"),
            data=[1, 2, 3],
        )
        computed_series = transform_time_index_series_into_period_index_series(
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

    @given(pd_series(dtype=float))
    def test_non_time_index_input(self, series: pd.Series):
        with pytest.raises(ValueError):
            transform_time_index_series_into_period_index_series(series)
