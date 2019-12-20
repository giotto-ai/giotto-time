from typing import Tuple

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal, assert_frame_equal
from hypothesis.extra.numpy import arrays
from hypothesis import given, settings, HealthCheck

from giottotime.utils.hypothesis.time_indexes import (
    series_with_timedelta_index,
    series_with_period_index,
    series_with_datetime_index,
    available_freqs,
)
from giottotime.time_series_preparation import TimeSeriesPreparation
from .utils import (
    pandas_series_with_period_index,
    datetime_index_series_to_period_index_series,
    timedelta_index_series_to_period_index_series,
)


class TestConstructor:
    @given(st.datetimes(), available_freqs(), st.booleans(), st.text())
    def test_constructor_does_not_fail(
        self,
        start: pd.datetime,
        freq: pd.Timedelta,
        resample_if_not_equispaced: bool,
        output_name: str,
    ):
        TimeSeriesPreparation(
            start=start,
            freq=freq,
            resample_if_not_equispaced=resample_if_not_equispaced,
            output_name=output_name,
        )

    @given(st.datetimes(), available_freqs(), st.booleans(), st.text())
    def test_constructor_initializes_parameters(
        self,
        start: pd.datetime,
        freq: pd.Timedelta,
        resample_if_not_equispaced: bool,
        output_name: str,
    ):
        time_series_preparation = TimeSeriesPreparation(
            start=start,
            freq=freq,
            resample_if_not_equispaced=resample_if_not_equispaced,
            output_name=output_name,
        )

        assert time_series_preparation.start == start
        assert time_series_preparation.end is None
        assert time_series_preparation.freq == freq
        assert (
            time_series_preparation.resample_if_not_equispaced
            == resample_if_not_equispaced
        )
        assert time_series_preparation.output_name == output_name


class TestToTimeIndexSeries:
    @given(st.lists(st.floats()), st.datetimes(), available_freqs())
    def test_list_as_input(
        self, input_list: pd.Series, start: pd.datetime, freq: pd.Timedelta,
    ):
        time_series_preparation = TimeSeriesPreparation(start=start, freq=freq)
        computed_time_series = time_series_preparation._to_time_index_series(input_list)
        expected_time_series = pandas_series_with_period_index(
            input_list, start, freq=freq
        )
        assert_series_equal(computed_time_series, expected_time_series)

    @given(
        arrays(shape=st.integers(0, 1000), dtype=float),
        st.datetimes(),
        available_freqs(),
    )
    def test_array_as_input(
        self, input_array: np.ndarray, start: pd.datetime, freq: pd.Timedelta,
    ):
        time_series_preparation = TimeSeriesPreparation(start=start, freq=freq)
        computed_time_series = time_series_preparation._to_time_index_series(
            input_array
        )
        expected_time_series = pandas_series_with_period_index(
            input_array, start, freq=freq
        )
        assert_series_equal(computed_time_series, expected_time_series)

    @given(series_with_period_index(), st.datetimes(), available_freqs())
    def test_period_index_series_unchanged(
        self, period_index_series: pd.Series, start: pd.datetime, freq: pd.Timedelta,
    ):
        time_series_preparation = TimeSeriesPreparation(start=start, freq=freq)
        computed_time_series = time_series_preparation._to_time_index_series(
            period_index_series
        )
        assert_series_equal(computed_time_series, period_index_series)

    @given(series_with_datetime_index(), st.datetimes(), available_freqs())
    def test_datetime_index_series_unchanged(
        self, datetime_index_series: pd.Series, start: pd.datetime, freq: pd.Timedelta,
    ):
        time_series_preparation = TimeSeriesPreparation(start=start, freq=freq)
        computed_time_series = time_series_preparation._to_time_index_series(
            datetime_index_series
        )
        assert_series_equal(computed_time_series, datetime_index_series)

    @given(series_with_timedelta_index(), st.datetimes(), available_freqs())
    def test_timedelta_index_series_unchanged(
        self, timedelta_index_series: pd.Series, start: pd.datetime, freq: pd.Timedelta,
    ):
        time_series_preparation = TimeSeriesPreparation(start=start, freq=freq)
        computed_time_series = time_series_preparation._to_time_index_series(
            timedelta_index_series
        )
        assert_series_equal(computed_time_series, timedelta_index_series)

    @given(st.tuples())
    def test_wrong_input_type(self, wrong_input: Tuple):
        time_series_preparation = TimeSeriesPreparation()
        with pytest.raises(TypeError):
            time_series_preparation._to_time_index_series(wrong_input)

    @given(series_with_period_index(), st.datetimes(), available_freqs())
    def test_period_index_dataframe_unchanged(
        self, period_index_series: pd.Series, start: pd.datetime, freq: pd.Timedelta,
    ):
        period_index_dataframe = pd.DataFrame(period_index_series)
        time_series_preparation = TimeSeriesPreparation(start=start, freq=freq)
        computed_time_series = time_series_preparation._to_time_index_series(
            period_index_dataframe
        )
        assert_series_equal(computed_time_series, period_index_series)

    @given(series_with_datetime_index(), st.datetimes(), available_freqs())
    def test_datetime_index_dataframe_unchanged(
        self, datetime_index_series: pd.Series, start: pd.datetime, freq: pd.Timedelta,
    ):
        datetime_index_dataframe = pd.DataFrame(datetime_index_series)
        time_series_preparation = TimeSeriesPreparation(start=start, freq=freq)
        computed_time_series = time_series_preparation._to_time_index_series(
            datetime_index_dataframe
        )
        assert_series_equal(computed_time_series, datetime_index_series)

    @given(series_with_timedelta_index(), st.datetimes(), available_freqs())
    def test_timedelta_index_dataframe_unchanged(
        self, timedelta_index_series: pd.Series, start: pd.datetime, freq: pd.Timedelta,
    ):
        timedelta_index_dataframe = pd.DataFrame(timedelta_index_series)
        time_series_preparation = TimeSeriesPreparation(start=start, freq=freq)
        computed_time_series = time_series_preparation._to_time_index_series(
            timedelta_index_dataframe
        )
        assert_series_equal(computed_time_series, timedelta_index_series)


class TestToEquispacedTimeSeries:
    @given(
        series_with_period_index(), st.datetimes(), available_freqs(), st.text(),
    )
    def test_with_resample_false(
        self,
        series: pd.Series,
        start: pd.datetime,
        freq: pd.Timedelta,
        output_name: str,
    ):
        time_series_preparation = TimeSeriesPreparation(
            start=start,
            freq=freq,
            resample_if_not_equispaced=False,
            output_name=output_name,
        )
        computed_series = time_series_preparation._to_equispaced_time_series(series)
        assert_series_equal(computed_series, series)

    @given(series_with_period_index(), st.datetimes(), available_freqs(), st.text())
    def test_with_resample_true(
        self,
        series: pd.Series,
        start: pd.datetime,
        freq: pd.Timedelta,
        output_name: str,
    ):
        time_series_preparation = TimeSeriesPreparation(
            start=start,
            freq=freq,
            resample_if_not_equispaced=True,
            output_name=output_name,
        )
        with pytest.raises(NotImplementedError):
            time_series_preparation._to_equispaced_time_series(series)


class TestToPeriodIndex:
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    @given(series_with_period_index())
    def test_with_period_index_as_input(self, period_index_series):
        time_series_preparation = TimeSeriesPreparation()
        computed_time_series = time_series_preparation._to_period_index_time_series(
            period_index_series
        )
        assert_series_equal(computed_time_series, period_index_series)

    @given(series_with_datetime_index())
    def test_with_datetime_index_as_input(self, datetime_index_series):
        time_series_preparation = TimeSeriesPreparation()
        computed_time_series = time_series_preparation._to_period_index_time_series(
            datetime_index_series
        )
        expected_time_series = datetime_index_series_to_period_index_series(
            computed_time_series
        )
        assert_series_equal(computed_time_series, expected_time_series)

    @given(series_with_timedelta_index())
    def test_with_timedelta_index_as_input(self, timedelta_index_series):
        time_series_preparation = TimeSeriesPreparation()
        computed_time_series = time_series_preparation._to_period_index_time_series(
            timedelta_index_series
        )
        expected_time_series = timedelta_index_series_to_period_index_series(
            timedelta_index_series
        )
        assert_series_equal(computed_time_series, expected_time_series)


class TestToPeriodIndexDataFrame:
    @given(series_with_period_index(), st.text())
    def test_output_dataframe_is_correct(
        self, period_index_series: pd.Series, output_name: str
    ):
        time_series_preparation = TimeSeriesPreparation(output_name=output_name)
        computed_time_series = time_series_preparation._to_period_index_dataframe(
            period_index_series
        )
        expected_time_series = pd.DataFrame({output_name: period_index_series})
        assert_frame_equal(computed_time_series, expected_time_series)


class TestTransform:
    @given(st.lists(st.floats()), st.datetimes(), available_freqs(), st.text())
    def test_list_as_input(
        self,
        input_list: pd.Series,
        start: pd.datetime,
        freq: pd.Timedelta,
        output_name: str,
    ):
        time_series_preparation = TimeSeriesPreparation(
            start=start, freq=freq, output_name=output_name
        )
        computed_time_series = time_series_preparation.transform(input_list)
        expected_series = pandas_series_with_period_index(input_list, start, freq=freq)
        expected_time_series = pd.DataFrame({output_name: expected_series})
        assert_frame_equal(computed_time_series, expected_time_series)

    @given(
        arrays(shape=st.integers(0, 1000), dtype=float),
        st.datetimes(),
        available_freqs(),
        st.text(),
    )
    def test_array_as_input(
        self,
        input_array: np.ndarray,
        start: pd.datetime,
        freq: pd.Timedelta,
        output_name: str,
    ):
        time_series_preparation = TimeSeriesPreparation(
            start=start, freq=freq, output_name=output_name
        )
        computed_time_series = time_series_preparation.transform(input_array)
        expected_series = pandas_series_with_period_index(input_array, start, freq=freq)
        expected_time_series = pd.DataFrame({output_name: expected_series})
        assert_frame_equal(computed_time_series, expected_time_series)

    @given(series_with_period_index(), st.datetimes(), available_freqs(), st.text())
    def test_period_index_as_input(
        self,
        period_index_series: pd.Series,
        start: pd.datetime,
        freq: pd.Timedelta,
        output_name: str,
    ):
        time_series_preparation = TimeSeriesPreparation(
            start=start, freq=freq, output_name=output_name
        )
        computed_time_series = time_series_preparation.transform(period_index_series)
        expected_time_series = pd.DataFrame({output_name: period_index_series})
        assert_frame_equal(computed_time_series, expected_time_series)

    @given(series_with_datetime_index(), st.datetimes(), available_freqs(), st.text())
    def test_datetime_index_as_input(
        self,
        datetime_index_series: pd.Series,
        start: pd.datetime,
        freq: pd.Timedelta,
        output_name: str,
    ):
        time_series_preparation = TimeSeriesPreparation(
            start=start, freq=freq, output_name=output_name
        )
        computed_time_series = time_series_preparation.transform(datetime_index_series)
        expected_series = datetime_index_series_to_period_index_series(
            datetime_index_series, freq=freq
        )
        expected_time_series = pd.DataFrame({output_name: expected_series})
        assert_frame_equal(computed_time_series, expected_time_series)

    @given(series_with_timedelta_index(), st.datetimes(), available_freqs(), st.text())
    def test_timedelta_index_as_input(
        self,
        timedelta_index_series: pd.Series,
        start: pd.datetime,
        freq: pd.Timedelta,
        output_name: str,
    ):
        time_series_preparation = TimeSeriesPreparation(
            start=start, freq=freq, output_name=output_name
        )
        computed_time_series = time_series_preparation.transform(timedelta_index_series)
        expected_series = timedelta_index_series_to_period_index_series(
            timedelta_index_series, freq=freq
        )
        expected_time_series = pd.DataFrame({output_name: expected_series})
        assert_frame_equal(computed_time_series, expected_time_series)
