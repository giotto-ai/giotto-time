from typing import Union, List

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
import pytest

from ..time_series_conversion import SequenceToPandasTimeSeries, \
    PandasSeriesToPandasTimeSeries


BAD_LENGTH_START_DATE_END_DATE_FREQ_PARAMETRIZATION = [
    (20, '2018-01-01', '2019-01-01', '1M'),
    (30, '2013-01-01', '2019-01-01', '1D'),
    (10, '2015-02-01', '2019-01-01', '1Y'),
]

LENGTH_START_DATE_END_DATE_FREQ_PARAMETRIZATION = [
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


class TestSequenceToPandasTimeSeries:

    @pytest.mark.parametrize('length, start_date, end_date, freq',
                             LENGTH_START_DATE_END_DATE_FREQ_PARAMETRIZATION)
    def test_list_as_input(
            self,
            length: int,
            start_date: str,
            end_date: str,
            freq: str
    ):
        input_list = random_list_of_length(length)
        compare_input_sequence_to_expected_one(
            input_list, start_date, end_date, freq
        )

    @pytest.mark.parametrize('length, start_date, end_date, freq',
                             BAD_LENGTH_START_DATE_END_DATE_FREQ_PARAMETRIZATION)
    def test_wrong_parameters_with_list_as_input(
            self,
            length: int,
            start_date: str,
            end_date: str,
            freq: str
    ):
        input_list = random_list_of_length(length)
        with pytest.raises(ValueError):
            compare_input_sequence_to_expected_one(
                input_list, start_date, end_date, freq
            )

    @pytest.mark.parametrize('length, start_date, end_date, freq',
                             LENGTH_START_DATE_END_DATE_FREQ_PARAMETRIZATION)
    def test_np_array_as_input(
            self,
            length: int,
            start_date: str,
            end_date: str,
            freq: str
    ):
        input_array = random_array_of_length(length)
        compare_input_sequence_to_expected_one(
            input_array, start_date, end_date, freq
        )

    @pytest.mark.parametrize('length, start_date, end_date, freq',
                             BAD_LENGTH_START_DATE_END_DATE_FREQ_PARAMETRIZATION)
    def test_wrong_parameters_with_np_array_as_input(
            self,
            length: int,
            start_date: str,
            end_date: str,
            freq: str
    ):
        input_array = random_array_of_length(length)
        with pytest.raises(ValueError):
            compare_input_sequence_to_expected_one(
                input_array, start_date, end_date, freq
            )


class TestPandasToPandasTimeSeries:

    @pytest.mark.parametrize('length, start_date, end_date, freq',
                             LENGTH_START_DATE_END_DATE_FREQ_PARAMETRIZATION)
    def test_non_time_index_as_input(
            self,
            length: int,
            start_date: str,
            end_date: str,
            freq: str,
    ):
        time_series = non_time_index_series(length)
        computed_pandas_series = transform_sequence_into_time_series(
            time_series, start_date, end_date, freq
        )
        expected_pandas_series_index = pd.period_range(
            start=start_date,
            end=end_date,
            freq=freq,
            periods=time_series.shape[0]
        )
        expected_pandas_series = pd.Series(data=time_series.values,
                                           index=expected_pandas_series_index)
        assert_series_equal(computed_pandas_series, expected_pandas_series)

    @pytest.mark.parametrize('length, start_date, end_date, freq',
                             BAD_LENGTH_START_DATE_END_DATE_FREQ_PARAMETRIZATION)
    def test_wrong_parameters_non_time_index_as_input(
            self,
            length: int,
            start_date: str,
            end_date: str,
            freq: str,
    ):
        time_series = non_time_index_series(length)
        with pytest.raises(ValueError):
            transform_sequence_into_time_series(
                time_series, start_date, end_date, freq
            )

    @pytest.mark.parametrize('length, start_date, end_date, freq',
                             LENGTH_START_DATE_END_DATE_FREQ_PARAMETRIZATION)
    def test_datetime_index_series_as_input(
            self,
            length: int,
            start_date: str,
            end_date: str,
            freq: str,
    ):
        time_series = datetime_index_series(length, start_date, end_date, freq)
        computed_pandas_series = transform_pandas_series_into_pandas_time_series(
            time_series
        )
        assert_series_equal(computed_pandas_series, time_series)

    @pytest.mark.parametrize('length, start_date, end_date, freq',
                             LENGTH_START_DATE_END_DATE_FREQ_PARAMETRIZATION)
    def test_period_index_series_as_input(
            self,
            length: int,
            start_date: str,
            end_date: str,
            freq: str,
    ):
        time_series = period_index_series(length, start_date, end_date, freq)
        computed_pandas_series = transform_pandas_series_into_pandas_time_series(
            time_series
        )
        assert_series_equal(computed_pandas_series, time_series)

    def test_timedelta_index_series_as_input(self):
        pass


def random_array_of_length(length: int):
    return np.random.random(length)


def random_list_of_length(length: int):
    return list(np.random.random(length))


def non_time_index_series(length: int):
    values = random_array_of_length(length)
    return pd.Series(values)


def datetime_index_series(
        length: int,
        start_date: str,
        end_date: str,
        freq: str,
):
    index = pd.date_range(start=start_date, end=end_date, periods=length,
                          freq=freq)
    values = random_array_of_length(length)
    return pd.Series(data=values, index=index)


def period_index_series(
        length: int,
        start_date: str,
        end_date: str,
        freq: str,
):
    index = pd.period_range(start=start_date, end=end_date, periods=length,
                          freq=freq)
    values = random_array_of_length(length)
    return pd.Series(data=values, index=index)


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


def compare_input_time_series(
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
