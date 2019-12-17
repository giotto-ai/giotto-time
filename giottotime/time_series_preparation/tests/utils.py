from typing import List, Union, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from giottotime.time_series_preparation.time_series_conversion import (
    SequenceToTimeIndexSeries,
    PandasSeriesToTimeIndexSeries,
    TimeIndexSeriesToPeriodIndexSeries,
    count_not_none,
)
from giottotime.utils.testing_constants import DEFAULT_START, DEFAULT_FREQ

PandasDate = Union[pd.datetime, pd.Timestamp, str]


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
    start: Optional[str] = None,
    end: Optional[str] = None,
    freq: Optional[str] = None,
) -> pd.Series:
    time_series_conversion = SequenceToTimeIndexSeries(start, end, freq)
    return time_series_conversion.transform(array_like_object)


def transform_series_into_time_index_series(
    array_like_object: Union[np.array, list, pd.Series],
    start: Optional[str] = None,
    end: Optional[str] = None,
    freq: Optional[str] = None,
) -> pd.Series:
    time_series_conversion = PandasSeriesToTimeIndexSeries(start, end, freq)
    return time_series_conversion.transform(array_like_object)


def transform_time_index_series_into_period_index_series(
    series: pd.Series, freq: pd.Timedelta = None,
) -> pd.Series:
    to_period_conversion = TimeIndexSeriesToPeriodIndexSeries(freq=freq)
    return to_period_conversion.transform(series)


def pandas_series_with_period_index(
    values: Union[np.array, List[float]],
    start: Optional[pd.datetime] = None,
    end: Optional[pd.datetime] = None,
    freq: Optional[pd.Timedelta] = None,
) -> pd.Series:
    start, end, freq = _initialize_start_end_freq(start, end, freq)
    index = pd.period_range(start=start, end=end, periods=len(values), freq=freq,)
    return pd.Series(index=index, data=values)


def _initialize_start_end_freq(
    start: PandasDate, end: PandasDate, freq: pd.Timedelta
) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timedelta]:
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


def _default_params_initialization() -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timedelta]:
    start = DEFAULT_START
    end = None
    freq = DEFAULT_FREQ
    return start, end, freq


def _one_not_none_param_initialization(
    start, end, freq
) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timedelta]:
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


def _two_not_none_params_initialization(
    start, end, freq
) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timedelta]:
    start = start
    end = end
    freq = freq
    return start, end, freq


def datetime_index_series_to_period_index_series(
    datetime_index_series: pd.Series, freq: Optional[pd.Timedelta] = None
) -> pd.Series:
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
) -> pd.Series:
    datetime_index = pd.to_datetime(timedelta_index_series.index)
    if datetime_index.freq is None:
        freq = "1D" if freq is None else freq
        period_index = pd.PeriodIndex(datetime_index, freq=freq)
    else:
        period_index = pd.PeriodIndex(datetime_index)
    return pd.Series(index=period_index, data=timedelta_index_series.values)
