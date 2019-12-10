from typing import Optional, Union, List

import numpy as np
import pandas as pd

from giottotime.time_series_preparation.base import TimeSeriesConversion

PandasTimeIndex = Union[pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex]
PandasDate = Union[pd.datetime, pd.Timestamp, str]


def count_not_none(*args):
    """Returns the count of arguments that are not None.
    """
    return sum(x is not None for x in args)


def check_period_range_parameters(
    start_date: PandasDate, end_date: PandasDate, periods: int
) -> None:
    """Check if the period range parameters given as input are compatible with the
    `pd.period_range` method.

    Of the three parameters: start, end, and periods, exactly two must be specified.

    Parameters
    ----------
    start_date : ``PandasDate``, required
    end_date : ``PandasDate``, required
    periods : ``int``, required

    Raises
    ------
    ``ValueError``  
        Of the three parameters: start, end, and periods, exactly two must be specified.
    """
    if count_not_none(start_date, end_date, periods) != 2:
        raise ValueError(
            "Of the three parameters: start, end, and periods, "
            "exactly two must be specified"
        )


class SequenceToTimeIndexSeries(TimeSeriesConversion):
    """Converts a np.array or list to Series with PeriodIndex.

    Parameters
    -----------
    start : ``PandasDate``, optional, (default=``None``)
        start date of the output time series. Not mandatory for all time series
        conversion.

    end : ``PandasDate``, optional, (default=``None``)
        end date of the output time series. Not mandatory for all time series
        conversion.

    freq : ``pd.Timedelta``, optional, (default=``None``)
        frequency of the output time series. Not mandatory for all time series
        conversion.
    """

    def __init__(
        self,
        start: Optional[PandasDate] = None,
        end: Optional[PandasDate] = None,
        freq: Optional[pd.Timedelta] = None,
    ) -> None:
        super().__init__(start, end, freq)

    def _get_index_from(
        self, array_like_object: Union[np.array, List[float]]
    ) -> pd.PeriodIndex:
        return self._compute_period_index_of_length(len(array_like_object))

    def _get_values_from(
        self, array_like_object: Union[np.array, List[float]]
    ) -> np.array:
        return np.array(array_like_object)


class PandasSeriesToTimeIndexSeries(TimeSeriesConversion):
    """Returns a Pandas Series with time index (DatetimeIndex, TimedeltaIndex or
    PeriodIndex from a standard Pandas Series

    Parameters
    ----------
    start : ``Union[pd.datetime, str]``, optional, (default=``None``)
    end : ``Union[pd.datetime, str]``, optional, (default=``None``)
    freq : ``pd.Timedelta``, optional, (default=``None``)

    """

    def __init__(
        self,
        start: Optional[Union[pd.datetime, str]] = None,
        end: Optional[Union[pd.datetime, str]] = None,
        freq: Optional[pd.Timedelta] = None,
    ) -> None:
        super().__init__(start, end, freq)

    def _get_index_from(
        self, array_like_object: Union[pd.Series, np.array, list]
    ) -> PandasTimeIndex:
        if self._has_time_index(array_like_object):
            return array_like_object.index
        else:
            return self._compute_period_index_of_length(array_like_object.shape[0])

    def _get_values_from(
        self, array_like_object: Union[pd.Series, np.array, list]
    ) -> np.array:
        return array_like_object.values

    def _has_time_index(self, time_series: pd.Series) -> bool:
        index = time_series.index
        return (
            isinstance(index, pd.DatetimeIndex)
            or isinstance(index, pd.TimedeltaIndex)
            or isinstance(index, pd.PeriodIndex)
        )


class TimeIndexSeriesToPeriodIndexSeries(TimeSeriesConversion):
    """Converts a series with a time index to a series with a PeriodIndex.

    It may be necessary to specify a `freq` if not already provided.

    Parameters
    ----------
    freq : ``pd.Timedelta``, optional, (default=``None``)
    """

    def __init__(self, freq: Optional[pd.Timedelta] = None):
        super().__init__(start=None, end=None, freq=freq)

    def _get_index_from(self, time_series: pd.Series) -> pd.PeriodIndex:
        index = time_series.index
        if isinstance(index, pd.PeriodIndex):
            return index
        elif isinstance(index, pd.DatetimeIndex):
            return self._datetime_index_to_period(index)
        elif isinstance(index, pd.TimedeltaIndex):
            return self._timedelta_index_to_period(index)
        else:
            raise ValueError(
                f"Only PeriodIndex, DatetimeIndex and "
                f"TimedeltaIndex are supported. Detected: "
                f"{type(index)}"
            )

    def _datetime_index_to_period(self, index: pd.DatetimeIndex) -> pd.PeriodIndex:
        if index.freq is None:
            return pd.PeriodIndex(index, freq=self.freq)
        else:
            return pd.PeriodIndex(index)

    def _timedelta_index_to_period(self, index: pd.TimedeltaIndex) -> pd.PeriodIndex:
        datetime_index = pd.to_datetime(index)
        return self._datetime_index_to_period(datetime_index)

    def _get_values_from(self, time_series: pd.Series) -> np.array:
        return time_series.values
