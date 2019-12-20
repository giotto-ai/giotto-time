from abc import ABC, abstractmethod
from typing import Optional, Union, List

import numpy as np
import pandas as pd

PandasTimeIndex = Union[pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex]
PandasDate = Union[pd.datetime, pd.Timestamp, str]

DEFAULT_START = pd.Timestamp("1970-01-01")
DEFAULT_END = pd.Timestamp("2020-01-01")
DEFAULT_FREQ = pd.Timedelta("1D")

__all__ = [
    "SequenceToTimeIndexSeries",
    "PandasSeriesToTimeIndexSeries",
    "TimeIndexSeriesToPeriodIndexSeries",
]


def count_not_none(*args):
    """Returns the count of arguments that are not None.
    """
    return sum(x is not None for x in args)


def check_period_range_parameters(
    start_date: PandasDate, end_date: PandasDate, periods: int
) -> None:
    """Check if the period range parameters given as input are compatible with the
    ``pd.period_range`` method.

    Of the three parameters: start, end, and periods, exactly two must be specified.

    Parameters
    ----------
    start_date : PandasDate, required
        The date to use as start date.

    end_date : PandasDate, required
        The date to use as end date.

    periods : int, required
        The number of periods.

    Raises
    ------
    ValueError
        Of the three parameters: start, end, and periods, exactly two must be specified.

    """
    if count_not_none(start_date, end_date, periods) != 2:
        raise ValueError(
            "Of the three parameters: start, end, and periods, "
            "exactly two must be specified"
        )


class TimeSeriesConversion(ABC):
    """Parent class for all time series type conversions.

    Subclasses must implement the two methods `_get_index_from` and `_get_values_from`.

    Parameters
    ----------
    start : PandasData, optional, default: ``None``
        start date of the output time series. Not mandatory for all time series
        conversion.

    end : PandasData, optional, default: ``None``
        end date of the output time series. Not mandatory for all time series
        conversion.

    freq : pd.Timedelta, optional, default: ``None``
        The frequency of the output time series. Not mandatory for all time series
        conversion.

    """

    def __init__(
        self,
        start: Optional[PandasDate] = None,
        end: Optional[PandasDate] = None,
        freq: Optional[pd.Timedelta] = None,
    ) -> None:
        self._initialize_start_end_freq(start, end, freq)

    def transform(self, time_series: Union[pd.Series, np.array, list]) -> pd.Series:
        """Transforms an array-like object (list, np.array, pd.Series) into a pd.Series
        with time index.

        It calls internally the abstract methods `_get_index_from()` and
        `_get_values_from()`. These are implemented in the subclasses.

        Parameters
        ----------
        time_series : Union[List, np.array, pd.Series], required
            It depends on the implementation of the subclasses.

        Returns
        -------
        time_series_t: pd.Series

        """
        index = self._get_index_from(time_series)
        values = self._get_values_from(time_series)
        return pd.Series(data=values, index=index)

    @abstractmethod
    def _get_index_from(
        self, array_like_object: Union[pd.Series, np.ndarray, list]
    ) -> PandasTimeIndex:
        raise NotImplementedError  # To exclude it from pytest coverage

    @abstractmethod
    def _get_values_from(
        self, array_like_object: Union[pd.Series, np.array, list]
    ) -> np.ndarray:
        raise NotImplementedError  # To exclude it from pytest coverage

    def _initialize_start_end_freq(
        self, start: PandasDate, end: PandasDate, freq: pd.Timedelta
    ) -> None:
        not_none_params = count_not_none(start, end, freq)
        if not_none_params == 0:
            self._default_params_initialization()
        elif not_none_params == 1:
            self._one_not_none_param_initialization(start, end, freq)
        elif not_none_params == 2:
            self._two_not_none_params_initialization(start, end, freq)
        else:
            raise ValueError(
                "Of the three parameters: start, end, and "
                "freq, exactly two must be specified"
            )

    def _default_params_initialization(self):
        self.start = DEFAULT_START
        self.end = None
        self.freq = DEFAULT_FREQ

    def _one_not_none_param_initialization(
        self, start: PandasDate, end: PandasDate, freq: pd.Timedelta
    ):
        if start is not None:
            self.start = start
            self.end = None
            self.freq = DEFAULT_FREQ
        elif end is not None:
            self.start = None
            self.end = end
            self.freq = DEFAULT_FREQ
        else:
            self.start = DEFAULT_START
            self.end = None
            self.freq = freq

    def _two_not_none_params_initialization(
        self, start: PandasDate, end: PandasDate, freq: pd.Timedelta
    ):
        self.start = start
        self.end = end
        self.freq = freq

    def _compute_period_index_of_length(self, length: int) -> pd.PeriodIndex:
        check_period_range_parameters(self.start, self.end, length)
        return pd.period_range(
            start=self.start, end=self.end, periods=length, freq=self.freq
        )


class SequenceToTimeIndexSeries(TimeSeriesConversion):
    """Converts a np.array or list to Series with PeriodIndex.

    Parameters
    -----------
    start : PandasDate, optional, default: ``None``
        start date of the output time series. Not mandatory for all time series
        conversion.

    end : PandasDate, optional, default: ``None``
        end date of the output time series. Not mandatory for all time series
        conversion.

    freq : pd.Timedelta, optional, default: ``None``
        frequency of the output time series. Not mandatory for all time series
        conversion.

    Examples
    --------
    >>> from giottotime.time_series_preparation import SequenceToTimeIndexSeries
    >>> time_series = [1,2,3,5,5,7]
    >>> sequence_to_time_index = SequenceToTimeIndexSeries(start='01-01-2010', freq='10D')
    >>> sequence_to_time_index.transform(time_series)
    2010-01-01    1
    2010-01-11    2
    2010-01-21    3
    2010-01-31    5
    2010-02-10    5
    2010-02-20    7
    Freq: 10D, dtype: int64
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
    PeriodIndex) from a standard Pandas Series

    Parameters
    ----------
    start: PandasDate, required
        The date to use as start date.

    end: PandasDate, required
        The date to use as end date.

    freq : pd.Timedelta``, optional, default: ``None``
        The frequency of the time series.

    Examples
    --------
    >>> import pandas as pd
    >>> from giottotime.time_series_preparation import PandasSeriesToTimeIndexSeries
    >>> time_series = pd.Series([1,2,3,5,5,7])
    >>> sequence_to_time_index = PandasSeriesToTimeIndexSeries(start='01-01-2010', freq='10D')
    >>> sequence_to_time_index.transform(time_series)
    2010-01-01    1
    2010-01-11    2
    2010-01-21    3
    2010-01-31    5
    2010-02-10    5
    2010-02-20    7
    Freq: 10D, dtype: int64
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
    """Converts a series with a time index (DatetimeIndex, TimedeltaIndex or
    PeriodIndex) to a series with a PeriodIndex.

    It may be necessary to specify a `freq` if not already provided.

    Parameters
    ----------
    freq : pd.Timedelta, optional, default: ``None``
        The frequency of the time series.

    Examples
    --------
    >>> import pandas as pd
    >>> from giottotime.time_series_preparation import TimeIndexSeriesToPeriodIndexSeries
    >>> period_index_time_series = pd.Series(
    ...     index = pd.period_range(start='01-01-2010', freq='10D', periods=6),
    ...     data=[1,2,3,5,5,7]
    ... )
    >>> datetime_index_time_series = pd.Series(
    ...     index = pd.date_range(start='01-01-2010', freq='10D', periods=6),
    ...     data=[1,2,3,5,5,7]
    ... )
    >>> timedelta_index_time_series = pd.Series(
    ...     index = pd.timedelta_range(start=pd.Timedelta(days=1), freq='10D', periods=6),
    ...     data=[1,2,3,5,5,7]
    ... )
    >>> sequence_to_time_index = TimeIndexSeriesToPeriodIndexSeries()
    >>> sequence_to_time_index.transform(period_index_time_series)
    2010-01-01    1
    2010-01-11    2
    2010-01-21    3
    2010-01-31    5
    2010-02-10    5
    2010-02-20    7
    freq: 10d, dtype: int64
    >>> sequence_to_time_index.transform(datetime_index_time_series)
    2010-01-01    1
    2010-01-11    2
    2010-01-21    3
    2010-01-31    5
    2010-02-10    5
    2010-02-20    7
    freq: 10d, dtype: int64
    >>> sequence_to_time_index.transform(timedelta_index_time_series)
    1970-01-02    1
    1970-01-12    2
    1970-01-22    3
    1970-02-01    5
    1970-02-11    5
    1970-02-21    7
    Freq: D, dtype: int64
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
