from abc import ABC, abstractmethod
from typing import Optional, Union, List

import numpy as np
import pandas as pd

from ..core.constants import DEFAULT_START, DEFAULT_FREQ

PandasTimeIndex = Union[pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex]
PandasDate = Union[pd.datetime, pd.Timestamp, str]


def count_not_none(*args):
    """Returns the count of arguments that are not None.
    """
    return sum(x is not None for x in args)


def check_period_range_parameters(
    start_date: PandasDate, end_date: PandasDate, periods: int
) -> None:
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
    start : ``PandasData``, optional, (default=``None``)
        start date of the output time series. Not mandatory for all time series
        conversion.

    end : ``PandasData``, optional, (default=``None``)
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
        self._initialize_start_end_freq(start, end, freq)

    def transform(self, X: Union[pd.Series, np.array, list]) -> pd.Series:
        """Transforms an array-like object (list, np.array, pd.Series)
        into a pd.Series with time index.

        It calls internally the abstract methods `_get_index_from()` and
        `_get_values_from()`. These are implemented in the subclasses.

        Parameters
        ----------
        X : Union[List, np.array, pd.Series], required.
            It depends on the implementation of the subclasses.

        Returns
        -------
        transformed series: pd.Series
        """
        index = self._get_index_from(X)
        values = self._get_values_from(X)
        return pd.Series(data=values, index=index)

    @abstractmethod
    def _get_index_from(
        self, array_like_object: Union[pd.Series, np.ndarray, list]
    ) -> PandasTimeIndex:
        """Abstract method that extract the index from an array-like object.
        It must return a PandasTimeIndex

        Parameters
        ----------
        array_like_object : Union[pd.Series, np.ndarray, list], required.

        Returns
        -------
        index : PandasTimeIndex
            the index of the returning Pandas Series.
        """
        pass

    @abstractmethod
    def _get_values_from(
        self, array_like_object: Union[pd.Series, np.array, list]
    ) -> np.ndarray:
        """Abstract method that extract the index from an array-like object.
        It must return a np.array

        Parameters
        ----------
        array_like_object : Union[pd.Series, np.ndarray, list], required.

        Returns
        -------
        values : np.array
            the values of the returning Pandas Series.
        """
        pass

    def _initialize_start_end_freq(
        self, start: PandasDate, end: PandasDate, freq: pd.Timedelta
    ) -> None:
        """Initialization of the parameters `start`, `end` and `freq`.

        Exactly two out of three must be specified.

        Parameters
        ----------
        start : PandasData, required.
        end : PandasData, required
        freq : pd.Timedelta, required

        Returns
        -------
        None

        Raises
        ------
        ValueError : if three of them are specified.
        """
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
        """ Returns a period index of the given length. It uses the private attributes
        ``start``, ``end``, ``freq``.

        Parameters
        ----------
        length : int, required.

        Returns
        -------
        pd.PeriodIndex
        """
        check_period_range_parameters(self.start, self.end, length)
        return pd.period_range(
            start=self.start, end=self.end, periods=length, freq=self.freq
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
        """Computes a pd.PeriodIndex for the output time series.

        Parameters
        ----------
        array_like_object: Union[np.array, List[float]], required

        Returns
        -------
        pd.PeriodIndex
        """
        return self._compute_period_index_of_length(len(array_like_object))

    def _get_values_from(
        self, array_like_object: Union[np.array, List[float]]
    ) -> np.array:
        """Computes the values for the output time series.

        Parameters
        ----------
        array_like_object: Union[np.array, List[float]], required

        Returns
        -------
        np.array
        """
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
        """Returns a time index from a pandas Series. It converts

        Parameters
        ----------
        array_like_object

        Returns
        -------

        """
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
    def __init__(self, freq: pd.Timedelta = None):
        super().__init__(start=None, end=None, freq=freq)

    def _get_index_from(self, time_series: pd.Series,) -> PandasTimeIndex:
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
