from abc import ABC, abstractmethod
from typing import Optional, Union, List

import numpy as np
import pandas as pd

PandasTimeIndex = Union[pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex]
PandasDate = Union[pd.datetime, pd.Timestamp, str]


def count_not_none(*args):
    """
    Returns the count of arguments that are not None.
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
    def __init__(
        self,
        start_date: Optional[PandasDate] = None,
        end_date: Optional[PandasDate] = None,
        freq: Optional[pd.DateOffset] = None,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq

    def fit(self, X, y=None):
        return self

    def transform(
        self, array_like_object: Union[pd.Series, np.array, list]
    ) -> pd.Series:
        """
        Transforms an array-like object (list, np.array, pd.Series)
        into a pd.Series with PeriodIndex

        Parameters
        ----------
        array_like_object: Union[List, np.array, pd.Series]

        Returns
        -------
        transformed series: pd.Series
        """
        index = self._get_index_from(array_like_object)
        values = self._get_values_from(array_like_object)
        return pd.Series(data=values, index=index)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    @abstractmethod
    def _get_index_from(
        self, array_like_object: Union[pd.Series, np.array, list]
    ) -> PandasTimeIndex:
        pass

    @abstractmethod
    def _get_values_from(
        self, array_like_object: Union[pd.Series, np.array, list]
    ) -> np.array:
        pass

    def _compute_index_of_length(self, length: int) -> pd.PeriodIndex:
        check_period_range_parameters(self.start_date, self.end_date, length)
        return pd.period_range(
            start=self.start_date, end=self.end_date, periods=length, freq=self.freq
        )


class SequenceToPandasTimeSeries(TimeSeriesConversion):
    def __init__(
        self,
        start_date: Optional[Union[pd.datetime, str]] = None,
        end_date: Optional[Union[pd.datetime, str]] = None,
        freq: Optional[pd.DateOffset] = None,
    ) -> None:
        super().__init__(start_date, end_date, freq)

    def _get_index_from(
        self, array_like_object: Union[np.array, List[float]]
    ) -> PandasTimeIndex:
        return self._compute_index_of_length(len(array_like_object))

    def _get_values_from(self, array_like_object: List[float]) -> np.array:
        return np.array(array_like_object)


class PandasSeriesToPandasTimeSeries(TimeSeriesConversion):
    def __init__(
        self,
        start_date: Optional[Union[pd.datetime, str]] = None,
        end_date: Optional[Union[pd.datetime, str]] = None,
        freq: Optional[pd.DateOffset] = None,
    ) -> None:
        super().__init__(start_date, end_date, freq)

    def _get_index_from(
        self, array_like_object: Union[pd.Series, np.array, list]
    ) -> PandasTimeIndex:
        if self._has_time_index(array_like_object):
            return array_like_object.index
        else:
            return self._compute_index_of_length(array_like_object.shape[0])

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
