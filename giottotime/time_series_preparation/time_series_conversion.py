from abc import ABC, abstractmethod
from typing import Optional, Union, List

import numpy as np
import pandas as pd

from giottotime.core.constants import DEFAULT_START, DEFAULT_END, DEFAULT_FREQ

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
        raise ValueError("Of the three parameters: start, end, and periods, " 
                         "exactly two must be specified")


class TimeSeriesConversion(ABC):

    def __init__(self,
                 start: Optional[PandasDate] = None,
                 end: Optional[PandasDate] = None,
                 freq: Optional[pd.Timedelta] = None) -> None:
        self._initialize_start_end_freq(start, end, freq)

    def fit(self, X, y=None):
        return self

    def transform(self,
                  array_like_object: Union[pd.Series, np.array, list]
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
    def _get_index_from(self,
                        array_like_object: Union[pd.Series, np.array, list]
                        ) -> PandasTimeIndex:
        pass

    @abstractmethod
    def _get_values_from(self,
                         array_like_object: Union[pd.Series, np.array, list]
                         ) -> np.array:
        pass

    def _initialize_start_end_freq(self,
                                   start: PandasDate,
                                   end: PandasDate,
                                   freq: pd.Timedelta):
        not_none_params = count_not_none(start, end, freq)
        if not_none_params == 0:
            self._default_params_initialization()
        elif not_none_params == 1:
            self._one_not_none_param_initialization(start, end, freq)
        elif not_none_params == 2:
            self._two_not_none_params_initialization(start, end, freq)
        else:
            raise ValueError("Of the three parameters: start, end, and "
                             "freq, exactly two must be specified")

    def _default_params_initialization(self):
        self.start = DEFAULT_START
        self.end = None
        self.freq = DEFAULT_FREQ

    def _one_not_none_param_initialization(self, start, end, freq):
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

    def _two_not_none_params_initialization(self, start, end, freq):
        self.start = start
        self.end = end
        self.freq = freq

    def _compute_index_of_length(self, length: int) -> pd.PeriodIndex:
        check_period_range_parameters(self.start, self.end, length)
        return pd.period_range(
            start=self.start,
            end=self.end,
            periods=length,
            freq=self.freq
        )


class SequenceToTimeIndexSeries(TimeSeriesConversion):

    def __init__(self,
                 start: Optional[Union[pd.datetime, str]] = None,
                 end: Optional[Union[pd.datetime, str]] = None,
                 freq: Optional[pd.DateOffset] = None) -> None:
        super().__init__(start, end, freq)

    def _get_index_from(self,
                        array_like_object: Union[np.array, List[float]]
                        ) -> PandasTimeIndex:
        return self._compute_index_of_length(len(array_like_object))

    def _get_values_from(self,
                         array_like_object: List[float]
                         ) -> np.array:
        return np.array(array_like_object)


class PandasSeriesToTimeIndexSeries(TimeSeriesConversion):
    def __init__(self,
                 start: Optional[Union[pd.datetime, str]] = None,
                 end: Optional[Union[pd.datetime, str]] = None,
                 freq: Optional[pd.DateOffset] = None) -> None:
        super().__init__(start, end, freq)

    def _get_index_from(self,
                        array_like_object: Union[pd.Series, np.array, list]
                        ) -> PandasTimeIndex:
        if self._has_time_index(array_like_object):
            return array_like_object.index
        else:
            return self._compute_index_of_length(array_like_object.shape[0])

    def _get_values_from(self,
                         array_like_object: Union[pd.Series, np.array, list]
                         ) -> np.array:
        return array_like_object.values

    def _has_time_index(self, time_series: pd.Series) -> bool:
        index = time_series.index
        return isinstance(index, pd.DatetimeIndex) or \
               isinstance(index, pd.TimedeltaIndex) or \
               isinstance(index, pd.PeriodIndex)


class TimeIndexSeriesToPeriodIndexSeries(TimeSeriesConversion):

    def __init__(self, freq: pd.Timedelta = None):
        super().__init__(start=None, end=None, freq=freq)

    def _get_index_from(self,
                        time_series: pd.Series,
                        ) -> PandasTimeIndex:
        index = time_series.index
        if isinstance(index, pd.PeriodIndex):
            return index
        elif isinstance(index, pd.DatetimeIndex):
            return self._datetime_index_to_period(index)
        elif isinstance(index, pd.TimedeltaIndex):
            return self._timedelta_index_to_period(index)
        else:
            raise ValueError(f'Only PeriodIndex, DatetimeIndex and '
                             f'TimedeltaIndex are supported. Detected: '
                             f'{type(index)}')

    def _datetime_index_to_period(self,
                                  index: pd.DatetimeIndex) -> pd.PeriodIndex:
        if index.freq is None:
            return pd.PeriodIndex(index, freq=self.freq)
        else:
            return pd.PeriodIndex(index)

    def _timedelta_index_to_period(self,
                                   index: pd.TimedeltaIndex) -> pd.PeriodIndex:
        datetime_index = pd.to_datetime(index)
        return self._datetime_index_to_period(datetime_index)

    def _get_values_from(self,
                         time_series: pd.Series
                         ) -> np.array:
        return time_series.values

