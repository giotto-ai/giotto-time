from typing import Union, List, Optional
import logging

import numpy as np
import pandas as pd

__all__ = ['TimeSeriesToPandasSeriesConversion']


def is_numpy_or_list(array_like_object: Union[List, np.array, pd.Series]):
    return isinstance(array_like_object, np.ndarray)\
           or isinstance(array_like_object, list)


def is_pd_series(array_like_object: Union[List, np.array, pd.Series]):
    return isinstance(array_like_object, pd.Series)


def compute_index_from_length_start_date_end_date(length: int,
                                                  start_date: str,
                                                  end_date: str):
    duration = pd.Timestamp(end_date) - pd.Timestamp(start_date)
    freq = duration / length
    # We cast the first {length} numbers since pd.PeriodIndex tends (not
    # always) to take {length+1} elements
    return pd.PeriodIndex(start=start_date,
                          end=end_date,
                          freq=freq)[:length]


class TimeSeriesToPandasSeriesConversion:
    """
    Convert an input sequence to a pandas Series with a period
    index.

    Two out of start_date, end_date, freq must be not None.

    Parameters
    ----------
    start_date: Union[pd.datetime, str], optional
        Output pandas series start date
    end_date: Union[pd.datetime, str], optional
        Output pandas end start date
    freq: pd.DateOffset, optional
        Offset between two consecutive time series elements
    """

    def __init__(self,
                 start_date: Optional[Union[pd.datetime, str]] = None,
                 end_date: Optional[Union[pd.datetime, str]] = None,
                 freq: Optional[pd.DateOffset] = None) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq

    def fit_transform(
            self, array_like_object: Union[List, np.array, pd.Series]
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

    def _get_index_from(
            self, array_like_object: Union[List, np.array, pd.Series]
    ) -> pd.PeriodIndex:
        if is_numpy_or_list(array_like_object):
            return self._compute_index_of_length(len(array_like_object))
        elif is_pd_series(array_like_object):
            return self._extract_index_from_series(array_like_object)
        else:
            raise TypeError(f'{type(array_like_object)} not supported')

    def _get_values_from(
            self, array_like_object: Union[List, np.array, pd.Series]
    ) -> np.array:
        if is_numpy_or_list(array_like_object):
            return array_like_object
        elif isinstance(array_like_object, pd.Series):
            return array_like_object.values
        else:
            raise TypeError(f'{type(array_like_object)} not supported')

    def _compute_index_of_length(self, length: int) -> pd.PeriodIndex:
        if self.start_date is not None and self.end_date is not None:
            return compute_index_from_length_start_date_end_date(
                length, self.start_date, self.end_date
            )
        elif self.start_date is not None and self.freq is not None:
            return pd.period_range(start=self.start_date, freq=self.freq,
                                   periods=length)
        elif self.end_date is not None and self.freq is not None:
            return pd.period_range(end=self.end_date, freq=self.freq,
                                   periods=length)
        else:
            raise ValueError(f'Of the three parameters: start_date, end_date, '
                             f'and freq, two must be specified')

    def _extract_index_from_series(self, series: pd.Series) -> pd.PeriodIndex:
        if isinstance(series.index, pd.DatetimeIndex):
            return self._convert_datetime_index_to_period(series.index)
        elif isinstance(series.index, pd.TimedeltaIndex):
            return series.index.to_period()
        elif isinstance(series.index, pd.PeriodIndex):
            return series.index
        else:
            return self._compute_index_of_length(series.shape[0])

    def _convert_datetime_index_to_period(self, index: pd.DatetimeIndex) -> pd.PeriodIndex:
        if index.freq is not None:
            return index.to_period()
        elif self.freq is not None:
            return index.to_period(self.freq)
        elif pd.infer_freq(index) is not None:
            return index.to_period(pd.infer_freq(index))
        else:
            logging.warning('Impossible to infer the frequency of the time ' 
                            'series. Setting freq = "1D"')
            return index.to_period(freq='1D')




