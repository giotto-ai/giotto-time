from typing import List, Union

import numpy as np
import pandas as pd

from giottotime.time_series_preparation.time_series_resampling import TimeSeriesResampler
from .time_series_conversion import PandasSeriesToTimeIndexSeries, \
    SequenceToTimeIndexSeries

SUPPORTED_SEQUENCE_TYPES = [
    np.ndarray,
    list,
]


class TimeSeriesPreparation:
    def __init__(self,
                 start_date: pd.datetime = None,
                 end_date: pd.datetime = None,
                 freq: pd.DateOffset = None,
                 resample_if_not_equispaced: bool = True):
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.resample_if_not_equispaced = resample_if_not_equispaced

        self.pandas_converter = PandasSeriesToTimeIndexSeries(
            self.start_date, self.end_date, self.freq
        )
        self.sequence_converter = SequenceToTimeIndexSeries(
            self.start_date, self.end_date, self.freq
        )
        self.resampler = TimeSeriesResampler()

    def fit_transform(self, array_like_object: Union[List, np.array, pd.Series]):
        pandas_time_series = self._to_pandas_time_series(array_like_object)
        equispaced_time_series = self._to_equispaced_time_series(pandas_time_series)
        period_index_time_series = self._to_period_index_time_series(equispaced_time_series)

        return period_index_time_series

    def _to_pandas_time_series(self, array_like_object):
        if isinstance(array_like_object, pd.Series):
            return self.pandas_converter.transform(array_like_object)
        elif any(isinstance(array_like_object, type_)
                 for type_ in SUPPORTED_SEQUENCE_TYPES):
            return self.sequence_converter.transform(array_like_object)
        else:
            raise TypeError(f'Type {type(array_like_object)} is not a '
                            f'supported time series type')

    def _to_equispaced_time_series(self, time_series):
        if self.resample_if_not_equispaced:
            self.resampler.transform(time_series)
        else:
            return time_series

    def _to_period_index_time_series(self, time_series):
        raise NotImplementedError

