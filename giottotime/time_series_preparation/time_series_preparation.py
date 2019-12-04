from typing import List, Union

import numpy as np
import pandas as pd

from giottotime.time_series_preparation.time_series_resampling import TimeSeriesResampler
from .time_series_conversion import PandasSeriesToTimeIndexSeries, \
    SequenceToTimeIndexSeries, TimeIndexSeriesToPeriodIndexSeries

SUPPORTED_SEQUENCE_TYPES = [
    np.ndarray,
    list,
]


class TimeSeriesPreparation:
    def __init__(self,
                 start_date: pd.datetime = None,
                 end_date: pd.datetime = None,
                 freq: pd.DateOffset = None,
                 resample_if_not_equispaced: bool = True,
                 output_name: str = 'time_series'):
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.resample_if_not_equispaced = resample_if_not_equispaced
        self.output_name = output_name

        self.pandas_converter = PandasSeriesToTimeIndexSeries(
            self.start_date, self.end_date, self.freq
        )
        self.sequence_converter = SequenceToTimeIndexSeries(
            self.start_date, self.end_date, self.freq
        )
        self.resampler = TimeSeriesResampler()
        self.to_period_index_series_converter = TimeIndexSeriesToPeriodIndexSeries(
            self.freq
        )

    def fit_transform(self,
                      array_like_object: Union[List, np.array, pd.Series]
                      ) -> pd.DataFrame:
        pandas_time_series = self._to_time_index_series(array_like_object)
        equispaced_time_series = self._to_equispaced_time_series(pandas_time_series)
        period_index_time_series = self._to_period_index_time_series(equispaced_time_series)
        period_index_dataframe = self._to_period_index_dataframe(period_index_time_series)
        return period_index_dataframe

    def _to_time_index_series(self,
                              array_like_object: Union[List, np.array, pd.Series]
                              ) -> pd.Series:
        if isinstance(array_like_object, pd.Series):
            return self.pandas_converter.transform(array_like_object)
        elif any(isinstance(array_like_object, type_)
                 for type_ in SUPPORTED_SEQUENCE_TYPES):
            return self.sequence_converter.transform(array_like_object)
        else:
            raise TypeError(f'Type {type(array_like_object)} is not a '
                            f'supported time series type')

    def _to_equispaced_time_series(self, time_series: pd.Series) -> pd.Series:
        if self.resample_if_not_equispaced:
            self.resampler.transform(time_series)
        else:
            return time_series

    def _to_period_index_time_series(self, time_series: pd.Series) -> pd.Series:
        return self.to_period_index_series_converter.transform(time_series)

    def _to_period_index_dataframe(self, time_series: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({self.output_name: time_series})

