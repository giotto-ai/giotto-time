from typing import Union

import numpy as np
import pandas as pd


class PeriodicSeasonalFeature:

    def __init__(self,
                 start_date: pd.Timestamp,
                 period: Union[pd.Timedelta, str],
                 amplitude: float):
        self.start_date = start_date
        self.period = pd.Timedelta(period)
        self.amplitude = amplitude

    def transform(self, X: pd.Series):
        datetime_index = self._convert_index_to_datetime(X.index)
        periodic_feature_values = self._compute_periodic_feature_of(datetime_index)
        periodic_feature_series = pd.Series(index=X.index,
                                            data=periodic_feature_values)
        return periodic_feature_series

    def _convert_index_to_datetime(self, index: pd.PeriodIndex):
        datetime_index = index.to_timestamp()
        self._check_sampling_frequency_of(datetime_index)
        return datetime_index

    def _check_sampling_frequency_of(self, datetime_index: pd.DatetimeIndex):
        sampling_frequency = pd.Timedelta(datetime_index.freq)
        if sampling_frequency < 2 * self.period:
            raise ValueError(f'Sampling frequency must be at least two times'
                             f'the period to obtain meaningful results. '
                             f'Sampling frequency = {sampling_frequency},'
                             f'period = {self.period}')

    def _compute_periodic_feature_of(self, datetime_index: pd.DatetimeIndex):
        return np.sin((datetime_index - self.start_date) / self.period)
