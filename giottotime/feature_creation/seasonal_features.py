from typing import Union

import numpy as np
import pandas as pd

from giottotime.feature_creation.base import TimeSeriesFeature

__all__ = [
    'PeriodicSeasonalFeature'
]


class PeriodicSeasonalFeature(TimeSeriesFeature):
    """Create a sinusoid from a given date and with a given period and
    amplitude.

    Parameters
    ----------
    start_date : ``pd.Timestamp``, required.
        The date from which to start generating the feature.

    period : ``Union[pd.Timedelta, str]``, required.
        The period of the generated time series.

    amplitude : ``float``, required.
        The amplitude of the time series.

    output_name : ``str``, required.
        The name of the output column.

    """
    def __init__(self,
                 start_date: pd.Timestamp,
                 period: Union[pd.Timedelta, str],
                 amplitude: float,
                 output_name: str):
        super().__init__(output_name)
        self.start_date = start_date
        self.period = pd.Timedelta(period)
        self.amplitude = amplitude

    # TODO: check that the explanaition
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Starting from the index of ``X``, generate a sinusoid,
        with the given ``period`` and ``amplitude``.

        Parameters
        ----------
        X : ``pd.DataFrame``, required.
            The input DataFrame, used only for its index.

        Returns
        -------
        periodic_feature : ``pd.DataFrame``, required.
            The DataFrame containing the generated period feature.

        """
        datetime_index = self._convert_index_to_datetime(X.index)
        periodic_feature_values = self._compute_periodic_feature(datetime_index)
        periodic_feature = pd.DataFrame(index=X.index,
                                        data=periodic_feature_values)
        periodic_feature = self._rename_columns(periodic_feature)
        return periodic_feature

    def _convert_index_to_datetime(self, index: pd.PeriodIndex) \
            -> pd.DatetimeIndex:
        """Convert a ``pd.PeriodIndex`` to a ``pd.DatetimeIndex``.

        Parameters
        ----------
        index : ``pd.PeriodIndex``, required.
            The index to convert to ``pd.DatetimeIndex``.

        Returns
        -------
        datetime_index : ``pd.DatetimeIndex``
            The original index, converted to ``pd.DatetimeIndex``.

        Raises
        ------
        ValueError
            Raised if the sampling frequency is not at least two times with
            respect to the period.

        """
        datetime_index = index.to_timestamp()
        self._check_sampling_frequency(datetime_index)
        return datetime_index

    def _check_sampling_frequency(self, datetime_index: pd.DatetimeIndex) \
            -> None:
        """Check that the sampling frequency is at least two times the period.

        Parameters
        ----------
        datetime_index : ``pd.DatetimeIndex``, required.
            The index on which to perform the check of the frequency.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            Raised if the sampling frequency is not at least two times with
            respect to the period.

        """
        sampling_frequency = pd.Timedelta(datetime_index.freq)
        if sampling_frequency < 2 * self.period:
            raise ValueError(f"Sampling frequency must be at least two times"
                             f"the period to obtain meaningful results. "
                             f"Sampling frequency = {sampling_frequency},"
                             f"period = {self.period}.")

    def _compute_periodic_feature(self, datetime_index: pd.DatetimeIndex):
        """Compute a sinusoid with the specified parameters.

        Parameters
        ----------
        datetime_index : ``pd.DatetimeIndex``, required.
            The index from which to compute the sinusoid.

        Returns
        -------
        sinusoid : ``np.ndarray``
            The generated sinusoid.

        """
        return (np.sin((datetime_index - self.start_date) / self.period)) * \
               self.amplitude
