from typing import Union, Optional, Callable

import numpy as np
import pandas as pd
from pandas import DatetimeIndex

from math import pi

from giottotime.feature_creation.standard_features.base import StandardFeature

__all__ = ["PeriodicSeasonalFeature", "ConstantFeature", "CustomFeature"]


class PeriodicSeasonalFeature(StandardFeature):
    """Create a sinusoid from a given date and with a given period and
    amplitude.

    Parameters
    ----------
    period : Union[pd.Timedelta, str], optional, default: ``'365 days'``
        The period of the generated time series.

    amplitude : float, optional, default: ``0.5``
        The amplitude of the time series.

    output_name : str, optional, default: ``"PeriodicSeasonalFeature"``
        The name of the output column.

    start_date : Union[pd.Timestamp, str], optional, default: ``None``
        The date from which to start generating the feature. This is used only if X is
        not passed in the ``transform`` method, otherwise the start date is inferred
        from it.

    length : int, optional, default: ``50``
        The length of the sinusoid. This is used only if X is not passed in the
        ``transform`` method, otherwise the length is inferred from it.

    index_period : Union[DatetimeIndex, int], optional, default: ``None``
        The period of the index of the output ``DataFrame``. This is used only if X is
        not passed in the ``transform`` method, otherwise the index period is taken
        from it.

    """

    def __init__(
        self,
        period: Union[pd.Timedelta, str] = "365 days",
        amplitude: float = 0.5,
        output_name: str = "PeriodicSeasonalFeature",
        start_date: Optional[Union[pd.Timestamp, str]] = None,
        length: Optional[int] = 50,
        index_period: Optional[Union[DatetimeIndex, int]] = None,
    ):
        super().__init__(output_name=output_name)
        self.start_date = start_date
        self.period = period
        self.amplitude = amplitude
        self.length = length
        self.index_period = index_period

    def transform(self, time_series: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Starting from the index of ``time_series``, generate a sinusoid, with the
        given ``period`` and ``amplitude``.

        Parameters
        ----------
        time_series : pd.DataFrame, shape (n_samples, 1), optional, default: ``None``
            The input DataFrame, used only for its index. If is not passed, then the
            ``start_date`` and ``index_period`` must have been passed in the constructor
            when the object was instantiated.

        Returns
        -------
        periodic_feature : pd.DataFrame, shape (n_samples, 1)
            The DataFrame containing the generated period feature.

        Raises
        ------
        ValueError
            Raised if ``time_series`` is not passed and the ``start_date`` or the
            ``index_period`` are not present.

        """
        self._validate_input(time_series)
        self._transform_inputs(time_series)
        datetime_index = self._get_time_index(time_series)

        periodic_feature_values = self._compute_periodic_feature(datetime_index)
        if time_series is not None:
            periodic_feature = pd.DataFrame(
                index=time_series.index, data=periodic_feature_values
            )

        else:
            periodic_feature = pd.DataFrame(data=periodic_feature_values[: self.length])

        periodic_feature = self._rename_columns(periodic_feature)
        return periodic_feature

    def _validate_input(self, X: pd.DataFrame) -> None:
        if X is None:
            if self.start_date is None:
                raise ValueError(
                    f"If X is not provided, the 'start_date' variable "
                    f"must be provided, but instead was "
                    f"{self.start_date}."
                )
            if self.index_period is None:
                raise ValueError(
                    f"If X is not provided, the 'index_period' variable "
                    f"must be provided, but instead was "
                    f"{self.index_period}."
                )

    def _transform_inputs(self, X: pd.DataFrame) -> None:
        if isinstance(self.period, str):
            self.period = pd.to_timedelta(self.period)

        if X is not None:
            self.start_date = X.index.values[0]
        else:
            if isinstance(self.start_date, str):
                self.start_date = pd.to_datetime(self.start_date)

    def _get_time_index(self, X: pd.DataFrame) -> pd.DatetimeIndex:
        if X is not None:
            datetime_index = X.index
        else:
            if isinstance(self.index_period, int):
                datetime_index = pd.date_range(
                    start=self.start_date, periods=self.index_period
                )
            else:
                datetime_index = self.index_period

        self._check_sampling_frequency(datetime_index)

        return datetime_index

    def _check_sampling_frequency(self, datetime_index: pd.DatetimeIndex) -> None:
        if isinstance(datetime_index, pd.PeriodIndex):
            datetime_index = datetime_index.to_timestamp()
        sampling_frequency = pd.Timedelta(datetime_index[1] - datetime_index[0])

        if self.period < 2 * sampling_frequency:
            raise ValueError(
                f"Sampling frequency must be at least two times"
                f"the period to obtain meaningful results. "
                f"Sampling frequency = {sampling_frequency},"
                f"period = {self.period}."
            )

    def _compute_periodic_feature(self, datetime_index: pd.DatetimeIndex):
        print((datetime_index - self.start_date) / self.period)
        return (
            np.sin(2 * pi * (datetime_index - self.start_date) / self.period)
        ) * self.amplitude


class ConstantFeature(StandardFeature):
    """Generate a ``pd.DataFrame`` with one column, of the same length as the
    input ``X`` and containing the value ``constant`` across the whole column.

    Parameters
    ----------
    constant : int, optional, default: ``2``
        The value to use to generate the constant column of the
        ``pd.DataFrame``.

    length: int, optional, default: ``50``
        The length of the DataFrame to generate. This is used only if X is not passed in
         the ``transform`` method, otherwise the length is inferred from it.

    output_name : str, optional, default: ``'ConstantFeature'``
        The name of the output column.

    """

    def __init__(
        self, constant: int = 2, length: int = 50, output_name: str = "ConstantFeature"
    ):
        super().__init__(output_name)
        self.length = length
        self.constant = constant

    def transform(self, time_series: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate a ``pd.DataFrame`` with one column with the same length as
        ``time_series`` and with the same index, containing a value equal to
        ``constant``.

        Parameters
        ----------
        time_series : pd.DataFrame, shape (n_samples, 1), optional, default: ``None``
            The input DataFrame. If passed, is used only for its index.

        Returns
        -------
        constant_series_renamed : pd.DataFrame, shape (length, 1)
            A constant series, with the same length of ``X`` and with the same index.

        """
        if time_series is not None:
            constant_series = pd.Series(
                data=self.constant, index=time_series.index
            ).to_frame()
        else:
            constant_series = pd.Series(data=[self.constant] * self.length).to_frame()

        constant_series_renamed = self._rename_columns(constant_series)
        return constant_series_renamed


class CustomFeature(StandardFeature):
    """Given a custom function, apply it to a time series and generate a
    ``pd.Dataframe``.

    Parameters
    ----------
    custom_feature_function`: Callable, required.
        The function to use to generate a ``pd.DataFrame`` containing the
        feature.

    output_name: str, optional, default: ``'CustomFeature'``.
        The name of the output column.

    kwargs : ``object``, optional.
        Optional arguments to pass to the function.

    """

    def __init__(
        self,
        custom_feature_function: Callable,
        output_name: str = "CustomFeature",
        **kwargs: object,
    ):
        super().__init__(output_name)
        self.custom_feature_function = custom_feature_function
        self.kwargs = kwargs

    def transform(self, time_series: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate a ``pd.DataFrame``, given ``time_series`` as input to the
        ``custom_feature_function``, as well as other optional arguments.

        Parameters
        ----------
        time_series : pd.DataFrame, shape (n_samples, 1), optional, default: ``None``
            The DataFrame from which to generate the feature_creation

        Returns
        -------
        custom_feature_renamed : pd.DataFrame, shape (length, 1)
            A DataFrame containing the generated feature_creation.

        """
        custom_feature = self.custom_feature_function(time_series, **self.kwargs)
        custom_feature_renamed = self._rename_columns(custom_feature)
        return custom_feature_renamed
