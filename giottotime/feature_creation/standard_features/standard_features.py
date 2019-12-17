from typing import Union, Optional, Callable

import numpy as np
import pandas as pd
from pandas import DatetimeIndex

from math import pi

from giottotime.feature_creation.standard_features.base import StandardFeature


class PeriodicSeasonalFeature(StandardFeature):
    """Create a sinusoid from a given date and with a given period and
    amplitude.

    Parameters
    ----------
    period : ``Union[pd.Timedelta, str]``, required.
        The period of the generated time series.

    amplitude : ``float``, required.
        The amplitude of the time series.

    output_name : ``str``, required.
        The name of the output column.

    start_date : ``Union[pd.Timestamp, str]``, optional, (default=``None``).
        The date from which to start generating the feature. This is used only if X is
        not passed in the ``transform`` method, otherwise the start date is inferred
        from it.

    length : ``int``, optional, (default=``50``).
        The length of the sinusoid. This is used only if X is not passed in the
        ``transform`` method, otherwise the length is inferred from it.

    index_period : ``Union[DatetimeIndex, int]``,
        The period of the index of the output ``DataFrame``. This is used only if X is
        not passed in the ``transform`` method, otherwise the index period is taken
        from it.

    """

    def __init__(
        self,
        period: Union[pd.Timedelta, str],
        amplitude: float,
        output_name: str,
        start_date: Optional[Union[pd.Timestamp, str]] = None,
        length: Optional[int] = 50,
        index_period: Optional[Union[DatetimeIndex, int]] = None,
    ):
        super().__init__(output_name=output_name)
        self._start_date = start_date
        self._period = period
        self._amplitude = amplitude
        self._length = length
        self._index_period = index_period

    def transform(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Starting from the index of ``X``, generate a sinusoid,
        with the given ``period`` and ``amplitude``.

        Parameters
        ----------
        X : ``pd.DataFrame``, optional, (default=``None``).
            The input DataFrame, used only for its index. If is not passed, then the
            ``start_date`` and ``index_period`` must have been passed in the constructor
            when the object was instantiated.

        Returns
        -------
        periodic_feature : ``pd.DataFrame``, required.
            The DataFrame containing the generated period feature.

        Raises
        ------
        ``ValueError``
            Raised if ``X`` is not passed and the ``start_date`` or the ``index_period``
            are not present.

        """
        self._validate_input(X)
        self._transform_inputs(X)
        datetime_index = self._get_time_index(X)

        periodic_feature_values = self._compute_periodic_feature(datetime_index)
        if X is not None:
            periodic_feature = pd.DataFrame(index=X.index, data=periodic_feature_values)

        else:
            periodic_feature = pd.DataFrame(
                data=periodic_feature_values[: self._length]
            )

        periodic_feature = self._rename_columns(periodic_feature)
        return periodic_feature

    def _validate_input(self, X: pd.DataFrame) -> None:
        if X is None:
            if self._start_date is None:
                raise ValueError(
                    f"If X is not provided, the 'start_date' variable "
                    f"must be provided, but instead was "
                    f"{self._start_date}."
                )
            if self._index_period is None:
                raise ValueError(
                    f"If X is not provided, the 'index_period' variable "
                    f"must be provided, but instead was "
                    f"{self._index_period}."
                )

    def _transform_inputs(self, X: pd.DataFrame) -> None:
        if isinstance(self._period, str):
            self._period = pd.Timedelta(self._period)

        if X is not None:
            self._start_date = X.index.values[0]

        else:
            if isinstance(self._start_date, str):
                self._start_date = pd.to_datetime(self._start_date)

    def _get_time_index(self, X: pd.DataFrame) -> pd.DatetimeIndex:
        if X is not None:
            datetime_index = X.index

        else:
            if isinstance(self._index_period, int):
                datetime_index = pd.date_range(
                    start=self._start_date, periods=self._index_period
                )
            else:
                datetime_index = self._index_period

        return datetime_index

    def _convert_index_to_datetime(self, index: pd.PeriodIndex) -> pd.DatetimeIndex:
        datetime_index = index.to_timestamp()
        self._check_sampling_frequency(datetime_index)
        return datetime_index

    def _check_sampling_frequency(self, datetime_index: pd.DatetimeIndex) -> None:
        sampling_frequency = pd.Timedelta(datetime_index.freq)
        if sampling_frequency < 2 * self._period:
            raise ValueError(
                f"Sampling frequency must be at least two times"
                f"the period to obtain meaningful results. "
                f"Sampling frequency = {sampling_frequency},"
                f"period = {self._period}."
            )

    def _compute_periodic_feature(self, datetime_index: pd.DatetimeIndex):
        return (
            np.sin(2 * pi * (datetime_index - self._start_date) / self._period)
        ) * self._amplitude


class ConstantFeature(StandardFeature):
    """Generate a ``pd.DataFrame`` with one column, of the same length as the
    input ``X`` and containing the value ``constant`` across the whole column.

    Parameters
    ----------
    constant : ``int``, required.
        The value to use to generate the constant column of the
        ``pd.DataFrame``.

    output_name : ``str``, required.
        The name of the output column.

    length: ``int``, optional, (default=``50``).
        The length of the DataFrame to generate. This is used only if X is not passed in
         the ``transform`` method, otherwise the length is inferred from it.

    """

    def __init__(self, constant: int, output_name: str, length: int = 50):
        super().__init__(output_name)
        self._length = length
        self.constant = constant

    def transform(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate a ``pd.DataFrame`` with one column with the same length as
        ``X`` and with the same index, containing a value equal to
        ``constant``.

        Parameters
        ----------
        X : ``pd.DataFrame``, required.
            The input DataFrame. It is used only for its index.

        Returns
        -------
        constant_series_renamed : ``pd.DataFrame``
            A constant series, with the same length of ``X`` and with the same
            index.

        """
        if X is not None:
            constant_series = pd.Series(data=self.constant, index=X.index).to_frame()
        else:
            constant_series = pd.Series(data=[self.constant] * self._length).to_frame()

        constant_series_renamed = self._rename_columns(constant_series)
        return constant_series_renamed


class CustomFeature(StandardFeature):
    """Given a custom function, apply it to a time series and generate a
    ``pd.Dataframe``.

    Parameters
    ----------
    custom_feature_function`: ``Callable`, required.
        The function to use to generate a ``pd.DataFrame`` containing the
        feature.

    output_name: ``str``, required.
        The name of the output column.

    kwargs : ``object``, optional.
        Optional arguments to pass to the function.

    """

    def __init__(
        self, custom_feature_function: Callable, output_name: str, **kwargs: object
    ):
        super().__init__(output_name)
        self.custom_feature_function = custom_feature_function
        self.kwargs = kwargs

    def transform(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate a ``pd.DataFrame``, given ``X`` as input to the
        ``custom_feature_function``, as well as other optional arguments.

        Parameters
        ----------
        X : ``pd.DataFrame``, required.
            The DataFrame from which to generate the feature_creation

        Returns
        -------
        custom_feature_renamed : ``pd.DataFrame``
            A DataFrame containing the generated feature_creation.

        """
        custom_feature = self.custom_feature_function(X, **self.kwargs)
        custom_feature_renamed = self._rename_columns(custom_feature)
        return custom_feature_renamed
