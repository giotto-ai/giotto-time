from typing import Union, Optional, Callable

import numpy as np
import pandas as pd
from pandas import DatetimeIndex

from math import pi

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from giottotime.base import FeatureMixin

__all__ = ["PeriodicSeasonal", "Constant"]


# TODO: add something like 'make_periodic_feature' or 'make_sinusoid'
class PeriodicSeasonal(BaseEstimator, TransformerMixin, FeatureMixin):
    """Create a sinusoid from a given date and with a given period and
    amplitude.

    Parameters
    ----------
    period : Union[pd.Timedelta, str], optional, default: ``'365 days'``
        The period of the generated time series.

    amplitude : float, optional, default: ``0.5``
        The amplitude of the time series.

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

    Examples
    --------
    >>> from giottotime.feature_extraction import PeriodicSeasonal
    >>> period_feature = PeriodicSeasonal(start_date="2020-01-01",
    ...                                          index_period=10)
    >>> period_feature.fit_transform()
    Float64Index([              0.0, 0.0027397260273972603,
               0.005479452054794521,   0.00821917808219178,
               0.010958904109589041,    0.0136986301369863,
                0.01643835616438356,  0.019178082191780823,
               0.021917808219178082,  0.024657534246575342],
             dtype='float64')

       PeriodicSeasonal
    0                 0.000000
    1                 0.008607
    2                 0.017211
    3                 0.025810
    4                 0.034401
    5                 0.042982
    6                 0.051551
    7                 0.060104
    8                 0.068639
    9                 0.077154

    """

    def __init__(
            self,
            period: Union[pd.Timedelta, str] = "365 days",
            amplitude: float = 0.5,
            start_date: Optional[Union[pd.Timestamp, str]] = None,
            length: Optional[int] = 50,
            index_period: Optional[Union[DatetimeIndex, int]] = None,
    ):
        self.start_date = start_date
        self.period = period
        self.amplitude = amplitude
        self.length = length
        self.index_period = index_period

    def fit(self, X, y=None):
        """Fit the estimator.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Input data.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        self.columns_ = X.columns.values
        return self

    def transform(self, time_series: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate a sinusoid, with the given ``period``, ``amplitude`` and ``length``,
        starting from the selected ``start_date``. If ``time_series`` is not ``None``,
        the ``start_date`` is replaced by the start date of the time series and the
        output sinusoid will have the same index as ``time_series``.

        Parameters
        ----------
        time_series : pd.DataFrame, shape (n_samples, 1), optional, default: ``None``
            The input DataFrame, If passed, the output DataFrame is going to have the
            same index as ``time_series``. If is not passed, then the ``start_date`` and
             ``index_period`` must have been passed in the constructor when the object
             was instantiated.

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

        return periodic_feature.add_suffix('__' + self.__class__.__name__)

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
        return (np.sin(2 * pi * (datetime_index - self.start_date) / self.period)) * self.amplitude


class Constant(BaseEstimator, TransformerMixin, FeatureMixin):
    """Generate a ``pd.DataFrame`` with one column, of the same length as the input
     ``X`` and containing the value ``constant`` across the whole column.

    Parameters
    ----------
    constant : int, optional, default: ``2``
        The value to use to generate the constant column of the ``pd.DataFrame``.

    length : int, optional, default: ``50``
        The length of the DataFrame to generate. This is used only if X is not passed in
        the ``transform`` method, otherwise the length is inferred from it.

    Examples
    --------
    >>> from giottotime.feature_extraction import ConstantFeature
    >>> constant_feature = ConstantFeature(constant=3, length=5)
    >>> constant_feature.transform()
      ConstantFeature
    0               3
    1               3
    2               3
    3               3
    4               3

    """

    def __init__(
            self, constant: int = 0, length: int = None
    ):
        super().__init__()
        self.length = length
        self.constant = constant

    def get_feature_names(self):
        """
        Return feature names for output features.

        Returns
        -------
        output_feature_names : ndarray of shape (n_output_features,)
            Array of feature names.
        """

        return [self.__class__.__name__]

    def fit(self, X, y=None):
        """Fit the estimator.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Input data.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        self.columns_ = X.columns.values
        self.rows_ = X.shape[0]
        self.length_ = np.min([self.length, self.rows_]) if self.length is not None else self.rows_
        return self

    def transform(self, time_series: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate a ``pd.DataFrame`` with one column with the same length as
        ``time_series`` and with the same index, containing a value equal to
        ``constant``.

        Parameters
        ----------
        time_series : pd.DataFrame, shape (n_samples, 1), optional, default: ``None``
            The input DataFrame. If passed, the output DataFrame is going to have the
            same index as ``time_series``.

        Returns
        -------
        constant_series_renamed : pd.DataFrame, shape (length, 1)
            A constant series, with the same length of ``X`` and with the same index.

        """
        check_is_fitted(self)

        constants = np.full(self.rows_, np.nan)
        constants[:self.length_] = self.constant
        if time_series is not None:
            constant_series = pd.Series(
                data=constants, index=time_series.index
            ).to_frame()
        else:
            constant_series = pd.Series(data=constants).to_frame()

        constant_series_renamed = constant_series.add_suffix('__' + self.__class__.__name__)
        return constant_series_renamed
