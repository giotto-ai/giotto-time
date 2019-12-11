from abc import abstractmethod
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

from giottotime.base.constants import DEFAULT_START, DEFAULT_FREQ

PandasTimeIndex = Union[pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex]
PandasDate = Union[pd.datetime, pd.Timestamp, str]

SUPPORTED_SEQUENCE_TYPES = [
    np.ndarray,
    list,
]


def count_not_none(*args):
    """Returns the count of arguments that are not None.
    """
    return sum(x is not None for x in args)


def check_period_range_parameters(
    start_date: PandasDate, end_date: PandasDate, periods: int
) -> None:
    """Check if the period range parameters given as input are compatible with the
    `pd.period_range` method.

    Of the three parameters: start, end, and periods, exactly two must be specified.

    Parameters
    ----------
    start_date : ``PandasDate``, required
    end_date : ``PandasDate``, required
    periods : ``int``, required

    Raises
    ------
    ``ValueError``
        Of the three parameters: start, end, and periods, exactly two must be specified.
    """
    if count_not_none(start_date, end_date, periods) != 2:
        raise ValueError(
            "Of the three parameters: start, end, and periods, "
            "exactly two must be specified"
        )


class TimeSeriesConversion(BaseEstimator, TransformerMixin):
    """Parent class for all time series type conversions.

    Subclasses must implement the two methods `_get_index_from` and `_get_values_from`.

    Parameters
    ----------
    start : ``PandasData``, optional, (default=``None``)
        start date of the output time series. Not mandatory for all time series
        conversion.

    end : ``PandasData``, optional, (default=``None``)
        end date of the output time series. Not mandatory for all time series
        conversion.

    freq : ``pd.Timedelta``, optional, (default=``None``)
        frequency of the output time series. Not mandatory for all time series
        conversion.
    """

    def __init__(
        self,
        start: Optional[PandasDate] = None,
        end: Optional[PandasDate] = None,
        freq: Optional[pd.Timedelta] = None,
    ) -> None:
        self._initialize_start_end_freq(start, end, freq)

    def fit(self, X: Union[pd.Series, np.array, list], y=None):
        """Do nothing and return the estimator unchanged.
        This method is there to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        check_array(X)

        return self

    def transform(self, X: Union[pd.Series, np.array, list]) -> pd.Series:
        """Transforms an array-like object (list, np.array, pd.Series)
        into a pd.Series with time index.

        It calls internally the abstract methods `_get_index_from()` and
        `_get_values_from()`. These are implemented in the subclasses.

        Parameters
        ----------
        X : ``Union[List, np.array, pd.Series]``, required.
            It depends on the implementation of the subclasses.

        Returns
        -------
        transformed series: ``pd.Series``
        """
        index = self._get_index_from(X)
        values = self._get_values_from(X)
        return pd.Series(data=values, index=index)

    @abstractmethod
    def _get_index_from(
        self, array_like_object: Union[pd.Series, np.ndarray, list]
    ) -> PandasTimeIndex:
        pass

    @abstractmethod
    def _get_values_from(
        self, array_like_object: Union[pd.Series, np.array, list]
    ) -> np.ndarray:
        pass

    def _initialize_start_end_freq(
        self, start: PandasDate, end: PandasDate, freq: pd.Timedelta
    ) -> None:
        not_none_params = count_not_none(start, end, freq)
        if not_none_params == 0:
            self._default_params_initialization()
        elif not_none_params == 1:
            self._one_not_none_param_initialization(start, end, freq)
        elif not_none_params == 2:
            self._two_not_none_params_initialization(start, end, freq)
        else:
            raise ValueError(
                "Of the three parameters: start, end, and "
                "freq, exactly two must be specified"
            )

    def _default_params_initialization(self):
        self.start = DEFAULT_START
        self.end = None
        self.freq = DEFAULT_FREQ

    def _one_not_none_param_initialization(
        self, start: PandasDate, end: PandasDate, freq: pd.Timedelta
    ):
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

    def _two_not_none_params_initialization(
        self, start: PandasDate, end: PandasDate, freq: pd.Timedelta
    ):
        self.start = start
        self.end = end
        self.freq = freq

    def _compute_period_index_of_length(self, length: int) -> pd.PeriodIndex:
        check_period_range_parameters(self.start, self.end, length)
        return pd.period_range(
            start=self.start, end=self.end, periods=length, freq=self.freq
        )

    def _more_tags(self):
        return {"requires_fit": False}
