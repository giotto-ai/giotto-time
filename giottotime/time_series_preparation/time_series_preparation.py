from typing import List, Union, Optional

import numpy as np
import pandas as pd

from .time_series_conversion import (
    PandasSeriesToTimeIndexSeries,
    SequenceToTimeIndexSeries,
    TimeIndexSeriesToPeriodIndexSeries,
)
from ..time_series_preparation.time_series_resampling import TimeSeriesResampler

SUPPORTED_SEQUENCE_TYPES = [
    np.ndarray,
    list,
]


class TimeSeriesPreparation:
    """Transforms an array-like sequence in a period-index DataFrame with a single
    column.

    Here is what happens:
    - if a `list` or `np.array` is passed, the PeriodIndex is built using the parameters
        `start`, `end` and `freq`
    - if a `pd.Series` is passed, it checks if the index is a time index
       (`DatetimeIndex`, `TimedeltaIndex`, `PeriodIndex`) or not. If not the index is
       built as if it were a `list` or `np.array. If yes the index is converted to
       PeriodIndex.

    Parameters
    ----------
    start : pd.datetime, optional, default: ``None``
        The date to use as start date.

    end : pd.datetime, optional, default: ``None``
        The date to use as end date.

    freq : pd.Timedelta, optional, default: ``None``
        The frequency of the output time series. Not mandatory for all time series
        conversion.

    resample_if_not_equispaced : bool, optional, default: ``False``
        Not supported yet, leave it as True

    output_name : str, optional, default: ``'time_series'``
        The name of the output column

    Raises
    ------
    ValueError
        Of the three parameters: start, end, and periods, exactly two must be specified.

    Examples
    --------
    >>> time_series = [1,2,3,5,5,7]
    >>> period_index_time_series = pd.Series(
    ...     index = pd.period_range(start='01-01-2010', freq='10D', periods=6),
    ...     data=[1,2,3,5,5,7]
    ... )
    >>> datetime_index_time_series = pd.Series(
    ...     index = pd.date_range(start='01-01-2010', freq='10D', periods=6),
    ...     data=[1,2,3,5,5,7]
    ... )
    >>> timedelta_index_time_series = pd.Series(
    ...     index = pd.timedelta_range(start=pd.Timedelta(days=1), freq='10D', periods=6),
    ...     data=[1,2,3,5,5,7]
    ... )
    >>> time_series_preparation = TimeSeriesPreparation()
    >>> time_series_preparation.transform(time_series)
                time_series
    1970-01-01            1
    1970-01-02            2
    1970-01-03            3
    1970-01-04            5
    1970-01-05            5
    1970-01-06            7
    >>> time_series_preparation.transform(period_index_time_series)
                time_series
    2010-01-01            1
    2010-01-11            2
    2010-01-21            3
    2010-01-31            5
    2010-02-10            5
    2010-02-20            7
    >>> time_series_preparation.transform(datetime_index_time_series)
                time_series
    2010-01-01            1
    2010-01-11            2
    2010-01-21            3
    2010-01-31            5
    2010-02-10            5
    2010-02-20            7
    >>> time_series_preparation.transform(timedelta_index_time_series)
                time_series
    1970-01-02            1
    1970-01-12            2
    1970-01-22            3
    1970-02-01            5
    1970-02-11            5
    1970-02-21            7

    """

    def __init__(
        self,
        start: Optional[pd.datetime] = None,
        end: Optional[pd.datetime] = None,
        freq: Optional[pd.Timedelta] = None,
        resample_if_not_equispaced: bool = False,
        output_name: str = "time_series",
    ):
        self.start = start
        self.end = end
        self.freq = freq
        self.resample_if_not_equispaced = resample_if_not_equispaced
        self.output_name = output_name

        self.pandas_converter = PandasSeriesToTimeIndexSeries(
            self.start, self.end, self.freq
        )
        self.sequence_converter = SequenceToTimeIndexSeries(
            self.start, self.end, self.freq
        )
        self.resampler = TimeSeriesResampler()
        self.to_period_index_series_converter = TimeIndexSeriesToPeriodIndexSeries(
            self.freq
        )

    def transform(
        self, time_series: Union[List, np.array, pd.Series, pd.DataFrame]
    ) -> pd.DataFrame:
        """Transforms an array-like sequence in a period-index DataFrame with a single
        column.

        Parameters
        ----------
        time_series : Union[List, np.array, pd.Series, pd.DataFrame], required
            The input time series.

        Returns
        -------
        period_index_dataframe : pd.DataFrame
            The output dataframe with a period index.

        """
        pandas_time_series = self._to_time_index_series(time_series)
        equispaced_time_series = self._to_equispaced_time_series(pandas_time_series)
        period_index_time_series = self._to_period_index_time_series(
            equispaced_time_series
        )
        period_index_dataframe = self._to_period_index_dataframe(
            period_index_time_series
        )
        return period_index_dataframe

    def _to_time_index_series(
        self, array_like_object: Union[List, np.array, pd.Series, pd.DataFrame]
    ) -> pd.Series:
        if isinstance(array_like_object, pd.DataFrame):
            return self.pandas_converter.transform(array_like_object.iloc[:, 0])
        elif isinstance(array_like_object, pd.Series):
            return self.pandas_converter.transform(array_like_object)
        elif any(
            isinstance(array_like_object, type_) for type_ in SUPPORTED_SEQUENCE_TYPES
        ):
            return self.sequence_converter.transform(array_like_object)
        else:
            raise TypeError(
                f"Type {type(array_like_object)} is not a "
                f"supported time series type"
            )

    def _to_equispaced_time_series(self, time_series: pd.Series) -> pd.Series:
        if self.resample_if_not_equispaced:
            self.resampler.transform(time_series)
        else:
            return time_series

    def _to_period_index_time_series(self, time_series: pd.Series) -> pd.Series:
        return self.to_period_index_series_converter.transform(time_series)

    def _to_period_index_dataframe(self, time_series: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({self.output_name: time_series})
