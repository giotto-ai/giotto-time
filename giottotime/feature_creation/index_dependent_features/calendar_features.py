import importlib
from typing import Optional, Union, List

import numpy as np
import pandas as pd
import workalendar

from .base import IndexDependentFeature


def check_index(time_series: pd.DataFrame) -> None:
    index_series = pd.Series(time_series.index)
    unique_differences = pd.unique(index_series.diff().dropna().values)

    if len(unique_differences) != 1:
        raise ValueError(
            "The time series should be evenly spaced in order to obtain "
            "meaningful results."
        )

    if not isinstance(time_series.index, pd.PeriodIndex):
        raise ValueError(
            "The input time series should have a index of type "
            f"PeriodIndex, got {type(time_series.index)} instead."
        )


class CalendarFeature(IndexDependentFeature):
    """Create a feature based on the national holidays of a specific country, based on
    a given kernel (if provided). The interface for this is based on the one of
    'workalendar'. To see which regions and countries are available, check the
    'workalendar' `documentation <https://peopledoc.github.io/workalendar/>`_.

    Parameters
    ----------
    region : str, optional, default: ``'america'``
        The region in which the ``country`` is located.

    country : str, optional, default: ``'Brazil'``
        The name of the country from which to retrieve the holidays. The country must be
         located in the given ``region``.

    start_date : str, optional, default: ``'01/01/2019'``
        The date starting from which to retrieve the holidays.

    end_date : str, optional, default: ``'01/01/2020'``
        The date until which to retrieve the holidays.

    kernel : list or np.ndarray, optional, default: ``None``
        The kernel to use when creating the feature.

    reindex_method : str, optional, default: ``pad``
        Used only if X is passed in the ``transform`` method. It is used as the method
        with which to reindex the holiday events with the index of X. This method
        should be compatible with the reindex methods provided by pandas. Please refer
        to the pandas `documentation
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reindex.html`_
        for further details.

    output_name : str, optional, default: ``'CalendarFeature'``
        The name of the output column.

    Examples
    --------
    >>> from giottotime.feature_creation import CalendarFeature
    >>> cal_feature = CalendarFeature(region="europe", country="Italy", kernel=[3, 2])
    >>> cal_feature.transform()
                CalendarFeature
    2018-01-01              2.0
    2018-01-02              3.0
    2018-01-03              0.0
    2018-01-04              0.0
    2018-01-05              0.0
    ...                     ...
    2019-12-28              0.0
    2019-12-29              0.0
    2019-12-30              0.0
    2019-12-31              0.0
    2020-01-01              2.0
    """

    def __init__(
        self,
        region: str = "america",
        country: str = "Brazil",
        start_date: str = "01/01/2018",
        end_date: str = "01/01/2020",
        kernel: Union[List, np.ndarray] = None,
        reindex_method: str = "pad",
        output_name: str = "CalendarFeature",
    ):
        super().__init__(output_name)
        self.region = region
        self.country = country
        self.start_date = start_date
        self.end_date = end_date
        self.reindex_method = reindex_method

        if kernel is None or len(kernel) == 0 or not np.isfinite(kernel).all():
            raise ValueError(
                "The kernel should be an array-like object, with at least 1 element "
                f"and should only contains finite values, got {kernel} instead."
            )
        self.kernel = kernel

    def transform(self, time_series: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate a DataFrame containing the events associated to the holidays of the
        selected ``country``.

        Parameters
        ----------
        time_series : pd.DataFrame, shape (n_samples, 1), optional, default: ``None``
            If provided, both ``start_date`` and ``end_date`` are going to be
            overwritten with the start and end date of the index of ``time_series``.
            Also, if provided the output DataFrame is going to be re-indexed with the
            index of ``time_series``, using the chosen ``reindex_method``.

        Returns
        -------
        events : pd.DataFrame, shape (length, 1)
            A DataFrame containing the events.

        """
        if time_series is not None:
            check_index(time_series)

        self._initialize_start_end_date(time_series)

        workalendar_region = importlib.import_module(f".{self.region}", "workalendar")
        workalendar_country = getattr(workalendar_region, self.country)()

        events = self._get_holiday_events(workalendar_country)
        if self.kernel is not None:
            events = self._apply_kernel(events)

        events_renamed = self._rename_columns(events)
        aligned_events = self._align_event_indices(time_series, events_renamed)

        return aligned_events

    def _initialize_start_end_date(self, X: pd.DataFrame):
        if X is not None:
            self.start_ = X.index.values[0]
            self.end_ = X.index.values[-1]
        else:
            self.start_ = pd.Timestamp(self.start_date)
            self.end_ = pd.Timestamp(self.end_date)

        slack_period = len(self.kernel)
        slack_days = np.timedelta64(slack_period, "D")

        slacked_start_date_ = self.start_ - slack_days
        slacked_end_date_ = self.end_ + slack_days

        self.slacked_start_date_ = pd.datetime(
            year=slacked_start_date_.year,
            month=slacked_start_date_.month,
            day=slacked_start_date_.day,
        )
        self.slacked_end_date_ = pd.datetime(
            year=slacked_end_date_.year,
            month=slacked_end_date_.month,
            day=slacked_end_date_.day,
        )

    def _get_holiday_events(self, country_mod):
        index = pd.date_range(
            start=self.slacked_start_date_, end=self.slacked_end_date_, freq="D"
        )
        years = index.year.unique()
        events = pd.DataFrame()
        for year in years:
            events = events.append(country_mod.holidays(year))

        events = events.rename(columns={0: "date", 1: "events"})
        events["date"] = pd.to_datetime(events["date"])
        events["status"] = 1
        events = events.set_index("date")
        events = self._group_by_event_name(events)

        events = events.reindex(index)
        events = events.drop(columns=["events"])

        events["status"] = events["status"].fillna(0).astype(int)
        events = events.sort_index()

        return events

    def _group_by_event_name(self, events_df):
        def aggregate_events_same_day(event):
            return pd.Series(
                {"status": 1, "events": " & ".join(event["events"].tolist())}
            )

        grouped_events = events_df.groupby(events_df.index).apply(
            aggregate_events_same_day
        )
        return grouped_events

    def _apply_kernel(self, events):
        kernel_events = events.copy()
        klen = len(self.kernel)

        def apply_kernel(w):
            if sum(w) != 0:
                return np.dot(w, self.kernel) / sum(w)
            else:
                return 0

        kernel_events["status"] = (
            kernel_events["status"]
            .rolling(klen, center=True)
            .apply(apply_kernel, raw=False)
        )
        return kernel_events

    def _align_event_indices(self, ts, events):
        if ts is not None:
            X = ts.copy()
            new_x_line = pd.DataFrame(
                columns=X.columns, index=pd.PeriodIndex([X.index[-1] + 1])
            )
            X_to_cut = pd.concat([X, new_x_line], axis=0)
            bins = pd.cut(events.index, X_to_cut.index.to_timestamp())

            grouped_events = events.groupby(bins).mean().ffill()

            grouped_events.index = pd.to_datetime(
                grouped_events.index.map(lambda row_index: row_index.left)
            )
            X.index = X.index.to_timestamp()

            events_renamed = pd.merge(
                X, grouped_events, left_index=True, right_index=True, how="inner"
            )
            events_renamed.index = ts.index
        else:
            events_renamed = events[self.start_ : self.end_]

        return events_renamed
