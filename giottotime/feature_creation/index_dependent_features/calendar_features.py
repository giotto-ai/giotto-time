import importlib
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
import workalendar

from .base import IndexDependentFeature

__all__ = ["CalendarFeature"]


def check_index(time_series: pd.DataFrame) -> None:
    index_series = pd.Series(time_series.index)
    unique_differences = pd.unique(index_series.diff().dropna().values)

    if len(unique_differences) != 1:
        raise ValueError(
            "The time series should be evenly spaced in order to obtain "
            "meaningful results."
        )


def get_period(X):
    if isinstance(X.index, pd.PeriodIndex):
        freq = X.index.freqstr
    else:
        warnings.warn("Frequency inferred from X.")
        freq = pd.Timedelta(X.index.values[1] - X.index.values[0])

    return freq


class CalendarFeature(IndexDependentFeature):
    """Create a feature based on the national holidays of a specific country, based on
    a given kernel (if provided). The interface for this is based on the one of
    'workalendar'. To see which regions and countries are available, check the
    'workalendar' `documentation <https://peopledoc.github.io/workalendar/>`_.

    Parameters
    ----------

    region : ``str``, required.
        The region in which the ``country`` is located.

    country : ``str``, required.
        The name of the country from which to retrieve the holidays. The country must be
         located in the given ``region``.

    kernel : ``np.ndarray``, required.
        The kernel to use when creating the feature.

    output_name : ``str``, required.
        The name of the output column.+

    start_date : ``str``, optional, (default=``'01/01/2018'``).
        The date starting from which to retrieve the holidays.

    end_date : ``str``, optional, (default=``'01/01/2020'``).
        The date until which to retrieve the holidays.

    reindex_method : ``str``, optional, (default=``pad``).
        Used only if X is passed in the ``transform`` method. It is used as the method
        with which to reindex the holiday events with the index of X. This method
        should be compatible with the reindex methods provided by pandas. Please refer
        to the pandas `documentation
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reindex.html`_
        for further details.

    return_name_event : ``bool``, optional, (default=``False``)
        If ``True``, also return a column containing the name of the event.

    """

    def __init__(
        self,
        region: str,
        country: str,
        kernel: Union[np.array, list, pd.Series],
        output_name: str,
        start_date: str = "01/01/2018",
        end_date: str = "01/01/2020",
        reindex_method: str = "pad",
    ):
        super().__init__(output_name)
        self._region = region
        self._country = country
        self._start_date = start_date
        self._end_date = end_date
        self._reindex_method = reindex_method

        if len(kernel) == 0 or not np.isfinite(kernel).all():
            raise ValueError(
                "The kernel should be an array-like object, with at least "
                "element and should only contains finite values, got "
                f"{kernel} instead."
            )
        self._kernel = kernel

    # TODO: write the description of the transform method
    def transform(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """

        Parameters
        ----------
        X : ``pd.DataFrame``, optional, (default=``None``).
            If provided, both ``start_date`` and ``end_date`` are going to be
            overwritten with the start and end date of the index of ``X``. Also, if
            provided the output DataFrame is going to be re-indexed with the index of
            ``X``, using the chosen ``reindex_method``.

        Returns
        -------
        events : ``pd.DataFrame``
            A DataFrame containing the events.

        """
        if X is not None:
            check_index(X)

        self._initialize_start_end_date(X)

        workalendar_region = importlib.import_module(f".{self._region}", "workalendar")
        workalendar_country = getattr(workalendar_region, self._country)()

        events = self._get_holiday_events(workalendar_country)

        if self._kernel is not None:
            klen = len(self._kernel)

            def apply_kernel(w):
                if sum(w) != 0:
                    return np.dot(w, self._kernel) / sum(w)
                else:
                    return 0

            events["status"] = (
                events["status"]
                .rolling(klen, center=True)
                .apply(apply_kernel, raw=False)
            )
        events_renamed = self._rename_columns(events)

        if X is not None:
            new_x_line = pd.DataFrame(
                columns=X.columns, index=pd.PeriodIndex([X.index[-1] + 1])
            )
            X_to_cut = pd.concat([X, new_x_line], axis=0)
            bins = pd.cut(events_renamed.index, X_to_cut.index.to_timestamp())
            grouped_events = events_renamed.groupby(bins).mean().ffill()

            grouped_events.index = pd.to_datetime(
                grouped_events.index.map(lambda row_index: row_index.left)
            )
            X.index = X.index.to_timestamp()

            events_renamed = pd.merge(
                X, grouped_events, left_index=True, right_index=True, how="inner"
            )
        return events_renamed

    def _initialize_start_end_date(self, X: pd.DataFrame):
        if X is not None:
            self._start_date = X.index.values[0]
            self._end_date = X.index.values[-1]
        else:
            self._start_date = pd.Timestamp(self._start_date)
            self._end_date = pd.Timestamp(self._end_date)

        slack_period = len(self._kernel)
        slack_days = np.timedelta64(slack_period, "D")

        slacked_start_date_ = self._start_date - slack_days
        slacked_end_date_ = self._end_date + slack_days

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
