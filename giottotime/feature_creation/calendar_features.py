import importlib
from typing import Optional

import numpy as np
import pandas as pd

from .base import TimeSeriesFeature

__all__ = [
    'CalendarFeature'
]


class CalendarFeature(TimeSeriesFeature):
    """Create a feature based on the national holidays of a specific country,
    based on a given kernel (if provided). The interface for this is based on
    the one of 'workalendar'. To see which regions and countries are available,
    check the 'workalendar' `documentation
    <https://peopledoc.github.io/workalendar/>`_.

    Parameters
    ----------
    output_name : ``str``, required.
        The name of the output column.

    region : ``str``, optional, (default=``'america'``).
        The region in which the ``country`` is located.

    country : ``str``, optional, (default=``'Brazil'`).
        The name of the country from which to retrieve the holidays. The
        country must be located in the given ``region``.

    start_date : ``str``, optional, (default=``'01/01/2018'``)
        The date starting from which to retrieve the holidays.

    end_date : ``str``, optional, (default=``'01/01/2020'``)
        The date until which to retrieve the holidays.

    kernel : ``np.ndarray``, optional, (default=``None``).
        The kernel to use when creating the feature.

    """
    def __init__(self,
                 output_name: str,
                 region: str = 'america',
                 country: str = 'Brazil',
                 start_date: str = '01/01/2018',
                 end_date: str = '01/01/2020',
                 kernel: Optional[np.ndarray] = None):
        super().__init__(output_name)
        self.region = region
        self.country = country
        self.start_date = start_date
        self.end_date = end_date
        self.kernel = kernel

    # TODO: finish docstrings
    def transform(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """

        Parameters
        ----------
        X : ``pd.DataFrame``, optional, (default=``None``).
            If provided, both ``start_date`` and ``end_date`` are going to be
            overwritten with the start and end date of the index of ``X``.

        Returns
        -------
        events : ``pd.DataFrame``

        """
        mod = importlib.import_module(f".{self.region}", 'workalendar')
        country_mod = getattr(mod, self.country)()

        index = pd.date_range(start=self.start_date,
                              end=self.end_date,
                              freq='D')

        years = index.year.unique()

        events = pd.DataFrame()

        for year in years:
            events = events.append(country_mod.holidays(year))

        events = events.rename(columns={0: 'date', 1: 'events'})
        events['date'] = pd.to_datetime(events['date'])
        events['status'] = 1
        events = events.set_index('date')

        events = events.reindex(index)

        events['events'] = events['events'].fillna('none')
        events['status'] = events['status'].fillna(0).astype(int)

        events = events.sort_index()

        klen = len(self.kernel)

        if self.kernel is None:
            return events

        else:
            def ip(w):
                if sum(w) != 0:
                    return np.dot(w, self.kernel) / sum(w)
                else:
                    return 0

            events['status'] = events['status'].rolling(klen, center=True)\
                .apply(ip)

            return events
