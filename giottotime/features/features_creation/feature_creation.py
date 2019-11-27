from typing import List

import pandas as pd

from giottotime.features.features_creation.base import TimeSeriesFeature
from giottotime.features.features_creation.time_series_features import ShiftFeature
from giottotime.features.utils import get_non_nan_values


class FeaturesCreation:
    """Class responsible for the generation of the features, starting from a
    list of ``TimeSeriesFeature``.

    Parameters
    ----------
    horizon : ``int``, required.
        It represents how much into the future is necessary to predict. This
        corresponds to the number of shifts that are going to be performed
        on y.

    time_series_features : ``List[TimeSeriesFeature]``, required.
        The list of ``TimeSeriesFeature`` from which to compute the features.

    """
    def __init__(self, horizon: int,
                 time_series_features: List[TimeSeriesFeature]):
        self.time_series_features = time_series_features
        self.horizon = horizon

    def fit_transform(self, time_series: pd.DataFrame) \
            -> (pd.DataFrame, pd.DataFrame):
        """Create the X matrix by generating the features, starting from the
        original ``time_series`` and using the list of ``time_series_features``.
        Also create the y matrix, by generating ``horizon`` number of shifts
        of the ``time_series``.
        Rows of ``X`` that contain at least a ``Nan`` value are
        dropped, as well as the corresponding rows of ``y``.
        Rows of ``y`` that contain a Nan are instead keep, since are going to
        be used in later steps as prediction time.

        Parameters
        ----------
        time_series : ``pd.DataFrame``, required.
            The time-series on which to compute the ``X`` and ``y`` matrices.

        Returns
        -------
        x_non_nans, y_non_nans : ``(pd.DataFrame, pd.DataFrame)``
            A tuple containing the ``X`` and ``y`` matrices.

        """
        x = self._create_x_features(time_series)
        y = self._create_y_shifts(time_series)
        x_non_nans, y_non_nans = get_non_nan_values(x, y)
        return x_non_nans, y_non_nans

    def _create_y_shifts(self, time_series: pd.DataFrame) -> pd.DataFrame:
        """Generate ``n`` shifts of ``time_series``, where ``n`` is equal to
        the ``horizon``.

        Parameters
        ----------
        time_series : ``pd.DataFrame``, required.
            The original DataFrame on which to generate the shifts.

        Returns
        -------
        y_shifted : ``pd.DataFrame``
            The DataFrame containing the shifts of ``time_series``.

        """
        y = pd.DataFrame(index=time_series.index)
        for k in range(self.horizon):
            shift_feature = ShiftFeature(-k, f'shift_{k}')
            y[f'shift_{k}'] = shift_feature.fit_transform(time_series)

        return y

    def _create_x_features(self, time_series: pd.DataFrame) -> pd.DataFrame:
        """Create a DataFrame, containing a set of features generated from the
        ``time_series`` and using the ``time_series_features``.

        Parameters
        ----------
        time_series : ``pd.DataFrame``, required.
            The original DataFrame on which to generate the shifts.

        Returns
        -------
        X : ``pd.DataFrame``
            The DataFrame containing the features.

        """
        X = pd.DataFrame(index=time_series.index)
        for time_series_feature in self.time_series_features:
            x_trasformed = time_series_feature.fit_transform(time_series)
            X = pd.concat([X, x_trasformed], axis=1)

        return X
