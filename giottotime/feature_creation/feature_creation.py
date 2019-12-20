from typing import List

import pandas as pd

from .base import Feature
from .index_dependent_features import ShiftFeature


def _check_feature_names(time_series_features: List[Feature]) -> None:
    feature_output_names = [feature.output_name for feature in time_series_features]
    if len(set(feature_output_names)) != len(feature_output_names):
        raise ValueError(
            "The input features should all have different names, instead "
            f"they are {feature_output_names}."
        )


class FeatureCreation:
    """Class responsible for the generation of the feature_creation, starting from a
    list of TimeSeriesFeature.

    Parameters
    ----------
    time_series_features : List[TimeSeriesFeature], required
        The list of ``TimeSeriesFeature`` from which to compute the feature_creation.

    horizon : int, optional, default: ``5``
        It represents how much into the future is necessary to predict. This corresponds
        to the number of shifts that are going to be performed on y.

    Examples
    --------
    >>> import pandas as pd
    >>> from giottotime.feature_creation import FeatureCreation
    >>> from giottotime.feature_creation import ShiftFeature, MovingAverageFeature
    >>> ts = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> shift_feature = ShiftFeature(shift=1)
    >>> mv_avg_feature = MovingAverageFeature(window_size=2)
    >>> feature_creation = FeatureCreation(horizon=3,
    ...                                    time_series_features=[shift_feature,
    ...                                                          mv_avg_feature])
    >>> X, y = feature_creation.fit_transform(ts)
    >>> X
       ShiftFeature  MovingAverageFeature
    0           NaN                   NaN
    1           0.0                   0.5
    2           1.0                   1.5
    3           2.0                   2.5
    4           3.0                   3.5
    5           4.0                   4.5
    6           5.0                   5.5
    7           6.0                   6.5
    8           7.0                   7.5
    9           8.0                   8.5
    >>> y
       y_1  y_2  y_3
    0  1.0  2.0  3.0
    1  2.0  3.0  4.0
    2  3.0  4.0  5.0
    3  4.0  5.0  6.0
    4  5.0  6.0  7.0
    5  6.0  7.0  8.0
    6  7.0  8.0  9.0
    7  8.0  9.0  NaN
    8  9.0  NaN  NaN
    9  NaN  NaN  NaN
    """

    def __init__(self, time_series_features: List[Feature], horizon: int = 5):
        _check_feature_names(time_series_features)

        self.time_series_features = time_series_features
        self.horizon = horizon

    def fit(self, time_series: pd.DataFrame) -> "FeatureCreation":
        """Fit the all the features inside ``time_series_features``.

        Parameters
        ----------
        time_series : pd.DataFrame, required
            The time series on which to fit the data.

        Returns
        -------
        self: FeatureCreation
            The FeatureCreation object itself, containing the fitted features.

        """
        for time_series_feature in self.time_series_features:
            time_series_feature.fit(time_series)
        return self

    def transform(self, time_series: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """Create the X matrix by generating the feature_creation, starting from the
        original ``time_series`` and using the list of ``time_series_features``. Also
        create the y matrix, by generating ``horizon`` number of shifts of the
         ``time_series``.

        Parameters
        ----------
        time_series : pd.DataFrame, shape (n_samples, 1), required
            The time series on which to compute the ``X`` and ``y`` matrices.

        Returns
        -------
        X, y: (pd.DataFrame, pd.DataFrame), shape ((n_samples, n_features), \
            n_samples, horizon))
            A tuple containing the ``X`` and ``y`` matrices.

        """
        X = self._create_x_features(time_series)
        y = self._create_y_shifts(time_series)
        return X, y

    def _create_y_shifts(self, time_series: pd.DataFrame) -> pd.DataFrame:
        y = pd.DataFrame(index=time_series.index)
        for k in range(1, self.horizon + 1):
            shift_feature = ShiftFeature(-k, f"shift_{k}")
            y[f"y_{k}"] = shift_feature.transform(time_series)

        return y

    def _create_x_features(self, time_series: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=time_series.index)
        for time_series_feature in self.time_series_features:
            x_transformed = time_series_feature.transform(time_series)
            features = pd.concat([features, x_transformed], axis=1)

        return features

    def fit_transform(self, time_series: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """Fit and then transform on the same time series.

        Parameters
        ----------
        time_series : pd.DataFrame, required
            The time series on which to fit and to transform.

        Returns
        -------
        X, y: (pd.DataFrame, pd.DataFrame), shape ((n_samples, n_features), \
            n_samples, horizon))
            A tuple containing the ``X`` and ``y`` matrices.

        """
        self.fit(time_series)
        X, y = self.transform(time_series)
        return X, y
