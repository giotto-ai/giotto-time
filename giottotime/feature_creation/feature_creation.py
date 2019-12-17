from typing import List

import pandas as pd

from .base import Feature
from .index_dependent_features import ShiftFeature

__all__ = ["FeatureCreation"]


def _check_feature_names(time_series_features: List[Feature]) -> None:
    feature_output_names = [feature.output_name for feature in time_series_features]
    if len(set(feature_output_names)) != len(feature_output_names):
        raise ValueError(
            "The input features should all have different names, instead "
            f"they are {feature_output_names}."
        )


class FeatureCreation:
    """Class responsible for the generation of the feature_creation, starting
    from a list of ``TimeSeriesFeature``.

    Parameters
    ----------
    horizon : ``int``, required.
        It represents how much into the future is necessary to predict. This
        corresponds to the number of shifts that are going to be performed
        on y.

    time_series_features : ``List[TimeSeriesFeature]``, required.
        The list of ``TimeSeriesFeature`` from which to compute the
        feature_creation.

    """

    def __init__(self, horizon: int, time_series_features: List[Feature]):
        _check_feature_names(time_series_features)

        self._time_series_features = time_series_features
        self._horizon = horizon

    def fit_transform(self, time_series: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """Create the X matrix by generating the feature_creation, starting
        from the original ``time_series`` and using the list of
        ``time_series_features``. Also create the y matrix, by generating
        ``horizon`` number of shifts of the ``time_series``.

        Parameters
        ----------
        time_series : ``pd.DataFrame``, required.
            The time-series on which to compute the ``X`` and ``y`` matrices.

        Returns
        -------
        x, y: ``(pd.DataFrame, pd.DataFrame)``
            A tuple containing the ``X`` and ``y`` matrices.

        """
        x = self._create_x_features(time_series)
        y = self._create_y_shifts(time_series)
        return x, y

    def _create_y_shifts(self, time_series: pd.DataFrame) -> pd.DataFrame:
        y = pd.DataFrame(index=time_series.index)
        for k in range(self._horizon):
            shift_feature = ShiftFeature(-k, f"shift_{k}")
            y[f"y_{k}"] = shift_feature.fit_transform(time_series)

        return y

    def _create_x_features(self, time_series: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=time_series.index)
        for time_series_feature in self._time_series_features:
            x_transformed = time_series_feature.fit_transform(time_series)
            features = pd.concat([features, x_transformed], axis=1)

        return features
