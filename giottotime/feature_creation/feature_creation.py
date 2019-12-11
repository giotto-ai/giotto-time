from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

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


class FeatureCreation(BaseEstimator, TransformerMixin):
    """Class responsible for the generation of the feature_creation, starting
    from a list of ``TimeSeriesFeature``.

    Parameters
    ----------
    horizon : ``int``, required.
        It represents how much into the future is necessary to predict. This
        corresponds to the number of shifts that are going to be performed
        on y.

    features : ``List[TimeSeriesFeature]``, required.
        The list of ``TimeSeriesFeature`` from which to compute the
        feature_creation.

    """

    def __init__(self, horizon: int, features: List[Feature]):
        self.features = features
        self._horizon = horizon

    def fit(
        self, time_series: Union[pd.Series, np.array, list], y=None
    ) -> "FeatureCreation":
        """Fit the given features.

        Parameters
        ----------
        time_series : ``np.ndarray``, required.
            Input data.

        y : ``None``
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        _check_feature_names(self.features)

        check_array(time_series)

        for feature in self.features:
            feature.fit(time_series)

        self.is_fitted_ = True

        return self

    def transform(self, time_series: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
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
        check_is_fitted(self, attributes=["is_fitted_"])

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
        for time_series_feature in self.features:
            x_transformed = time_series_feature.fit_transform(time_series)
            features = pd.concat([features, x_transformed], axis=1)

        return features
