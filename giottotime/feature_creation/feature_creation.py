import pandas as pd

from giottotime.feature_creation.time_series_features import ShiftFeature
from giottotime.feature_creation.utils import get_non_nan_values


class FeaturesCreation:
    def __init__(self, horizon, time_series_features):
        self.time_series_features = time_series_features
        self.horizon = horizon

    def fit_transform(self, time_series):
        y = self._create_y_shifts(time_series)
        x = self._create_x_features(time_series)
        x_non_nans, y_non_nans = get_non_nan_values(x, y)

        return x_non_nans, y_non_nans

    def _create_y_shifts(self, time_series):
        y = pd.DataFrame(index=time_series.index)
        for k in range(self.horizon):
            shift_feature = ShiftFeature(-k)
            y[str(shift_feature)] = shift_feature.fit_transform(time_series)

        return y

    def _create_x_features(self, time_series):
        x = pd.DataFrame(index=time_series.index)
        for time_series_feature in self.time_series_features:
            x[str(time_series_feature)] = time_series_feature.fit_transform(time_series)

        return x
