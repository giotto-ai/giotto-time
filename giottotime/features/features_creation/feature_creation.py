import pandas as pd

from giottotime.feature_creation.time_series_features import ShiftFeature
from giottotime.feature_creation.utils import get_train_test_features


class FeaturesCreation:
    """
    Tentative docstring

    """
    def __init__(self, horizon, time_series_features):
        if horizon <= 0 or not isinstance(horizon, int):
            raise ValueError("The horizon should be an int greater than 0, "
                             f"but has value {horizon}.")

        self.horizon = horizon
        self.time_series_features = time_series_features

    def _create_y_shifts(self, time_series):
        y = pd.DataFrame(index=time_series.index)
        for k in range(self.horizon):
            shift_feature = ShiftFeature(-k, f'shift_{k}')
            y[f'shift_{k}'] = shift_feature.fit_transform(time_series)

        return y

    def _create_x_features(self, time_series):
        x = pd.DataFrame(index=time_series.index)
        for time_series_feature in self.time_series_features:
            x[str(time_series_feature)] = time_series_feature.fit_transform(time_series)

        return x

    def fit_transform(self, time_series):
        x = self._create_x_features(time_series)
        y = self._create_y_shifts(time_series)

        train_x, train_y, test_x = get_train_test_features(x, y)

        return train_x, train_y, test_x
