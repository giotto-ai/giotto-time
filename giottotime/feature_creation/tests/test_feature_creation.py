import pandas as pd
import pytest
import numpy as np

from giottotime.feature_creation import ShiftFeature
from giottotime.feature_creation.feature_creation import (
    check_feature_names,
    FeatureCreation,
)


def test_wrong_feature_naming():
    features = [ShiftFeature(k, output_name=f"{k}") for k in range(50)]
    features_same_name = ShiftFeature(shift=100, output_name="1")
    features.append(features_same_name)

    with pytest.raises(ValueError):
        check_feature_names(features)


def test_correct_feature_names():
    features = [ShiftFeature(k, output_name=f"{k}") for k in range(50)]
    check_feature_names(features)


class TestFeatureCreation:
    def test_wrong_feature_naming(self):
        time_series_features = [
            ShiftFeature(shift=2, output_name="same_name"),
            ShiftFeature(shift=5, output_name="same_name"),
        ]

        with pytest.raises(ValueError):
            FeatureCreation(horizon=1, time_series_features=time_series_features)

    def test_correct_fit_transform(self):
        horizon = 5
        ts = pd.DataFrame.from_dict({"ignored": range(50)})
        time_series_features = [
            ShiftFeature(shift=2, output_name="shift_2"),
            ShiftFeature(shift=5, output_name="shift_5"),
        ]

        fc = FeatureCreation(horizon=horizon, time_series_features=time_series_features)
        x, y = fc.fit_transform(ts)

        assert horizon == y.shape[1]
        assert len(time_series_features) == x.shape[1]
        assert x.shape[0] == y.shape[0]

        assert (ts.index == x.index).all()
