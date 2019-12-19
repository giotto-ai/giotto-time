from hypothesis import given, strategies as st, settings
import pandas as pd
import pytest
import numpy as np
from hypothesis._settings import duration

from giottotime.utils.hypothesis.time_indexes import giotto_time_series
from giottotime.feature_creation import ShiftFeature, MovingAverageFeature
from giottotime.feature_creation.feature_creation import (
    _check_feature_names,
    FeatureCreation,
)


def test_wrong_feature_naming():
    features = [ShiftFeature(k, output_name=f"{k}") for k in range(50)]
    features_same_name = ShiftFeature(shift=100, output_name="1")
    features.append(features_same_name)

    with pytest.raises(ValueError):
        _check_feature_names(features)


def test_correct_feature_names():
    features = [ShiftFeature(k, output_name=f"{k}") for k in range(50)]
    _check_feature_names(features)


class TestFeatureCreation:
    def _correct_x(self, ts, time_series_features):
        features = pd.DataFrame(index=ts.index)
        for time_series_feature in time_series_features:
            x_transformed = time_series_feature.fit_transform(ts)
            features = pd.concat([features, x_transformed], axis=1)
        return features

    def _correct_y(self, ts, horizon):
        y = pd.DataFrame(index=ts.index)
        for k in range(1, horizon + 1):
            shift_feature = ShiftFeature(-k, f"shift_{k}")
            y[f"y_{k}"] = shift_feature.fit_transform(ts)
        return y

    def test_wrong_feature_naming(self):
        time_series_features = [
            ShiftFeature(shift=2, output_name="same_name"),
            ShiftFeature(shift=5, output_name="same_name"),
        ]

        with pytest.raises(ValueError):
            FeatureCreation(horizon=1, time_series_features=time_series_features)

    def test_correct_y_shifts(self):
        horizon = 2
        ts = pd.DataFrame.from_dict({"ignored": range(10)})
        ts.index = np.random.random(len(ts))

        feature_creation = FeatureCreation(horizon=horizon, time_series_features=[])
        y_shifts = feature_creation._create_y_shifts(ts)
        expected_y_shifts = pd.DataFrame.from_dict(
            {
                "y_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan],
                "y_2": [2, 3, 4, 5, 6, 7, 8, 9, np.nan, np.nan],
            }
        )
        expected_y_shifts.index = ts.index

        pd.testing.assert_frame_equal(expected_y_shifts, y_shifts)

    @given(giotto_time_series(), st.integers(1, 10))
    def test_correct_y_shifts_random_ts(self, ts, horizon):
        feature_creation = FeatureCreation(horizon=horizon, time_series_features=[])
        y_shifts = feature_creation._create_y_shifts(ts)
        expected_y_shifts = self._correct_y(ts, horizon)

        pd.testing.assert_frame_equal(expected_y_shifts, y_shifts)

    def test_correct_x_shifts(self):
        horizon = 2
        ts = pd.DataFrame.from_dict({"ignored": range(10)})
        ts.index = np.random.random(len(ts))
        features = [
            MovingAverageFeature(window_size=2, output_name="mov_avg_2"),
            MovingAverageFeature(window_size=5, output_name="mov_avg_5"),
            ShiftFeature(shift=3, output_name="shift_3"),
        ]

        feature_creation = FeatureCreation(
            horizon=horizon, time_series_features=features
        )
        x_shifts = feature_creation._create_x_features(ts)
        expected_x_shifts = pd.DataFrame.from_dict(
            {
                "mov_avg_2": [np.nan, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5,],
                "mov_avg_5": [np.nan, np.nan, np.nan, np.nan, 2, 3, 4, 5, 6, 7,],
                "shift_3": [np.nan, np.nan, np.nan, 0, 1, 2, 3, 4, 5, 6,],
            }
        )
        expected_x_shifts.index = ts.index

        pd.testing.assert_frame_equal(expected_x_shifts, x_shifts)

    @given(giotto_time_series(), st.integers(1, 10))
    def test_correct_y_shifts_random_ts(self, ts, horizon):
        features = [
            MovingAverageFeature(window_size=2, output_name="mov_avg_2"),
            MovingAverageFeature(window_size=5, output_name="mov_avg_5"),
            ShiftFeature(shift=3, output_name="shift_3"),
        ]

        feature_creation = FeatureCreation(
            horizon=horizon, time_series_features=features
        )
        x_shifts = feature_creation._create_x_features(ts)
        expected_x_shifts = self._correct_x(ts, features)

        pd.testing.assert_frame_equal(expected_x_shifts, x_shifts)

    def test_correct_fit_transform(self):
        np.random.seed(0)

        random_index = np.random.random(10)
        horizon = 3
        ts = pd.DataFrame.from_dict({"ignored": range(10)})
        ts.index = random_index

        time_series_features = [
            ShiftFeature(shift=2, output_name="shift_2"),
            ShiftFeature(shift=-5, output_name="shift_5"),
        ]

        fc = FeatureCreation(horizon=horizon, time_series_features=time_series_features)
        x, y = fc.fit_transform(ts)

        expected_x = pd.DataFrame.from_dict(
            {
                "shift_2": [np.nan, np.nan, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,],
                "shift_5": [
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
            }
        )
        expected_x.index = random_index
        pd.testing.assert_frame_equal(expected_x, x)

        expected_y = pd.DataFrame.from_dict(
            {
                "y_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan,],
                "y_2": [2, 3, 4, 5, 6, 7, 8, 9, np.nan, np.nan,],
                "y_3": [3, 4, 5, 6, 7, 8, 9, np.nan, np.nan, np.nan,],
            }
        )
        expected_y.index = random_index
        pd.testing.assert_frame_equal(expected_y, y)

    @given(giotto_time_series(min_length=1), st.integers(1, 10))
    def test_correct_fit_random_ts(self, ts, horizon):
        time_series_features = [
            ShiftFeature(shift=2, output_name="shift_2"),
            ShiftFeature(shift=-5, output_name="shift_5"),
        ]

        fc = FeatureCreation(horizon=horizon, time_series_features=time_series_features)
        x, y = fc.fit_transform(ts)

        expected_x = self._correct_x(ts, time_series_features)
        pd.testing.assert_frame_equal(expected_x, x)

        expected_y = self._correct_y(ts, horizon)
        pd.testing.assert_frame_equal(expected_y, y)
