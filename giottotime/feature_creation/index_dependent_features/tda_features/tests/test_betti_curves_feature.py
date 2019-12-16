import numpy as np
import pandas as pd
import pytest

from giottotime.feature_creation.index_dependent_features.tda_features import (
    BettiCurvesFeature,
)


def test_correct_betti_curves_mean():
    np.random.seed(0)

    output_name = "betti_mean"
    betti_curve_feature = BettiCurvesFeature(output_name=output_name, betti_mode="mean")
    df = pd.DataFrame(np.random.randint(0, 100, size=(14, 1)), columns=["old_name"])

    betti_curves_mean = betti_curve_feature.fit_transform(df)

    expected_betti_curves = pd.DataFrame.from_dict(
        {
            f"{output_name}_0": [
                49.0,
                49.0,
                49.0,
                49.0,
                49.0,
                49.0,
                49.0,
                49.0,
                49.0,
                49.0,
                49.0,
                49.0,
                49.0,
                49.0,
            ],
            f"{output_name}_1": [
                59.325,
                59.325,
                59.325,
                59.325,
                59.325,
                59.325,
                59.325,
                45.640,
                45.640,
                45.640,
                45.640,
                45.640,
                45.640,
                45.640,
            ],
            f"{output_name}_2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
    )

    assert np.allclose(expected_betti_curves, betti_curves_mean)


def test_correct_betti_curves_argmax():
    np.random.seed(0)

    output_name = "betti_mean"
    betti_curve_feature = BettiCurvesFeature(
        output_name=output_name, betti_mode="arg_max"
    )
    df = pd.DataFrame(np.random.randint(0, 100, size=(14, 1)), columns=["old_name"])

    betti_curves_arg_max = betti_curve_feature.fit_transform(df)

    expected_betti_curves = pd.DataFrame.from_dict(
        {
            f"{output_name}_0": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            f"{output_name}_1": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            f"{output_name}_2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
    )

    assert np.allclose(expected_betti_curves, betti_curves_arg_max)


def test_wrong_inputs_for_betti_mode():
    df = pd.DataFrame(np.random.randint(0, 100, size=(14, 1)), columns=["old_name"])

    betti_mode_1 = "wrong_mode"
    betti_curve_feature_1 = BettiCurvesFeature(
        output_name="ignored", betti_mode=betti_mode_1
    )
    betti_mode_2 = ""
    betti_curve_feature_2 = BettiCurvesFeature(
        output_name="ignored", betti_mode=betti_mode_2
    )
    with pytest.raises(ValueError):
        betti_curve_feature_1.fit_transform(df)

    with pytest.raises(ValueError):
        betti_curve_feature_2.fit_transform(df)


def test_correct_mean_mode_betti_features():
    np.random.seed(0)

    df = pd.DataFrame(np.random.randint(0, 100, size=(14, 1)), columns=["old_name"])

    betti_mode_mean = "mean"

    betti_curves_mean = BettiCurvesFeature(
        output_name="ignored", betti_mode=betti_mode_mean
    )

    persistence_diagrams = betti_curves_mean._compute_persistence_diagrams(df)
    betti_curves = betti_curves_mean._compute_betti_curves(persistence_diagrams)

    betti_features = betti_curves_mean._compute_betti_features(betti_curves)
    expected_betti_features = [
        np.array([49.0, 49.0]),
        np.array([59.325, 45.64]),
        np.array([0.0, 0.0]),
    ]

    assert np.allclose(expected_betti_features, betti_features)


def test_correct_argmax_mode_betti_features():
    np.random.seed(0)

    df = pd.DataFrame(np.random.randint(0, 100, size=(14, 1)), columns=["old_name"])

    betti_mode_mean = "arg_max"

    betti_curves_mean = BettiCurvesFeature(
        output_name="ignored", betti_mode=betti_mode_mean
    )

    persistence_diagrams = betti_curves_mean._compute_persistence_diagrams(df)
    betti_curves = betti_curves_mean._compute_betti_curves(persistence_diagrams)

    betti_features = betti_curves_mean._compute_betti_features(betti_curves)
    expected_betti_features = [
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
    ]

    assert np.allclose(expected_betti_features, betti_features)
