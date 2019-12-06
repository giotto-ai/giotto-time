import numpy as np
import pandas as pd

from giottotime.feature_creation.index_dependent_features.tda_features import (
    AmplitudeFeature,
)


def test_correct_amplitude_computation():
    np.random.seed(0)

    output_name = "ignored"
    amplitude_feature = AmplitudeFeature(output_name=output_name)
    df = pd.DataFrame(np.random.randint(0, 100, size=(30, 1)), columns=["ignored"])

    persistence_diagrams = amplitude_feature._compute_persistence_diagrams(df)
    amplitudes = amplitude_feature._calculate_amplitude_feature(persistence_diagrams)

    expected_amplitudes = np.array(
        [
            [0.42390011],
            [0.42392732],
            [0.48618924],
            [0.46564008],
            [0.46508716],
            [0.33761475],
            [0.33761475],
            [0.33680451],
            [0.29547906],
            [0.30680885],
            [0.29692516],
            [0.29494336],
            [0.35830464],
            [0.35827269],
            [0.35806485],
            [0.37202706],
            [0.3731072],
        ]
    )

    assert np.allclose(expected_amplitudes, amplitudes)


def test_correct_amplitude_feature():
    np.random.seed(0)

    output_name = "amplitude_feature"
    amplitude_feature = AmplitudeFeature(output_name=output_name)
    df = pd.DataFrame(np.random.randint(0, 100, size=(30, 1)), columns=["old_name"])

    amplitudes = amplitude_feature.fit_transform(df)
    expected_amplitudes = pd.DataFrame.from_dict(
        {
            output_name: [
                0.423900,
                0.423900,
                0.423927,
                0.423927,
                0.486189,
                0.486189,
                0.465640,
                0.465640,
                0.465087,
                0.465087,
                0.337615,
                0.337615,
                0.337615,
                0.337615,
                0.336805,
                0.336805,
                0.295479,
                0.295479,
                0.306809,
                0.306809,
                0.296925,
                0.296925,
                0.294943,
                0.294943,
                0.358305,
                0.358305,
                0.358273,
                0.358065,
                0.372027,
                0.373107,
            ]
        }
    )
    expected_amplitudes.index = df.index

    assert np.allclose(expected_amplitudes.values, amplitudes.values)
    assert np.array_equal(expected_amplitudes.index, amplitudes.index)
    assert expected_amplitudes.columns == amplitudes.columns
