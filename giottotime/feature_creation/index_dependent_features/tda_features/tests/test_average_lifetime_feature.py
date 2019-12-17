import numpy as np
import pandas as pd

from giottotime.feature_creation.index_dependent_features.tda_features import (
    AvgLifeTimeFeature,
)


def test_correct_average_lifetime_computation():
    np.random.seed(0)

    output_name = "ignored"
    avg_lifetime_feature = AvgLifeTimeFeature(output_name=output_name)
    df = pd.DataFrame(np.random.randint(0, 100, size=(30, 1)), columns=["ignored"])

    persistence_diagrams = avg_lifetime_feature._compute_persistence_diagrams(df)

    average_lifetime = avg_lifetime_feature._compute_average_lifetime(
        persistence_diagrams
    )

    expected_avg_lifetime = np.array(
        [
            0.8719302650092168,
            0.9375827753658988,
            0.9898459500730074,
            0.9927494731020051,
            0.9657270652747636,
            0.9135946743014184,
            0.838502504813531,
            0.8052276794850292,
            0.7657986690498374,
            0.8041319195886067,
            0.8091595180105875,
            0.8212692086741538,
            0.8097153100855028,
            0.8103762891856373,
            0.7947070443375637,
            0.8403939703821199,
            0.8371402400850239,
        ]
    )

    assert np.allclose(expected_avg_lifetime, average_lifetime)


def test_correct_avg_lifetime_feature():
    np.random.seed(0)

    output_name = "avg_lifetime_feature"
    avg_lifetime_feature = AvgLifeTimeFeature(output_name=output_name)
    df = pd.DataFrame(np.random.randint(0, 100, size=(30, 1)), columns=["old_name"])

    avg_lifetime = avg_lifetime_feature.fit_transform(df)

    expected_avg_lifetime = pd.DataFrame.from_dict(
        {
            output_name: [
                0.871930,
                0.871930,
                0.937583,
                0.937583,
                0.989846,
                0.989846,
                0.992749,
                0.992749,
                0.965727,
                0.965727,
                0.913595,
                0.913595,
                0.838503,
                0.838503,
                0.805228,
                0.805228,
                0.765799,
                0.765799,
                0.804132,
                0.804132,
                0.809160,
                0.809160,
                0.821269,
                0.821269,
                0.809715,
                0.809715,
                0.810376,
                0.794707,
                0.840394,
                0.837140,
            ]
        }
    )
    expected_avg_lifetime.index = df.index

    assert np.allclose(expected_avg_lifetime.values, avg_lifetime.values)
    assert np.array_equal(expected_avg_lifetime.index, avg_lifetime.index)
    assert expected_avg_lifetime.columns == avg_lifetime.columns
