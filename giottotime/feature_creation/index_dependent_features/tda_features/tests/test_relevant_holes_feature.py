import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st

from giottotime.feature_creation.index_dependent_features.tda_features import (
    NumberOfRelevantHolesFeature,
)


@given(st.integers(min_value=3))
def test_wrong_h_dim(h_dim):
    with pytest.raises(ValueError):
        NumberOfRelevantHolesFeature(output_name="ignored", h_dim=h_dim)


@given(st.floats(max_value=0))
def test_wrong_theta(theta):
    with pytest.raises(ValueError):
        NumberOfRelevantHolesFeature(output_name="ignored", theta=theta)


def test_correct_n_holes_feature():
    np.random.seed(0)

    h_dim = 0
    theta = 0.4
    output_name = "n_holes"
    n_holes_feature = NumberOfRelevantHolesFeature(
        output_name=output_name, h_dim=h_dim, theta=theta
    )
    df = pd.DataFrame(np.random.randint(0, 100, size=(30, 1)), columns=["old_name"])

    n_holes = n_holes_feature.transform(df)

    expected_n_holes = pd.DataFrame.from_dict(
        {
            output_name: [
                9.0,
                9.0,
                10.0,
                10.0,
                9.0,
                9.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
            ]
        }
    )

    assert np.allclose(expected_n_holes, n_holes)


def test_correct_n_relevant_holes():
    np.random.seed(0)

    h_dim = 1
    theta = 0.8
    output_name = "n_holes"
    n_holes_feature = NumberOfRelevantHolesFeature(
        output_name=output_name, h_dim=h_dim, theta=theta
    )
    df = pd.DataFrame(np.random.randint(0, 100, size=(30, 1)), columns=["old_name"])

    persistence_diagrams = n_holes_feature._compute_persistence_diagrams(df)
    n_holes = n_holes_feature._compute_num_relevant_holes(persistence_diagrams)

    expected_n_holes = [1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 2, 1]

    assert np.allclose(expected_n_holes, n_holes)
