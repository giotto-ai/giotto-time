from typing import Iterable

import giotto.diagrams as diag
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st

from giottotime.feature_creation.index_dependent_features.tda_features import (
    TDAFeatures,
)


class BaseTDAFeature(TDAFeatures):
    def __init__(
        self,
        output_name: str,
        takens_parameters_type: str = "search",
        takens_dimension: int = 5,
        takens_stride: int = 1,
        takens_time_delay: int = 1,
        takens_n_jobs: int = 1,
        sliding_window_width: int = 10,
        sliding_stride: int = 1,
        diags_metric: str = "euclidean",
        diags_coeff: int = 2,
        diags_max_edge_length: float = np.inf,
        diags_homology_dimensions: Iterable = (0, 1, 2),
        diags_infinity_values: float = None,
        diags_n_jobs: int = 1,
    ):
        super().__init__(
            output_name,
            takens_parameters_type,
            takens_dimension,
            takens_stride,
            takens_time_delay,
            takens_n_jobs,
            sliding_window_width,
            sliding_stride,
            diags_metric,
            diags_coeff,
            diags_max_edge_length,
            diags_homology_dimensions,
            diags_infinity_values,
            diags_n_jobs,
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    def _correct_n_points_formula(self, n_windows: int) -> int:
        embedder_length = (
            self.sliding_stride * (n_windows - 1) + self.sliding_window_width
        )

        n_used_points = self.takens_stride * (embedder_length - 1) + (
            self.takens_dimension * self.takens_time_delay
        )
        return n_used_points

    def _correct_compute_persistence_diagrams(self, X: pd.DataFrame) -> np.ndarray:
        X_embedded = self._takens_embedding.fit_transform(X,)
        self.X_embedded_dims_ = X_embedded.shape

        X_windows = self.sliding_window.fit_transform(X_embedded,)
        X_diagrams = self.vietoris_rips_persistence.fit_transform(X_windows,)

        diagram_scaler = diag.Scaler()
        diagram_scaler.fit(X_diagrams)

        return diagram_scaler.transform(X_diagrams)


def test_correct_compute_n_points():
    sliding_stride = 3
    sliding_window_width = 2
    takens_stride = 5
    takens_dimension = 2
    takens_time_delay = 3
    n_windows = 9

    tda_feature = BaseTDAFeature(
        output_name="ignored",
        sliding_stride=sliding_stride,
        sliding_window_width=sliding_window_width,
        takens_stride=takens_stride,
        takens_dimension=takens_dimension,
        takens_time_delay=takens_time_delay,
    )
    n_points = tda_feature._compute_n_points(n_windows)
    expected_n_points = 131

    assert expected_n_points == n_points


def test_negative_or_zero_n_windows():
    tda_feature = BaseTDAFeature(output_name="ignored")

    with pytest.raises(ValueError):
        tda_feature._compute_n_points(0)

    with pytest.raises(ValueError):
        tda_feature._compute_n_points(-4)


@given(
    st.integers(1, 20),
    st.integers(1, 20),
    st.integers(1, 20),
    st.integers(1, 20),
    st.integers(1, 20),
    st.integers(1, 20),
)
def test_correct_n_points_random_ts_and_values(
    n_windows,
    sliding_stride,
    sliding_window_width,
    takens_stride,
    takens_dimension,
    takens_time_delay,
):
    tda_feature = BaseTDAFeature(
        output_name="ignored",
        sliding_stride=sliding_stride,
        sliding_window_width=sliding_window_width,
        takens_stride=takens_stride,
        takens_dimension=takens_dimension,
        takens_time_delay=takens_time_delay,
    )
    n_points = tda_feature._compute_n_points(n_windows)
    expected_n_points = tda_feature._correct_n_points_formula(n_windows)
    assert expected_n_points == n_points
    assert expected_n_points > 0


def test_correct_persistence_diagrams():
    np.random.seed(0)
    df = pd.DataFrame(np.random.randint(0, 100, size=(13, 1)), columns=list("A"))

    tda_feature = BaseTDAFeature(output_name="ignored")
    persistence_diagrams = tda_feature._compute_persistence_diagrams(df)
    expected_diagrams = np.array(
        [
            [
                [0.0, 0.435255, 0.0],
                [0.0, 0.44177876, 0.0],
                [0.0, 0.62426053, 0.0],
                [0.0, 0.64725938, 0.0],
                [0.0, 0.76393602, 0.0],
                [0.0, 0.86575012, 0.0],
                [0.0, 1.21810876, 0.0],
                [0.0, 1.28787198, 0.0],
                [0.0, 1.40406276, 0.0],
                [0.0, 1.41421356, 0.0],
                [1.54689366, 1.57882526, 1.0],
                [1.46434634, 1.4783885, 1.0],
                [0.0, 0.0, 2.0],
            ]
        ]
    )

    assert np.allclose(expected_diagrams, persistence_diagrams)
