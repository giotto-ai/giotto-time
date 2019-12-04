from typing import Iterable

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st

from giottotime.feature_creation.tda_features import TDAFeatures


class TestTDAFeature(TDAFeatures):
    def __init__(self, output_name: str,
                 takens_parameters_type: str = 'search',
                 takens_dimension: int = 5, takens_stride: int = 1,
                 takens_time_delay: int = 1, takens_n_jobs: int = 1,
                 sliding_window_width: int = 10, sliding_stride: int = 1,
                 diags_metric: str = 'euclidean', diags_coeff: int = 2,
                 diags_max_edge_length: float = np.inf,
                 diags_homology_dimensions: Iterable = (0, 1, 2),
                 diags_infinity_values: float = None, diags_n_jobs: int = 1):
        super().__init__(output_name, takens_parameters_type, takens_dimension,
                         takens_stride, takens_time_delay, takens_n_jobs,
                         sliding_window_width, sliding_stride, diags_metric,
                         diags_coeff, diags_max_edge_length,
                         diags_homology_dimensions, diags_infinity_values,
                         diags_n_jobs)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass


def test_correct_compute_n_points():
    sliding_stride = 3
    sliding_window_width = 2
    takens_stride = 5
    takens_dimension = 2
    takens_time_delay = 3
    n_windows = 9

    tda_feature = TestTDAFeature(output_name='ignored',
                                 sliding_stride=sliding_stride,
                                 sliding_window_width=sliding_window_width,
                                 takens_stride=takens_stride,
                                 takens_dimension=takens_dimension,
                                 takens_time_delay=takens_time_delay
                                 )
    n_points = tda_feature._compute_n_points(n_windows)
    expected_n_points = 131

    assert expected_n_points == n_points


def test_negative_or_zero_n_windows():
    tda_feature = TestTDAFeature(output_name='ignored')

    with pytest.raises(ValueError):
        tda_feature._compute_n_points(0)

    with pytest.raises(ValueError):
        tda_feature._compute_n_points(-4)


def _correct_n_points_formula(n_windows, sliding_stride, sliding_window_width,
                              takens_stride, takens_dimension,
                              takens_time_delay):
    embedder_length = sliding_stride * (n_windows - 1) + \
                      sliding_window_width

    n_used_points = takens_stride * (embedder_length - 1) + \
                    takens_dimension * takens_time_delay
    return n_used_points


@given(st.integers(1, 20), st.integers(1, 20), st.integers(1, 20),
       st.integers(1, 20), st.integers(1, 20), st.integers(1, 20))
def test_correct_n_points_random_ts_and_values(n_windows, sliding_stride,
                                               sliding_window_width,
                                               takens_stride, takens_dimension,
                                               takens_time_delay):
    tda_feature = TestTDAFeature(output_name='ignored',
                                 sliding_stride=sliding_stride,
                                 sliding_window_width=sliding_window_width,
                                 takens_stride=takens_stride,
                                 takens_dimension=takens_dimension,
                                 takens_time_delay=takens_time_delay
                                 )
    n_points = tda_feature._compute_n_points(n_windows)
    expected_n_points = _correct_n_points_formula(n_windows, sliding_stride,
                                                  sliding_window_width,
                                                  takens_stride,
                                                  takens_dimension,
                                                  takens_time_delay)
    assert expected_n_points == n_points
    assert expected_n_points > 0
