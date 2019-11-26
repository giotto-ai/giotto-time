from abc import ABCMeta
from collections import Iterable

import numpy as np
from giotto.time_series import TakensEmbedding, SlidingWindow
import giotto.diagrams as diag
import giotto.homology as hl

from giottotime.features.features_creation.base import TimeSeriesFeature


class TDAFeatures(TimeSeriesFeature, metaclass=ABCMeta):
    def __init__(self,
                 output_name: str,
                 takens_parameters_type: str = 'search',
                 takens_dimension: int = 5,
                 takens_stride: int = 1,
                 takens_time_delay: int = 1,
                 takens_n_jobs: int = 1,
                 sliding_window_width: int = 10,
                 sliding_stride: int = 1,
                 diags_metric: str = 'euclidean',
                 diags_max_edge_length: float = np.inf,
                 diags_homology_dimensions: Iterable = (0, 1, 2),
                 diags_infinity_values: float = None,
                 diags_n_jobs: int = 1,
                 ):
        super().__init__(output_name)

        self._takens_embedding = TakensEmbedding(
            parameters_type=takens_parameters_type,
            dimension=takens_dimension,
            stride=takens_stride,
            time_delay=takens_time_delay,
            n_jobs=takens_n_jobs
        )
        self.takens_dimension = takens_dimension
        self.takens_stride = takens_stride
        self.takens_time_delay = takens_time_delay
        self.takens_dimension = takens_dimension

        self._sliding_window = SlidingWindow(
            width=sliding_window_width,
            stride=sliding_stride
        )
        self.sliding_window_width = sliding_window_width
        self.sliding_stride = sliding_stride

        self._vietoris_rips_persistence = hl.VietorisRipsPersistence(
            metric=diags_metric,
            max_edge_length=diags_max_edge_length,
            homology_dimensions=diags_homology_dimensions,
            infinity_values=diags_infinity_values,
            n_jobs=diags_n_jobs
        )

    def fit(self, X, y=None):
        return self

    def _compute_indices(self, windows_points):
        windows_points = self.sliding_stride * (windows_points - 1) + \
                         self.sliding_window_width

        original_points = self.takens_stride * (windows_points - 1) + \
                          (self.takens_dimension * self.takens_time_delay)

        return original_points

    def _compute_persistence_diagrams(self, X):
        X_embedded = self._takens_embedding.fit_transform(X)
        self.X_embedded_dims_ = X_embedded.shape

        X_windows = self._sliding_window.fit_transform(X_embedded)
        X_diagrams = self._vietoris_rips_persistence.fit_transform(X_windows)

        diagram_scaler = diag.Scaler()
        diagram_scaler.fit(X_diagrams)

        return diagram_scaler.transform(X_diagrams)