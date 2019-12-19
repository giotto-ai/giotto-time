from abc import ABCMeta, abstractmethod
from collections import Iterable
from typing import Union, List, Optional

import numpy as np
from giotto.time_series import TakensEmbedding, SlidingWindow
import giotto.diagrams as diag
import giotto.homology as hl
import pandas as pd

from ..base import IndexDependentFeature

__all__ = ["TDAFeatures"]


def _align_indices(
    X: pd.DataFrame, n_points: int, tda_feature_values: Union[List, np.ndarray]
) -> pd.DataFrame:
    output_X = X.copy()

    output_X.iloc[:-n_points] = np.nan

    splits = np.array_split(
        output_X.iloc[-n_points:].index.values, len(tda_feature_values)
    )

    for index, split in enumerate(splits):
        if isinstance(tda_feature_values[index], list) or isinstance(
            tda_feature_values[index], np.ndarray
        ):
            target_value = tda_feature_values[index][0]
        else:
            target_value = tda_feature_values[index]
        output_X.loc[split] = target_value

    return output_X


class TDAFeatures(IndexDependentFeature, metaclass=ABCMeta):
    """Base class for all the TDA feature_creation contained in the package.

    """

    @abstractmethod
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
        super().__init__(output_name)

        self._takens_embedding = TakensEmbedding(
            parameters_type=takens_parameters_type,
            dimension=takens_dimension,
            stride=takens_stride,
            time_delay=takens_time_delay,
            n_jobs=takens_n_jobs,
        )
        self.takens_dimension = takens_dimension
        self.takens_stride = takens_stride
        self.takens_time_delay = takens_time_delay
        self.takens_dimension = takens_dimension

        self.sliding_window = SlidingWindow(
            width=sliding_window_width, stride=sliding_stride
        )
        self.sliding_window_width = sliding_window_width
        self.sliding_stride = sliding_stride

        self.vietoris_rips_persistence = hl.VietorisRipsPersistence(
            metric=diags_metric,
            coeff=diags_coeff,
            max_edge_length=diags_max_edge_length,
            homology_dimensions=diags_homology_dimensions,
            infinity_values=diags_infinity_values,
            n_jobs=diags_n_jobs,
        )

    def fit(self, time_series: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        return self

    def _compute_n_points(self, n_windows: int) -> int:
        if n_windows <= 0:
            raise ValueError(
                f"The number of windows should be greater than "
                f"0, instead was {n_windows}."
            )
        embedder_length = (
            self.sliding_stride * (n_windows - 1) + self.sliding_window_width
        )

        n_used_points = (
            self.takens_stride * (embedder_length - 1)
            + self.takens_dimension * self.takens_time_delay
        )

        return n_used_points

    def _compute_persistence_diagrams(self, X: pd.DataFrame) -> np.ndarray:
        X_embedded = self._takens_embedding.fit_transform(X)
        self.X_embedded_dims_ = X_embedded.shape

        X_windows = self.sliding_window.fit_transform(X_embedded)
        X_diagrams = self.vietoris_rips_persistence.fit_transform(X_windows)

        diagram_scaler = diag.Scaler()
        diagram_scaler.fit(X_diagrams)

        return diagram_scaler.transform(X_diagrams)
