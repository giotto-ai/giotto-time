from abc import ABCMeta, abstractmethod
from collections import Iterable
from typing import Union, List

import numpy as np
from giotto.time_series import TakensEmbedding, SlidingWindow
import giotto.diagrams as diag
import giotto.homology as hl
import pandas as pd

from giottotime.features.base import TimeSeriesFeature


def _align_indices(X: pd.DataFrame, n_points: int,
                   tda_feature_values: Union[List, np.ndarray]) -> pd.DataFrame:
    """Given ``X`` of length ``n_samples``, set the first
    ``n_samples - n_points`` to ``np.nan``. Then, split the remaining points in
    ``len(tda_feature_values)`` chunks and, to each data-point in a chunk, set
    its value to the corresponding value in ``tda_feature_values``.
    Parameters
    ----------
    X : ``pd.DataFrame``, required.
        The input DataFrame. Only the indices of the DataFrame are used
    n_points : ``int``, required.
        The number of points on which to apply the values
    tda_feature_values : ``Union[List, np.ndarray]``, required.
        The List or np.ndarray containing the values to put in ``output_X``.
    Returns
    -------
    output_X : ``pd.DataFrame``
        A ``pd.DataFrame`` with the same index as ``X`` and with the values
        set according to ``n_points`` and ``tda_feature_values``.
    """
    output_X = X.copy()

    output_X.iloc[:-n_points] = np.nan

    splits = np.array_split(output_X.iloc[-n_points:].index,
                            len(tda_feature_values))

    for index, split in enumerate(splits):
        output_X.loc[split] = tda_feature_values[index]

    return output_X


class TDAFeatures(TimeSeriesFeature, metaclass=ABCMeta):
    """Base class for all the TDA features_creation contained in the package.
    Parameter documentation is in the derived classes.
    """
    @abstractmethod
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
            coeff=diags_coeff,
            max_edge_length=diags_max_edge_length,
            homology_dimensions=diags_homology_dimensions,
            infinity_values=diags_infinity_values,
            n_jobs=diags_n_jobs
        )

    def fit(self, X, y=None):
        return self

    def _compute_n_points(self, n_windows: int) -> int:
        """Given the initial parameters used in the TakensEmbedding and
        SlidingWindow steps, compute the total number of points that have been
        used during the computation.
        Parameters
        ----------
        n_windows : ``int``, required.
            The number of windows after the SlidingWindow step.
        Returns
        -------
        n_used_points : ``int``
            The total number of points that have been used in the
            TakensEmbedding and SlidingWindow steps.
        """
        embedder_length = self.sliding_stride * (n_windows-1) + \
                          self.sliding_window_width

        n_used_points = self.takens_stride * (embedder_length-1) + \
                          self.takens_dimension*self.takens_time_delay

        return n_used_points

    def _compute_persistence_diagrams(self, X: Union[pd.DataFrame, pd.Series])\
            -> np.ndarray:
        """Compute the persistence diagrams starting from a time-series using
        the Vietoris Rips algorithm. The resulting diagrams are then scaled.
        Parameters
        ----------
        X : ``Union[pd.DataFrame, pd.Series]``, required.
            The time-series on which to compute the persistence diagrams.
        Returns
        -------
        X_scaled : ``np.ndarray``
            The scaled persistence diagrams.
        """
        X_embedded = self._takens_embedding.fit_transform(X)
        self.X_embedded_dims_ = X_embedded.shape

        X_windows = self._sliding_window.fit_transform(X_embedded)
        X_diagrams = self._vietoris_rips_persistence.fit_transform(X_windows)

        diagram_scaler = diag.Scaler()
        diagram_scaler.fit(X_diagrams)

        return diagram_scaler.transform(X_diagrams)