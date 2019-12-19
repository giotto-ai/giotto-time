from typing import Iterable, List, Optional, Union, Callable

import numpy as np
import pandas as pd

from .base import TDAFeatures, _align_indices


class NumberOfRelevantHolesFeature(TDAFeatures):
    """Compute the list of average lifetime for each time window, starting from the
    persistence diagrams.

    Parameters
    ----------
    h_dim: int, optional, default: ``0``
        The homology dimension on which to compute the average lifetime.

    theta: float, optional, default: ``0.7``
        Constant used to set the threshold in the computation of the holes

    output_name : str, optional, default: ``'NumberOfRelevantHolesFeature'``
        The name of the output column.

    takens_parameters_type: ``'search'`` | ``'fixed'``, optional, default: ``'search'``
        If set to ``'fixed'``, the values of `time_delay` and `dimension are used
        directly in :meth:`transform`. If set to ``'search'``, those values are only
        used as upper bounds in a search as follows: first, an optimal time delay is
        found by minimising the time delayed mutual information; then, a heuristic based
        on an algorithm in [2]_ is used to select an embedding dimension which, when
        increased, does not reveal a large proportion of "false nearest neighbors".

    takens_time_delay : int, optional, default: ``1``
        Time delay between two consecutive values for constructing one embedded point.
        If `parameters_type` is ``'search'``, it corresponds to the maximal embedding
        time delay that will be considered.

    takens_dimension : int, optional, default: ``5``
        Dimension of the embedding space. If `parameters_type` is ``'search'``, it
        corresponds to the maximum embedding dimension that will be considered.

    takens_stride : int, optional, default: ``1``
        Stride duration between two consecutive embedded points. It defaults to 1 as
        this is the usual value in the statement of Takens's embedding theorem.

    takens_n_jobs : int: , optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.

    sliding_window_width : int, optional, default: ``10``
        Width of each sliding window. Each window contains ``width + 1`` objects from
        the original time series.

    sliding_stride : int, optional, default: ``1``
        Stride between consecutive windows.

    diags_metric : Union[str, Callable], optional, default: ``'euclidean'``
        If set to `'precomputed'`, input data is to be interpreted as a collection of
        distance matrices. Otherwise, input data is to be interpreted as a collection
        of point clouds (i.e. feature arrays), and `metric` determines a rule with which
        to calculate distances between pairs of instances (i.e. rows) in these arrays.
        If `metric` is a string, it must be one of the options allowed by
        :obj:`scipy.spatial.distance.pdist` for its metric parameter, or a metric listed
        in :obj:`sklearn.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`, including "euclidean",
        "manhattan", or "cosine". If `metric` is a callable function, it is called on
        each pair of instances and the resulting value recorded. The callable should
        take two arrays from the entry in `X` as input, and return a value indicating
        the distance between them.

    diags_homology_dimensions : Iterable, optional, default: ``(0, 1)``
        Dimensions (non-negative integers) of the topological feature_creation to be
        detected.

    diags_coeff : int prime, optional, default: ``2``
        Compute homology with coefficients in the prime field
        :math:`\\mathbb{F}_p = \\{ 0, \\ldots, p - 1 \\}` where
        :math:`p` equals `coeff`.

    diags_max_edge_length : float, optional, default: ``np.inf``
        Upper bound on the maximum value of the Vietoris-Rips filtration parameter.
        Points whose distance is greater than this value will never be connected by an
        edge, and topological feature_creation at scales larger than this value will not
        be detected.

    diags_infinity_values : float, optional, default: ``None``
        Which death value to assign to feature_creation which are still alive at
        filtration value `max_edge_length`. ``None`` has the same behaviour as
        `max_edge_length`.

    diags_n_jobs : int, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.

    Examples
    --------
    >>> import pandas as pd
    >>> from giottotime.feature_creation import NumberOfRelevantHolesFeature
    >>> X = pd.DataFrame(range(0, 15))
    >>> relevant_holes = NumberOfRelevantHolesFeature()
    >>> relevant_holes.transform(X)
        NumberOfRelevantHolesFeature
    0                           10.0
    1                           10.0
    2                           10.0
    3                           10.0
    4                           10.0
    5                           10.0
    6                           10.0
    7                           10.0
    8                           10.0
    9                           10.0
    10                          10.0
    11                          10.0
    12                          10.0
    13                          10.0
    14                          10.0

    """

    def __init__(
        self,
        h_dim: int = 0,
        theta: float = 0.7,
        output_name: str = "NumberOfRelevantHolesFeature",
        takens_parameters_type: str = "search",
        takens_dimension: int = 5,
        takens_stride: int = 1,
        takens_time_delay: int = 1,
        takens_n_jobs: Optional[int] = 1,
        sliding_window_width: int = 10,
        sliding_stride: int = 1,
        diags_metric: Union[str, Callable] = "euclidean",
        diags_coeff: int = 2,
        diags_max_edge_length: float = np.inf,
        diags_homology_dimensions: Iterable = (0, 1, 2),
        diags_infinity_values: Optional[float] = None,
        diags_n_jobs: Optional[int] = 1,
    ):
        super().__init__(
            output_name=output_name,
            takens_parameters_type=takens_parameters_type,
            takens_dimension=takens_dimension,
            takens_stride=takens_stride,
            takens_time_delay=takens_time_delay,
            takens_n_jobs=takens_n_jobs,
            sliding_window_width=sliding_window_width,
            sliding_stride=sliding_stride,
            diags_metric=diags_metric,
            diags_coeff=diags_coeff,
            diags_max_edge_length=diags_max_edge_length,
            diags_homology_dimensions=diags_homology_dimensions,
            diags_infinity_values=diags_infinity_values,
            diags_n_jobs=diags_n_jobs,
        )
        self._validate_inputs(h_dim=h_dim, theta=theta)

        self.h_dim = h_dim
        self.theta = theta

    def transform(self, time_series: pd.DataFrame) -> pd.DataFrame:
        """From the initial DataFrame ``time_series``, compute the persistence diagrams
        and detect the relevant number of holes. Then, assign a value to each initial
        data points.

        Parameters
        ----------
        time_series : pd.DataFrame, shape (n_samples, 1), required
            The DataFrame on which to compute the feature_creation.

        Returns
        -------
        time_series_t : pd.DataFrame, shape (n_samples, 1)
            A DataFrame containing, for each original data-point, the relevant number of
            holes associated to it. If, given the initial parameters, a point was
            excluded from the computation, its value is set to ``Nan``.

        """
        persistence_diagrams = self._compute_persistence_diagrams(time_series)
        n_holes = self._compute_num_relevant_holes(persistence_diagrams)
        n_points = self._compute_n_points(len(n_holes))

        time_series_aligned = _align_indices(time_series, n_points, n_holes)
        time_series_t = self._rename_columns(time_series_aligned)

        return time_series_t

    def _compute_num_relevant_holes(self, persistence_diagrams: np.ndarray) -> List:
        n_rel_holes = []
        for i in range(persistence_diagrams.shape[0]):
            pers_table = pd.DataFrame(
                persistence_diagrams[i], columns=["birth", "death", "homology"]
            )

            pers_table["lifetime"] = pers_table["death"] - pers_table["birth"]
            threshold = (
                pers_table[pers_table["homology"] == self.h_dim]["lifetime"].max()
                * self.theta
            )
            n_rel_holes.append(
                pers_table[
                    (pers_table["lifetime"] > threshold)
                    & (pers_table["homology"] == self.h_dim)
                ].shape[0]
            )

        return n_rel_holes

    def _validate_inputs(self, h_dim: int, theta: float) -> None:
        if h_dim != 0 and h_dim != 1 and h_dim != 2:
            raise ValueError(
                f"'h_dim' must have be either 0, 1 or 2, " f"but has value {h_dim}."
            )

        if not theta > 0:
            raise ValueError(
                f"'theta' must be greater than 0, but instead " f"has value {theta}."
            )
