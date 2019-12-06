from typing import Iterable, Dict, Optional, Union, Callable

import giotto.diagrams as diag
import numpy as np
import pandas as pd

from .base import TDAFeatures, align_indices

__all__ = ["AmplitudeFeature"]


class AmplitudeFeature(TDAFeatures):
    """Compute the list of average lifetime for each time window, starting from
    the persistence diagrams.

    Parameters
    ----------
    output_name: ``str``, required,
        The name of the output column.

    metric : ``'bottleneck'`` | ``'wasserstein'`` | ``'landscape'`` | \
        ``'betti'`` | ``'heat'``, optional, (default=``'landscape'``)
        Distance or dissimilarity function used to define the amplitude of
        a subdiagram as its distance from the diagonal diagram:
        - ``'bottleneck'`` and ``'wasserstein'`` refer to the identically named
          perfect-matching--based notions of distance.
        - ``'landscape'`` refers to the :math:`L^p` distance between
          persistence landscapes.
        - ``'betti'`` refers to the :math:`L^p` distance between Betti curves.
        - ``'heat'`` refers to the :math:`L^p` distance between
          Gaussian-smoothed diagrams.

    amplitude_metric_params : ``Dict``, optional, (default=``None``)
        Additional keyword arguments for the metric function:
        - If ``metric == 'bottleneck'`` there are no available arguments.
        - If ``metric == 'wasserstein'`` the only argument is `p` (int,
          default: ``2``).
        - If ``metric == 'betti'`` the available arguments are `p` (float,
          default: ``2.``) and `n_values` (int, default: ``100``).
        - If ``metric == 'landscape'`` the available arguments are `p`
          (float, default: ``2.``), `n_values` (int, default: ``100``) and
          `n_layers` (int, default: ``1``).
        - If ``metric == 'heat'`` the available arguments are `p` (float,
          default: ``2.``), `sigma` (float, default: ``1.``) and `n_values`
          (int, default: ``100``).

    amplitude_order : ``float``, optional, (default=``2.``)
        If ``None``, :meth:`transform` returns for each diagram a vector of
        amplitudes corresponding to the dimensions in
        :attr:`homology_dimensions_`. Otherwise, the :math:`p`-norm of
        these vectors with :math:`p` equal to `order` is taken.

    amplitude_n_jobs : ``int``, optional, (default=``None``)
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    takens_parameters_type: ``'search'`` | ``'fixed'``, optional,
        (default=``'search'``)
        If set to ``'fixed'``, the values of `time_delay` and `dimension`
        are used directly in :meth:`transform`. If set to ``'search'``,
        those values are only used as upper bounds in a search as follows:
        first, an optimal time delay is found by minimising the time delayed
        mutual information; then, a heuristic based on an algorithm in [2]_ is
        used to select an embedding dimension which, when increased, does not
        reveal a large proportion of "false nearest neighbors".

    takens_time_delay : ``int``, optional, (default=``1``)
        Time delay between two consecutive values for constructing one
        embedded point. If `parameters_type` is ``'search'``,
        it corresponds to the maximal embedding time delay that will be
        considered.

    takens_dimension : ``int``, optional, (default=``5``)
        Dimension of the embedding space. If `parameters_type` is ``'search'``,
        it corresponds to the maximum embedding dimension that will be
        considered.

    takens_stride : ``int``, optional, (default=``1``)
        Stride duration between two consecutive embedded points. It defaults
        to 1 as this is the usual value in the statement of Takens's embedding
        theorem.

    takens_n_jobs : ``int``, optional, (default=``None``)
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    sliding_window_width : ``int``, optional, (default=``10``)
        Width of each sliding window. Each window contains ``width + 1``
        objects from the original time series.

    sliding_stride : ``int``, optional, (default=``1``)
        Stride between consecutive windows.

    diags_metric : ``Union[str, Callable]``, optional,
    (default=``'euclidean'``)
        If set to `'precomputed'`, input data is to be interpreted as a
        collection of distance matrices. Otherwise, input data is to be
        interpreted as a collection of point clouds (i.e. feature arrays),
        and `metric` determines a rule with which to calculate distances
        between pairs of instances (i.e. rows) in these arrays.
        If `metric` is a string, it must be one of the options allowed by
        :obj:`scipy.spatial.distance.pdist` for its metric parameter, or a
        metric listed in :obj:`sklearn.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`,
        including "euclidean", "manhattan", or "cosine".
        If `metric` is a callable function, it is called on each pair of
        instances and the resulting value recorded. The callable should take
        two arrays from the entry in `X` as input, and return a value
        indicating the distance between them.

    diags_homology_dimensions : ``Iterable``, optional, (default=``(0, 1)``)
        Dimensions (non-negative integers) of the topological feature_creation
        to be detected.

    diags_coeff : ``int`` prime, optional, (default=``2``)
        Compute homology with coefficients in the prime field
        :math:`\\mathbb{F}_p = \\{ 0, \\ldots, p - 1 \\}` where
        :math:`p` equals `coeff`.

    diags_max_edge_length : ``float``, optional, (default=``np.inf``)
        Upper bound on the maximum value of the Vietoris-Rips filtration
        parameter. Points whose distance is greater than this value will
        never be connected by an edge, and topological feature_creation at
        scales larger than this value will not be detected.

    diags_infinity_values : ``float``, optional, (default=``None``)
        Which death value to assign to feature_creation which are still alive
        at filtration value `max_edge_length`. ``None`` has the same behaviour
        as `max_edge_length`.

    diags_n_jobs : ``int``, optional, (default=``None``)
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    """

    def __init__(
        self,
        output_name: str,
        metric: str = "landscape",
        amplitude_metric_params: Optional[Dict] = None,
        amplitude_order: Dict = 2,
        amplitude_n_jobs: Optional[float] = None,
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
        self._metric = metric
        self._amplitude_metric_params = amplitude_metric_params
        self._amplitude_order = amplitude_order
        self._amplitude_n_jobs = amplitude_n_jobs

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """From the initial DataFrame ``X``, compute the persistence diagrams
        and detect the average lifetime for a given homology dimension.
        Then, assign a value to each initial data points.

        Parameters
        ----------
        X : ``pd.DataFrame``, required.
            The DataFrame on which to compute the feature_creation.

        Returns
        -------
        X_renamed : ``pd.DataFrame``
            A DataFrame containing, for each original data-point, the average
            lifetime associated to it. If, given the initial parameters, a
            point was excluded from the computation, its value is set to
            ``Nan``.

        """
        persistence_diagrams = self._compute_persistence_diagrams(X)
        amplitudes = self._calculate_amplitude_feature(persistence_diagrams)

        original_points = self._compute_n_points(len(amplitudes))

        X_aligned = align_indices(X, original_points, amplitudes)
        X_renamed = self._rename_columns(X_aligned)

        return X_renamed

    def _calculate_amplitude_feature(self, diagrams: np.ndarray) -> np.ndarray:
        """Calculate the amplitude of the persistence diagrams

        Parameters
        ----------
        diagrams : ``np.ndarray``, required.
            Array containing the persistence diagrams.

        Returns
        -------
        amplitudes : ``np.ndarray``
            Array containing, for each diagrams, the corresponding amplitude,
            calculated with with the given ``metric`` and ``amplitude_order``.

        """
        amplitude = diag.Amplitude(
            metric=self._metric,
            order=self._amplitude_order,
            metric_params=self._amplitude_metric_params,
            n_jobs=self._amplitude_n_jobs,
        )
        return amplitude.fit_transform(diagrams)
