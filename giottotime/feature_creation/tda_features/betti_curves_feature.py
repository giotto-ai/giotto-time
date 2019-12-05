from typing import Iterable, List, Optional, Callable, Union

import pandas as pd
import numpy as np
import giotto.diagrams as diag

from giottotime.feature_creation.tda_features.base import TDAFeatures, align_indices

__all__ = ["BettiCurvesFeature"]


def _find_mean_nonzero(g):
    if g.to_numpy().nonzero()[1].any():
        return g.to_numpy().nonzero()[1].mean()
    else:
        return 0


class BettiCurvesFeature(TDAFeatures):
    """Compute the list of average lifetime for each time window, starting
    from the persistence diagrams.

    Parameters
    ----------
    betti_mode : ``'mean'`` | ``'arg_max'``, required.
        If ``mean``, compute the mean.

    output_name : ``str``, required.
        The name of the output column.

    betti_homology_dimensions : ``Iterable`, optional, (default=``(0, 1)``)
        Dimensions (non-negative integers) of the topological feature_creation
        to be detected.

    betti_n_values : ``int``, optional, (default=``100``)
        The number of filtration parameter values, per available homology
        dimension, to sample during :meth:`fit`.

    betti_rolling : ``int``, optional, (default=``1``)
        Used only if ``betti_mode`` is set to ``mean``. When computing the
        betti surfaces, used to set the rolling parameter.

    betti_n_jobs : ``int``, optional, (default=``None``)
        The number of jobs to use for the computation. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means
        using all processors.

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
        betti_mode: str,
        output_name: str,
        betti_homology_dimensions: Iterable = (0, 1, 2),
        betti_n_values: int = 100,
        betti_rolling: int = 1,
        betti_n_jobs: Optional[int] = None,
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
        self._betti_mode = betti_mode
        self._betti_homology_dimensions = betti_homology_dimensions
        self._betti_n_values = betti_n_values
        self._betti_n_jobs = betti_n_jobs
        self._betti_rolling = betti_rolling

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
        betti_curves = self._compute_betti_curves(persistence_diagrams)

        betti_features = self._compute_betti_features(betti_curves)

        output_dfs = []
        for betti_feature in betti_features:
            original_points = self._compute_n_points(len(betti_feature))

            output_dfs.append(align_indices(X, original_points, betti_feature))

        X_aligned = pd.concat(output_dfs, axis=1)
        X_renamed = self._rename_columns(X_aligned)

        return X_renamed

    def _compute_betti_curves(self, diagrams: np.ndarray) -> List:
        """Given a list of diagrams, compute the betti curves for each of them.

        Parameters
        ----------
        diagrams : ``np.ndarray``, required.
            Compute the betti curves of the diagrams.

        Returns
        -------
        betti_curves : ``List``
            The ``List`` containing the Betti curves.

        """
        betti_curves = diag.BettiCurve()
        betti_curves.fit(diagrams)
        X_betti_curves = betti_curves.transform(diagrams)

        betti_curves = []
        for h_dim in self._betti_homology_dimensions:
            betti_curves.append(pd.DataFrame(X_betti_curves[:, h_dim, :]))

        return betti_curves

    def _compute_betti_features(
        self, betti_curves: List[pd.DataFrame]
    ) -> List[np.ndarray]:
        """Compute the betti feature_creation, depending on the values of
        ``self._betti_mode``. If the value is set to ``mean`` compute the
        rolling mean, if set to ``arg_max`` compute the argmax along the
        epsilon axis.

        Parameters
        ----------
        betti_curves : ``List[pd.DataFrame]``, required.
            A list containing the betti surfaces, one for each homology
            dimension.

        Returns
        -------
        betti_features : ``List[np.ndarray]``
            The feature_creation extracted from the betti curves.

        Raises
        ------
        ``ValueError``
            Raised if a ``self._betti_mode`` has a value which is different
            from ``mean`` or ``arg_max``.

        """
        if self._betti_mode == "mean":
            betti_features = self._compute_betti_mean(betti_curves)

        elif self._betti_mode == "arg_max":
            betti_features = self._compute_arg_max_by_time(betti_curves)

        else:
            raise ValueError(
                f"The valid values for 'betti_mode' are 'mean' "
                f"or 'arg_max', instead has value "
                f"{self._betti_mode}."
            )

        return betti_features

    def _compute_betti_mean(
        self, betti_surfaces: List[pd.DataFrame]
    ) -> List[pd.DataFrame]:
        """Compute the mean along the epsilon axis of the non-zero elements of
        the betti surface.

        Parameters
        ----------
        betti_surfaces : ``List[pd.DataFrame]``, required.
            A list containing the betti surfaces, one for each homology
            dimension.

        Returns
        -------
        betti_means : ``List[np.ndarray]``
            The mean of each betti surfaces.

        """
        betti_means = []
        for betti_surface in betti_surfaces:
            betti_means.append(
                betti_surface.groupby(betti_surface.index)
                .apply(lambda g: _find_mean_nonzero(g))
                .rolling(self._betti_rolling)
                .mean()
                .values
            )

        return betti_means

    def _compute_arg_max_by_time(
        self, betti_surfaces: List[pd.DataFrame]
    ) -> List[np.ndarray]:
        """For each surface in ``betti_surfaces``, compute the argmax along the
         epsilon axis.

        Parameters
        ----------
        betti_surfaces : ``List[pd.DataFrame]``, required.
            A list containing the betti surfaces, one for each homology
            dimension.

        Returns
        -------
        betti_arg_maxes : ``List[np.ndarray]``
            The argmax of each betti surfaces.

        """
        betti_arg_maxes = []
        for betti_surface in betti_surfaces:
            arg_max = np.argmax(np.array(betti_surface), axis=1)
            betti_arg_maxes.append(arg_max)

        return betti_arg_maxes
