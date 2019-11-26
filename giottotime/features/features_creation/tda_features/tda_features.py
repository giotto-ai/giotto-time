from typing import Iterable, Union, List

import numpy as np
import pandas as pd

from giottotime.features.features_creation.tda_features.base import TDAFeatures
from giottotime.features.features_creation.feature_creation import \
    FeaturesCreation


class NumberOfRelevantHolesFeature(TDAFeatures):
    """Compute the list of average lifetime for each time window, starting
    from the persistence diagrams.

    Parameters
    ----------
    output_name: str
        The name of the output column

    h_dim: int, default: ``0``
        The homology dimension on which to compute the average lifetime.

    theta: float, default: ``0.7``
        Constant used to set the threshold in the computation of the holes

    interpolation_strategy: str
        The interpolation strategy to use to fill the values

    takens_parameters_type: ``'search'`` | ``'fixed'``, optional,
        default: ``'search'``
        If set to ``'fixed'``, the values of `time_delay` and `dimension`
        are used directly in :meth:`transform`. If set to ``'search'``,
        those values are only used as upper bounds in a search as follows:
        first, an optimal time delay is found by minimising the time delayed
        mutual information; then, a heuristic based on an algorithm in [2]_ is
        used to select an embedding dimension which, when increased, does not
        reveal a large proportion of "false nearest neighbors".

    takens_time_delay : int, optional, default: ``1``
        Time delay between two consecutive values for constructing one
        embedded point. If `parameters_type` is ``'search'``,
        it corresponds to the maximal embedding time delay that will be
        considered.

    takens_dimension : int, optional, default: ``5``
        Dimension of the embedding space. If `parameters_type` is ``'search'``,
        it corresponds to the maximum embedding dimension that will be
        considered.

    takens_stride : int, optional, default: ``1``
        Stride duration between two consecutive embedded points. It defaults
        to 1 as this is the usual value in the statement of Takens's embedding
        theorem.

    takens_n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    sliding_window_width : int, optional, default: ``10``
        Width of each sliding window. Each window contains ``width + 1``
        objects from the original time series.

    sliding_stride : int, optional, default: ``1``
        Stride between consecutive windows.

    diags_metric : string or callable, optional, default: ``'euclidean'``
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

    diags_homology_dimensions : iterable, optional, default: ``(0, 1)``
        Dimensions (non-negative integers) of the topological features to be
        detected.

    diags_coeff : int prime, optional, default: ``2``
        Compute homology with coefficients in the prime field
        :math:`\\mathbb{F}_p = \\{ 0, \\ldots, p - 1 \\}` where
        :math:`p` equals `coeff`.

    diags_max_edge_length : float, optional, default: ``numpy.inf``
        Upper bound on the maximum value of the Vietoris-Rips filtration
        parameter. Points whose distance is greater than this value will
        never be connected by an edge, and topological features at scales
        larger than this value will not be detected.

    diags_infinity_values : float or None, default : ``None``
        Which death value to assign to features which are still alive at
        filtration value `max_edge_length`. ``None`` has the same behaviour
        as `max_edge_length`.

    diags_n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    """
    def __init__(self,
                 output_name: str,
                 h_dim: int = 0,
                 theta: float = 0.7,
                 interpolation_strategy: str = 'ffill',
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
                 diags_n_jobs: int = 1
                 ):
        super().__init__(output_name,
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
                         diags_n_jobs
                         )

        self._h_dim = h_dim
        self._theta = theta
        self.interpolation_strategy = interpolation_strategy

    def _compute_num_relevant_holes(self, X_scaled: np.ndarray) -> List:
        """Compute the number of relevant holes in the point cloud.

        Parameters
        ----------
        X_scaled: numpy.ndarray
            The array containing the scaled persistent diagrams.

        Returns
        -------
        n_rel_holes: List
            For each diagram present in ``X_scaled``, return the number of
            relevant holes that have been found.

        """
        n_rel_holes = []
        for i in range(X_scaled.shape[0]):
            pers_table = pd.DataFrame(X_scaled[i], columns=['birth',
                                                            'death',
                                                            'homology'])

            pers_table['lifetime'] = pers_table['death'] - pers_table['birth']
            threshold = pers_table[pers_table['homology'] == self._h_dim][
                            'lifetime'].max() * self._theta
            n_rel_holes.append(pers_table[
                                   (pers_table['lifetime'] > threshold) & (
                                           pers_table[
                                               'homology'] == self._h_dim)].shape[0])

        return n_rel_holes

    # TODO: Finish this method
    def transform(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        X_scaled = self._compute_persistence_diagrams(X)
        print(type(X_scaled))
        n_holes = self._compute_num_relevant_holes(X_scaled)
        original_points= self._compute_indices(len(n_holes))
        # TODO: check this is correct with Matteo
        return


class AvgLifeTimeFeature(TDAFeatures):
    """Compute the list of average lifetime for each time window, starting
    from the persistence diagrams.

    Parameters
    ----------
    output_name: str
        The name of the output column

    h_dim: int, default: ``0``
        The homology dimension on which to compute the average lifetime.

    takens_parameters_type: ``'search'`` | ``'fixed'``, optional,
        default: ``'search'``
        If set to ``'fixed'``, the values of `time_delay` and `dimension`
        are used directly in :meth:`transform`. If set to ``'search'``,
        those values are only used as upper bounds in a search as follows:
        first, an optimal time delay is found by minimising the time delayed
        mutual information; then, a heuristic based on an algorithm in [2]_ is
        used to select an embedding dimension which, when increased, does not
        reveal a large proportion of "false nearest neighbors".

    takens_time_delay : int, optional, default: ``1``
        Time delay between two consecutive values for constructing one
        embedded point. If `parameters_type` is ``'search'``,
        it corresponds to the maximal embedding time delay that will be
        considered.

    takens_dimension : int, optional, default: ``5``
        Dimension of the embedding space. If `parameters_type` is ``'search'``,
        it corresponds to the maximum embedding dimension that will be
        considered.

    takens_stride : int, optional, default: ``1``
        Stride duration between two consecutive embedded points. It defaults
        to 1 as this is the usual value in the statement of Takens's embedding
        theorem.

    takens_n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    sliding_window_width : int, optional, default: ``10``
        Width of each sliding window. Each window contains ``width + 1``
        objects from the original time series.

    sliding_stride : int, optional, default: ``1``
        Stride between consecutive windows.

    diags_metric : string or callable, optional, default: ``'euclidean'``
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

    diags_homology_dimensions : iterable, optional, default: ``(0, 1)``
        Dimensions (non-negative integers) of the topological features to be
        detected.

    diags_coeff : int prime, optional, default: ``2``
        Compute homology with coefficients in the prime field
        :math:`\\mathbb{F}_p = \\{ 0, \\ldots, p - 1 \\}` where
        :math:`p` equals `coeff`.

    diags_max_edge_length : float, optional, default: ``numpy.inf``
        Upper bound on the maximum value of the Vietoris-Rips filtration
        parameter. Points whose distance is greater than this value will
        never be connected by an edge, and topological features at scales
        larger than this value will not be detected.

    diags_infinity_values : float or None, default : ``None``
        Which death value to assign to features which are still alive at
        filtration value `max_edge_length`. ``None`` has the same behaviour
        as `max_edge_length`.

    diags_n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    """
    def __init__(self,
                 output_name: str,
                 h_dim: int = 0,
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
                 diags_n_jobs: int = 1
                 ):
        super().__init__(output_name,
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
                         diags_n_jobs
                         )
        self._h_dim = h_dim

    def _average_lifetime(self, X_scaled: np.ndarray) -> List:
        """Compute the average lifetime of a given homology dimension in the
        point cloud.

        Parameters
        ----------
        X_scaled: numpy.ndarray
            The array containing the scaled persistent diagrams.

        Returns
        -------
        avg_lifetime: List
            For each diagram present in ``X_scaled``, return the average
            lifetime of a given homology dimension.

        """
        avg_lifetime = []

        for i in range(X_scaled.shape[0]):
            persistence_table = pd.DataFrame(X_scaled[i],
                                             columns=['birth', 'death',
                                                      'homology'])
            persistence_table['lifetime'] = persistence_table['death'] - \
                                            persistence_table['birth']
            avg_lifetime.append(
                persistence_table[persistence_table['homology']
                                  == self._h_dim]['lifetime'].mean())

        return avg_lifetime

    def transform(self, X) -> pd.Series:
        X_scaled = self._compute_persistence_diagrams(X)
        avg_lifetime = self._average_lifetime(X_scaled)
        original_points = self._compute_indices(len(avg_lifetime))
        return


if __name__ == "__main__":
    df = pd.read_pickle(
        "/Users/alessiobaccelli/PycharmProjects/giotto-time/giotto-time-notebooks/tda_feature/duffing_0.0.pickle")
    df.rename({'label': 'y', 'coord_0': 'x'}, axis='columns', inplace=True)
    df['idx'] = np.arange(len(df))

    ts = df['x'].iloc[:100]

    time_series_features = [NumberOfRelevantHolesFeature(takens_dimension=4,
                                                         takens_stride=10,
                                                         takens_time_delay=3,
                                                         sliding_window_width=7,
                                                         diags_max_edge_length=100,
                                                         sliding_stride=4,
                                                         output_name="num_holes")]

    horizon_main = 4
    feature_creation = FeaturesCreation(horizon_main, time_series_features)
    X, y = feature_creation.fit_transform(ts)
