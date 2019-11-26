from typing import Iterable

import numpy as np
import pandas as pd

from giottotime.features.features_creation.tda_features.base import TDAFeatures
from giottotime.features.features_creation.feature_creation import \
    FeaturesCreation


class NumberOfRelevantHolesFeature(TDAFeatures):
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
                 h_dim: int = 0,
                 theta: float = 0.7,
                 interpolation_strategy: str = 'ffill'):
        super().__init__(output_name,
                         takens_parameters_type,
                         takens_dimension,
                         takens_stride,
                         takens_time_delay,
                         takens_n_jobs,
                         sliding_window_width,
                         sliding_stride,
                         diags_metric,
                         diags_max_edge_length,
                         diags_homology_dimensions,
                         diags_infinity_values,
                         diags_n_jobs
                         )

        self._h_dim = h_dim
        self._theta = theta
        self.interpolation_strategy = interpolation_strategy

    def _compute_num_relevant_holes(self, X_scaled):
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
    def transform(self, X):
        X_scaled = self._compute_persistence_diagrams(X)
        n_holes = self._compute_num_relevant_holes(X_scaled)
        original_points= self._compute_indices(len(n_holes))
        # TODO: check this is correct with Matteo
        return


class AvgLifeTimeFeature(TDAFeatures):
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
                 h_dim: int = 0
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
                         diags_max_edge_length,
                         diags_homology_dimensions,
                         diags_infinity_values,
                         diags_n_jobs
                         )
        self._h_dim = h_dim

    def _average_lifetime(self, X_scaled):
        avg_lifetime_list = []

        for i in range(X_scaled.shape[0]):
            persistence_table = pd.DataFrame(X_scaled[i],
                                             columns=['birth', 'death',
                                                      'homology'])
            persistence_table['lifetime'] = persistence_table['death'] - \
                                            persistence_table['birth']
            avg_lifetime_list.append(
                persistence_table[persistence_table['homology']
                                  == self._h_dim]['lifetime'].mean())

        return avg_lifetime_list

    def transform(self, X):
        X_scaled = self._compute_persistence_diagrams(X)
        avg_lifetime = self._average_lifetime(X_scaled)
        original_points = self._compute_indices(len(avg_lifetime))
        pass





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
