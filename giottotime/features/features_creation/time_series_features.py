from typing import List, Optional
from collections import namedtuple

import numpy as np
import pandas as pd
from giotto.time_series import TakensEmbedding, SlidingWindow
import giotto.homology as hl

from giottotime.features.features_creation.base import TimeSeriesFeature


class ShiftFeature(TimeSeriesFeature):
    """
    Tentative Docstring
    """
    def __init__(self, shift):
        self.shift = shift

    def transform(self, time_series):
        return time_series.shift(self.shift)


class MovingAverageFeature(TimeSeriesFeature):
    def __init__(self, window_size):
        self.window_size = window_size

    def transform(self, time_series):
        return time_series.rolling(self.window_size).mean().shift(1)


class ConstantFeature(TimeSeriesFeature):
    def __init__(self, constant=1):
        self.constant = constant

    def transform(self, time_series):
        return pd.Series(data=self.constant, index=time_series.index)


class PolynomialFeature(TimeSeriesFeature):
    def __init__(self, degree=2):
        self.degree = degree

    def fit_transform(self, time_series):
        pass


class ExogenousFeature(TimeSeriesFeature):
    def __init__(self, exogenous_time_series, name):
        self.exogenous_time_series = exogenous_time_series
        self.name = name

    def __repr__(self):
        return "{class_name}({name})".format(class_name=self.__class__.__name__,
                                             name=self.name)

    def transform(self, time_series):
        return self.exogenous_time_series.reindex(index=time_series.index)


class CustomFeature(TimeSeriesFeature):
    def __init__(self, custom_feature_function, **kwargs):
        self.custom_feature_function = custom_feature_function
        self.kwargs = kwargs

    def __repr__(self):
        return "{class_name=self.__class__.__name__}" \
               "({self.custom_feature_function.__name__})"

    def transform(self, time_series):
        return self.custom_feature_function(time_series, **self.kwargs)


TakensEmbeddingParams = namedtuple('TakensEmbeddingParams', [
    'parameters_type',
    'dimension',
    'stride',
    'n_jobs'
])

def default_takens_embedding_params() -> TakensEmbeddingParams:
    return TakensEmbeddingParams(
        parameters_type='search',
        dimension=5,
        stride=1,
        n_jobs=1,
    )


SlidingWindowParams = namedtuple('SlidingWindowParams', [
    'window_width',
    'stride',
    'n_jobs',
])

def default_sliding_window_params() -> SlidingWindowParams:
    return SlidingWindowParams(
        window_width=10,
        stride=1,
        n_jobs=1,
    )


PersistenceDiagramParams = namedtuple('PersistenceDiagramParams', [
    'metric',
    'max_edge_length',
    'homology_dimension',
    'persistence_infinity_values',
    'n_jobs',
])

def default_persistence_diagram_params() -> PersistenceDiagramParams:
    return PersistenceDiagramParams(
        metric='euclidean',
        max_edge_length=np.inf,
        homology_dimension=[0, 1, 2],
        persistence_infinity_values=None,
        n_jobs=1,
    )


class TDAFeature(TimeSeriesFeature):
    available_tda_features = [
        'number_of_holes',
        'average_lifetime',
        'betti',
        'amplitude',
    ]

    def __init__(
            self,
            tda_features: List[str],
            takens_embedding_params: Optional[TakensEmbeddingParams] = None,
            sliding_window_params: Optional[SlidingWindowParams] = None,
            persistence_diagram_params: Optional[PersistenceDiagramParams] = None,
    ):
        self.takens_embedding_params = takens_embedding_params \
            if takens_embedding_params is not None \
            else default_takens_embedding_params()
        self.sliding_window_params = sliding_window_params \
            if sliding_window_params is not None \
            else default_sliding_window_params()
        self.persistence_diagram_params = persistence_diagram_params \
            if persistence_diagram_params is not None \
            else default_persistence_diagram_params()

        self.takens_embedding = TakensEmbedding(**self.takens_embedding_params)
        self.sliding_window = SlidingWindow(**self.sliding_window_params)
        self.vietoris_rips_persistence = hl.VietorisRipsPersistence(**persistence_diagram_params)

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        pass


class _NumberOfRelevantHoles(TimeSeriesFeature):

    def __init__(self, homology_dimension: int = 0, theta: float = 0.7):
        self.homology_dimension = homology_dimension
        self.theta = theta





