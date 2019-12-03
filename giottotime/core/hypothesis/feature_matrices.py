from typing import List

import hypothesis.strategies as st

from giottotime.features.features_creation import TimeSeriesFeature
from giottotime.features.feature_creation import FeaturesCreation
from .time_indexes import giotto_time_series


@st.composite
def X_y_matrices(draw,
                 horizon: int,
                 time_series_features: List[TimeSeriesFeature]):
    """ Returns a strategy that generates X and y feature matrices.

    Parameters
    ----------
    horizon : ``int``, required
        the number of steps to forecast in the future. It affects the y shape.

    time_series_features : ``List[TimeSeriesFeature]``, required
        the list of TimeSeriesFeature that is given as input to the
        FeaturesCreation

    Returns
    -------
    X : ``pd.DataFrame``
        X feature matrix
    y : ``pd.DataFrame``
        y feature matrix
    """
    period_index_series = draw(giotto_time_series())
    feature_creation = FeaturesCreation(horizon=horizon,
                                        time_series_features=time_series_features)
    X, y = feature_creation.fit_transform(period_index_series)
    return X, y


@st.composite
def y_matrices(draw,
               horizon: int = 3):
    """ Returns a strategy that generates the y feature matrix.

    Parameters
    ----------
    horizon : ``int``, optional, (default=3)
        the number of steps to forecast in the future. It affects the y shape.

    Returns
    -------
    y : ``pd.DataFrame``
            y feature matrix
    """
    X, y = draw(X_y_matrices(horizon=horizon, time_series_features=[]))
    return y


@st.composite
def X_matrices(draw,
               time_series_features: List[TimeSeriesFeature]):
    """ Returns a strategy that generates the X feature matrix.

    Parameters
    ----------
    time_series_features : ``List[TimeSeriesFeature]``, required
        the list of TimeSeriesFeature that is given as input to the
        FeaturesCreation

    Returns
    -------
    X : ``pd.DataFrame``
            X feature matrix
    """
    X, y = draw(X_y_matrices(horizon=1, time_series_features=time_series_features))
    return X
