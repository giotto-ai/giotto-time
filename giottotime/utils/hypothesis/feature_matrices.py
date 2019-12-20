from typing import List, Optional

import hypothesis.strategies as st

from giottotime.feature_creation import FeatureCreation
from giottotime.feature_creation.base import Feature
from .time_indexes import giotto_time_series


@st.composite
def X_y_matrices(
    draw,
    horizon: int,
    time_series_features: List[Feature],
    min_length: Optional[int] = None,
    allow_nan_infinity: bool = True,
):
    """ Returns a strategy that generates X and y feature matrices.

    Parameters
    ----------
    horizon : ``int``, required
        The number of steps to forecast in the future. It affects the y shape.

    time_series_features : ``List[TimeSeriesFeature]``, required
        The list of TimeSeriesFeature that is given as input to the FeaturesCreation

    min_length : ``int``, optional, (default=``None``)
        Minimum length of the matrices

    allow_nan_infinity : ``bool``, optional, (default=``True``)
        Allow nan and infinity in the starting time series

    Returns
    -------
    X : pd.DataFrame
        X feature matrix

    y : pd.DataFrame
        y feature matrix
    """
    min_length = min_length if min_length is not None else 1
    period_index_series = draw(
        giotto_time_series(
            min_length=min_length,
            allow_nan=allow_nan_infinity,
            allow_infinity=allow_nan_infinity,
        )
    )
    feature_creation = FeatureCreation(
        horizon=horizon, time_series_features=time_series_features
    )
    X, y = feature_creation.fit_transform(period_index_series)
    return X, y


@st.composite
def y_matrices(
    draw,
    horizon: int = 3,
    min_length: Optional[int] = None,
    allow_nan_infinity: bool = True,
):
    """ Returns a strategy that generates the y feature matrix.

    Parameters
    ----------
    horizon : ``int``, optional, (default=3)
        the number of steps to forecast in the future. It affects the y shape.

    min_length : ``int``, optional, (default=``None``)
        minimum length of the matrices

    allow_nan_infinity : ``bool``, optional, (default=``True``)
        allow nan and infinity in the starting time series

    Returns
    -------
    y : ``pd.DataFrame``
            y feature matrix
    """
    X, y = draw(
        X_y_matrices(
            horizon=horizon,
            time_series_features=[],
            min_length=min_length,
            allow_nan_infinity=allow_nan_infinity,
        )
    )
    return y


@st.composite
def X_matrices(
    draw,
    time_series_features: List[Feature],
    min_length: Optional[int] = None,
    allow_nan_infinity: bool = True,
):
    """ Returns a strategy that generates the X feature matrix.

    Parameters
    ----------
    time_series_features : ``List[TimeSeriesFeature]``, required
        the list of TimeSeriesFeature that is given as input to the
        FeaturesCreation

    min_length : ``int``, optional, (default=``None``)
        minimum length of the matrices

    allow_nan_infinity : ``bool``, optional, (default=``True``)
        allow nan and infinity in the starting time series

    Returns
    -------
    X : ``pd.DataFrame``
            X feature matrix
    """
    X, y = draw(
        X_y_matrices(
            horizon=1,
            time_series_features=time_series_features,
            min_length=min_length,
            allow_nan_infinity=allow_nan_infinity,
        )
    )
    return X
