from typing import Optional

import hypothesis.strategies as st
import pandas as pd
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import tuples, integers, floats

from gtime.utils.hypothesis.general_strategies import shape_X_y_matrices, shape_matrix
from .time_indexes import giotto_time_series
from ...compose import FeatureCreation
from ...model_selection import horizon_shift


@st.composite
def X_y_matrices(
    draw,
    horizon: int,
    df_transformer: FeatureCreation,
    min_length: Optional[int] = None,
    allow_nan_infinity: bool = True,
):
    """ Returns a strategy that generates X and y feature matrices.

    Parameters
    ----------
    horizon : ``int``, required
        The number of steps to forecast in the future. It affects the y shape.

    df_transformer : ``List[TimeSeriesFeature]``, required
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
    #  feature_creation = FeatureCreation(horizon=horizon, time_series_features=time_series_features)
    #  X, y = feature_creation.fit_transform(period_index_series)

    X = df_transformer.fit_transform(period_index_series)

    y = horizon_shift(period_index_series, horizon=horizon)

    return X, y


@st.composite
def X_matrices(
    draw,
    df_transformer: FeatureCreation,
    min_length: Optional[int] = None,
    allow_nan_infinity: bool = True,
):
    """ Returns a strategy that generates the X feature matrix.

    Parameters
    ----------
    df_transformer : ``List[TimeSeriesFeature]``, required
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
    min_length = min_length if min_length is not None else 1
    period_index_series = draw(
        giotto_time_series(
            min_length=min_length,
            allow_nan=allow_nan_infinity,
            allow_infinity=allow_nan_infinity,
        )
    )

    X = df_transformer.fit_transform(period_index_series)
    return X


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
    min_length = min_length if min_length is not None else 1
    period_index_series = draw(
        giotto_time_series(
            min_length=min_length,
            allow_nan=allow_nan_infinity,
            allow_infinity=allow_nan_infinity,
        )
    )

    y = horizon_shift(period_index_series, horizon=horizon)

    return y


@st.composite
def numpy_X_y_matrices(
    draw,
    X_y_shapes=shape_X_y_matrices(),
    y_vector=True,
    min_value: float = None,
    max_value: float = None,
    allow_nan: bool = False,
    allow_infinity: bool = False,
):
    if isinstance(X_y_shapes, tuple) or isinstance(X_y_shapes, list):
        X_shape, y_shape = X_y_shapes
    else:
        X_shape, y_shape = draw(X_y_shapes)
    if X_shape[0] != y_shape[0]:
        raise ValueError(f"X.shape[0] must be == y.shape[0]: {X_shape}, {y_shape}")
    if X_shape[0] <= X_shape[1]:
        raise ValueError(f"X.shape[0] must be <= X.shape[1]: {X_shape}")
    if y_vector and len(y_shape) > 1:
        y_shape = (y_shape[0],)

    elements = floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=allow_nan,
        allow_infinity=allow_infinity,
    )
    X = draw(arrays(dtype=float, shape=X_shape, elements=elements,))
    y = draw(arrays(dtype=float, shape=y_shape, elements=elements,))
    return X, y


@st.composite
def numpy_X_matrices(
    draw,
    shape=shape_matrix(),
    min_value: float = None,
    max_value: float = None,
    allow_nan: bool = False,
    allow_infinity: bool = False,
):
    if not isinstance(shape, tuple) and not isinstance(shape, list):
        shape = draw(shape)
    if shape[0] <= shape[1]:
        raise ValueError(f"X.shape[0] must be <= X.shape[1]: {shape}")

    elements = floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=allow_nan,
        allow_infinity=allow_infinity,
    )

    X = draw(arrays(dtype=float, shape=shape, elements=elements,))
    return X
