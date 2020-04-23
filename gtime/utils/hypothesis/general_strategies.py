from hypothesis import assume
from hypothesis.strategies import tuples, integers, floats, sampled_from
import hypothesis.strategies as st
from sklearn.ensemble import (
    BaggingRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.tree import ExtraTreeRegressor


def ordered_pair(min_value: int, max_value: int):
    if min_value == max_value:
        raise ValueError("min_value and max_value can not be the same")
    return (
        tuples(integers(min_value, max_value), integers(min_value, max_value))
        .map(sorted)
        .filter(lambda x: x[0] < x[1])
    )


def shape_matrix(min_shape_0=30, max_shape_0=200, min_shape_1=5, max_shape_1=10):
    return tuples(
        integers(min_shape_0, max_shape_0), integers(min_shape_1, max_shape_1)
    ).filter(lambda x: x[0] > x[1])


@st.composite
def shape_X_y_matrices(
    draw,
    min_shape_0=30,
    max_shape_0=200,
    min_shape_1_X=5,
    max_shape_1_X=10,
    min_shape_1_y=1,
    max_shape_1_y=3,
    y_as_vector=True,
):
    if max_shape_0 <= min_shape_1_X:
        raise ValueError(
            f"max_shape_0 must be greater than min_shape_1_X: "
            f"{max_shape_0}, {min_shape_1_X}"
        )
    shape_0 = draw(integers(min_shape_0, max_shape_0))
    shape_X = draw(shape_matrix(shape_0, shape_0, min_shape_1_X, max_shape_1_X))
    if y_as_vector:
        shape_y = (shape_0,)
    else:
        shape_y = draw(shape_matrix(shape_0, shape_0, min_shape_1_y, max_shape_1_y))
    assume(shape_X[1] < shape_X[0])
    return shape_X, shape_y


@st.composite
def regressors(draw):
    regressors = [
        LinearRegression(),
        Ridge(alpha=draw(floats(0.00001, 2))),
        BayesianRidge(),
        ExtraTreeRegressor(),
        GradientBoostingRegressor(),
        RandomForestRegressor(),
    ]
    return draw(sampled_from(regressors))
