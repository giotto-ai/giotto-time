from typing import Tuple

import pytest
from hypothesis import given
from hypothesis.strategies import integers, data

from gtime.utils.hypothesis.general_strategies import (
    ordered_pair,
    shape_matrix,
    shape_X_y_matrices,
)


@given(ordered_pair(0, 10))
def test_ordered_pair(pair: Tuple[int, int]):
    assert pair[0] < pair[1]


@given(ordered_pair(27, 132))
def test_ordered_pair_values(pair: Tuple[int, int]):
    assert pair[0] >= 27
    assert pair[1] <= 132


@given(data=data(), value=integers(0, 10))
def test_ordered_pair_min_equal_max(data, value):
    with pytest.raises(ValueError):
        data.draw(ordered_pair(value, value))


@given(data=data(), shape_0=ordered_pair(10, 100), shape_1=ordered_pair(1, 8))
def test_shape_X(data, shape_0, shape_1):
    shape = data.draw(shape_matrix(*shape_0, *shape_1))
    assert shape_0[0] <= shape[0] <= shape_0[1]
    assert shape_1[0] <= shape[1] <= shape_1[1]


@given(shape_X_y_matrices(123, 243, 12, 34, 1, 6))
def test_shape_X_y_matrices(shape_X_y):
    shape_X, shape_y = shape_X_y
    assert shape_X[0] == shape_y[0]
    assert 12 <= shape_X[1] <= 34
    assert 1 <= shape_y[1] <= 6


@given(shape_X_y_matrices(10, 20, 10, 20, 1, 6))
def test_shape_1_X_smaller_shape_0(shape_X_y):
    shape_X, shape_y = shape_X_y
    assert shape_X[0] > shape_X[1]


@given(data=data())
def test_shape_X_Y_value_error(data):
    with pytest.raises(ValueError):
        data.draw(shape_X_y_matrices(1, 8, 9, 10, 10, 20))
