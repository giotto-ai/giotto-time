from typing import List

import pytest
from pytest import fixture
import numpy as np
from sklearn.compose import make_column_selector
from sklearn.linear_model import LinearRegression, Ridge

from gtime.feature_extraction import Shift, MovingAverage
from gtime.forecasting import GAR
from gtime.time_series_models import TimeSeriesForecastingModel


@fixture(scope="function")
def features1():
    return [
        ("shift_0", Shift(0), make_column_selector(dtype_include=np.number)),
        ("shift_1", Shift(1), make_column_selector(dtype_include=np.number)),
        (
            "moving_average_3",
            MovingAverage(window_size=3),
            make_column_selector(dtype_include=np.number),
        ),
    ]


@fixture(scope="function")
def features2():
    return [
        ("shift_0", Shift(0), make_column_selector(dtype_include=np.number)),
        ("shift_1", Shift(1), make_column_selector(dtype_include=np.number)),
    ]


@fixture(scope="function")
def model1():
    lr = LinearRegression()
    return GAR(lr)


@fixture(scope="function")
def model2():
    lr = Ridge(alpha=0.1)
    return GAR(lr)


@fixture(scope="function")
def time_series_forecasting_model1_no_cache(features1, model1):
    return TimeSeriesForecastingModel(
        features=features1, horizon=2, model=model1, cache_features=False,
    )


@fixture(scope="function")
def time_series_forecasting_model1_cache(features1, model1):
    return TimeSeriesForecastingModel(
        features=features1, horizon=2, model=model1, cache_features=True,
    )


def _single_element_lazy_fixtures(*args):
    return [pytest.lazy_fixture(arg.__name__) for arg in args]


def lazy_fixtures(*args) -> List:
    if isinstance(args[0], tuple):
        return [tuple([pytest.lazy_fixture(arg[0].__name__), *arg[1:]]) for arg in args]
    else:
        return _single_element_lazy_fixtures(*args)
