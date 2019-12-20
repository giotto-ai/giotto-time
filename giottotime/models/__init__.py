"""
The :mod:`giottotime.feature_creation` module contains a collection of different kind
of machine learning models, for dealing with time series and non time series data.
"""

from .regressors import LinearRegressor
from .time_series_models import GAR
from .trend_models import PolynomialTrend, ExponentialTrend

__all__ = [
    "LinearRegressor",
    "GAR",
    "PolynomialTrend",
    "ExponentialTrend",
]
