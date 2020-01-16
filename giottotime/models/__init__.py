"""
The :mod:`giottotime.feature_extraction` module contains a collection of different kind
of machine learning models, for dealing with time series and non time series data.
"""

from .regressors import LinearRegressor
from .forecasting import GAR, GARFF
from .trend_models import PolynomialTrend, ExponentialTrend

__all__ = [
    "LinearRegressor",
    "GAR",
    "GARFF",
    "PolynomialTrend",
    "ExponentialTrend",
]
