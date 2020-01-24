"""
The :mod:`giottotime.feature_extraction` module contains a collection of different kind
of machine learning models, for dealing with time series and non time series data.
"""

from giottotime.regressors import LinearRegressor
from .gar import GAR, GARFF
from .trend_models import TrendForecaster

__all__ = [
    "LinearRegressor",
    "GAR",
    "GARFF",
    "TrendForecaster",
]
