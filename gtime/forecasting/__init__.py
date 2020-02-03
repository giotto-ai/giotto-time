"""
The :mod:`gtime.forecasting` module contains a collection of different kind
of machine learning models, for dealing with time series and non time series data.
"""

from .gar import GAR, GARFF
from .trend_models import TrendForecaster

__all__ = [
    "GAR",
    "GARFF",
    "TrendForecaster",
]
