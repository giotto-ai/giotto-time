"""
The :mod:`gtime.forecasting` module contains a collection of machine learning models,
for dealing with time series data.
"""

from .gar import GAR, GARFF
from .trend_models import TrendForecaster
from .online import HedgeForecaster

__all__ = [
    "GAR",
    "GARFF",
    "TrendForecaster",
    "HedgeForecaster",
]
