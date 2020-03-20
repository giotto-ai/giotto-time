"""
The :mod:`gtime.forecasting` module contains a collection of machine learning models,
for dealing with time series data.
"""

from .gar import GAR, GARFF
from .trend_models import TrendForecaster
from .online import HedgeForecaster
from .simple_models import NaiveModel, SeasonalNaiveModel, DriftModel, AverageModel

__all__ = [
    "GAR",
    "GARFF",
    "TrendForecaster",
    "HedgeForecaster",
    'NaiveModel',
    'SeasonalNaiveModel',
    'DriftModel',
    'AverageModel'
]
