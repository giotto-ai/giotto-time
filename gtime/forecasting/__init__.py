"""
The :mod:`gtime.forecasting` module contains a collection of machine learning models,
for dealing with time series data.
"""

from .gar import GAR, GARFF, MultiFeatureMultiOutputRegressor
from .trend_models import TrendForecaster
from .online import HedgeForecaster
from .simple_models import (
    NaiveForecaster,
    SeasonalNaiveForecaster,
    DriftForecaster,
    AverageForecaster,
)

__all__ = [
    "GAR",
    "GARFF",
    "MultiFeatureMultiOutputRegressor",
    "TrendForecaster",
    "HedgeForecaster",
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "DriftForecaster",
    "AverageForecaster",
]
