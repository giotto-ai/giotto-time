"""
The :mod:`gtime.forecasting` module contains a collection of machine learning models,
for dealing with time series data.
"""

from .gar import GAR, GARFF, MultiFeatureMultiOutputRegressor, MultiFeatureGAR
from .trend import TrendForecaster
from .online import HedgeForecaster
from .naive import (
    NaiveForecaster,
    SeasonalNaiveForecaster,
    DriftForecaster,
    AverageForecaster,
)
from .arima import ARIMAForecaster

__all__ = [
    "GAR",
    "GARFF",
    "MultiFeatureGAR",
    "TrendForecaster",
    "HedgeForecaster",
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "DriftForecaster",
    "AverageForecaster",
    "MultiFeatureMultiOutputRegressor",
    "ARIMAForecaster"
]
