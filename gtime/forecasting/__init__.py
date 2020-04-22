"""
The :mod:`gtime.forecasting` module contains a collection of machine learning models,
for dealing with time series data.
"""

from .gar import GAR, GARFF, MultiFeatureGAR
from gtime.forecasting.multi_output import MultiFeatureMultiOutputRegressor
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
    "MultiFeatureGAR" "TrendForecaster",
    "HedgeForecaster",
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "DriftForecaster",
    "AverageForecaster",
]
