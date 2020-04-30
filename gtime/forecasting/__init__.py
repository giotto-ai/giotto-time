"""
The :mod:`gtime.forecasting` module contains a collection of machine learning models,
for dealing with time series data.
"""

from .gar import GAR, GARFF, MultiFeatureGAR
from gtime.regressors.multi_output import MultiFeatureMultiOutputRegressor
from .trend_models import TrendForecaster
from .online import HedgeForecaster
from .simple_models import (
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
