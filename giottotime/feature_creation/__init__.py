"""
The :mod:`giottotime.feature_creation` module deals with the creation of features
starting from a time series.
"""

from .feature_creation import FeatureCreation
from .index_dependent_features import (
    CalendarFeature,
    DetrendedFeature,
    RemovePolynomialTrend,
    RemoveExponentialTrend,
    tda_features,
)
from .index_dependent_features import (
    ShiftFeature,
    MovingAverageFeature,
    PolynomialFeature,
    ExogenousFeature,
    AmplitudeFeature,
    AvgLifeTimeFeature,
    BettiCurvesFeature,
    NumberOfRelevantHolesFeature,
)

from .standard_features import ConstantFeature, PeriodicSeasonalFeature, CustomFeature

__all__ = [
    "FeatureCreation",
    "ShiftFeature",
    "MovingAverageFeature",
    "ConstantFeature",
    "PolynomialFeature",
    "ExogenousFeature",
    "tda_features",
    "CalendarFeature",
    "PeriodicSeasonalFeature",
    "DetrendedFeature",
    "RemovePolynomialTrend",
    "RemoveExponentialTrend",
    "AmplitudeFeature",
    "AvgLifeTimeFeature",
    "BettiCurvesFeature",
    "NumberOfRelevantHolesFeature",
]
