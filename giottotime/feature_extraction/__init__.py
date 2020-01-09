"""
The :mod:`giottotime.feature_extraction` module deals with the creation of features
starting from a time series.
"""

from .feature_creation import FeatureCreation
from .calendar import CalendarFeature
from .time_series import (
    ShiftFeature,
    MovingAverageFeature,
    PolynomialFeature,
    ExogenousFeature,
)

from .topology import AmplitudeFeature, AvgLifeTimeFeature, \
    BettiCurvesFeature, NumberOfRelevantHolesFeature

from .standard import ConstantFeature, PeriodicSeasonalFeature, \
    CustomFeature

from .trend import (
    DetrendedFeature,
    RemovePolynomialTrend,
    RemoveExponentialTrend)

__all__ = [
    "FeatureCreation",
    "ShiftFeature",
    "MovingAverageFeature",
    "ConstantFeature",
    "PolynomialFeature",
    "ExogenousFeature",
    "topology",
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
