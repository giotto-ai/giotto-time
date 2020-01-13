"""
The :mod:`giottotime.feature_extraction` module deals with the creation of features
starting from a time series.
"""

from .feature_creation import FeatureCreation
from .calendar import CalendarFeature
from .time_series import (
    Shift,
    MovingAverage,
    Polynomial,
    Exogenous,
)

from .topology import AmplitudeFeature, AvgLifeTimeFeature, \
    BettiCurvesFeature, NumberOfRelevantHolesFeature

from .trend import (
    DetrendedFeature,
    RemovePolynomialTrend,
    RemoveExponentialTrend)

__all__ = [
    "FeatureCreation",
    "Shift",
    "MovingAverage",
    "Polynomial",
    "Exogenous",
    "topology",
    "CalendarFeature",
    "DetrendedFeature",
    "RemovePolynomialTrend",
    "RemoveExponentialTrend",
    "AmplitudeFeature",
    "AvgLifeTimeFeature",
    "BettiCurvesFeature",
    "NumberOfRelevantHolesFeature",
]
