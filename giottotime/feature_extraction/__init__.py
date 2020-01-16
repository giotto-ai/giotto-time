"""
The :mod:`giottotime.feature_extraction` module deals with the creation of features
starting from a time series.
"""

from giottotime.feature_generation.calendar import Calendar
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
    "Shift",
    "MovingAverage",
    "Polynomial",
    "Exogenous",
    "topology",
    "Calendar",
    "DetrendedFeature",
    "RemovePolynomialTrend",
    "RemoveExponentialTrend",
    "AmplitudeFeature",
    "AvgLifeTimeFeature",
    "BettiCurvesFeature",
    "NumberOfRelevantHolesFeature",
]
