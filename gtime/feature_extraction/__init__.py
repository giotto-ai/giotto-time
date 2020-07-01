"""
The :mod:`gtime.feature_extraction` module deals with the creation of features
starting from a time series.
"""

from gtime.feature_generation.calendar import Calendar
from .standard import (
    Shift,
    MovingAverage,
    MovingMedian,
    Max,
    Min,
    MovingCustomFunction,
    Polynomial,
    Exogenous,
    CustomFeature,
)
from .custom import SortedDensity, CrestFactorDetrending

from .trend import Detrender

__all__ = [
    "Shift",
    "MovingAverage",
    "MovingMedian",
    "Max",
    "Min",
    "MovingCustomFunction",
    "Polynomial",
    "Exogenous",
    "Calendar",
    "Detrender",
    "CustomFeature",
    "SortedDensity",
    "CrestFactorDetrending",
]
