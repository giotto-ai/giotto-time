"""
The :mod:`gtime.feature_extraction` module deals with the creation of features
starting from a time series.
"""

from gtime.feature_generation.calendar import Calendar
from .standard import (
    Shift,
    MovingAverage,
    MovingCustomFunction,
    Polynomial,
    Exogenous,
    CustomFeature,
)

from .trend import Detrender

__all__ = [
    "Shift",
    "MovingAverage",
    "MovingCustomFunction",
    "Polynomial",
    "Exogenous",
    "Calendar",
    "Detrender",
    "CustomFeature",
]
