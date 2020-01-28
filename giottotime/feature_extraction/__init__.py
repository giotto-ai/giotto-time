"""
The :mod:`giottotime.feature_extraction` module deals with the creation of features
starting from a time series.
"""

from giottotime.feature_generation.calendar import Calendar
from .standard import (
    Shift,
    MovingAverage,
    Polynomial,
    Exogenous,
    CustomFeature,
)


from .trend import Detrender


__all__ = [
    "Shift",
    "MovingAverage",
    "Polynomial",
    "Exogenous",
    "Calendar",
    "Detrender",
    "CustomFeature",
]
