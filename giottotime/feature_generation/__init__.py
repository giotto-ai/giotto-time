"""
The :mod:`giottotime.feature_generation` module deals with the creation of features
that do not necessary require a time series in order to be created.
"""

from .external import PeriodicSeasonal, Constant
from .calendar import Calendar

__all__ = ["PeriodicSeasonal", "Constant", "Calendar"]
