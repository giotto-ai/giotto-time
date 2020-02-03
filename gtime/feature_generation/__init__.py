"""
The :mod:`gtime.feature_generation` module deals with the creation of features that do
not depend on the input data, but just on its index.
"""

from .external import PeriodicSeasonal, Constant
from .calendar import Calendar

__all__ = ["PeriodicSeasonal", "Constant", "Calendar"]
