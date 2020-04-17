"""
The :mod:`gtime.custom` module implements custom methods for time
series.
"""

from .crest_factor_detrending import CrestFactorDetrending
from .sorted_density import SortedDensity

__all__ = [
    "CrestFactorDetrending",
    "SortedDensity",
]
