"""
The :mod:`giottotime.feature_extraction` module contains a collection of different loss
functions.
"""

from .metrics import smape, max_error

__all__ = [
    "smape",
    "max_error",
]
