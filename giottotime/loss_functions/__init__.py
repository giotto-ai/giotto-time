"""
The :mod:`giottotime.feature_extraction` module contains a collection of different loss
functions.
"""

from .loss_functions import smape, max_error

__all__ = [
    "smape",
    "max_error",
]
