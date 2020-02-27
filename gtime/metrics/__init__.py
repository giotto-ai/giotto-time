"""
The :mod:`gtime.metrics` module contains a collection of different metrics.
"""

from .metrics import smape, max_error, mse, log_mse, r_square

__all__ = [
    "smape",
    "max_error",
    "mse",
    "log_mse",
    "r_square",
]
