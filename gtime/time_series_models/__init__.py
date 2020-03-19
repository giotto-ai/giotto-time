"""
The :mod:`gtime.time_series_models` module contains time series models.
"""

from .base import TimeSeriesForecastingModel
from .ar import AR

__all__ = ["TimeSeriesForecastingModel", "AR"]
