"""
The :mod:`gtime.model_selection` module deals with model selection.
"""

from .horizon_shift import horizon_shift
from .splitters import FeatureSplitter

__all__ = ["FeatureSplitter", "horizon_shift"]
