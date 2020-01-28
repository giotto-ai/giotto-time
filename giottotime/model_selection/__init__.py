"""
The :mod:`giottotime.feature_extraction` module deals with model selection.
"""

from .splitters import FeatureSplitter
from .horizon_shift import horizon_shift

__all__ = ["FeatureSplitter", "horizon_shift"]
