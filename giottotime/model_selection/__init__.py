"""
The :mod:`giottotime.feature_extraction` module deals with model selection.
"""

from .feature_splitters import FeatureSplitter
from .walk_forward_split import walk_forward_split

__all__ = ["FeatureSplitter", "walk_forward_split"]
