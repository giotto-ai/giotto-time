import inspect
from abc import ABCMeta, abstractmethod

import pandas as pd


class TimeSeriesFeature(metaclass=ABCMeta):
    """Base class for all the feature classes in this package.

    Parameter documentation is in the derived classes.
    """
    @abstractmethod
    def __init__(self, output_name):
        self.output_name = output_name

    def fit(self, X, y=None):
        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.Series:
        pass

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
