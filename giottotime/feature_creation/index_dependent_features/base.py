from abc import ABCMeta, abstractmethod

import pandas as pd

from giottotime.feature_creation.base import Feature


class IndexDependentFeature(Feature, metaclass=ABCMeta):
    """Base class for all the feature classes in this package.

    """

    def fit(self, X, y=None):
        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError  # to exclude from pytest coverage

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
