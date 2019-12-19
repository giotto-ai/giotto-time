from abc import abstractmethod
from typing import Optional

import pandas as pd

from giottotime.feature_creation.base import Feature


class StandardFeature(Feature):
    """Base class for all the feature classes in this package.

    """

    def fit(self, X, y=None):
        return self

    @abstractmethod
    def transform(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        pass

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
