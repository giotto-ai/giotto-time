from abc import abstractmethod

import pandas as pd

from giottotime.feature_creation.base import Feature


class IndexDependentFeature(Feature):
    """Base class for all the feature classes in this package.

    Parameters documentation is in the derived classes.
    """

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass
