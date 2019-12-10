from abc import abstractmethod
from typing import Optional

import pandas as pd

from giottotime.feature_creation.base import Feature


class StandardFeature(Feature):
    """Base class for all the feature classes in this package.

    Parameters documentation is in the derived classes.
    """

    @abstractmethod
    def transform(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        pass
