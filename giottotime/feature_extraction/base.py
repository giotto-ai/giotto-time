from abc import abstractmethod, ABCMeta
from typing import Union, Optional

import pandas as pd


class Feature:
    """Base class for all the features defined in this package.

    """

    @abstractmethod
    def __init__(self, output_name):
        self.output_name = output_name

    def _rename_columns(self, X: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        if isinstance(X, pd.Series):
            X = X.to_frame()

        suffix = ""

        X_renamed = X.T.reset_index(drop=True).T
        for index, col in enumerate(X_renamed.columns):
            if len(X.columns) > 1:
                suffix = "_" + str(index)
            X_renamed.rename(
                columns={col: self.output_name + str(suffix)}, inplace=True
            )

        return X_renamed


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