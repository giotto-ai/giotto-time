from abc import abstractmethod

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TrendModel(BaseEstimator, TransformerMixin):
    def fit(self, time_series: pd.DataFrame) -> "TrendModel":
        return self

    @abstractmethod
    def predict(self, t):
        pass

    @abstractmethod
    def transform(self, time_series: pd.DataFrame):
        pass
