import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class NaiveModel(BaseEstimator, RegressorMixin):

    def fit(self, X: pd.DataFrame, y=None):

        self.last_value_ = X.iloc[-1]

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:

        check_is_fitted(self)
        y_pred = self.last_value_.to_numpy()
        predictions = pd.DataFrame(data=np.tile(y_pred, len(X)), index=X.index, columns=X.columns)

        return predictions


class SeasonalNaiveModel(BaseEstimator, RegressorMixin):


    def __init__(self, seasonal_length = 1):
        super().__init__()
        self.lag_ = seasonal_length

    def fit(self, X: pd.DataFrame, y=None):

        self.last_value_ = X.iloc[-self.lag_:]

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:

        check_is_fitted(self)
        len_x = len(X)
        len_s = len(self.last_value_)
        y_pred = self.last_value_[X.columns].to_numpy()
        cycles = len_x // len_s
        extra = len_x % len_s
        y_pred = np.concatenate((np.tile(y_pred, (cycles, 1)), y_pred[:extra, :]), axis=0)
        predictions = pd.DataFrame(data=y_pred, index=X.index, columns=X.columns)

        return predictions


class DriftModel(BaseEstimator, RegressorMixin):

    def fit(self, X: pd.DataFrame, y=None):

        self.drift_ = (X.iloc[-1] - X.iloc[0]) / len(X)
        self.last_value_ = X.iloc[-1]
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:

        check_is_fitted(self)
        len_x = len(X)
        y_pred = [self.last_value_ + i * self.drift_ for i in range(len_x)]
        predictions = pd.DataFrame(data=y_pred, index=X.index, columns=X.columns)

        return predictions