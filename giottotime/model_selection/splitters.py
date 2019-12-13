import numpy as np
import pandas as pd


class Splitter:
    def __init__(self, drop_na_mode: str = "any"):
        self.drop_na_mode = drop_na_mode

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame, **kwargs
    ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):

        X, y = self._trim_nan(X, y)
        X_train, y_train, X_test, y_test = self._split_train_test(X, y)
        return X_train, y_train, X_test, y_test

    def _trim_nan(
        self, X: pd.DataFrame, y: pd.DataFrame
    ) -> (pd.DataFrame, pd.DataFrame):

        X = X.dropna(axis=0, how=self.drop_na_mode)
        y = y.loc[X.index]
        return X, y

    def _split_train_test(
        self, X: pd.DataFrame, y: pd.DataFrame
    ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):

        train_indexes, test_indexes = self._get_train_test_indexes_from_y(y)
        X_train, y_train = X.loc[train_indexes], y.loc[train_indexes]
        X_test, y_test = X.loc[test_indexes], y.loc[test_indexes]
        return X_train, y_train, X_test, y_test

    def _get_train_test_indexes_from_y(self, y):
        last_train_index = self._last_non_nan_y_index(y)
        train_indexes = y.index[:last_train_index]
        test_indexes = y.index.difference(train_indexes)
        return train_indexes, test_indexes

    def _last_non_nan_y_index(self, y: pd.DataFrame) -> pd.Period:
        y_nan = y.isnull().any(axis=1).replace(True, np.nan)
        return y_nan.last_valid_index()
