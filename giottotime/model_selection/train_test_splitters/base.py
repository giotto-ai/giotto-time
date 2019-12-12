import numpy as np
import pandas as pd


class Splitter:
    def __init__(self, drop_na_mode: str = "any"):
        self.drop_na_mode = drop_na_mode

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame, **kwargs
    ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
        X, y = self.trim(X, y)
        X_train, y_train, X_test, y_test = self.split_train_test(X, y)
        return X_train, y_train, X_test, y_test

    def trim(self, X: pd.DataFrame, y: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        X = X.dropna(axis=0, how=self.drop_na_mode)
        y = y.loc[X.index]
        return X, y

    def split_train_test(
        self, X: pd.DataFrame, y: pd.DataFrame
    ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
        y_nan = y.isnull().any(axis=1).replace(True, np.nan)
        last_valid_index = y_nan.last_valid_index()
        X_train, y_train = X.loc[:last_valid_index], y.loc[:last_valid_index]
        test_indexes = y.index.difference(y_train.index)
        X_test, y_test = X.loc[test_indexes], y.loc[test_indexes]
        return X_train, y_train, X_test, y_test
