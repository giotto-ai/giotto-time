from typing import Tuple

import pandas as pd

from .base import Splitter

FourPandasDataFrames = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]


class DatetimeSplitter(Splitter):
    """ Splits

    """

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame, split_at_time: pd.Timestamp = None
    ) -> FourPandasDataFrames:
        if split_at_time is None:
            X_test, y_test = (
                pd.DataFrame(columns=X.columns),
                pd.DataFrame(columns=y.columns),
            )
            return X, y, X_test, y_test

        X_train, X_test = X[X.index <= split_at_time], X[X.index > split_at_time]
        y_train, y_test = y[y.index <= split_at_time], y[y.index > split_at_time]

        return X_train, y_train, X_test, y_test


class PercentageSplitter(Splitter):
    def transform(self, X, y, split_at_percentage: float = 1):
        if not 0 <= split_at_percentage <= 1:
            raise ValueError(
                f"split_at_percentage has to be between 0"
                f"and 1. Detected: {split_at_percentage}"
            )

        train_max_index = int(X.shape[0] * split_at_percentage) + 1
        X_train, X_test = X[:train_max_index], X[train_max_index:]
        y_train, y_test = y[:train_max_index], y[train_max_index:]

        return X_train, y_train, X_test, y_test


class TrainSizeSplitter(Splitter):
    def transform(self, X, y, train_elements: int = None):
        train_elements = train_elements if train_elements is not None else X.shape[0]
        if train_elements < 0:
            raise ValueError(
                f"train_elements must be positive. " f"Detected: {train_elements}"
            )
        X_train, X_test = X[:train_elements], X[train_elements:]
        y_train, y_test = y[:train_elements], y[train_elements:]

        return X_train, y_train, X_test, y_test
