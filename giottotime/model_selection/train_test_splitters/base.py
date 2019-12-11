import pandas as pd


class Splitter:
    def __init__(
        self, train_test_split_mode: str = "train_size", drop_na_mode: str = "any"
    ):
        self.train_test_split_mode = train_test_split_mode
        self.drop_na_mode = drop_na_mode

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame, **kwargs
    ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
        X, y = self.drop_na(X, y)
        X_train, y_train, X_test, y_test = self.split_train_test(X, y)
        return X_train, y_train, X_test, y_test

    def drop_na(self, X: pd.DataFrame, y: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        X = X.dropna(axis=0, how=self.drop_na_mode)
        y = y.loc[X.index]
        return X, y

    def split_train_test(
        self, X: pd.DataFrame, y: pd.DataFrame
    ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
        pass
