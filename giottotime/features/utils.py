import numpy as np
import pandas as pd


def split_train_test(x, y, split_percentage=0.6):
    train_index = int(split_percentage * len(x))

    x_train = x.iloc[:train_index]
    y_train = y.iloc[:train_index]

    x_test = x.iloc[train_index:]
    y_test = y.iloc[train_index:]

    return x_train, y_train, x_test, y_test


def get_non_nan_values(X: pd.DataFrame, y: pd.DataFrame) \
        -> (pd.DataFrame, pd.DataFrame):
    """Find all rows of X that have at least a ``Nan``value and drop them. Drop
    also the corresponding rows of y.

    Parameters
    ----------
    X: pd.DataFrame
        The DataFrame in which to look and remove for ``Nan`` values.
    y: pd.DataFrame
        The DataFrame in which to remove the rows that correspond to a row of
        X that contain at least a ``Nan`` value.

    Returns
    -------
    X_non_nans, y_non_nans: (pd.DataFrame, pd.DataFrame)
        A tuple containing the two DataFrame.

    """
    X_non_nans = X.dropna(axis='index', how='any')
    y_non_nans = y.loc[X_non_nans.index]

    return X_non_nans, y_non_nans
