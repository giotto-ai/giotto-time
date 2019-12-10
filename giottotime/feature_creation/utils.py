from typing import Tuple

import pandas as pd


def trim_feature_nans(
    X: pd.DataFrame, y: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the ``X`` and ``y`` in train and test set. First, the rows of
    ``X`` that contain a ``Nan`` value are dropped, as well as the
    corresponding rows of ``y``. Then, the training set is composed of all the
    rows that don't have any ``Nan`` values in the ``y``, while the remaining
    rows are used as test set.

    Parameters
    ----------
    X : ``pd.DataFrame``, required.
        The ``pd.DataFrame`` containing ``X``.

    y : ``pd.DataFrame``, required.
        The ``pd.DataFrame`` containing ``y``.

    Returns
    -------
    X_train, y_train, X_test, y_test : ``Tuple[pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame]``
        The ``X`` and ``y``, split in train and test set according to the
        ``split_percentage``.

    """
    X_non_nans, y_non_nans = _get_non_nan_values(X, y)
    last_valid_y_index = y.iloc[:, -1].last_valid_index()

    X_train = X_non_nans.loc[:last_valid_y_index]
    y_train = y_non_nans.loc[:last_valid_y_index]

    X_test = X_non_nans.loc[last_valid_y_index:]
    y_test = y_non_nans.loc[last_valid_y_index:]

    return X_train, y_train, X_test, y_test


def _get_non_nan_values(
    X: pd.DataFrame, y: pd.DataFrame
) -> (pd.DataFrame, pd.DataFrame):
    X_non_nans = X.dropna(axis="index", how="any")
    y_non_nans = y.loc[X_non_nans.index]

    return X_non_nans, y_non_nans
