from typing import Tuple

import pandas as pd


def trim_feature_nans(
    X: pd.DataFrame, y: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ``X`` and ``y`` in train and test set. First, the rows of ``X`` that
    contain a ``Nan`` value are dropped, as well as the corresponding rows of ``y``.
    Then, the training set is composed of all the rows that don't have any ``Nan``
    values in the ``y``, while the remaining rows are used as test set.

    Parameters
    ----------
    X : pd.DataFrame, shape (n_samples, n_features), required
        The ``pd.DataFrame`` containing ``X``.

    y : pd.DataFrame, shape (n_samples, horizon), required
        The ``pd.DataFrame`` containing ``y``.

    Returns
    -------
    X_train, y_train, X_test, y_test : Tuple[pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame]
        The ``X`` and ``y``, split in train and test set according to the
        ``split_percentage``.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from giottotime.feature_creation import trim_feature_nans
    >>> X = pd.DataFrame.from_dict({"feature_0": [np.nan, 0, 1, 2, 3, 4, 5, 6, 7, 8],
    ...                             "feature_1": [np.nan, np.nan, 0.5, 1.5, 2.5, 3.5,
    ...                                            4.5, 5.5, 6.5, 7.5, ]
    ...                            })
    >>> y = pd.DataFrame.from_dict({"y_0": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    ...                             "y_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan],
    ...                             "y_2": [2, 3, 4, 5, 6, 7, 8, 9, np.nan, np.nan]
    ...                            })
    >>> X_train, y_train, X_test, y_test = trim_feature_nans(X, y)
    >>> X_train
           feature_0            feature_1
    2           1.0                   0.5
    3           2.0                   1.5
    4           3.0                   2.5
    5           4.0                   3.5
    6           5.0                   4.5
    7           6.0                   5.5
    >>> y_train
       y_0  y_1  y_2
    2    2  3.0  4.0
    3    3  4.0  5.0
    4    4  5.0  6.0
    5    5  6.0  7.0
    6    6  7.0  8.0
    7    7  8.0  9.0
    >>> X_test
          feature_0             feature_1
    7           6.0                   5.5
    8           7.0                   6.5
    9           8.0                   7.5
    >>> y_test
       y_0  y_1  y_2
    7    7  8.0  9.0
    8    8  9.0  NaN
    9    9  NaN  NaN
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
