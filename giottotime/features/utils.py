from typing import Tuple

import pandas as pd


def split_train_test(X: pd.DataFrame, y: pd.DataFrame,
                     split_percentage: float = 0.6) \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the ``X`` and ``y`` in train and test set. The ratio of the
    training set is set with ``split_percentage``.

    Parameters
    ----------
    X : ``pd.DataFrame``, required.
        The ``pd.DataFrame`` containing ``X``.

    y : ``pd.DataFrame``, required.
     The ``pd.DataFrame`` containing ``y``.

    split_percentage : ``float``, optional, (default=``0.6``).
        The percentage of the training set with respect to the original ``X``.

    Returns
    -------
    X_train, y_train, X_test, y_test : ``Tuple[pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame]``
        The ``X`` and ``y``, split in train and test set according to the
        ``split_percentage``.

    """
    train_index = int(split_percentage * len(X))

    X_train = X.iloc[:train_index]
    y_train = y.iloc[:train_index]

    X_test = X.iloc[train_index:]
    y_test = y.iloc[train_index:]

    return X_train, y_train, X_test, y_test


def get_non_nan_values(X: pd.DataFrame, y: pd.DataFrame) \
        -> (pd.DataFrame, pd.DataFrame):
    """Find all rows of X that have at least a ``Nan``value and drop them. Drop
    also the corresponding rows of y.

    Parameters
    ----------
<<<<<<< HEAD
    X : ``pd.DataFrame``, required.
        The DataFrame in which to look and remove for ``Nan`` values.

    y : ``pd.DataFrame``, required.
=======
    X: pd.DataFrame
        The DataFrame in which to look and remove for ``Nan`` values.
    y: pd.DataFrame
>>>>>>> Added docstrings
        The DataFrame in which to remove the rows that correspond to a row of
        X that contain at least a ``Nan`` value.

    Returns
    -------
<<<<<<< HEAD
    X_non_nans, y_non_nans : ``(pd.DataFrame, pd.DataFrame)``
=======
    X_non_nans, y_non_nans: (pd.DataFrame, pd.DataFrame)
>>>>>>> Added docstrings
        A tuple containing the two DataFrame.

    """
    X_non_nans = X.dropna(axis='index', how='any')
    y_non_nans = y.loc[X_non_nans.index]

    return X_non_nans, y_non_nans
