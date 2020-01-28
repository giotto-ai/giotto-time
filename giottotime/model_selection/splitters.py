import numpy as np
import pandas as pd


class FeatureSplitter:
    """Splits the feature matrices X and y in X_train, y_train, X_test, y_test.

    X and y are the feature matrices obtained from the FeatureCreation class.

    Parameters
    ----------
    drop_na_mode: str, optional, default: ``'any'``
        How to drop the Nan contained in the ``X`` and ``y`` matrices. Only 'any' is
        supported for the moment.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from giottotime.model_selection import FeatureSplitter
    >>> X = pd.DataFrame.from_dict({"feature_0": [np.nan, 0, 1, 2, 3, 4, 5, 6, 7, 8],
    ...                             "feature_1": [np.nan, np.nan, 0.5, 1.5, 2.5, 3.5,
    ...                                            4.5, 5.5, 6.5, 7.5, ]
    ...                            })
    >>> y = pd.DataFrame.from_dict({"y_0": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    ...                             "y_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan],
    ...                             "y_2": [2, 3, 4, 5, 6, 7, 8, 9, np.nan, np.nan]
    ...                            })
    >>> feature_splitter = FeatureSplitter()
    >>> X_train, y_train, X_test, y_test = feature_splitter.transform(X, y)
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
    8           7.0                   6.5
    9           8.0                   7.5
    >>> y_test
       y_0  y_1  y_2
    8    8  9.0  NaN
    9    9  NaN  NaN

    """

    def __init__(self, drop_na_mode: str = "any"):
        if drop_na_mode != "any":
            raise ValueError(
                f'Only drop_na_mode="any" is supported. Detected: {drop_na_mode}'
            )
        self.drop_na_mode = drop_na_mode

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame
    ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """Split the feature matrices X and y in X_train, y_train, X_test, y_test.

        ``X`` and ``y`` are the feature matrices obtained from the FeatureCreation
        class.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), required
            The feature matrix.

        y : ``pd.DataFrame``, shape (n_samples, horizon), required
            The y matrix.

        Returns
        -------
        X_train, y_train, X_test, y_test : ``Tuple[pd.DataFrame, pd.DataFrame, \
            pd.DataFrame, pd.DataFrame]``
            The X and y, split between train and test.

        """
        X, y = self._drop_X_na(X, y)
        X_train, y_train, X_test, y_test = self._split_train_test(X, y)
        return X_train, y_train, X_test, y_test

    def _drop_X_na(
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
        train_indexes = y.loc[:last_train_index].index if last_train_index else []
        test_indexes = y.index.difference(train_indexes)
        return train_indexes, test_indexes

    def _last_non_nan_y_index(self, y: pd.DataFrame) -> pd.Period:
        y_nan = y.isnull().any(axis=1).replace(True, np.nan)
        return y_nan.last_valid_index()
