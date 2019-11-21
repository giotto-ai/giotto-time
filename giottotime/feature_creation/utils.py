import numpy as np
import pandas as pd


def split_train_test(x, y, split_percentage=0.6):
    train_index = int(split_percentage * len(x))

    x_train = x.iloc[:train_index]
    y_train = y.iloc[:train_index]

    x_test = x.iloc[train_index:]
    y_test = y.iloc[train_index:]

    return x_train, y_train, x_test, y_test


def get_train_test_features(base_features: pd.DataFrame,
                            y: pd.DataFrame = None):
    """
    If y is provided, get the training and testing features, starting from the
    ``base_features``. If y is not provided, then only the testing features are
    provided.

    Parameters
    ----------
    base_features
    y

    Returns
    -------

    """

    features_non_na = base_features.dropna(axis='index')

    if y is not None:
        y_n_cols = y.shape[1]
        y_non_na = y.dropna(axis='index')
        x_y_train = features_non_na.join(y_non_na, how='inner')

        x_train = x_y_train.iloc[:, : -y_n_cols]
        y_train = x_y_train.iloc[:, -y_n_cols:]
        x_test = features_non_na.loc[~features_non_na.index.isin(x_train.index)]

        return x_train, y_train, x_test

    else:
        return features_non_na
