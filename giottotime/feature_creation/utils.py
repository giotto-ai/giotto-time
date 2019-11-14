import numpy as np
import pandas as pd


def split_train_test(x, y, split_percentage=0.6):
    train_index = int(split_percentage * len(x))

    x_train = x.iloc[:train_index]
    y_train = y.iloc[:train_index]

    x_test = x.iloc[train_index:]
    y_test = y.iloc[train_index:]

    return x_train, y_train, x_test, y_test


def get_non_nan_values(x: pd.DataFrame, y: pd.DataFrame):
    x_non_nans = x.dropna(axis='index', how='any')
    y_non_nans = y.loc[x_non_nans.index]

    return x_non_nans, y_non_nans
