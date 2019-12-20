from typing import Union, List

import numpy as np
import pandas as pd


def _check_input(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if len(y_pred) != len(y_true):
        raise ValueError(
            f"The arrays must have the same length, but they "
            f"have length {len(y_pred)} and {len(y_true)}."
        )
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError(
            "The two arrays should not contain Nan values, but they are "
            f"{y_true}, {y_pred}."
        )

    if np.isinf(y_true).any() or np.isinf(y_pred).any():
        raise ValueError(
            "The two arrays should not contain Inf values, but they are "
            f"{y_true}, {y_pred}."
        )


def _convert_to_ndarray(
    y_true: Union[pd.DataFrame, List, np.ndarray],
    y_pred: Union[pd.DataFrame, List, np.ndarray],
):
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    elif isinstance(y_true, List):
        y_true = np.array(y_true)

    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values
    elif isinstance(y_pred, List):
        y_pred = np.array(y_pred)

    return y_true, y_pred


def smape(
    y_true: Union[pd.DataFrame, List, np.ndarray],
    y_pred: Union[pd.DataFrame, List, np.ndarray],
) -> float:
    """Compute the 'Symmetric mean absolute percentage error' (SMAPE) between two
    vectors.

    Parameters
    ----------
    y_true : Union[pd.DataFrame, List, np.ndarray], shape (length, 1), required
        The first vector.

    y_pred : Union[pd.DataFrame, List, np.ndarray], shape (length, 1), required
        The second vector.

    Returns
    -------
    smape : float
        The smape between the two input vectors.

    Examples
    --------
    >>> from giottotime.loss_functions import smape
    >>> y_true = [0, 1, 2, 3, 4, 5]
    >>> y_pred = [1.1, 2.3, 0.4, 3.9, 3.1, 4.6]
    >>> smape(y_true, y_pred)
    0.7864893577539014

    """
    y_true, y_pred = _convert_to_ndarray(y_true, y_pred)
    _check_input(y_true, y_pred)

    non_normalized_smape = sum(
        np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))
    )
    non_normalized_smape_filled = np.nan_to_num(non_normalized_smape, nan=0)
    smape = (2 / len(y_pred)) * non_normalized_smape_filled
    return smape


def max_error(
    y_true: Union[pd.DataFrame, List, np.ndarray],
    y_pred: Union[pd.DataFrame, List, np.ndarray],
) -> float:
    """Compute the maximum error between two vectors.

    Parameters
    ----------
    y_true : Union[pd.DataFrame, List, np.ndarray], shape (length, 1), required
        The first vector.

    y_pred : Union[pd.DataFrame, List, np.ndarray], shape (length, 1), required
        The second vector.

    Returns
    -------
    error : float
        The maximum error between the two vectors.

    Examples
    --------
    >>> from giottotime.loss_functions import max_error
    >>> y_true = [0, 1, 2, 3, 4, 5]
    >>> y_pred = [1.1, 2.3, 0.4, 3.9, 3.1, 4.6]
    >>> max_error(y_true, y_pred)
    1.6

    """
    y_true, y_pred = _convert_to_ndarray(y_true, y_pred)
    _check_input(y_true, y_pred)

    error = np.amax(np.absolute(np.subtract(y_true, y_pred)))
    return error
