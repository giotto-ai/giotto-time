from typing import Union, List

import numpy as np


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


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the 'Symmetric mean absolute percentage error' (SMAPE) between
    two vectors.

    Parameters
    ----------
    y_true : ``np.ndarray``, required.
        The first vector.

    y_pred : ``np.ndarray``, required.
        The second vector.

    Returns
    -------
    smape : ``float``
        The smape between the two input vectors.

    """
    _check_input(y_true, y_pred)

    non_normalized_smape = sum(
        np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))
    )
    non_normalized_smape_filled = np.nan_to_num(non_normalized_smape, nan=0)
    smape = (2 / len(y_pred)) * non_normalized_smape_filled
    return smape


def max_error(
    y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]
) -> float:

    """Compute the maximum error between two vectors.

    Parameters
    ----------
    y_true : ``Union[List, np.ndarray]``, required.
        The first vector

    y_pred : ``Union[List, np.ndarray]``, required.
        The second vector

    Returns
    -------
    error : ``float``
        The maximum error between the two vectors.

    """
    _check_input(y_true, y_pred)

    error = np.amax(np.absolute(np.subtract(y_true, y_pred)))
    return error
