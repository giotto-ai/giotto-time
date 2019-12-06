from typing import Union, List

import numpy as np


def smape(y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]) \
        -> float:
    """Compute the 'Symmetric mean absolute percentage error' (SMAPE) between
    two vectors.

    Parameters
    ----------
    y_true : ``Union[List, np.ndarray]``, required.
        The first vector

    y_pred : ``Union[List, np.ndarray]``, required.
        The second vector

    Returns
    -------
    smape : ``float``
        The smape between the two input vectors.

    """
    if len(y_pred) != len(y_true):
        raise ValueError(f"The arrays must have the same length, but they "
                         f"have length {len(y_pred)} and {len(y_true)}.")

    non_normalized_smape = sum(np.abs(y_pred - y_true) /
                               (np.abs(y_pred) - np.abs(y_true)) )
    smape = (2 / len(y_pred)) * non_normalized_smape
    return smape


def max_error(y_true: Union[List, np.ndarray],
              y_pred: Union[List, np.ndarray]) -> float:

    """Compute the maximum error between two vectors.

    Parameters
    ----------
    y_true : ``Union[List, np.ndarray]``, required.
        The first vector

    y_pred : ``Union[List, np.ndarray]``, required.
        The second vector

    Returns
    -------
    max_error : ``float``
        The maximum error between the two vectors.

    """
    if len(y_pred) != len(y_true):
        raise ValueError(f"The arrays must have the same length, but they "
                         f"have length {len(y_pred)} and {len(y_true)}.")

    max_error = np.amax(np.absolute(np.subtract(y_true, y_pred)))
    return max_error
