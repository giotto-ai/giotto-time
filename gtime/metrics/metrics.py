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
    """Compute the 'Symmetric Mean Absolute Percentage Error' (SMAPE) between two
    vectors. Documentation
    `here <https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error>_`.

    Parameters
    ----------
    y_true : array-like, shape (length, 1), required
        The first vector.

    y_pred : array-like, shape (length, 1), required
        The second vector.

    Returns
    -------
    smape : float
        The smape between the two input vectors.

    Examples
    --------
    >>> from gtime.metrics import smape
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
    y_true : array-like, shape (length, 1), required
        The first vector.

    y_pred : array-like, shape (length, 1), required
        The second vector.

    Returns
    -------
    error : float
        The maximum error between the two vectors.

    Examples
    --------
    >>> from gtime.metrics import max_error
    >>> y_true = [0, 1, 2, 3, 4, 5]
    >>> y_pred = [1.1, 2.3, 0.4, 3.9, 3.1, 4.6]
    >>> max_error(y_true, y_pred)
    1.6

    """
    y_true, y_pred = _convert_to_ndarray(y_true, y_pred)
    _check_input(y_true, y_pred)

    error = np.amax(np.absolute((y_true - y_pred)))
    return error


def mse(
    y_true: Union[pd.DataFrame, List, np.ndarray],
    y_pred: Union[pd.DataFrame, List, np.ndarray],
    rmse = False,
) -> float:
    """Compute the Mean Squared Error(MSE) or Root Mean Squared Error(RMSE) between two vectors.

    Parameters
    ----------
    y_true : array-like, shape (length, 1), required.
        The first vector.

    y_pred : array-like, shape (length, 1), required.
        The second vector.

    rmse : boolean, default is False.
        To be set to True if Root Mean Squared Error is required

    Returns
    -------
    mse : float
        The Mean Squared Error between the two vectors.
        If rmse is True then the function returns Root Mean Squared Error

    Examples
    --------
    >>> from gtime.metrics import mse
    >>> y_true = [0, 1, 2, 3, 4, 5]
    >>> y_pred = [1.1, 2.3, 0.4, 3.9, 3.1, 4.6]
    >>> mse(y_true, y_pred)
    1.20

    >>> from gtime.metrics import mse
    >>> y_true = [0, 1, 2, 3, 4, 5]
    >>> y_pred = [1.1, 2.3, 0.4, 3.9, 3.1, 4.6]
    >>> mse(y_true, y_pred, rmse=True)
    1.098

    """
    y_true, y_pred = _convert_to_ndarray(y_true, y_pred)
    _check_input(y_true, y_pred)
    
    sum_squared_error = sum((y_true - y_pred) ** 2)
    mse = sum_squared_error / len(y_true)
    return np.sqrt(mse) if rmse else mse


def log_mse(
    y_true: Union[pd.DataFrame, List, np.ndarray],
    y_pred: Union[pd.DataFrame, List, np.ndarray],
    rmsle = False,
) -> float:
    """Compute the Mean Squared Log Error(MSLE) or Root Mean Squared Log Error(RMSLE) between two vectors.
    Note: Log_mse accepts only positive numbers as input

    Parameters
    ----------
    y_true : array-like, shape (length, 1), required
        The first vector.

    y_pred : array-like, shape (length, 1), required
        The second vector.
    
    rmsle : boolean, default is False.
        To be set to True if Root Mean Squared Log Error is required

    Returns
    -------
    log_mse : float
        The mean squared log error between the two vectors.
        If rmsle is True then the function returns Root Mean Squared Log Error

    Examples
    --------
    >>> from gtime.metrics import log_mse
    >>> y_true = [0, 1, 2, 3, 4, 5]
    >>> y_pred = [1.1, 2.3, 0.4, 3.9, 3.1, 4.6]
    >>> log_mse(y_true, y_pred)
    0.244

    >>> from gtime.metrics import log_mse
    >>> y_true = [0, 1, 2, 3, 4, 5]
    >>> y_pred = [1.1, 2.3, 0.4, 3.9, 3.1, 4.6]
    >>> log_mse(y_true, y_pred, rmsle=True)
    0.49

    """
    y_true, y_pred = _convert_to_ndarray(y_true, y_pred)
    _check_input(y_true, y_pred)
    
    if (np.any(y_true < 0)) or (np.any(y_pred < 0)):
        raise ValueError("MSLE can not be used when inputs contain Negative values") 
    log_y_true = np.log(y_true + 1)
    log_y_pred = np.log(y_pred + 1)
    log_mse = mse(log_y_true, log_y_pred)
    return np.sqrt(log_mse) if rmsle else log_mse


def r_square(
    y_true: Union[pd.DataFrame, List, np.ndarray],
    y_pred: Union[pd.DataFrame, List, np.ndarray],
) -> float:
    """Compute the R squared (Coefficient of Determination) between two vectors.

    Parameters
    ----------
    y_true : array-like, shape (length, 1), required
        The first vector.

    y_pred : array-like, shape (length, 1), required
        The second vector.

    Returns
    -------
    r_square : float
        The R squared between the two vectors.

    Examples
    --------
    >>> from gtime.metrics import r_square
    >>> y_true = [0, 1, 2, 3, 4, 5]
    >>> y_pred = [1.1, 2.3, 0.4, 3.9, 3.1, 4.6]
    >>> r_square(y_true, y_pred)
    0.586

    """
    y_true, y_pred = _convert_to_ndarray(y_true, y_pred)
    _check_input(y_true, y_pred)
    
    ss_res = sum((y_true - y_pred) ** 2)
    ss_tot = sum((y_true - np.mean(y_true)) ** 2)
    if not np.any(ss_tot):
        if not np.any(ss_res):
            return 1.0
        else:
            return 0.0 
    if np.isnan(ss_res / ss_tot):
        return np.NINF
    r_square = 1 - (ss_res / ss_tot)
    return r_square


def mae(
    y_true: Union[pd.DataFrame, List, np.ndarray],
    y_pred: Union[pd.DataFrame, List, np.ndarray],
) -> float:
    """Compute the Mean Absolute Error(also called, Mean Absolute Deviation(MAD)) between two vectors.

    Parameters
    ----------
    y_true : array-like, shape (length, 1), required
        The first vector.

    y_pred : array-like, shape (length, 1), required
        The second vector.

    Returns
    -------
    mae_value : float
        The mean absolute error between the two vectors.

    Examples
    --------
    >>> from gtime.metrics import mae
    >>> y_true = [0, 1, 2, 3, 4, 5]
    >>> y_pred = [1.1, 2.3, 0.4, 3.9, 3.1, 4.6]
    >>> mae(y_true, y_pred)
    1.033

    """
    y_true, y_pred = _convert_to_ndarray(y_true, y_pred)
    _check_input(y_true, y_pred)
    
    mae_value = np.mean(np.abs(y_pred - y_true))
    return mae_value


def mape(
    y_true: Union[pd.DataFrame, List, np.ndarray],
    y_pred: Union[pd.DataFrame, List, np.ndarray],
) -> float:
    """Compute the Mean Absolute Percentage Error(MAPE) between two vectors.

    Parameters
    ----------
    y_true : array-like, shape (length, 1), required.
        The first vector.

    y_pred : array-like, shape (length, 1), required.
        The second vector.

    Returns
    -------
    mape_value : float
        The mean absolute percentage error between the two vectors.

    Examples
    --------
    >>> from gtime.metrics import mape
    >>> y_true = [1, 1, 2, 3, 4, 5]
    >>> y_pred = [1, 2.3, 0.4, 3.9, 3.1, 4.6]
    >>> mape(y_true, y_pred)
    45.08

    """
    y_true, y_pred = _convert_to_ndarray(y_true, y_pred)
    _check_input(y_true, y_pred)
    
    ratio_list = np.abs((y_pred - y_true)/y_true)
    if (y_true == 0).any():
        if (np.nan == ratio_list).any():
            raise ValueError("MAPE can not be calculated due to Zero/Zero")
        else:
            return np.inf
    else:
        mape_value = np.mean(ratio_list) * 100
    return mape_value
