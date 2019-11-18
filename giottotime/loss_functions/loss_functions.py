import numpy as np

def smape(y_true, y_pred):
    if len(y_pred) != len(y_true):
        raise ValueError('{len_pred} != {len_true}'.format(len_pred=len(y_pred), len_true=len(y_true)))

    non_normalized_smape = sum(np.abs(y_pred - y_true) / np.abs(y_pred) - np.abs(y_true))
    return (2 / len(y_pred)) * non_normalized_smape

def max_error(y_true, y_pred):
    if len(y_pred) != len(y_true):
        raise ValueError('{len_pred} != {len_true}'.format(len_pred=len(y_pred), len_true=len(y_true)))

    return np.amax( np.absolute(np.subtract(y_true, y_pred)) )



#
