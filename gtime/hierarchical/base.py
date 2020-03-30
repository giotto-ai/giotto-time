from abc import abstractmethod
from typing import Any, Dict, Union

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class HierarchicalBase(BaseEstimator, RegressorMixin):
    """ Base class for hierarchical models.

    Parameters
    ----------
    model : BaseEstimator, required
        base model applied to all the time series
    hierarchy_tree: Union[str, Dict[str, Any]], optional, default = ``'infer'``
        hierarchy structure between time series. If 'infer' a standard structure if inferred. It
        depends on the subclass the implementation of infer.
    """

    def __init__(
        self, model: BaseEstimator, hierarchy_tree: Union[str, Dict[str, Any]] = "infer"
    ):
        self.model = model
        self.hierarchy_tree = hierarchy_tree

    @abstractmethod
    def fit(self, X: Dict[str, pd.DataFrame], y=None):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Dict[str, pd.DataFrame] = None):
        raise NotImplementedError

    @staticmethod
    def _check_is_dict_of_dataframes_with_str_key(X: Any):
        if not isinstance(X, dict):
            raise ValueError(
                f"X must be a dictionary of pd.DataFrame. Detected: {type(X)}"
            )
        if not all(isinstance(key, str) for key in X):
            raise ValueError("All X keys must be string")
        if not all(isinstance(df, pd.DataFrame) for df in X.values()):
            raise ValueError("All values of X must be pd.DataFrame")
