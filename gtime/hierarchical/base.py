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
