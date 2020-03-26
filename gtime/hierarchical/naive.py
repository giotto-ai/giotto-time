from copy import deepcopy
from typing import Dict

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from gtime.hierarchical.base import HierarchicalBase


class HierarchicalNaive(HierarchicalBase):
    def __init__(self, model: BaseEstimator):
        super().__init__(model=model, hierarchy_tree="infer")

    def fit(self, X: Dict[str, pd.DataFrame], y=None):
        self._infer_hierarchy_tree(X)
        self._initialize_models(X)
        for key, time_series in X.items():
            self.models_[key].fit(time_series)

    def predict(self, X: Dict[str, pd.DataFrame] = None):
        check_is_fitted(self)
        if X is None:
            return {key: model.predict() for key, model in self.models_.items()}
        else:
            return {
                key: self.models_[key].predict(time_series)
                for key, time_series in X.items()
            }

    def _initialize_models(self, X: Dict[str, pd.DataFrame]):
        self.models_ = {key: deepcopy(self.model) for key in X}

    def _infer_hierarchy_tree(self, X: Dict[str, pd.DataFrame]):
        self.hierarchy_tree_ = set(
            X.keys()
        )  # No need of a proper hierarchy tree for HierarchicalNaive
