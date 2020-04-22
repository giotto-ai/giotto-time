from typing import Union, List

from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from sklearn.utils.validation import check_is_fitted

from gtime.explainability import _LimeExplainer, _ShapExplainer


class ExplainableRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, estimator: RegressorMixin, explainer_type: str):
        self.estimator = self._check_estimator(estimator)
        self.explainer_type = explainer_type
        self.explainer = self._initialize_explainer()

    def _check_estimator(self, estimator: RegressorMixin) -> RegressorMixin:
        if not hasattr(estimator, 'fit') or not hasattr(estimator, 'predict'):
            raise TypeError(f'Estimator not compatible: {estimator}')
        return estimator

    def _initialize_explainer(self) -> Union[_LimeExplainer, _ShapExplainer]:
        if self.explainer_type == "lime":
            return _LimeExplainer()
        elif self.explainer_type == "shap":
            return _ShapExplainer()
        else:
            raise ValueError(f"Explainer not available: {self.explainer_type}")

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        self.estimator_ = self.estimator.fit(X, y)
        self.explainer_ = self.explainer.fit(self.estimator_, X, feature_names=feature_names)
        return self

    def predict(self, X: np.ndarray):
        check_is_fitted(self)
        return self.explainer_.predict(X)
