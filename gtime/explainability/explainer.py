from abc import abstractmethod
from typing import List, Tuple, Dict

import numpy as np
import shap
from lime import lime_tabular
from lime.explanation import Explanation
from shap.explainers.explainer import Explainer
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted


class _RegressorExplainer:
    @abstractmethod
    def fit(
        self, model: RegressorMixin, X: np.ndarray, feature_names: List[str] = None
    ):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def plot_explanation(self, i: int):
        raise NotImplementedError

    def _define_feature_names(self, X: np.ndarray):
        return [f"{i}" for i in range(X.shape[1])]


class _LimeExplainer(_RegressorExplainer):
    def fit(
        self, model: RegressorMixin, X: np.ndarray, feature_names: List[str] = None
    ):
        check_is_fitted(model)
        if feature_names is None:
            feature_names = self._define_feature_names(X)

        self.model_ = model
        self.explainer_ = lime_tabular.LimeTabularExplainer(
            X, feature_names=feature_names, mode="regression"
        )
        self.feature_names_ = feature_names
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)

        self._explanations_ = self._compute_lime_explanations(X)
        predictions = self._extract_predictions_from_explanations(self._explanations_)
        self.explanations_ = self._reformat_explanations(self._explanations_)

        return predictions

    def plot_explanation(self, i: int):
        raise NotImplementedError
        # self._explanations_[i].show_in_notebook(show_tale=True)

    def _compute_lime_explanations(self, X: np.ndarray) -> List[Explanation]:
        return [
            self.explainer_.explain_instance(item, self.model_.predict) for item in X
        ]

    def _extract_predictions_from_explanations(
        self, explanations: List[Explanation]
    ) -> np.ndarray:
        return np.asarray([exp.predicted_value for exp in explanations])

    def _reformat_explanations(
        self, explanations: List[Explanation]
    ) -> List[Dict[str, float]]:
        explanations_lists = [
            exp.as_map()[0] for exp in explanations
        ]  # not clear why 0
        return [
            {self.feature_names_[name]: value for name, value in exp}
            for exp in explanations_lists
        ]


class _ShapExplainer(_RegressorExplainer):
    def __init__(self):
        self.allowed_explainer = [shap.LinearExplainer, shap.TreeExplainer]

    def fit(
        self, model: RegressorMixin, X: np.ndarray, feature_names: List[str] = None
    ):
        check_is_fitted(model)
        if feature_names is None:
            feature_names = self._define_feature_names(X)

        self.model_ = model
        self.explainer_ = self._infer_explainer(
            model=model, X=X, feature_names=feature_names
        )
        self.feature_names_ = feature_names
        return self

    def predict(self, X: np.ndarray):
        check_is_fitted(self)

        self.shap_values_ = self.explainer_.shap_values(X)
        predictions = self._compute_predictions_from_shap_values(self.shap_values_)
        self.explanations_ = self._reformat_shap_values(self.shap_values_)
        return predictions

    def plot_explanation(self, i: int):
        raise NotImplementedError

    def _infer_explainer(
        self, model: RegressorMixin, X: np.ndarray, feature_names: List[str]
    ) -> Explainer:
        for explainer in self.allowed_explainer:
            try:
                return explainer(model=model, data=X, feature_names=feature_names)
            except Exception:
                pass
        raise ValueError(f"Model not available: {type(model)}")

    def _compute_predictions_from_shap_values(
        self, shap_values: np.ndarray
    ) -> np.ndarray:
        return np.sum(shap_values, axis=1) + self.explainer_.expected_value

    def _reformat_shap_values(self, shap_values: np.ndarray) -> List[Dict[str, float]]:
        return [
            {
                feature: shap_value
                for feature, shap_value in zip(self.feature_names_, shap_values_row)
            }
            for shap_values_row in shap_values
        ]
