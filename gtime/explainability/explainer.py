from abc import abstractmethod
from typing import List, Tuple, Dict

import numpy as np
import shap
from lime import lime_tabular
from lime.explanation import Explanation
from shap.explainers._explainer import Explainer
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
    """ LIME explainer for the predictions of a scikit-learn regressor.
    It is built around lime_tabular.LimeTabularExplainer.
    This class is not supposed to be used directly by the user. Use `ExplainableRegressor` instead.

    The choice of a superset of LimeTabularExplainer has been made to provide a unique interface, compatible
    also with the SHAP explainer.

    This is used as a backbone of gtime.regressors.ExplainableRegressor.

    Examples
    --------
    >>> import numpy as np
    >>> from gtime.explainability import _LimeExplainer
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> X = np.random.random((30, 5))
    >>> y = np.random.random(30)
    >>> X_train, y_train = X[:20], y[:20]
    >>> X_test, y_test = X[20:], y[20:]
    >>>
    >>> random_forest = RandomForestRegressor()
    >>> random_forest.fit(X, y)
    >>> explainer = _LimeExplainer()
    >>>
    >>> explainer.fit(random_forest, X_train, feature_names=['a', 'b', 'c', 'd', 'e'])
    >>> explainer.predict(X_test)
    array([0.52152573, 0.66515405, 0.49843039, 0.36233978, 0.48702524,
           0.49387234, 0.50727344, 0.74125629, 0.73243989, 0.59242504])
    >>> explainer.explanations_[0]
    {'d': -0.10406889434277307, 'c': 0.07973507022816899, 'b': 0.02312395991550859, 'a': 0.006403509251399996, 'e': 0.006272607738125953}
    """

    def fit(
        self, model: RegressorMixin, X: np.ndarray, feature_names: List[str] = None
    ):
        """ Fit function. The initialization of `LimeTabular` is made here.
        This choice has been made, since it needs a fitted scikit-learn model as input.

        Parameters
        ----------
        model: RegressorMixin, required
            scikit-learn model given as input
        X: np.ndarray, required
            train matrix
        feature_names: List[str], optional, (default=``None``)
            the name of the feature column of X

        Returns
        -------
        Fitted _LimeExplainer
        """
        check_is_fitted(model)
        if feature_names is None:
            feature_names = self._define_feature_names(X)
        else:
            feature_names = list(feature_names)

        self.model_ = model
        self.explainer_ = lime_tabular.LimeTabularExplainer(
            X, feature_names=feature_names, mode="regression"
        )
        self.feature_names_ = feature_names
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Predict function. It returns the predictions of the model given as input in fit()
        It stores in  `self.explanations_` a float value per each feature that gives an indication of the
        impact of that feature on the predictions.

        Parameters
        ----------
        X: np.ndarray, required
            test matrix

        Returns
        -------
        predictions: np.ndarray
        """
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
    """ SHAP explainer for the predictions of a scikit-learn regressor.
    It is built around `shap.Explainer`. The choice of the explainer is made automatically based on
    the scikit-learn model.
    This class is not supposed to be used directly by the user. Use `ExplainableRegressor` instead.

    The choice of a superset of shap.Explainer has been made to provide a unique interface, compatible
    also with the LIME explainer.

    This is used as a backbone of gtime.regressors.ExplainableRegressor.

    Examples
    --------
    >>> import numpy as np
    >>> from gtime.explainability import _ShapExplainer
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> X = np.random.random((30, 5))
    >>> y = np.random.random(30)
    >>> X_train, y_train = X[:20], y[:20]
    >>> X_test, y_test = X[20:], y[20:]
    >>>
    >>> random_forest = RandomForestRegressor()
    >>> random_forest.fit(X, y)
    >>> explainer = _ShapExplainer()
    >>>
    >>> explainer.fit(random_forest, X_train, feature_names=['a', 'b', 'c', 'd', 'e'])
    >>> explainer.predict(X_test)
    array([0.52152573, 0.66515403, 0.4984304 , 0.36233976, 0.48702522,
           0.49387233, 0.50727345, 0.74125631, 0.7324399 , 0.59242503])
    >>> explainer.explanations_[0]
    {'a': -0.04801958111784188, 'b': -0.03872758470097324, 'c': -0.09502661560291017, 'd': 0.07797206658942742, 'e': -0.015306016394606558}
    """

    allowed_explainer = [shap.LinearExplainer, shap.TreeExplainer]

    def fit(
        self, model: RegressorMixin, X: np.ndarray, feature_names: List[str] = None
    ):
        """ Fit function. The initialization of `shap.Explainer` is made here.
        This choice has been made, since it needs a fitted scikit-learn model as input.

        Parameters
        ----------
        model: RegressorMixin, required
           scikit-learn model given as input
        X: np.ndarray, required
           train matrix
        feature_names: List[str], optional, (default=``None``)
           the name of the feature column of X

        Returns
        -------
        Fitted _ShapExplainer
        """
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
        """ Predict function. It returns the predictions of the model given as input in fit()
        It stores in  `self.explanations_` a float value per each feature that gives an indication of the
        impact of that feature on the predictions.

        Parameters
        ----------
        X: np.ndarray, required
            test matrix

        Returns
        -------
        predictions: np.ndarray
        """
        check_is_fitted(self)

        try:
            self.shap_values_ = self.explainer_.shap_values(X, check_additivity=False)
        except TypeError:
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
                return explainer(model, X, feature_names=feature_names)
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
