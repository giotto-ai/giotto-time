from typing import Union, List

from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from sklearn.utils.validation import check_is_fitted

from gtime.explainability import _LimeExplainer, _ShapExplainer


class ExplainableRegressor(BaseEstimator, RegressorMixin):
    """ Wraps the most commons scikit-learn regressor to offer a nice to use interface to fit/predict
    models and at the same time to explain the predictions.

    Since it follows the fit/predict interface of scikit-learn model it is compatible with
    scikit-learn pipelines, etc..

    2 explainers are available: LIME and SHAP

    You can get the explanation by accessing to `regressor.explainer_.explanations_` after
    the predict function,

    Parameters
    ----------
    estimator: RegressorMixin, required
        the scikit-learn model
    explainer_type: str, required
        'lime' or 'shap'

    Examples
    --------
    >>> import numpy as np
    >>> from gtime.regressors import ExplainableRegressor
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> X = np.random.random((30, 5))
    >>> y = np.random.random(30)
    >>> X_train, y_train = X[:20], y[:20]
    >>> X_test, y_test = X[20:], y[20:]
    >>>
    >>> random_forest = RandomForestRegressor()
    >>> explainable_regressor = ExplainableRegressor(random_forest, 'shap')
    >>>
    >>> explainable_regressor.fit(X_train, y_train, feature_names=['a', 'b', 'c', 'd', 'e'])
    >>> explainable_regressor.predict(X_test)
    array([0.41323105, 0.40386639, 0.46462663, 0.3795568 , 0.57571486,
           0.37079003, 0.54756082, 0.35160197, 0.30881165, 0.48201442])
    >>> explainable_regressor.explainer_.explanations_[0]
    {'a': -0.019896434698603117, 'b': 0.029814649814215954, 'c': 0.02447547087613202, 'd': 0.021313815648682066, 'e': -0.10778800140251406}
    """

    def __init__(self, estimator: RegressorMixin, explainer_type: str):
        self.estimator = self._check_estimator(estimator)
        self.explainer_type = explainer_type
        self.explainer = self._initialize_explainer()

    def _check_estimator(self, estimator: RegressorMixin) -> RegressorMixin:
        if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
            raise TypeError(f"Estimator not compatible: {estimator}")
        return estimator

    def _initialize_explainer(self) -> Union[_LimeExplainer, _ShapExplainer]:
        if self.explainer_type == "lime":
            return _LimeExplainer()
        elif self.explainer_type == "shap":
            return _ShapExplainer()
        else:
            raise ValueError(f"Explainer not available: {self.explainer_type}")

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """ Fit function that calls the fit on the estimator and on the explainer.

        Parameters
        ----------
        X: np.ndarray, required
            train matrix
        y: np.ndarray, required
            train true values
        feature_names: List[str], optional, (default=`None`)
            the name of the feature column of X

        Returns
        -------
        Fitted `ExplainableRegressor`
        """
        self.estimator_ = self.estimator.fit(X, y)
        self.explainer_ = self.explainer.fit(
            self.estimator_, X, feature_names=feature_names
        )
        return self

    def predict(self, X: np.ndarray):
        """ Predict function that call the predict function of the explainer.

        You can access to the explanation of the predictions via
        `regressor.explainer_.explanations_` attribute

        Parameters
        ----------
        X: np.ndarray, required
            test matrix

        Returns
        -------
        predictions: np.ndarray
        """
        check_is_fitted(self)
        return self.explainer_.predict(X)
