from typing import Dict, List

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.multioutput import (
    MultiOutputRegressor,
    _MultiOutputEstimator,
    _fit_estimator,
)
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted

from gtime.explainability.explainer import Explainer, _LimeExplainer, _ShapExplainer


class MultiFeatureMultiOutputRegressor(RegressorMixin, _MultiOutputEstimator):
    """ Multi target regression with option to choose the features for each target.

    This strategy consists of fitting one regressor per target. It is built over
    sklearn.multioutput.MultiOutputRegressor. Compared to this, it allows to choose
    different features for each regressor.

    Parameters
    ----------
    estimator: RegressorMixin, required
        An estimator object implementing fit and predict.

    Examples
    --------
    >>> import numpy as np
    >>> from gtime.regressors import MultiFeatureMultiOutputRegressor
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> X = np.random.random((30, 5))
    >>> y = np.random.random((30, 3))
    >>> X_train, y_train = X[:20], y[:20]
    >>> X_test, y_test = X[20:], y[20:]
    >>>
    >>> random_forest = RandomForestRegressor()
    >>> regressor = MultiFeatureMultiOutputRegressor(estimator=random_forest)
    >>>
    >>> target_to_features_dict = {0: [0,1,2], 1: [0,1,3], 2: [0,1,4]}
    >>> regressor.fit(X_train, y_train, target_to_features_dict=target_to_features_dict)
    >>>
    >>> predictions = regressor.predict(X_test)
    >>> predictions.shape
    (10, 3)

    """

    def __init__(
        self,
        estimator: RegressorMixin,
        target_to_features_dict: Dict[int, List[int]] = None,
    ):
        super().__init__(estimator=estimator, n_jobs=1)
        self.target_to_features_dict = target_to_features_dict

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit the model.

        Train the models, one for each target variable in y.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features), required.
            The data.
        y : np.ndarray, shape (n_samples, horizon), required.
            The matrix containing the target variables.

        Returns
        -------
        self : object


        """
        if self.target_to_features_dict is None:
            super().fit(X, y)
            self.target_to_features_dict_ = None
            return self

        X, y = check_X_y(X, y, multi_output=True, accept_sparse=True)

        if y.ndim == 1:
            raise ValueError("y must have at least two dimensions")

        self.estimators_ = [
            _fit_estimator(
                self.estimator, X[:, self.target_to_features_dict[i]], y[:, i]
            )
            for i in range(y.shape[1])
        ]
        self.target_to_features_dict_ = self.target_to_features_dict
        self.expected_X_shape_ = X.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """For each row in ``X``, make a prediction for each fitted model

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features), required
            The data.

        Returns
        -------
        predictions : np.ndarray, shape (n_samples, horizon)
            The predictions

        """
        check_is_fitted(self)
        if self.target_to_features_dict_ is None:
            return super().predict(X)

        X = check_array(X, accept_sparse=True)
        if X.shape[1] != self.expected_X_shape_:
            raise ValueError(
                f"Expected X shape is {self.expected_X_shape_}. Detected {X.shape[1]}"
            )
        y = [
            estimator.predict(X[:, self.target_to_features_dict_[i]])
            for i, estimator in enumerate(self.estimators_)
        ]

        return np.asarray(y).T
