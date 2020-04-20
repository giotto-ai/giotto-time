from typing import Dict, List

import pandas as pd
from sklearn.base import RegressorMixin, is_classifier
from sklearn.multioutput import (
    MultiOutputRegressor,
    RegressorChain,
    _MultiOutputEstimator,
    _fit_estimator,
)
import numpy as np
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted


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
    >>> from gtime.forecasting import MultiFeatureMultiOutputRegressor
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

    def __init__(self, estimator: RegressorMixin):
        super().__init__(estimator=estimator, n_jobs=1)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_to_features_dict: Dict[int, List[int]] = None,
    ):
        """Fit the model.

        Train the models, one for each target variable in y.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features), required.
            The data.
        y : np.ndarray, shape (n_samples, horizon), required.
            The matrix containing the target variables.
        target_to_features_dict: Dict[int, List[int]], optional, (default=``None``)
            dictionary that assign to each target column the feature columns to be used for training.
            If None, all the features are used for each target.

        Returns
        -------
        self : object


        """
        if target_to_features_dict is None:
            super().fit(X, y)
            self.target_to_features_dict_ = None
            return self

        X, y = check_X_y(X, y, multi_output=True, accept_sparse=True)

        if y.ndim == 1:
            raise ValueError("y must have at least two dimensions")

        self.estimators_ = [
            _fit_estimator(self.estimator, X[:, target_to_features_dict[i]], y[:, i])
            for i in range(y.shape[1])
        ]
        self.target_to_features_dict_ = target_to_features_dict
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


class GAR(MultiOutputRegressor):
    """Generalized Auto Regression model.

    This model is a wrapper of ``sklearn.multioutput.MultiOutputRegressor`` but returns
    a ``pd.DataFrame``.

    Fit one model for each target variable contained in the ``y`` matrix.

    Parameters
    ----------
    estimator : estimator object, required
        The model used to make the predictions step by step. Regressor object such as
        derived from ``RegressorMixin``.

    n_jobs : int, optional, default: ``None``
        The number of jobs to use for the parallelization.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from gtime.forecasting import GAR
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> time_index = pd.date_range("2020-01-01", "2020-01-30")
    >>> X = pd.DataFrame(np.random.random((30, 5)), index=time_index)
    >>> y_columns = ["y_1", "y_2", "y_3"]
    >>> y = pd.DataFrame(np.random.random((30, 3)), index=time_index, columns=y_columns)
    >>> X_train, y_train = X[:20], y[:20]
    >>> X_test, y_test = X[20:], y[20:]
    >>> random_forest = RandomForestRegressor()
    >>> gar = GAR(estimator=random_forest)
    >>> gar.fit(X_train, y_train)
    >>> predictions = gar.predict(X_test)
    >>> predictions.shape
    (10, 3)

    """

    def __init__(self, estimator, n_jobs: int = None):
        super().__init__(estimator, n_jobs)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight=None):
        """Fit the model.

        Train the models, one for each target variable in y.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), required.
            The data.

        y : pd.DataFrame, shape (n_samples, horizon), required.
            The matrix containing the target variables.

        Returns
        -------
        self : object

        """
        self._y_columns = y.columns
        return super().fit(X, y, sample_weight)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """For each row in ``X``, make a prediction for each fitted model, from 1 to
        ``horizon``.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), required
            The data.

        Returns
        -------
        y_p_df : pd.DataFrame, shape (n_samples, horizon)
            The predictions, one for each timestep in horizon.

        """
        y_p = super().predict(X)
        y_p_df = pd.DataFrame(data=y_p, columns=self._y_columns, index=X.index)

        return y_p_df


# TODO: See #99
class GARFF(RegressorChain):
    """Generalized Auto Regression model with feedforward training. This model is a
    wrapper of ``sklearn.multioutput.RegressorChain`` but returns  a ``pd.DataFrame``.

    Fit one model for each target variable contained in the ``y`` matrix, also using the
    predictions of the previous model.

    Parameters
    ----------
    estimator : estimator object, required
        The model used to make the predictions step by step. Regressor object such as
        derived from ``RegressorMixin``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from gtime.forecasting import GARFF
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> time_index = pd.date_range("2020-01-01", "2020-01-30")
    >>> X = pd.DataFrame(np.random.random((30, 5)), index=time_index)
    >>> y_columns = ["y_1", "y_2", "y_3"]
    >>> y = pd.DataFrame(np.random.random((30, 3)), index=time_index, columns=y_columns)
    >>> X_train, y_train = X[:20], y[:20]
    >>> X_test, y_test = X[20:], y[20:]
    >>> random_forest = RandomForestRegressor()
    >>> garff = GARFF(estimator=random_forest)
    >>> garff.fit(X_train, y_train)
    >>> predictions = garff.predict(X_test)
    >>> predictions.shape
    (10, 3)

    Notes
    -----
    ``sklearn.multioutput.RegressorChain`` order, cv and random_state parameters were
    set to None due to target order importance in a time-series forecasting context.

    """

    def __init__(self, estimator):
        super().__init__(
            base_estimator=estimator, order=None, cv=None, random_state=None
        )

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Fit the models, one for each time step. Each model is trained on the initial
        set of features and on the true values of the previous steps.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), required
            The data.

        y : pd.DataFrame, shape (n_samples, horizon), required
            The matrix containing the target variables.

        Returns
        -------
        self : object
            The fitted object.

        """
        self._y_columns = y.columns
        return super().fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """For each row in ``X``, make a prediction for each fitted model, from 1 to
        ``horizon``.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            The data.

        Returns
        -------
        y_p_df : pd.DataFrame, shape (n_samples, horizon)
            The predictions, one for each timestep in horizon.

        """
        y_p = super().predict(X)
        y_p_df = pd.DataFrame(data=y_p, columns=self._y_columns, index=X.index)

        return y_p_df


class MultiFeatureGAR(MultiFeatureMultiOutputRegressor):
    """Generalized Auto Regression model.

    This model is a wrapper of ``MultiFeatureMultiOutputRegressor`` but returns
    a ``pd.DataFrame``.

    Fit one model for each target variable contained in the ``y`` matrix. You can select
    the feature columns to use for each model

    Parameters
    ----------
    estimator : estimator object, required
        The model used to make the predictions step by step. Regressor object such as
        derived from ``RegressorMixin``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from gtime.forecasting import MultiFeatureGAR
    >>> from sklearn.ensemble import RandomForestRegressor
    >>>
    >>> time_index = pd.date_range("2020-01-01", "2020-01-30")
    >>> X_columns = ['c1', 'c2', 'c3', 'c4', 'c5']
    >>> X = pd.DataFrame(np.random.random((30, 5)), index=time_index, columns=X_columns)
    >>> y_columns = ["y_1", "y_2", "y_3"]
    >>> y = pd.DataFrame(np.random.random((30, 3)), index=time_index, columns=y_columns)
    >>> X_train, y_train = X[:20], y[:20]
    >>> X_test, y_test = X[20:], y[20:]
    >>>
    >>> random_forest = RandomForestRegressor()
    >>> gar = MultiFeatureGAR(estimator=random_forest)
    >>>
    >>> target_to_features_dict = {'y_1': ['c1','c2','c3'], 'y_2': ['c1','c2','c4'], 'y_3': ['c1','c2','c5']}
    >>> gar.fit(X_train, y_train, target_to_features_dict)
    >>>
    >>> predictions = gar.predict(X_test)
    >>> predictions.shape
    (10, 3)

    """

    def __init__(self, estimator: RegressorMixin):
        super().__init__(estimator)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        target_to_features_dict: Dict[str, List[str]] = None,
    ):
        """Fit the model.

        Train the models, one for each target variable in y.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), required.
            The data.
        y : pd.DataFrame, shape (n_samples, horizon), required.
            The matrix containing the target variables.

        Returns
        -------
        self : object

        """
        self.X_columns_ = X.columns
        self.y_columns_ = y.columns
        if target_to_features_dict is not None:
            target_to_features_dict = self._feature_name_to_index(
                target_to_features_dict, X.columns, y.columns
            )
        return super().fit(X.values, y.values, target_to_features_dict)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """For each row in ``X``, make a prediction for each fitted model, from 1 to
        ``horizon``.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), required
            The data.

        Returns
        -------
        y_p_df : pd.DataFrame, shape (n_samples, horizon)
            The predictions, one for each timestep in horizon.

        """
        check_is_fitted(self)
        self._check_X_columns(X)
        y_p = super().predict(X.values)
        y_p_df = pd.DataFrame(data=y_p, columns=self.y_columns_, index=X.index)

        return y_p_df

    @staticmethod
    def _feature_name_to_index(
        target_to_features_dict: Dict[str, List[str]],
        X_columns: List[str],
        y_columns: List[str],
    ):
        X_str_to_int = {column: i for i, column in enumerate(X_columns)}
        y_str_to_int = {column: i for i, column in enumerate(y_columns)}
        return {
            y_str_to_int[target]: [X_str_to_int[feature] for feature in features]
            for target, features in target_to_features_dict.items()
        }

    def _check_X_columns(self, X: pd.DataFrame):
        for column1, column2 in zip(X.columns, self.X_columns_):
            if column1 != column2:
                raise ValueError(f"X columns are not the same: {column1}, {column2}")
