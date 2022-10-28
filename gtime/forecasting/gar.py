from copy import deepcopy
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.multioutput import (
    MultiOutputRegressor,
    RegressorChain,
)
from sklearn.utils.validation import check_is_fitted

from gtime.regressors import ExplainableRegressor
from gtime.regressors.multi_output import MultiFeatureMultiOutputRegressor


def initialize_estimator(
    estimator: RegressorMixin, explainer_type: Optional[str]
) -> RegressorMixin:
    if explainer_type is None:
        return estimator
    else:
        return ExplainableRegressor(estimator, explainer_type)


class _ExplanationsMixin:
    def _explanations_as_dataframe(
        self,
        index: pd.Index,
        y_columns: List[str],
        X_columns: Union[List[str], Dict[str, List[str]]],
    ) -> Dict[str, pd.DataFrame]:
        explanations = self._dict_explanations(index, y_columns)
        explanations = self._rename_columns(explanations, X_columns)
        return explanations

    def _dict_explanations(self, index: pd.Index, y_columns: List[str]):
        return {
            y_column: pd.DataFrame(estimator.explanations_, index=index)
            for y_column, estimator in zip(y_columns, self.estimators_)
        }

    def _rename_columns(
        self,
        dict_explanations: Dict[str, pd.DataFrame],
        X_columns: Union[List[str], Dict[str, List[str]]],
    ):
        if isinstance(X_columns, list):
            for column, explanation in dict_explanations.items():
                explanation.columns = X_columns
        elif isinstance(X_columns, dict):
            for y_column, columns in X_columns.items():
                dict_explanations[y_column].columns = columns
        else:
            raise TypeError(
                f"X_columns must be a list or a dict. Detected: {type(X_columns)}"
            )
        return dict_explanations


class GAR(MultiOutputRegressor, _ExplanationsMixin):
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

    def __init__(self, estimator, explainer_type: str = None, n_jobs: int = None):
        self.explainer_type = explainer_type
        estimator = initialize_estimator(estimator, explainer_type)
        super().__init__(estimator, n_jobs=n_jobs)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight=None, **kwargs):
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
        self.y_columns_ = y.columns
        return super().fit(X, y)

    def predict(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
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
        y_p_df = pd.DataFrame(data=y_p, columns=self.y_columns_, index=X.index)

        if self.explainer_type is not None:
            self.explanations_ = self._explanations_as_dataframe(
                index=y_p_df.index,
                y_columns=self.y_columns_,
                X_columns=list(X.columns),
            )
        return y_p_df


# TODO: See #99
class GARFF(RegressorChain, _ExplanationsMixin):
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

    def __init__(self, estimator, explainer_type: str = None):
        self.explainer_type = explainer_type
        estimator = initialize_estimator(estimator, explainer_type)
        super().__init__(
            base_estimator=estimator, order=None, cv=None, random_state=None
        )

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs):
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
        self.y_columns_ = y.columns
        self.target_to_features_dict_ = self._compute_target_to_features_dict(
            X.columns, y.columns
        )
        return super().fit(X, y)

    def predict(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
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
        y_p_df = pd.DataFrame(data=y_p, columns=self.y_columns_, index=X.index)

        if self.explainer_type is not None:
            self.explanations_ = self._explanations_as_dataframe(
                index=y_p_df.index,
                y_columns=self.y_columns_,
                X_columns=self.target_to_features_dict_,
            )
        return y_p_df

    def _compute_target_to_features_dict(
        self, X_columns: List[str], y_columns: List[str]
    ) -> Dict[str, List[str]]:
        X_columns, y_columns = list(X_columns), list(y_columns)

        target_to_features_dict = {}
        for i, y_column in enumerate(y_columns):
            target_to_features_dict[y_column] = X_columns + y_columns[:i]
        return target_to_features_dict


class MultiFeatureGAR(MultiFeatureMultiOutputRegressor, _ExplanationsMixin):
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

    def __init__(
        self,
        estimator: RegressorMixin,
        explainer_type: str = None,
        target_to_features_dict: Dict[str, List[str]] = None,
    ):
        self.explainer_type = explainer_type
        estimator = initialize_estimator(estimator, explainer_type)
        super().__init__(estimator, target_to_features_dict=target_to_features_dict)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs):
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
        if self.target_to_features_dict is not None:
            numeric_target_to_features_dict = self._feature_name_to_index(
                self.target_to_features_dict, X.columns, y.columns
            )
            return super().fit(
                X.values,
                y.values,
                target_to_features_dict=numeric_target_to_features_dict,
            )
        else:
            return super().fit(X.values, y.values)

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

        if self.explainer_type is not None:
            if self.target_to_features_dict_ is None:
                self.explanations_ = self._explanations_as_dataframe(
                    index=y_p_df.index,
                    y_columns=self.y_columns_,
                    X_columns=list(X.columns),
                )
            else:
                self.explanations_ = self._explanations_as_dataframe(
                    index=y_p_df.index,
                    y_columns=self.y_columns_,
                    X_columns=self.target_to_features_dict,
                )
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
