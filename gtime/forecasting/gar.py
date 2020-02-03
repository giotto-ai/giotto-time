import pandas as pd
from sklearn.multioutput import MultiOutputRegressor, RegressorChain


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
