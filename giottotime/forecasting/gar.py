import pandas as pd
from sklearn.multioutput import MultiOutputRegressor, RegressorChain


# TODO: retest example + docs
class GAR(MultiOutputRegressor):
    """Generalized Auto Regression model also known as MultiOutputRegressor in scikit-learn

    Parameters
    ----------
    estimator: object, required
        The model used to make the predictions step by step. This object must be inherits from RegressorMixin.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from giottotime.forecasting import GAR
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

    def __init__(self, estimator, n_jobs=None):
        super().__init__(estimator, n_jobs)

    def fit(self, X, y, sample_weight=None):
        self._y_columns = y.columns
        return super().fit(X, y, sample_weight)

    def predict(self, X):
        X_p = super().predict(X)
        X_p_df = pd.DataFrame(data=X_p,
                              columns=self._y_columns,
                              index=X.index)

        return X_p_df


# FIXME: See #99
class GARFF(RegressorChain):
    def __init__(self, base_estimator, random_state=None):
        # TODO: justify order=None and cv in documentation
        super().__init__(base_estimator, order=None, cv=None, random_state=random_state)

    def fit(self, X, y):
        self._y_columns = y.columns
        return super().fit(X, y)

    def predict(self, X):
        X_p = super().predict(X)
        X_p_df = pd.DataFrame(data=X_p,
                              columns=self._y_columns,
                              index=X.index)

        return X_p_df
