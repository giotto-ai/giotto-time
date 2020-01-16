import pandas as pd
from sklearn.multioutput import MultiOutputRegressor, RegressorChain


class GAR(MultiOutputRegressor):
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


# FIXME: GARFF != oldGAR(feed_forward=True)
class GARFF(RegressorChain):
    def __init__(self, base_estimator, random_state=None):
        #  TODO: justify order=None and cv in documentation
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
