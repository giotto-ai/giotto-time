import pandas as pd
from sklearn.compose import ColumnTransformer


class DataFrameTransformer(ColumnTransformer):
    def fit_transform(self, X, y=None):
        X_t = super().fit_transform(X, y)
        X_t_df = pd.DataFrame(data=X_t,
                              columns=self.get_feature_names(),
                              index=X.index)
        return X_t_df

    def transform(self, X):
        X_t = super().transform(X)
        X_t_df = pd.DataFrame(data=X_t,
                              columns=self.get_feature_names(),
                              index=X.index)
        return X_t_df

