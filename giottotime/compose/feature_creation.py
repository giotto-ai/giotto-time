import pandas as pd
from sklearn.compose import ColumnTransformer


class FeatureCreation(ColumnTransformer):
    """Applies transformers to columns of a pandas DataFrame.

    This estimator is a wrapper of sklearn.compose.ColumnTransformer, the only difference is the output type of
    fit_transform and transform methods which is a DataFrame instead of an array.
    """

    def fit_transform(self, X, y=None):
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : DataFrame of shape [n_samples, n_features]
            Input data, of which specified subsets are used to fit the
            transformers.

        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        X_t_df : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.

        """
        X_t = super().fit_transform(X, y)
        X_t_df = pd.DataFrame(data=X_t, columns=self.get_feature_names(), index=X.index)
        return X_t_df

    def transform(self, X):
        """Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : DataFrame of shape [n_samples, n_features]
            The data to be transformed by subset.

        Returns
        -------
        X_t_df : DataFrame, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers. If
            any result is a sparse matrix, everything will be converted to
            sparse matrices.

        """
        X_t = super().transform(X)
        X_t_df = pd.DataFrame(data=X_t, columns=self.get_feature_names(), index=X.index)
        return X_t_df
