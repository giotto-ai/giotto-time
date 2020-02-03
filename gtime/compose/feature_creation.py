import pandas as pd
from sklearn.compose import ColumnTransformer


class FeatureCreation(ColumnTransformer):
    """Applies transformers to columns of a pandas DataFrame.

    This estimator is a wrapper of sklearn.compose.ColumnTransformer, the only
    difference is the output type of fit_transform and transform methods which is a
    DataFrame instead of an array.

    """

    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame = None):
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), required
            Input data, of which specified subsets are used to fit the
            transformers.

        y : pd.DataFrame, shape (n_samples, ...), optional, default: ``None``
            Targets for supervised learning.

        Examples
        --------
        >>> import pandas.util.testing as testing
        >>> from gtime.compose import FeatureCreation
        >>> from gtime.feature_extraction import Shift, MovingAverage
        >>> data = testing.makeTimeDataFrame(freq="s")
        >>> fc = FeatureCreation([
        ...     ('s1', Shift(1), ['A']),
        ...     ('ma3', MovingAverage(window_size=3), ['B']),
        ... ])
        >>> fc.fit_transform(data).head()
                             s1__A__Shift  ma3__B__MovingAverage
        2000-01-01 00:00:00           NaN                    NaN
        2000-01-01 00:00:01      0.211403                    NaN
        2000-01-01 00:00:02     -0.313854               0.085045
        2000-01-01 00:00:03      0.502018              -0.239269
        2000-01-01 00:00:04     -0.225324              -0.144625

        Returns
        -------
        X_t_df : pd.DataFrame, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.

        """
        X_t = super().fit_transform(X, y)
        X_t_df = pd.DataFrame(data=X_t, columns=self.get_feature_names(), index=X.index)
        return X_t_df

    def transform(self, X: pd.DataFrame):
        """Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), required
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
