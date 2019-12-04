from abc import ABCMeta, abstractmethod

import pandas as pd


class TimeSeriesFeature(metaclass=ABCMeta):
    """Base class for all the feature classes in this package.

    Parameters documentation is in the derived classes.
    """
    @abstractmethod
    def __init__(self, output_name):
        self.output_name = output_name

    def fit(self, X, y=None):
        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def _rename_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Rename (in place) the column of the DataFrame with the
        ``output_name``. In case the output columns are more than one, a suffix
         is appended to the name, from ``_0`` to ``_n``, where ``n`` is the
         number of output columns.

        Parameters
        ----------
        X : ``pd.DataFrame``, required.
            The DataFrame to be renamed.

        Returns
        -------
        X_renamed : ``pd.DataFrame``
            The original DataFrame ``X``, with the columns renamed.

        """
        suffix = ''

        X_renamed = X.T.reset_index(drop=True).T
        for index, col in enumerate(X_renamed.columns):
            if len(X.columns) > 1:
                suffix = '_' + str(index)
            X_renamed.rename(columns={col: self.output_name + str(suffix)},
                             inplace=True)

        return X_renamed
