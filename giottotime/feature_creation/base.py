from abc import abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array


class Feature(BaseEstimator, TransformerMixin):
    @abstractmethod
    def __init__(self, output_name):
        self.output_name = output_name

    def fit(self, time_series: Union[pd.Series, np.array, list], y=None) -> "Feature":
        """Do nothing and return the estimator unchanged.
        This method is there to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        time_series : ``np.ndarray``, required.
            Input data.

        y : ``None``
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        check_array(time_series)

        return self

    def _rename_columns(self, X: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        if isinstance(X, pd.Series):
            X = X.to_frame()

        suffix = ""

        X_renamed = X.T.reset_index(drop=True).T
        for index, col in enumerate(X_renamed.columns):
            if len(X.columns) > 1:
                suffix = "_" + str(index)
            X_renamed.rename(
                columns={col: self.output_name + str(suffix)}, inplace=True
            )

        return X_renamed
