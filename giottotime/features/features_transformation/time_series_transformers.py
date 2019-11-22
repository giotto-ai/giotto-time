from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures as ScikitPolynomialFeatures

from .base import TimeSeriesTransformer


StringOrList = Union[List[str], str]


class ColumnsProduct(TimeSeriesTransformer):

    def __init__(
            self,
            input_columns: StringOrList,
            output_columns: StringOrList
    ):
        super().__init__(input_columns, output_columns, False)
        self.output_column = output_columns[0]

    def transform(self, X) -> pd.DataFrame:
        return X[self.input_columns].prod(axis=1)


class PolynomialFeatures(TimeSeriesTransformer):

    def __init__(
            self,
            input_columns: StringOrList,
            output_columns: StringOrList,
            drop_input_columns: bool,
            degree: int = 2,
            interaction_only: bool = False,
            include_bias: bool = True,
    ):
        super().__init__(input_columns, output_columns, drop_input_columns)

        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias

        self._scikit_polynomial_features = ScikitPolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias,
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        polynomial_features = self._compute_polynomial_features_from(X)
        polynomial_features_dataframe = pd.DataFrame(
            data=polynomial_features,
            index=X.index,
            columns=self.output_columns
        )
        return polynomial_features_dataframe

    def _compute_polynomial_features_from(self, df: pd.DataFrame) -> np.array:
        input_features = df[self.input_columns]
        polynomial_features = self._scikit_polynomial_features.fit_transform(
            input_features)
        if polynomial_features.shape[1] != len(self.output_columns):
            raise ValueError(f'The number of polynomial feature must match'
                             f'the number of output columns. '
                             f'There are {polynomial_features.shape[1]} '
                             f'polynomial features and '
                             f'{len(self.output_columns)} output columns')





