from typing import List

import pandas as pd

from .base import TimeSeriesTransformer


class FeaturesTransformation:

    def __init__(self, feature_transformers: List[TimeSeriesTransformer]):
        self.feature_transformers = feature_transformers

    def fit(self, X: pd.DataFrame):
        self.fit_transform(X)

    def transform(self, X: pd.DataFrame):
        transformed_df = X.copy()
        for transformer in self.feature_transformers:
            transformed_df = self._transform_df_with_transformer(
                transformed_df, transformer)
        return transformed_df

    def fit_transform(self, X: pd.DataFrame):
        transformed_df = X.copy()
        for transformer in self.feature_transformers:
            transformed_df = self._fit_transform_transformer_to(transformer,
                                                                transformed_df)
        return transformed_df

    def _transform_df_with_transformer(self, df: pd.DataFrame,
                            transformer: TimeSeriesTransformer):
        df = pd.concat([df, transformer.transform(df)], axis=1)
        if transformer.drop_input_column:
            df = df.drop(transformer.input_columns, axis=1)
        return df

    def _fit_transform_transformer_to(self, transformer: TimeSeriesTransformer,
                                      df: pd.DataFrame):
        df = pd.concat([df, transformer.fit_transform(df)], axis=1)
        if transformer.drop_input_column:
            df = df.drop(transformer.input_columns, axis=1)
        return df
