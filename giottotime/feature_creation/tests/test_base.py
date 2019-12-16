import numpy as np
import pandas as pd
import pandas.util.testing as testing

from giottotime.feature_creation import MovingAverageFeature
from giottotime.feature_creation.base import Feature


class BaseFeature(Feature):
    def __init__(self, output_name):
        super().__init__(output_name=output_name)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass


def test_correct_renaming_single_col():
    n_cols = 1
    testing.N, testing.K = 500, n_cols
    df = testing.makeTimeDataFrame(freq="MS")

    output_name = "shift"
    shift_feature = BaseFeature(output_name=output_name)
    df_renamed = shift_feature._rename_columns(df)

    assert df.shape == df_renamed.shape

    expected_cols = output_name

    np.testing.assert_array_equal(expected_cols, df_renamed.columns)


def test_correct_renaming_series():
    n_cols = 1
    testing.N, testing.K = 500, n_cols
    df = testing.makeTimeSeries(freq="MS")

    output_name = "shift"
    shift_feature = BaseFeature(output_name=output_name)
    df_renamed = shift_feature._rename_columns(df)

    assert df.shape[0] == df_renamed.shape[0]
    assert pd.DataFrame == type(df_renamed)

    expected_cols = output_name

    np.testing.assert_array_equal(expected_cols, df_renamed.columns)


def test_correct_renaming_multiple_columns():
    n_cols = 10
    testing.N, testing.K = 500, n_cols
    df = testing.makeTimeDataFrame(freq="MS")

    output_name = "shift"
    shift_feature = BaseFeature(output_name=output_name)
    df_renamed = shift_feature._rename_columns(df)

    assert df.shape == df_renamed.shape

    expected_cols = [f"{output_name}_{k}" for k in range(n_cols)]

    np.testing.assert_array_equal(expected_cols, df_renamed.columns)


def test_correct_fit_transform():
    n_cols = 10
    testing.N, testing.K = 500, n_cols
    df = testing.makeTimeDataFrame(freq="MS")

    output_name = "shift"
    ma_feature = MovingAverageFeature(window_size=2, output_name=output_name)

    fit_transform_res = ma_feature.fit_transform(df)

    transform_res = ma_feature.fit(df).transform(df)

    testing.assert_frame_equal(fit_transform_res, transform_res)
