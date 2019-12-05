import pandas as pd
import pandas.util.testing as testing

from giottotime.feature_creation import MovingAverageFeature
from giottotime.feature_creation.base import TimeSeriesFeature


class TestTimeSeriesFeature(TimeSeriesFeature):
    def __init__(self, output_name):
        super().__init__(output_name=output_name)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass


def test_correct_renaming_single_col():
    n_cols = 1
    testing.N, testing.K = 500, n_cols
    df = testing.makeTimeDataFrame(freq="MS")

    output_name = "shift"
    shift_feature = TestTimeSeriesFeature(output_name=output_name)
    df_renamed = shift_feature._rename_columns(df)

    assert df.shape == df_renamed.shape

    expected_cols = output_name

    assert (expected_cols == df_renamed.columns).all()


def test_correct_renaming_multiple_columns():
    n_cols = 10
    testing.N, testing.K = 500, n_cols
    df = testing.makeTimeDataFrame(freq="MS")

    output_name = "shift"
    shift_feature = TestTimeSeriesFeature(output_name=output_name)
    df_renamed = shift_feature._rename_columns(df)

    assert df.shape == df_renamed.shape

    expected_cols = [f"{output_name}_{k}" for k in range(n_cols)]

    assert (expected_cols == df_renamed.columns).all()


def test_correct_fit_transform():
    n_cols = 10
    testing.N, testing.K = 500, n_cols
    df = testing.makeTimeDataFrame(freq="MS")

    output_name = "shift"
    ma_feature = MovingAverageFeature(window_size=2, output_name=output_name)

    fit_transform_res = ma_feature.fit_transform(df)

    transform_res = ma_feature.fit(df).transform(df)

    testing.assert_frame_equal(fit_transform_res, transform_res)
