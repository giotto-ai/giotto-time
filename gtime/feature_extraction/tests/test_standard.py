import numpy as np
import pandas as pd
import pandas.util.testing as testing
import pytest
from hypothesis import given, strategies as st
from sklearn.preprocessing import PolynomialFeatures

from gtime.utils.hypothesis.time_indexes import giotto_time_series
from gtime.feature_extraction import (
    Shift,
    MovingAverage,
    Exogenous,
    Polynomial,
    MovingCustomFunction,
)

df = pd.DataFrame.from_dict({"x": [0, 1, 2, 3, 4, 5]})

shift_class_name = Shift().__class__.__name__
df_shift_1 = pd.DataFrame.from_dict({f"x__{shift_class_name}": [np.nan, 0, 1, 2, 3, 4]})
df_shift_m2 = pd.DataFrame.from_dict(
    {f"x__{shift_class_name}": [2, 3, 4, 5, np.nan, np.nan]}
)
df_shift_0 = pd.DataFrame.from_dict({f"x__{shift_class_name}": [0, 1, 2, 3, 4, 5]})


# FIXME: shift a + shift b = shift a+b instead
class TestShift:
    def _correct_shift(self, df: pd.DataFrame, shift: int) -> pd.DataFrame:
        return df.shift(shift)

    @pytest.mark.parametrize(
        ("shift", "expected"), [(1, df_shift_1), (-2, df_shift_m2), (0, df_shift_0)]
    )
    def test_shift_transform(self, shift, expected):
        shift = Shift(shift=shift)
        testing.assert_frame_equal(shift.fit_transform(df), expected)

    def test_multi_columns_time_shift_feature(self):
        shift = Shift(shift=-2)
        df_multi = pd.DataFrame({"x0": [0, 1, 2, 3, 4, 5], "x1": [7, 8, 9, 10, 11, 12]})

        expected_df = pd.DataFrame.from_dict(
            {
                f"x0__{shift_class_name}": [2, 3, 4, 5, np.nan, np.nan],
                f"x1__{shift_class_name}": [9, 10, 11, 12, np.nan, np.nan],
            }
        )

        testing.assert_frame_equal(shift.fit_transform(df_multi), expected_df)

    @given(
        giotto_time_series(
            start_date=pd.Timestamp(2000, 1, 1), end_date=pd.Timestamp(2010, 1, 1)
        ),
        st.integers(0, 200),
    )
    def test_random_ts_and_shifts(self, df: pd.DataFrame, shift: int):
        shift_feature = Shift(shift=shift)

        df_shifted = shift_feature.fit_transform(df)
        correct_df_shifted = self._correct_shift(df, shift)

        #  testing.assert_frame_equal(correct_df_shifted, df_shifted)


# FIXME: mean(df_k, df_k+1) = dft_k
class TestMovingAverage:
    def _correct_ma(self, df: pd.DataFrame, window_size: int) -> pd.DataFrame:
        return (
            df.rolling(window_size)
            .mean()
            .add_suffix("__" + MovingAverage().__class__.__name__)
        )

    def test_invalid_window_size(self):
        window_size = -1
        df = pd.DataFrame.from_dict({"x0": [0, 1, 2, 3, 4, 5]})

        ma_feature = MovingAverage(window_size=window_size)

        with pytest.raises(ValueError):
            ma_feature.fit_transform(df)

    def test_positive_window_size(self):
        window_size = 2
        df = pd.DataFrame.from_dict({"x": [0, 1, 2, 3, 4, 5]})

        ma_feature = MovingAverage(window_size=window_size)
        df_ma = ma_feature.fit_transform(df)
        output_name = "x__" + ma_feature.__class__.__name__
        expected_df_ma = pd.DataFrame.from_dict(
            {output_name: [np.nan, 0.5, 1.5, 2.5, 3.5, 4.5]}
        )

        testing.assert_frame_equal(expected_df_ma, df_ma, check_names=False)

    def test_multi_columns_window_size(self):
        window_size = 2
        df = pd.DataFrame.from_dict(
            {"x0": [0, 1, 2, 3, 4, 5], "x1": [7, 8, 9, 10, 11, 12]}
        )

        ma_feature = MovingAverage(window_size=window_size)
        feature_name = ma_feature.__class__.__name__

        df_ma = ma_feature.fit_transform(df)
        expected_df_ma = pd.DataFrame(
            {
                f"x0__{feature_name}": [np.nan, 0.5, 1.5, 2.5, 3.5, 4.5],
                f"x1__{feature_name}": [np.nan, 7.5, 8.5, 9.5, 10.5, 11.5],
            }
        )

        testing.assert_frame_equal(expected_df_ma, df_ma, check_names=False)

    @given(
        giotto_time_series(
            start_date=pd.Timestamp(2000, 1, 1), end_date=pd.Timestamp(2010, 1, 1)
        ),
        st.integers(0, 100),
    )
    def test_random_ts_and_window_size(self, df: pd.DataFrame, window_size: int):
        ma_feature = MovingAverage(window_size=window_size)
        df_ma = ma_feature.fit_transform(df)
        expected_df_ma = self._correct_ma(df, window_size)

        testing.assert_frame_equal(expected_df_ma, df_ma)


# FIXME: refactor with pytest parametrize
@pytest.mark.skip(reason="TODO: Use pytest parametrize")
class TestExogenous:
    def _correct_exog(self, exog: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        return exog.reindex(index=df.index)

    def test_correct_exog_none_method(self):
        method = None
        exog = pd.DataFrame.from_dict({"x0": [0, 1, 2, 3]})
        exog.index = [
            pd.Timestamp(2000, 1, 1),
            pd.Timestamp(2000, 2, 1),
            pd.Timestamp(2000, 3, 1),
            pd.Timestamp(2000, 4, 1),
        ]

        df = pd.DataFrame.from_dict({"x1": [10, 11, 12, 13]})
        df.index = [
            pd.Timestamp(2000, 1, 1),
            pd.Timestamp(2000, 1, 2),
            pd.Timestamp(2000, 1, 3),
            pd.Timestamp(2000, 1, 4),
        ]

        exog_feature = Exogenous(exogenous_time_series=exog, method=method)
        feature_name = exog_feature.__class__.__name__

        new_exog_feature = exog_feature.fit_transform(df)
        expected_exog = pd.DataFrame.from_dict(
            {f"x0__{feature_name}": [0, np.nan, np.nan, np.nan]}
        )
        expected_exog.index = [
            pd.Timestamp(2000, 1, 1),
            pd.Timestamp(2000, 1, 2),
            pd.Timestamp(2000, 1, 3),
            pd.Timestamp(2000, 1, 4),
        ]

        testing.assert_frame_equal(expected_exog, new_exog_feature)

    def test_correct_exog_backfill_method(self):
        method = "backfill"

        exog = pd.DataFrame.from_dict({"x0": [0, 1, 2, 3]})
        exog.index = [
            pd.Timestamp(2000, 1, 1),
            pd.Timestamp(2000, 2, 1),
            pd.Timestamp(2000, 3, 1),
            pd.Timestamp(2000, 4, 1),
        ]

        df = pd.DataFrame.from_dict({"x1": [10, 11, 12, 13]})
        df.index = [
            pd.Timestamp(2000, 1, 1),
            pd.Timestamp(2000, 1, 2),
            pd.Timestamp(2000, 1, 3),
            pd.Timestamp(2000, 1, 4),
        ]

        exog_feature = Exogenous(exogenous_time_series=exog, method=method)

        new_exog_feature = exog_feature.fit_transform(df)
        expected_exog = pd.DataFrame.from_dict({"output_name": [0, 1, 1, 1]})
        expected_exog.index = [
            pd.Timestamp(2000, 1, 1),
            pd.Timestamp(2000, 1, 2),
            pd.Timestamp(2000, 1, 3),
            pd.Timestamp(2000, 1, 4),
        ]

        testing.assert_frame_equal(expected_exog, new_exog_feature)

    def test_correct_exog_pad_method(self):
        method = "pad"
        exog = pd.DataFrame.from_dict({"x_1": [0, 1, 2, 3]})
        exog.index = [
            pd.Timestamp(2000, 1, 1),
            pd.Timestamp(2000, 2, 1),
            pd.Timestamp(2000, 3, 1),
            pd.Timestamp(2000, 4, 1),
        ]

        df = pd.DataFrame.from_dict({"x_2": [10, 11, 12, 13]})
        df.index = [
            pd.Timestamp(2000, 1, 1),
            pd.Timestamp(2000, 1, 2),
            pd.Timestamp(2000, 1, 3),
            pd.Timestamp(2000, 1, 4),
        ]

        exog_feature = Exogenous(exogenous_time_series=exog, method=method)

        new_exog_feature = exog_feature.fit_transform(df)
        expected_exog = pd.DataFrame.from_dict({"output_name": [0, 0, 0, 0]})
        expected_exog.index = [
            pd.Timestamp(2000, 1, 1),
            pd.Timestamp(2000, 1, 2),
            pd.Timestamp(2000, 1, 3),
            pd.Timestamp(2000, 1, 4),
        ]

        testing.assert_frame_equal(expected_exog, new_exog_feature)

    def test_correct_nearest_pad_method(self):
        method = "nearest"
        exog = pd.DataFrame.from_dict({"x_1": [0, 1, 2, 3]})
        exog.index = [
            pd.Timestamp(2000, 1, 1),
            pd.Timestamp(2000, 2, 1),
            pd.Timestamp(2000, 3, 1),
            pd.Timestamp(2000, 4, 1),
        ]

        df = pd.DataFrame.from_dict({"x_2": [10, 11, 12, 13]})
        df.index = [
            pd.Timestamp(2000, 1, 1),
            pd.Timestamp(2000, 1, 2),
            pd.Timestamp(2000, 1, 3),
            pd.Timestamp(2000, 1, 29),
        ]

        exog_feature = Exogenous(exogenous_time_series=exog, method=method)

        new_exog_feature = exog_feature.fit_transform(df)
        expected_exog = pd.DataFrame.from_dict({"output_name": [0, 0, 0, 1]})
        expected_exog.index = [
            pd.Timestamp(2000, 1, 1),
            pd.Timestamp(2000, 1, 2),
            pd.Timestamp(2000, 1, 3),
            pd.Timestamp(2000, 1, 29),
        ]

        testing.assert_frame_equal(expected_exog, new_exog_feature)

    def test_correct_multi_columns_exog(self):
        output_name = "exog"
        exog = pd.DataFrame.from_dict({"x_0": [0, 1, 2, 3], "x_1": [5, 6, 7, 8]})
        exog.index = [
            pd.Timestamp(2000, 1, 1),
            pd.Timestamp(2000, 2, 1),
            pd.Timestamp(2000, 3, 1),
            pd.Timestamp(2000, 4, 1),
        ]

        df = pd.DataFrame.from_dict({"x_2": [10, 11, 12, 13]})
        df.index = [
            pd.Timestamp(2000, 1, 1),
            pd.Timestamp(2000, 1, 2),
            pd.Timestamp(2000, 1, 3),
            pd.Timestamp(2000, 1, 29),
        ]

        exog_feature = Exogenous(exogenous_time_series=exog)

        new_exog_feature = exog_feature.fit_transform(df)
        expected_exog = pd.DataFrame.from_dict(
            {
                f"{output_name}_0": [0, np.nan, np.nan, np.nan],
                f"{output_name}_1": [5, np.nan, np.nan, np.nan],
            }
        )
        expected_exog.index = [
            pd.Timestamp(2000, 1, 1),
            pd.Timestamp(2000, 1, 2),
            pd.Timestamp(2000, 1, 3),
            pd.Timestamp(2000, 1, 29),
        ]

        testing.assert_frame_equal(expected_exog, new_exog_feature)


class TestPolynomial:
    def test_correct_pol_features_single_column(self):
        degree = 3
        df = pd.DataFrame.from_dict({"x": [0, 1, 2, 3]})
        df.index = [
            pd.Timestamp(2000, 1, 1),
            pd.Timestamp(2000, 2, 1),
            pd.Timestamp(2000, 3, 1),
            pd.Timestamp(2000, 4, 1),
        ]

        pol_feature = Polynomial(degree=degree)
        feature_name = pol_feature.__class__.__name__

        pol_df = pol_feature.fit_transform(df)

        expected_pol_df = pd.DataFrame.from_dict(
            {
                f"1__{feature_name}": [1.0, 1.0, 1.0, 1.0],
                f"x0__{feature_name}": [0.0, 1.0, 2.0, 3.0],
                f"x0^2__{feature_name}": [0.0, 1.0, 4.0, 9.0],
                f"x0^3__{feature_name}": [0.0, 1.0, 8.0, 27.0],
            }
        )
        expected_pol_df.index = [
            pd.Timestamp(2000, 1, 1),
            pd.Timestamp(2000, 2, 1),
            pd.Timestamp(2000, 3, 1),
            pd.Timestamp(2000, 4, 1),
        ]

        testing.assert_frame_equal(expected_pol_df, pol_df)

    def test_correct_pol_features_multi_columns(self):
        degree = 2
        df = pd.DataFrame.from_dict({"x_1": [0, 2, 4], "x_2": [1, 3, 5]})
        df.index = [
            pd.Timestamp(2000, 1, 1),
            pd.Timestamp(2000, 2, 1),
            pd.Timestamp(2000, 3, 1),
        ]

        pol_feature = Polynomial(degree=degree)
        feature_name = pol_feature.__class__.__name__

        pol_df = pol_feature.fit_transform(df)

        expected_pol_df = pd.DataFrame.from_dict(
            {
                f"1__{feature_name}": [1.0, 1.0, 1],
                f"x0__{feature_name}": [0.0, 2.0, 4],
                f"x1__{feature_name}": [1.0, 3.0, 5],
                f"x0^2__{feature_name}": [0.0, 4.0, 16.0],
                f"x0 x1__{feature_name}": [0.0, 6.0, 20.0],
                f"x1^2__{feature_name}": [1.0, 9.0, 25.0],
            }
        )
        expected_pol_df.index = [
            pd.Timestamp(2000, 1, 1),
            pd.Timestamp(2000, 2, 1),
            pd.Timestamp(2000, 3, 1),
        ]

        testing.assert_frame_equal(expected_pol_df, pol_df)


class TestMovingCustomFunction:
    def test_correct_moving_custom_function(self):
        df = pd.DataFrame.from_dict({"x_1": [0, 7, 2], "x_2": [2, 10, 4]})
        df.index = [
            pd.Timestamp(2000, 1, 1),
            pd.Timestamp(2000, 2, 1),
            pd.Timestamp(2000, 3, 1),
        ]
        custom_feature = MovingCustomFunction(
            custom_feature_function=np.diff, window_size=2
        )
        custom_output = custom_feature.fit_transform(df)

        feature_name = custom_feature.__class__.__name__
        expected_custom_df = pd.DataFrame.from_dict(
            {
                f"x_1__{feature_name}": [np.nan, 7.0, -5],
                f"x_2__{feature_name}": [np.nan, 8.0, -6],
            }
        )
        expected_custom_df.index = [
            pd.Timestamp(2000, 1, 1),
            pd.Timestamp(2000, 2, 1),
            pd.Timestamp(2000, 3, 1),
        ]

        testing.assert_frame_equal(expected_custom_df, custom_output)
