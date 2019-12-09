import numpy as np
import pandas as pd
from hypothesis import given, strategies as st
from pandas.util import testing as testing

from giottotime.utils.hypothesis.time_indexes import giotto_time_series
from giottotime.feature_creation import ConstantFeature, CustomFeature


class TestConstantFeature:
    def _correct_constant(
        self, df: pd.DataFrame, constant: int, output_name: str
    ) -> pd.DataFrame:
        constant_series = pd.Series(data=constant, index=df.index)
        constant_series.name = output_name
        return constant_series.to_frame()

    def test_correct_constant_feature(self):
        output_name = "constant_feature"
        constant = 12
        df = pd.DataFrame.from_dict({"old_name": [0, 1, 2, 3, 4, 5]})

        constant_feature = ConstantFeature(constant=constant, output_name=output_name)

        df_constant = constant_feature.fit_transform(df)
        expected_df_constant = pd.DataFrame.from_dict(
            {output_name: [constant, constant, constant, constant, constant, constant]}
        )

        testing.assert_frame_equal(expected_df_constant, df_constant)

    @given(
        giotto_time_series(
            min_length=1,
            start_date=pd.Timestamp(2000, 1, 1),
            end_date=pd.Timestamp(2010, 1, 1),
        ),
        st.integers(0, 100),
    )
    def test_random_ts_and_constant(self, df: pd.DataFrame, constant: int):
        output_name = "time_series"

        constant_feature = ConstantFeature(constant=constant, output_name=output_name)
        df_constant = constant_feature.fit_transform(df)
        expected_df_constant = self._correct_constant(df, constant, output_name)

        testing.assert_frame_equal(expected_df_constant, df_constant)


class TestCustomFeature:
    def _df_to_power(self, df: pd.DataFrame, power: int) -> pd.DataFrame:
        return np.power(df, power)

    def test_correct_custom_feature(self):
        output_name = "custom"
        power = 3
        df = pd.DataFrame.from_dict({"old_name": [0, 1, 2, 3, 4, 5]})

        custom_feature = CustomFeature(
            custom_feature_function=self._df_to_power,
            output_name=output_name,
            power=power,
        )

        output_custom_feature = custom_feature.fit_transform(df)
        expected_custom_output = pd.DataFrame.from_dict(
            {output_name: [0, 1, 8, 27, 64, 125]}
        )

        testing.assert_frame_equal(expected_custom_output, output_custom_feature)

    def test_multi_columns_custom_feature(self):
        output_name = "custom"
        power = 2
        df = pd.DataFrame.from_dict(
            {"old_name": [0, 1, 2, 3, 4, 5], "old_name_1": [7, 8, 9, 10, 11, 12]}
        )

        custom_feature = CustomFeature(
            custom_feature_function=self._df_to_power,
            output_name=output_name,
            power=power,
        )

        output_custom_feature = custom_feature.fit_transform(df)
        expected_custom_output = pd.DataFrame.from_dict(
            {
                f"{output_name}_0": [0, 1, 4, 9, 16, 25],
                f"{output_name}_1": [49, 64, 81, 100, 121, 144],
            }
        )

        testing.assert_frame_equal(expected_custom_output, output_custom_feature)

    @given(
        giotto_time_series(
            start_date=pd.Timestamp(2000, 1, 1), end_date=pd.Timestamp(2010, 1, 1)
        ),
        st.integers(0, 10),
    )
    def test_random_ts_and_power(self, df: pd.DataFrame, power: int):
        output_name = "time_series"

        custom_feature = CustomFeature(self._df_to_power, output_name, power=power)

        output_custom = custom_feature.fit_transform(df)
        expected_custom_output = self._df_to_power(df, power)

        testing.assert_frame_equal(expected_custom_output, output_custom)
