import pandas as pd
import numpy as np
import pytest
import pandas.util.testing as testing
from hypothesis import given, strategies as st, settings

from giottotime.core.hypothesis.time_indexes import giotto_time_series
from giottotime.feature_creation import ShiftFeature, MovingAverageFeature, \
    ConstantFeature, ExogenousFeature, CustomFeature


class TestShiftFeature:
    def _correct_shift(self, df: pd.DataFrame, shift: int) -> pd.DataFrame:
        return df.shift(shift)

    def test_forward_time_shift_feature(self):
        output_name = "shift"
        df = pd.DataFrame.from_dict({output_name: [0, 1, 2, 3, 4, 5]})

        shift_feature = ShiftFeature(shift=1, output_name=output_name)
        expected_df = pd.DataFrame.from_dict(
            {output_name: [np.nan, 0, 1, 2, 3, 4]})

        df_shifted = shift_feature.fit_transform(df)
        testing.assert_frame_equal(expected_df, df_shifted)

    def test_backwards_time_shift_feature(self):
        output_name = "shift"
        df = pd.DataFrame.from_dict({output_name: [0, 1, 2, 3, 4, 5]})

        shift_feature = ShiftFeature(shift=-2, output_name=output_name)
        expected_df = pd.DataFrame.from_dict(
            {output_name: [2, 3, 4, 5, np.nan, np.nan]})

        df_shifted = shift_feature.fit_transform(df)
        testing.assert_frame_equal(expected_df, df_shifted)

    def test_zero_shift_feature(self):
        output_name = "shift"
        df = pd.DataFrame.from_dict({output_name: [0, 1, 2, 3, 4, 5]})

        shift_feature = ShiftFeature(shift=0, output_name=output_name)
        expected_df = pd.DataFrame.from_dict({output_name: [0, 1, 2, 3, 4, 5]})

        df_shifted = shift_feature.fit_transform(df)
        testing.assert_frame_equal(expected_df, df_shifted)

    @given(giotto_time_series(start_date=pd.Timestamp(2000, 1, 1),
                              end_date=pd.Timestamp(2010, 1, 1)
                              ),
           st.integers(0, 200))
    def test_random_ts_and_shifts(self, df: pd.DataFrame, shift: int):
        output_name = "time_series"
        shift_feature = ShiftFeature(shift=shift, output_name=output_name)

        df_shifted = shift_feature.fit_transform(df)
        correct_df_shifted = self._correct_shift(df, shift)

        testing.assert_frame_equal(correct_df_shifted, df_shifted)


class TestMovingAverageFeature:
    def _correct_ma(self, df: pd.DataFrame, window_size: int) -> pd.DataFrame:
        return df.rolling(window_size).mean().shift(1)

    def test_invalid_window_size(self):
        output_name = "moving_average"
        window_size = -1
        df = pd.DataFrame.from_dict({output_name: [0, 1, 2, 3, 4, 5]})

        ma_feature = MovingAverageFeature(window_size=window_size,
                                          output_name=output_name)
        with pytest.raises(ValueError):
            ma_feature.fit_transform(df)

    def test_positive_window_size(self):
        output_name = "moving_average"
        window_size = 2
        df = pd.DataFrame.from_dict({output_name: [0, 1, 2, 3, 4, 5]})

        ma_feature = MovingAverageFeature(window_size=window_size,
                                          output_name=output_name)
        df_ma = ma_feature.fit_transform(df)
        expected_df_ma = pd.DataFrame.from_dict(
            {output_name: [np.nan, np.nan, 0.5, 1.5, 2.5, 3.5]})

        testing.assert_frame_equal(expected_df_ma, df_ma)

    @given(giotto_time_series(start_date=pd.Timestamp(2000, 1, 1),
                              end_date=pd.Timestamp(2010, 1, 1)
                              ),
           st.integers(0, 100))
    def test_random_ts_and_window_size(self, df: pd.DataFrame,
                                       window_size: int):
        output_name = "time_series"

        ma_feature = MovingAverageFeature(window_size=window_size,
                                          output_name=output_name)
        df_ma = ma_feature.fit_transform(df)
        expected_df_ma = self._correct_ma(df, window_size)

        testing.assert_frame_equal(expected_df_ma, df_ma)


class TestConstantFeature:
    def _correct_constant(self, df: pd.DataFrame, constant: int,
                          output_name: str) -> pd.DataFrame:
        constant_series = pd.Series(data=constant, index=df.index)
        constant_series.name = output_name
        return constant_series.to_frame()

    def test_correct_constant_feature(self):
        output_name = "constant_feature"
        constant = 12
        df = pd.DataFrame.from_dict({output_name: [0, 1, 2, 3, 4, 5]})

        constant_feature = ConstantFeature(constant=constant,
                                           output_name=output_name)

        df_constant = constant_feature.fit_transform(df)
        expected_df_constant = pd.DataFrame.from_dict(
            {output_name: [constant, constant, constant, constant, constant,
                           constant]})

        testing.assert_frame_equal(expected_df_constant, df_constant)

    @given(giotto_time_series(start_date=pd.Timestamp(2000, 1, 1),
                              end_date=pd.Timestamp(2010, 1, 1)
                              ),
           st.integers(0, 100))
    def test_random_ts_and_constant(self, df: pd.DataFrame, constant: int):
        output_name = "time_series"

        constant_feature = ConstantFeature(constant=constant,
                                           output_name=output_name)
        df_constant = constant_feature.fit_transform(df)
        expected_df_constant = self._correct_constant(df, constant,
                                                      output_name)

        testing.assert_frame_equal(expected_df_constant, df_constant)


class TestExogenousFeature:
    def _correct_exog(self, exog: pd.DataFrame, df: pd.DataFrame) \
            -> pd.DataFrame:
        return exog.reindex(index=df.index)

    def test_correct_exog(self):
        output_name = "exog"
        exog = pd.DataFrame.from_dict({output_name: [0, 1, 2, 3]})
        exog.index = ([pd.Timestamp(2000, 1, 1),
                        pd.Timestamp(2000, 2, 1),
                        pd.Timestamp(2000, 3, 1),
                        pd.Timestamp(2000, 4, 1)])

        df = pd.DataFrame.from_dict({output_name: [10, 11, 12, 13]})
        df.index = ([pd.Timestamp(2000, 1, 1),
                      pd.Timestamp(2000, 1, 2),
                      pd.Timestamp(2000, 1, 3),
                      pd.Timestamp(2000, 1, 4)])

        exog_feature = ExogenousFeature(exogenous_time_series=exog,
                                        output_name=output_name)

        new_exog_feature = exog_feature.fit_transform(df)
        expected_exog = self._correct_exog(exog, df)

        testing.assert_frame_equal(expected_exog, new_exog_feature)

    @pytest.mark.skip(reason='to be fixed')
    @settings(max_examples=10)
    @given(giotto_time_series(start_date=pd.Timestamp(2000, 1, 1),
                              end_date=pd.Timestamp(2010, 1, 1)
                              ),
           giotto_time_series(start_date=pd.Timestamp(2000, 1, 1),
                              end_date=pd.Timestamp(2010, 1, 1)
                              )
           )
    def test_random_ts_and_exog(self, exog: pd.DataFrame, df: pd.DataFrame):
        output_name = "time_series"

        exog_feature = ExogenousFeature(exogenous_time_series=exog,
                                        output_name=output_name)

        print(df)
        print(exog)
        new_exog_feature = exog_feature.fit_transform(df)
        expected_exog = self._correct_exog(exog, df)

        testing.assert_frame_equal(expected_exog, new_exog_feature)


class TestPolynomialFeature:
    pass


class TestCustomFeature:
    def df_to_power(self, df: pd.DataFrame, power: int) -> pd.DataFrame:
        return np.power(df, power)

    def test_correct_custom_feature(self):
        output_name = "custom"
        power = 3
        df = pd.DataFrame.from_dict({output_name: [0, 1, 2, 3, 4, 5]})

        custom_feature = CustomFeature(
            custom_feature_function=self.df_to_power,
            output_name=output_name,
            power=power)

        output_custom_feature = custom_feature.fit_transform(df)
        expected_custom_output = pd.DataFrame.from_dict(
            {output_name: [0, 1, 8, 27, 64, 125]})

        testing.assert_frame_equal(expected_custom_output,
                                   output_custom_feature)

    @given(giotto_time_series(start_date=pd.Timestamp(2000, 1, 1),
                              end_date=pd.Timestamp(2010, 1, 1)
                              ),
           st.integers(0, 10)
           )
    def test_random_ts_and_power(self, df: pd.DataFrame, power: int):
        output_name = "time_series"

        custom_feature = CustomFeature(self.df_to_power, output_name,
                                       power=power)

        output_custom = custom_feature.fit_transform(df)
        expected_custom_output = self.df_to_power(df, power)

        testing.assert_frame_equal(expected_custom_output, output_custom)
