import numpy as np
import pandas as pd
import pandas.util.testing as testing
import pytest
from hypothesis import given, strategies as st
from sklearn.preprocessing import PolynomialFeatures

from giottotime.core.hypothesis.time_indexes import giotto_time_series
from giottotime.feature_creation import ShiftFeature, MovingAverageFeature, \
    ConstantFeature, ExogenousFeature, CustomFeature, PolynomialFeature


class TestShiftFeature:
    def _correct_shift(self, df: pd.DataFrame, shift: int) -> pd.DataFrame:
        return df.shift(shift)

    def test_forward_time_shift_feature(self):
        output_name = 'shift'
        df = pd.DataFrame.from_dict({'old_name': [0, 1, 2, 3, 4, 5]})

        shift_feature = ShiftFeature(shift=1, output_name=output_name)
        expected_df = pd.DataFrame.from_dict(
            {output_name: [np.nan, 0, 1, 2, 3, 4]})

        df_shifted = shift_feature.fit_transform(df)
        testing.assert_frame_equal(expected_df, df_shifted)

    def test_backwards_time_shift_feature(self):
        output_name = 'shift'
        df = pd.DataFrame.from_dict({'old_name': [0, 1, 2, 3, 4, 5]})

        shift_feature = ShiftFeature(shift=-2, output_name=output_name)
        expected_df = pd.DataFrame.from_dict(
            {output_name: [2, 3, 4, 5, np.nan, np.nan]})

        df_shifted = shift_feature.fit_transform(df)
        testing.assert_frame_equal(expected_df, df_shifted)

    def test_zero_shift_feature(self):
        output_name = 'shift'
        df = pd.DataFrame.from_dict({'old_name': [0, 1, 2, 3, 4, 5]})

        shift_feature = ShiftFeature(shift=0, output_name=output_name)
        expected_df = pd.DataFrame.from_dict({output_name: [0, 1, 2, 3, 4, 5]})

        df_shifted = shift_feature.fit_transform(df)
        testing.assert_frame_equal(expected_df, df_shifted)

    def test_multi_columns_time_shift_feature(self):
        output_name = 'shift'
        df = pd.DataFrame.from_dict({'old_name_0': [0, 1, 2, 3, 4, 5],
                                     'old_name_1': [7, 8, 9, 10, 11, 12]})

        shift_feature = ShiftFeature(shift=-2, output_name=output_name)
        expected_df = pd.DataFrame.from_dict(
            {f'{output_name}_0': [2, 3, 4, 5, np.nan, np.nan],
             f'{output_name}_1': [9, 10, 11, 12, np.nan, np.nan]})

        df_shifted = shift_feature.fit_transform(df)
        testing.assert_frame_equal(expected_df, df_shifted)

    @given(giotto_time_series(start_date=pd.Timestamp(2000, 1, 1),
                              end_date=pd.Timestamp(2010, 1, 1)
                              ),
           st.integers(0, 200))
    def test_random_ts_and_shifts(self, df: pd.DataFrame, shift: int):
        output_name = 'time_series'
        shift_feature = ShiftFeature(shift=shift, output_name=output_name)

        df_shifted = shift_feature.fit_transform(df)
        correct_df_shifted = self._correct_shift(df, shift)

        testing.assert_frame_equal(correct_df_shifted, df_shifted)


class TestMovingAverageFeature:
    def _correct_ma(self, df: pd.DataFrame, window_size: int) -> pd.DataFrame:
        return df.rolling(window_size).mean().shift(1)

    def test_invalid_window_size(self):
        output_name = 'moving_average'
        window_size = -1
        df = pd.DataFrame.from_dict({'old_name': [0, 1, 2, 3, 4, 5]})

        ma_feature = MovingAverageFeature(window_size=window_size,
                                          output_name=output_name)
        with pytest.raises(ValueError):
            ma_feature.fit_transform(df)

    def test_positive_window_size(self):
        output_name = 'moving_average'
        window_size = 2
        df = pd.DataFrame.from_dict({'old_name': [0, 1, 2, 3, 4, 5]})

        ma_feature = MovingAverageFeature(window_size=window_size,
                                          output_name=output_name)
        df_ma = ma_feature.fit_transform(df)
        expected_df_ma = pd.DataFrame.from_dict(
            {output_name: [np.nan, np.nan, 0.5, 1.5, 2.5, 3.5]})

        testing.assert_frame_equal(expected_df_ma, df_ma)

    def test_multi_columns_window_size(self):
        output_name = 'moving_average'
        window_size = 2
        df = pd.DataFrame.from_dict({'old_name_0': [0, 1, 2, 3, 4, 5],
                                     'old_name_1': [7, 8, 9, 10, 11, 12]})

        ma_feature = MovingAverageFeature(window_size=window_size,
                                          output_name=output_name)
        df_ma = ma_feature.fit_transform(df)
        expected_df_ma = pd.DataFrame.from_dict(
            {f'{output_name}_0': [np.nan, np.nan, 0.5, 1.5, 2.5, 3.5],
             f'{output_name}_1': [np.nan, np.nan, 7.5, 8.5, 9.5, 10.5]})

        testing.assert_frame_equal(expected_df_ma, df_ma)

    @given(giotto_time_series(start_date=pd.Timestamp(2000, 1, 1),
                              end_date=pd.Timestamp(2010, 1, 1)
                              ),
           st.integers(0, 100))
    def test_random_ts_and_window_size(self, df: pd.DataFrame,
                                       window_size: int):
        output_name = 'time_series'

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
        output_name = 'constant_feature'
        constant = 12
        df = pd.DataFrame.from_dict({'old_name': [0, 1, 2, 3, 4, 5]})

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
        output_name = 'time_series'

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

    def test_correct_exog_none_method(self):
        output_name = 'exog'
        method = None
        exog = pd.DataFrame.from_dict({'old_name_1': [0, 1, 2, 3]})
        exog.index = ([pd.Timestamp(2000, 1, 1),
                       pd.Timestamp(2000, 2, 1),
                       pd.Timestamp(2000, 3, 1),
                       pd.Timestamp(2000, 4, 1)])

        df = pd.DataFrame.from_dict({'old_name_2': [10, 11, 12, 13]})
        df.index = ([pd.Timestamp(2000, 1, 1),
                     pd.Timestamp(2000, 1, 2),
                     pd.Timestamp(2000, 1, 3),
                     pd.Timestamp(2000, 1, 4)])

        exog_feature = ExogenousFeature(exogenous_time_series=exog,
                                        output_name=output_name,
                                        method=method)

        new_exog_feature = exog_feature.fit_transform(df)
        expected_exog = pd.DataFrame.from_dict({output_name: [0,
                                                              np.nan,
                                                              np.nan,
                                                              np.nan]})
        expected_exog.index = ([pd.Timestamp(2000, 1, 1),
                                pd.Timestamp(2000, 1, 2),
                                pd.Timestamp(2000, 1, 3),
                                pd.Timestamp(2000, 1, 4)])

        testing.assert_frame_equal(expected_exog, new_exog_feature)

    def test_correct_exog_backfill_method(self):
        output_name = 'exog'
        method = 'backfill'
        exog = pd.DataFrame.from_dict({'old_name_1': [0, 1, 2, 3]})
        exog.index = ([pd.Timestamp(2000, 1, 1),
                       pd.Timestamp(2000, 2, 1),
                       pd.Timestamp(2000, 3, 1),
                       pd.Timestamp(2000, 4, 1)])

        df = pd.DataFrame.from_dict({'old_name_2': [10, 11, 12, 13]})
        df.index = ([pd.Timestamp(2000, 1, 1),
                     pd.Timestamp(2000, 1, 2),
                     pd.Timestamp(2000, 1, 3),
                     pd.Timestamp(2000, 1, 4)])

        exog_feature = ExogenousFeature(exogenous_time_series=exog,
                                        output_name=output_name,
                                        method=method)

        new_exog_feature = exog_feature.fit_transform(df)
        expected_exog = pd.DataFrame.from_dict({output_name: [0, 1, 1, 1]})
        expected_exog.index = ([pd.Timestamp(2000, 1, 1),
                                pd.Timestamp(2000, 1, 2),
                                pd.Timestamp(2000, 1, 3),
                                pd.Timestamp(2000, 1, 4)])

        testing.assert_frame_equal(expected_exog, new_exog_feature)

    def test_correct_exog_pad_method(self):
        output_name = 'exog'
        method = 'pad'
        exog = pd.DataFrame.from_dict({'old_name_1': [0, 1, 2, 3]})
        exog.index = ([pd.Timestamp(2000, 1, 1),
                       pd.Timestamp(2000, 2, 1),
                       pd.Timestamp(2000, 3, 1),
                       pd.Timestamp(2000, 4, 1)])

        df = pd.DataFrame.from_dict({'old_name_2': [10, 11, 12, 13]})
        df.index = ([pd.Timestamp(2000, 1, 1),
                     pd.Timestamp(2000, 1, 2),
                     pd.Timestamp(2000, 1, 3),
                     pd.Timestamp(2000, 1, 4)])

        exog_feature = ExogenousFeature(exogenous_time_series=exog,
                                        output_name=output_name,
                                        method=method)

        new_exog_feature = exog_feature.fit_transform(df)
        expected_exog = pd.DataFrame.from_dict({output_name: [0, 0, 0, 0]})
        expected_exog.index = ([pd.Timestamp(2000, 1, 1),
                                pd.Timestamp(2000, 1, 2),
                                pd.Timestamp(2000, 1, 3),
                                pd.Timestamp(2000, 1, 4)])

        testing.assert_frame_equal(expected_exog, new_exog_feature)

    def test_correct_nearest_pad_method(self):
        output_name = 'exog'
        method = 'nearest'
        exog = pd.DataFrame.from_dict({'old_name_1': [0, 1, 2, 3]})
        exog.index = ([pd.Timestamp(2000, 1, 1),
                       pd.Timestamp(2000, 2, 1),
                       pd.Timestamp(2000, 3, 1),
                       pd.Timestamp(2000, 4, 1)])

        df = pd.DataFrame.from_dict({'old_name_2': [10, 11, 12, 13]})
        df.index = ([pd.Timestamp(2000, 1, 1),
                     pd.Timestamp(2000, 1, 2),
                     pd.Timestamp(2000, 1, 3),
                     pd.Timestamp(2000, 1, 29)])

        exog_feature = ExogenousFeature(exogenous_time_series=exog,
                                        output_name=output_name,
                                        method=method)

        new_exog_feature = exog_feature.fit_transform(df)
        expected_exog = pd.DataFrame.from_dict({output_name: [0, 0, 0, 1]})
        expected_exog.index = ([pd.Timestamp(2000, 1, 1),
                                pd.Timestamp(2000, 1, 2),
                                pd.Timestamp(2000, 1, 3),
                                pd.Timestamp(2000, 1, 29)])

        testing.assert_frame_equal(expected_exog, new_exog_feature)

    def test_correct_multi_columns_exog(self):
        output_name = 'exog'
        exog = pd.DataFrame.from_dict({'old_name_0': [0, 1, 2, 3],
                                       'old_name_1': [5, 6, 7, 8]})
        exog.index = ([pd.Timestamp(2000, 1, 1),
                       pd.Timestamp(2000, 2, 1),
                       pd.Timestamp(2000, 3, 1),
                       pd.Timestamp(2000, 4, 1)])

        df = pd.DataFrame.from_dict({'old_name_2': [10, 11, 12, 13]})
        df.index = ([pd.Timestamp(2000, 1, 1),
                     pd.Timestamp(2000, 1, 2),
                     pd.Timestamp(2000, 1, 3),
                     pd.Timestamp(2000, 1, 29)])

        exog_feature = ExogenousFeature(exogenous_time_series=exog,
                                        output_name=output_name)

        new_exog_feature = exog_feature.fit_transform(df)
        expected_exog = pd.DataFrame.from_dict(
            {f'{output_name}_0': [0, np.nan, np.nan, np.nan],
             f'{output_name}_1': [5, np.nan, np.nan, np.nan]})
        expected_exog.index = ([pd.Timestamp(2000, 1, 1),
                                pd.Timestamp(2000, 1, 2),
                                pd.Timestamp(2000, 1, 3),
                                pd.Timestamp(2000, 1, 29)])

        testing.assert_frame_equal(expected_exog, new_exog_feature)


class TestPolynomialFeature:
    def _correct_pol_features(self, df: pd.DataFrame, degree: int) \
            -> pd.DataFrame:
        poly = PolynomialFeatures(degree)
        pol_features_arr = poly.fit_transform(df)
        pol_features = pd.DataFrame(pol_features_arr, index=df.index)
        return pol_features

    def test_correct_pol_features_single_column(self):
        output_name = 'pol_features'
        degree = 3
        df = pd.DataFrame.from_dict({'old_name': [0, 1, 2, 3]})
        df.index = ([pd.Timestamp(2000, 1, 1),
                     pd.Timestamp(2000, 2, 1),
                     pd.Timestamp(2000, 3, 1),
                     pd.Timestamp(2000, 4, 1)])

        pol_feature = PolynomialFeature(degree=degree, output_name=output_name)
        pol_df = pol_feature.fit_transform(df)

        expected_pol_df = pd.DataFrame.from_dict(
            {f'{output_name}_0': [1., 1., 1., 1.],
             f'{output_name}_1': [0., 1., 2., 3.],
             f'{output_name}_2': [0., 1., 4., 9.],
             f'{output_name}_3': [0., 1., 8., 27.]})
        expected_pol_df.index = ([pd.Timestamp(2000, 1, 1),
                                  pd.Timestamp(2000, 2, 1),
                                  pd.Timestamp(2000, 3, 1),
                                  pd.Timestamp(2000, 4, 1)])

        testing.assert_frame_equal(expected_pol_df, pol_df)

    def test_correct_pol_features_multi_columns(self):
        output_name = 'pol_features'
        degree = 2
        df = pd.DataFrame.from_dict({'old_name_1': [0, 2, 4],
                                     'old_name_2': [1, 3, 5]})
        df.index = ([pd.Timestamp(2000, 1, 1),
                     pd.Timestamp(2000, 2, 1),
                     pd.Timestamp(2000, 3, 1)])

        pol_feature = PolynomialFeature(degree=degree, output_name=output_name)
        pol_df = pol_feature.fit_transform(df)

        expected_pol_df = pd.DataFrame.from_dict(
            {f'{output_name}_0': [1., 1., 1],
             f'{output_name}_1': [0., 2., 4],
             f'{output_name}_2': [1., 3., 5],
             f'{output_name}_3': [0., 4., 16.],
             f'{output_name}_4': [0., 6., 20.],
             f'{output_name}_5': [1., 9., 25.]})
        expected_pol_df.index = ([pd.Timestamp(2000, 1, 1),
                                  pd.Timestamp(2000, 2, 1),
                                  pd.Timestamp(2000, 3, 1)])

        testing.assert_frame_equal(expected_pol_df, pol_df)


class TestCustomFeature:
    def _df_to_power(self, df: pd.DataFrame, power: int) -> pd.DataFrame:
        return np.power(df, power)

    def test_correct_custom_feature(self):
        output_name = 'custom'
        power = 3
        df = pd.DataFrame.from_dict({'old_name': [0, 1, 2, 3, 4, 5]})

        custom_feature = CustomFeature(
            custom_feature_function=self._df_to_power,
            output_name=output_name,
            power=power)

        output_custom_feature = custom_feature.fit_transform(df)
        expected_custom_output = pd.DataFrame.from_dict(
            {output_name: [0, 1, 8, 27, 64, 125]})

        testing.assert_frame_equal(expected_custom_output,
                                   output_custom_feature)

    def test_multi_columns_custom_feature(self):
        output_name = 'custom'
        power = 2
        df = pd.DataFrame.from_dict({'old_name': [0, 1, 2, 3, 4, 5],
                                     'old_name_1': [7, 8, 9, 10, 11, 12]})

        custom_feature = CustomFeature(
            custom_feature_function=self._df_to_power,
            output_name=output_name,
            power=power)

        output_custom_feature = custom_feature.fit_transform(df)
        expected_custom_output = pd.DataFrame.from_dict(
            {f'{output_name}_0': [0, 1, 4, 9, 16, 25],
             f'{output_name}_1': [49, 64, 81, 100, 121, 144]})

        testing.assert_frame_equal(expected_custom_output,
                                   output_custom_feature)

    @given(giotto_time_series(start_date=pd.Timestamp(2000, 1, 1),
                              end_date=pd.Timestamp(2010, 1, 1)
                              ),
           st.integers(0, 10)
           )
    def test_random_ts_and_power(self, df: pd.DataFrame, power: int):
        output_name = 'time_series'

        custom_feature = CustomFeature(self._df_to_power, output_name,
                                       power=power)

        output_custom = custom_feature.fit_transform(df)
        expected_custom_output = self._df_to_power(df, power)

        testing.assert_frame_equal(expected_custom_output, output_custom)
