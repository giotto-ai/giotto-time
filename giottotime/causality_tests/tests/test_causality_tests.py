from random import randint
from typing import List

import numpy as np
import pandas as pd
from pandas.util import testing as testing

from giottotime.causality_tests import (
    ShiftedLinearCoefficient,
    ShiftedPearsonCorrelation,
)


class TestCausalityTest:
    def test_causality_tests_shifts_shifted_pearson_correlation(self):
        expected_shifts = [randint(2, 9) * 2 for _ in range(3)]
        df = make_df_from_expected_shifts(expected_shifts)

        shifted_test = ShiftedPearsonCorrelation(target_col="A", max_shift=20)
        shifted_test.fit(df)

        shifts = shifted_test.best_shifts_.loc["A"][1:].values
        np.testing.assert_array_equal(shifts, expected_shifts)

    def test_causality_tests_shifts_shifted_linear_coefficient(self):
        expected_shifts = [randint(2, 9) * 2 for _ in range(3)]
        df = make_df_from_expected_shifts(expected_shifts)

        shifted_test = ShiftedLinearCoefficient(target_col="A", max_shift=20)
        shifted_test.fit(df)

        shifts = shifted_test.best_shifts_.loc["A"][1:].values
        np.testing.assert_array_equal(shifts, expected_shifts)

    def test_causality_tests_shifts_shifted_pearson_correlation_transform(self):
        expected_shifts = [randint(2, 9) * 2 for _ in range(3)]
        df = make_df_from_expected_shifts(expected_shifts)

        shifted_test = ShiftedPearsonCorrelation(target_col="A", max_shift=20)
        shifted_test.fit(df)

        transformation = shifted_test.transform(df).dropna()
        expected_transformation = shift_df_from_expected_shifts(df, expected_shifts)

        np.testing.assert_array_equal(transformation, expected_transformation)

    def test_causality_tests_shifts_shifted_linear_coefficient_transform(self):
        expected_shifts = [randint(2, 9) * 2 for _ in range(3)]
        df = make_df_from_expected_shifts(expected_shifts)

        shifted_test = ShiftedLinearCoefficient(target_col="A", max_shift=20)
        shifted_test.fit(df)

        transformation = shifted_test.transform(df).dropna()
        expected_transformation = shift_df_from_expected_shifts(df, expected_shifts)

        np.testing.assert_array_equal(transformation, expected_transformation)

    def test_linear_p_values(self):
        # This test and the next one just test if the p_values on the diagonal are equal
        # to 0. Is hard to implement other unittets, since the bootstrapping always
        # gives different result. However, other properties could be tested
        expected_shifts = [randint(2, 9) * 2 for _ in range(3)]
        df = make_df_from_expected_shifts(expected_shifts)
        shifted_test = ShiftedLinearCoefficient(
            target_col="A",
            max_shift=5,
            bootstrap_iterations=500,
            bootstrap_samples=1000,
        )
        shifted_test.fit(df)

        linear_p_values = shifted_test.p_values_
        for col_index in range(len(linear_p_values.columns)):
            assert linear_p_values.iloc[col_index, col_index] == 0

    def test_pearson_p_values(self):
        expected_shifts = [randint(2, 9) * 2 for _ in range(3)]
        df = make_df_from_expected_shifts(expected_shifts)
        shifted_test = ShiftedPearsonCorrelation(
            target_col="A",
            max_shift=5,
            bootstrap_iterations=500,
            bootstrap_samples=1000,
        )
        shifted_test.fit(df)

        pearson_p_values = shifted_test.p_values_
        for col_index in range(len(pearson_p_values.columns)):
            assert pearson_p_values.iloc[col_index, col_index] == 0


def make_df_from_expected_shifts(expected_shifts: List[int]) -> pd.DataFrame:
    testing.N, testing.K = 500, 1

    df = testing.makeTimeDataFrame(freq="D")
    for sh, k in zip(expected_shifts, range(3)):
        df[f"shift_{k}"] = df["A"].shift(sh)
    df = df.dropna()

    return df


def shift_df_from_expected_shifts(
    df: pd.DataFrame, expected_shifts: List[int]
) -> pd.DataFrame:
    for sh, k in zip(expected_shifts, range(3)):
        df[f"shift_{k}"] = df[f"shift_{k}"].shift(-sh)
    return df.dropna()
