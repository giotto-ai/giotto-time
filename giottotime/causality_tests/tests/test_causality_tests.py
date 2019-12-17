from random import randint

import numpy as np
from pandas.util import testing as testing

from giottotime.causality_tests import (
    ShiftedLinearCoefficient,
    ShiftedPearsonCorrelation,
)


class TestCausalityTest:
    def test_causality_tests_shifts_shifted_pearson_correlation(self):
        testing.N, testing.K = 500, 1
        df = testing.makeTimeDataFrame(freq="D")

        correct_shifts = [randint(2, 9) * 2 for _ in range(3)]

        for sh, k in zip(correct_shifts, range(3)):
            df[f"shift_{k}"] = df["A"].shift(sh)

        df = df.dropna()

        shifted_test = ShiftedPearsonCorrelation(target_col="A", max_shift=20)
        shifted_test.fit(df)

        shifts = shifted_test.best_shifts_.loc["A"][1:].values
        np.testing.assert_array_equal(shifts, correct_shifts)

    def test_causality_tests_shifts_shifted_linear_coefficient(self):
        testing.N, testing.K = 500, 1
        df = testing.makeTimeDataFrame(freq="D")

        correct_shifts = [randint(2, 9) * 2 for _ in range(3)]

        for sh, k in zip(correct_shifts, range(3)):
            df[f"shift_{k}"] = df["A"].shift(sh)

        df = df.dropna()

        shifted_test = ShiftedLinearCoefficient(target_col="A", max_shift=20)
        shifted_test.fit(df)

        shifts = shifted_test.best_shifts_.loc["A"][1:].values
        print(shifts)
        print(correct_shifts)
        np.testing.assert_array_equal(shifts, correct_shifts)

    def test_causality_tests_shifts_shifted_pearson_correlation_transform(self):
        testing.N, testing.K = 500, 1
        df = testing.makeTimeDataFrame(freq="D")

        correct_shifts = [randint(2, 9) * 2 for _ in range(3)]

        for sh, k in zip(correct_shifts, range(3)):
            df[f"shift_{k}"] = df["A"].shift(sh)

        df = df.dropna()

        shifted_test = ShiftedPearsonCorrelation(target_col="A", max_shift=20)
        shifted_test.fit(df)

        trans = shifted_test.transform(df)

        for sh, k in zip(correct_shifts, range(3)):
            df[f"shift_{k}"] = df[f"shift_{k}"].shift(sh)

        df = df.dropna()
        trans = trans.dropna()

        np.testing.assert_array_equal(trans, df)

    def test_causality_tests_shifts_shifted_linear_coefficient_transform(self):
        testing.N, testing.K = 500, 1
        df = testing.makeTimeDataFrame(freq="D")

        correct_shifts = [randint(2, 9) * 2 for _ in range(3)]

        for sh, k in zip(correct_shifts, range(3)):
            df[f"shift_{k}"] = df["A"].shift(sh)

        df = df.dropna()

        shifted_test = ShiftedLinearCoefficient(target_col="A", max_shift=20)
        shifted_test.fit(df)

        trans = shifted_test.transform(df)

        for sh, k in zip(correct_shifts, range(3)):
            df[f"shift_{k}"] = df[f"shift_{k}"].shift(sh)

        df = df.dropna()
        trans = trans.dropna()

        np.testing.assert_array_equal(trans, df)
