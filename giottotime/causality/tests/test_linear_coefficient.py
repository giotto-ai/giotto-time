from random import randint

import numpy as np
import pytest
from hypothesis import given, strategies as st
from pandas.util import testing as testing

from giottotime.causality import ShiftedLinearCoefficient

from giottotime.causality.tests.common import make_df_from_expected_shifts, shift_df_from_expected_shifts


def test_linear_coefficient():
    expected_shifts = [randint(2, 9) * 2 for _ in range(3)]
    df = make_df_from_expected_shifts(expected_shifts)

    slc = ShiftedLinearCoefficient(target_col="A", max_shift=20)
    slc.fit(df)

    shifts = slc.best_shifts_.loc["A"][1:].values
    np.testing.assert_array_equal(shifts, expected_shifts)


# TODO: tests refactor TBD
@given(st.integers(1, 20))
def test_linear_coefficient_hyp(shift):
    testing.N, testing.K = 500, 1
    df = testing.makeTimeDataFrame(freq="D")
    df["shifted"] = df["A"].shift(shift)

    slc = ShiftedLinearCoefficient(target_col="A", max_shift=20)
    a = slc.fit(df).transform(df)
    print(slc.best_shifts_)


def test_linear_p_values():
    # This test and the next one just test if the p_values on the diagonal are equal
    # to 0. Is hard to implement other unittest, since the bootstrapping always
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
