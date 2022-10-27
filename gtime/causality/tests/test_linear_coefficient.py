from random import randint

import numpy as np
import pytest
from hypothesis import given, strategies as st
from pandas.util import testing as testing

from gtime.causality import ShiftedLinearCoefficient
from gtime.causality.tests.common import make_df_from_expected_shifts


def test_linear_coefficient():
    expected_shifts = [randint(2, 6) * 2 for _ in range(3)]

    df = make_df_from_expected_shifts(expected_shifts)
    slc = ShiftedLinearCoefficient(target_col="A", max_shift=12)
    slc.fit(df)

    shifts = slc.best_shifts_["A"][4:].values
    np.testing.assert_array_equal(shifts, expected_shifts)


# TODO: tests refactor TBD
@given(st.integers(1, 20))
@pytest.mark.skip(reason="TODO: Write proper test, increase hypothesis max duration")
def test_linear_coefficient_hyp(shift):
    testing.N, testing.K = 500, 1
    df = testing.makeTimeDataFrame(freq="D")
    df["shifted"] = df["A"].shift(shift)

    slc = ShiftedLinearCoefficient(target_col="A", max_shift=20)
    slc.fit(df).transform(df)


def test_linear_bootstrap_p_values():
    # This test and the next one just test if the p_values on the diagonal are equal
    # to 0. Is hard to implement other unittest, since the bootstrapping always
    # gives different result. However, other properties could be tested
    expected_shifts = [randint(2, 4) * 2 for _ in range(3)]
    df = make_df_from_expected_shifts(expected_shifts)
    shifted_test = ShiftedLinearCoefficient(
        target_col="A", max_shift=8, bootstrap_iterations=500,
    )
    shifted_test.fit(df)

    linear_p_values = shifted_test.bootstrap_p_values_
    for col_index in range(len(linear_p_values.columns)):
        assert linear_p_values.iloc[col_index, col_index] == 0


def test_linear_permutation_p_values():
    expected_shifts = [randint(2, 4) * 2 for _ in range(3)]
    df = make_df_from_expected_shifts(expected_shifts)
    shifted_test = ShiftedLinearCoefficient(
        target_col="A", max_shift=8, permutation_iterations=50,
    )
    shifted_test.fit(df)

    linear_p_values = shifted_test.permutation_p_values_
    for col_index in range(len(linear_p_values.columns)):
        assert linear_p_values.iloc[col_index, col_index] == 0
