from random import randint

import numpy as np
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

    transformation = slc.transform(df).dropna()
    expected_transformation = shift_df_from_expected_shifts(df, expected_shifts)

    np.testing.assert_array_equal(transformation, expected_transformation)


# TODO: tests refactor TBD
@given(st.integers(1, 20))
def test_linear_coefficient_hyp(shift):
    testing.N, testing.K = 500, 1
    df = testing.makeTimeDataFrame(freq="D")
    df["shifted"] = df["A"].shift(shift)

    slc = ShiftedLinearCoefficient(target_col="A", max_shift=20)
    a = slc.fit(df).transform(df)
    print(slc.best_shifts_)
