from random import randint

import numpy as np

from gtime.causality import ShiftedPearsonCorrelation
from gtime.causality.tests.common import make_df_from_expected_shifts


def test_pearson_correlation():
    expected_shifts = [randint(2, 6) * 2 for _ in range(3)]
    df = make_df_from_expected_shifts(expected_shifts)

    spc = ShiftedPearsonCorrelation(target_col="A", max_shift=12)
    spc.fit(df)

    shifts = spc.best_shifts_["A"][4:].values
    np.testing.assert_array_equal(shifts, expected_shifts)


def test_pearson_bootstrap_p_values():
    expected_shifts = [randint(2, 9) * 2 for _ in range(3)]
    df = make_df_from_expected_shifts(expected_shifts)
    shifted_test = ShiftedPearsonCorrelation(
        target_col="A", max_shift=5, bootstrap_iterations=500,
    )
    shifted_test.fit(df)

    pearson_p_values = shifted_test.bootstrap_p_values_
    for col_index in range(len(pearson_p_values.columns)):
        assert pearson_p_values.iloc[col_index, col_index] == 0


def test_pearson_permutation_p_values():
    expected_shifts = [randint(2, 9) * 2 for _ in range(3)]
    df = make_df_from_expected_shifts(expected_shifts)
    shifted_test = ShiftedPearsonCorrelation(
        target_col="A", max_shift=5, permutation_iterations=50,
    )
    shifted_test.fit(df)

    pearson_p_values = shifted_test.permutation_p_values_
    for col_index in range(len(pearson_p_values.columns)):
        assert pearson_p_values.iloc[col_index, col_index] == 0
