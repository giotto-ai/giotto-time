from random import randint

import numpy as np

from giottotime.causality import ShiftedPearsonCorrelation
from giottotime.causality.tests.common import make_df_from_expected_shifts, shift_df_from_expected_shifts


def test_pearson_correlation():
    expected_shifts = [randint(2, 9) * 2 for _ in range(3)]
    df = make_df_from_expected_shifts(expected_shifts)

    spc = ShiftedPearsonCorrelation(target_col="A", max_shift=20)
    spc.fit(df)

    shifts = spc.best_shifts_.loc["A"][1:].values
    np.testing.assert_array_equal(shifts, expected_shifts)

    transformation = spc.transform(df).dropna()
    expected_transformation = shift_df_from_expected_shifts(df, expected_shifts)

    np.testing.assert_array_equal(transformation, expected_transformation)


