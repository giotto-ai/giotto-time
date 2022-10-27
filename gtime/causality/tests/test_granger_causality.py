import numpy as np
import pytest
import pandas as pd


if pd.__version__ >= "1.0.0":
    import pandas._testing as testing
else:
    import pandas.util.testing as testing
from gtime.causality import GrangerCausality


# Expected values from results of statstools
@pytest.mark.parametrize(
    "test_input, expected",
    [
        (["ssr_f"], 0.8420421667509344),
        (["ssr_chi2"], 0.8327660223526767),
        (["likelihood_chi2"], 0.8341270186135072),
        (["zero_f"], 0.8420421667508992),
    ],
)
def test_granger_pvalues_ssr_f(test_input, expected):
    # Set random seed, otherwise testing creates a new dataframe each time.
    np.random.seed(12)

    data = testing.makeTimeDataFrame(freq="s", nper=1000)
    granger = (
        GrangerCausality(target_col="A", x_col="B", max_shift=10, statistics=test_input)
        .fit(data)
        .results_[0]
    )

    p_value = granger.values[1]
    # Not exactly equal but up test to 7 digits
    np.testing.assert_almost_equal(p_value, expected, decimal=7)
