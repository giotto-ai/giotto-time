import pandas as pd
if pd.__version__ >= '1.0.0':
    import pandas._testing as testing
else:
    import pandas.util.testing as testing
from numpy.testing import assert_array_equal

from gtime.compose import FeatureCreation
from gtime.feature_extraction import Shift, MovingAverage


def test_feature_creation_transform():
    data = testing.makeTimeDataFrame(freq="s")

    shift = Shift(1)
    ma = MovingAverage(window_size=3)

    col_name = "A"

    fc = FeatureCreation([("s1", shift, [col_name]), ("ma3", ma, [col_name]),])
    res = fc.fit(data).transform(data)

    assert_array_equal(
        res.columns.values,
        [
            f"s1__{col_name}__{shift.__class__.__name__}",
            f"ma3__{col_name}__{ma.__class__.__name__}",
        ],
    )
