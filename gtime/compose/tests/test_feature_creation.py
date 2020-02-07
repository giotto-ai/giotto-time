import pandas.util.testing as testing
from numpy.testing import assert_array_equal

from gtime.compose import FeatureCreation
from gtime.feature_extraction import Shift, MovingAverage


def test_feature_creation_transform():
    data = testing.makeTimeDataFrame(freq="s")
    fc = FeatureCreation([
        ('s1', Shift(1), ['A']),
        ('ma3', MovingAverage(window_size=3), ['B']),
    ])
    res = fc.fit(data).transform(data)

    assert_array_equal(res.columns, fc.fit_transform(data).columns.values)
