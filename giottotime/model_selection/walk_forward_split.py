import pandas as pd

from giottotime.feature_extraction import Shift


def walk_forward_split(time_series: pd.DataFrame, horizon) -> pd.DataFrame:
    """
    TODO: doc

    Parameters
    ----------
    time_series : List[TimeSeriesFeature], required
        The list of ``TimeSeriesFeature`` from which to compute the feature_extraction.

    horizon : int, optional, default: ``5``
        It represents how much into the future is necessary to predict. This corresponds
        to the number of shifts that are going to be performed on y.
    """

    y = pd.DataFrame(index=time_series.index)
    for k in range(1, horizon + 1):
        shift_feature = Shift(-k)
        y[f"y_{k}"] = shift_feature.fit_transform(time_series)

    return y
