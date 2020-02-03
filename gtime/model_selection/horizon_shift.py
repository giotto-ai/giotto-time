import pandas as pd

from gtime.feature_extraction import Shift


def horizon_shift(time_series: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Perform a shift of the original ``time_series`` for each time step between 1 and
    ``horizon``.

    Parameters
    ----------
    time_series : pd.DataFrame, shape (n_samples, n_features), required
        The list of ``TimeSeriesFeature`` from which to compute the feature_extraction.

    horizon : int, optional, default: ``5``
        It represents how much into the future is necessary to predict. This corresponds
        to the number of shifts that are going to be performed on y.
        
    Returns
    -------
    y : pd.DataFrame, shape (n_samples, horizon)
        The shifted time series.

    Examples
    --------
    >>> import pandas as pd
    >>> from gtime.model_selection import horizon_shift
    >>> X = pd.DataFrame(range(0, 5), index=pd.date_range("2020-01-01", "2020-01-05"))
    >>> horizon_shift(X, horizon=2)
                y_1  y_2
    2020-01-01  1.0  2.0
    2020-01-02  2.0  3.0
    2020-01-03  3.0  4.0
    2020-01-04  4.0  NaN
    2020-01-05  NaN  NaN

    """
    y = pd.DataFrame(index=time_series.index)
    for k in range(1, horizon + 1):
        shift_feature = Shift(-k)
        y[f"y_{k}"] = shift_feature.fit_transform(time_series)

    return y
