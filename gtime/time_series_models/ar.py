from typing import List, Union

import numpy as np
from sklearn.compose import make_column_selector
from sklearn.linear_model import LinearRegression

from gtime.feature_extraction import Shift
from gtime.forecasting import GAR
from gtime.time_series_models import TimeSeriesForecastingModel


class AR(TimeSeriesForecastingModel):
    """ Standard AR model for time series

    Parameters
    ----------
    p: int, required
        p parameter in AR
    horizon: int, required
        how many steps to predict in the future

    Examples
    --------
    >>> import pandas._testing as testing
    >>> from gtime.time_series_models import AR
    >>>
    >>> testing.N, testing.K = 20, 1
    >>> data = testing.makeTimeDataFrame(freq="s")
    >>> ar = AR(p=2, horizon=3, column_name='A')
    >>>
    >>> ar.fit(data)
    >>> ar.predict()
                              y_1       y_2       y_3
    2000-01-01 00:00:17  0.037228  0.163446 -0.237299
    2000-01-01 00:00:18 -0.139627 -0.018082  0.063273
    2000-01-01 00:00:19 -0.107707  0.052031 -0.105526
    """

    def __init__(self, p: int, horizon: Union[int, List[int]]):
        features = [
            tuple((f"s{i}", Shift(i), make_column_selector(dtype_include=np.number)))
            for i in range(p)
        ]
        model = GAR(LinearRegression())
        super().__init__(features=features, horizon=horizon, model=model)
