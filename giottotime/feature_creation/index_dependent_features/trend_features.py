from abc import ABC
from typing import Callable

from sklearn.metrics import mean_squared_error
import pandas as pd

from .base import IndexDependentFeature
from giottotime.models.trend_models import TrendModel
from giottotime.models.trend_models import PolynomialTrend
from giottotime.models.trend_models import ExponentialTrend

__all__ = [
    "DetrendedFeature",
    "RemovePolynomialTrend",
    "RemoveExponentialTrend",
]


class DetrendedFeature(IndexDependentFeature):
    """Apply a de-trend transformation to a time series.

    Parameters
    ----------
    trend_model : TrendModel, optional, default: ``PolynomialTrend()``
        The kind of trend removal to apply.

    output_name : str, optional, default: ``'DetrendedFeature'``
        The name of the output column.

    Examples
    --------
    >>> import pandas as pd
    >>> from giottotime.feature_creation import DetrendedFeature
    >>> from giottotime.models import PolynomialTrend
    >>> model = PolynomialTrend(order=2)
    >>> detrend_feature = DetrendedFeature(trend_model=model)
    >>> time_index = pd.date_range("2020-01-01", "2020-01-10")
    >>> X = pd.DataFrame(range(0, 10), index=time_index)
    >>> detrend_feature.transform(X)
                DetrendedFeature
    2020-01-01      2.092234e-06
    2020-01-02      6.590209e-07
    2020-01-03     -4.104701e-07
    2020-01-04     -1.116238e-06
    2020-01-05     -1.458284e-06
    2020-01-06     -1.436607e-06
    2020-01-07     -1.051207e-06
    2020-01-08     -3.020852e-07
    2020-01-09      8.107597e-07
    2020-01-10      2.287327e-06

    """

    def __init__(
        self, trend_model: TrendModel, output_name: str = "DetrendedFeature",
    ):
        super().__init__(output_name)
        self.trend_model = trend_model

    def transform(self, time_series: pd.DataFrame) -> pd.DataFrame:
        """Apply a de-trend transformation to the input ``time_series``.

        Parameters
        ----------
        time_series : pd.DataFrame, shape (n_samples, 1), required
            The time series on which to apply the de-trend transformation.

        Returns
        -------
        time_series_t : pd.DataFrame, shape (n_samples, 1)
            The de-trended time series.

        """
        self.trend_model.fit(time_series)
        time_series_t = self.trend_model.transform(time_series)
        time_series_t = self._rename_columns(time_series_t)
        return time_series_t


class RemovePolynomialTrend(DetrendedFeature):
    """Apply a de-trend transformation to a time series using a polynomial with a given
    degree.

    Parameters
    ----------
    polynomial_order : int, optional, default: ``1``
        The order of the polynomial to use when de-trending the time series.

    loss : Callable, optional, default: ``mean_squared_error``
        The function to use in order to minimize the loss.

    output_name : str, optional, default: ``'RemovePolynomialTrend'``
        The name of the output column.

    Examples
    --------
    >>> import pandas as pd
    >>> from giottotime.feature_creation import RemovePolynomialTrend
    >>> detrend_feature = RemovePolynomialTrend(polynomial_order=4)
    >>> time_index = pd.date_range("2020-01-01", "2020-01-10")
    >>> X = pd.DataFrame(range(0, 10), index=time_index)
    >>> detrend_feature.transform(X)
                RemovePolynomialTrend
    2020-01-01               0.000969
    2020-01-02              -0.001216
    2020-01-03              -0.000915
    2020-01-04               0.000201
    2020-01-05               0.001021
    2020-01-06               0.000995
    2020-01-07               0.000133
    2020-01-08              -0.000993
    2020-01-09              -0.001249
    2020-01-10               0.001059

    """

    def __init__(
        self,
        polynomial_order: int = 2,
        loss: Callable = mean_squared_error,
        output_name: str = "RemovePolynomialTrend",
    ):
        trend_model = PolynomialTrend(order=polynomial_order, loss=loss)
        super().__init__(trend_model=trend_model, output_name=output_name)


class RemoveExponentialTrend(DetrendedFeature):
    """Apply a de-trend transformation to a time series using an exponential function.

    Parameters
    ----------
    loss : Callable, optional, default: ``mean_squared_error``
        The function to use in order to minimize the loss.

    output_name : str, optional, default: ``'RemoveExponentialTrend'``
        The name of the output column.

    Examples
    --------
    >>> import pandas as pd
    >>> from giottotime.feature_creation import RemoveExponentialTrend
    >>> detrend_feature = RemoveExponentialTrend()
    >>> time_index = pd.date_range("2020-01-01", "2020-01-10")
    >>> X = pd.DataFrame(range(0, 10), index=time_index)
    >>> detrend_feature.transform(X)
                RemoveExponentialTrend
    2020-01-01               -1.000000
    2020-01-02               -0.295788
    2020-01-03                0.320933
    2020-01-04                0.824285
    2020-01-05                1.180734
    2020-01-06                1.346829
    2020-01-07                1.266264
    2020-01-08                0.866080
    2020-01-09                0.051740
    2020-01-10               -1.299262

    """

    def __init__(
        self,
        loss: Callable = mean_squared_error,
        output_name: str = "RemoveExponentialTrend",
    ):
        trend_model = ExponentialTrend(loss=loss)
        super().__init__(trend_model=trend_model, output_name=output_name)
