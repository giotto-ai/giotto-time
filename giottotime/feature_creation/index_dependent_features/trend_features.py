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

    """

    def __init__(
        self,
        trend_model: TrendModel = PolynomialTrend(),
        output_name: str = "DetrendedFeature",
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

    """

    def __init__(
        self,
        polynomial_order: int = 1,
        loss: Callable = mean_squared_error,
        output_name: str = "RemovePolynomialTrend",
    ):
        self.trend_model = PolynomialTrend(order=polynomial_order, loss=loss)
        super().__init__(trend_model=self.trend_model, output_name=output_name)


class RemoveExponentialTrend(DetrendedFeature):
    """Apply a de-trend transformation to a time series using an exponential function.

    Parameters
    ----------
    loss : Callable, optional, default: ``mean_squared_error``
        The function to use in order to minimize the loss.

    output_name : str, optional, default: ``'RemoveExponentialTrend'``
        The name of the output column.

    """

    def __init__(
        self,
        loss: Callable = mean_squared_error,
        output_name: str = "RemoveExponentialTrend",
    ):
        self.trend_model = ExponentialTrend(loss=loss)
        super().__init__(trend_model=self.trend_model, output_name=output_name)
