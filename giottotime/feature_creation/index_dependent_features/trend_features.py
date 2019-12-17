from typing import Callable

from sklearn.metrics import mean_squared_error
import pandas as pd

from .base import IndexDependentFeature
from giottotime.models.trend_models.base import TrendModel
from giottotime.models.trend_models.polynomial_trend import PolynomialTrend
from giottotime.models.trend_models.exponential_trend import ExponentialTrend

__all__ = [
    "DetrendedFeature",
    "RemovePolynomialTrend",
    "RemoveExponentialTrend",
]


class DetrendedFeature(IndexDependentFeature):
    """Apply a de-trend transformation to a time series.

    Parameters
    ----------
    trend_model : ``TrendModel``, required.
        The kind of trend removal to apply.

    output_name : ``str``, required.
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
        time_series : ``pd.DataFrame``, required.
            The time series on which to apply the de-trend transformation.

        Returns
        -------
        de-trended_time_series : ``pd.DataFrame``
            The de-trended time series.

        """
        self.trend_model.fit(time_series)
        return self.trend_model.transform(time_series)


class RemovePolynomialTrend(DetrendedFeature):
    def __init__(
        self,
        polynomial_order: int = 1,
        loss: Callable = mean_squared_error,
        output_name: str = "RemovePolynomialTrend",
    ):
        self.trend_model = PolynomialTrend(order=polynomial_order, loss=loss)
        super().__init__(trend_model=self.trend_model, output_name=output_name)


class RemoveExponentialTrend(DetrendedFeature):
    def __init__(
        self,
        loss: Callable = mean_squared_error,
        output_name: str = "RemoveExponentialTrend",
    ):
        self.trend_model = ExponentialTrend(loss=loss)
        super().__init__(trend_model=self.trend_model, output_name=output_name)
