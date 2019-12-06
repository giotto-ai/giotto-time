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
    "RemoveFunctionTrend",
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

    def __init__(self, trend_model: TrendModel, output_name: str):
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
    def __init__(self, output_name: str, polynomial_order=1, loss=mean_squared_error):
        self.trend_model = PolynomialTrend(order=polynomial_order, loss=loss)
        super().__init__(trend_model=self.trend_model, output_name=output_name)


class RemoveExponentialTrend(DetrendedFeature):
    def __init__(self, output_name: str, loss=mean_squared_error):
        self.trend_model = ExponentialTrend(loss=loss)
        super().__init__(trend_model=self.trend_model, output_name=output_name)


class RemoveFunctionTrend(DetrendedFeature):
    def __init__(self, output_name: str, loss=mean_squared_error):
        self.trend_model = ExponentialTrend(loss=loss)
        super().__init__(trend_model=self.trend_model, output_name=output_name)
