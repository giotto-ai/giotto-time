from typing import Optional

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from .base import IndexDependentFeature

__all__ = [
    "ShiftFeature",
    "MovingAverageFeature",
    "PolynomialFeature",
    "ExogenousFeature",
]


class ShiftFeature(IndexDependentFeature):
    """Perform a shift of a DataFrame of size equal to ``shift``.

    Parameters
    ----------
    shift : int, optional, default: ``1``
        How much to shift.

    output_name : str, optional, default: ``'ShiftFeature'``
        The name of the output column.

    """

    def __init__(self, shift: int = 1, output_name: str = "ShiftFeature"):
        super().__init__(output_name)
        self.shift = shift

    def transform(self, ts: pd.DataFrame) -> pd.DataFrame:
        """Create a shifted version of ``ts``.

        Parameters
        ----------
        ts : pd.DataFrame, shape (n_samples, 1), required
            The DataFrame to shift.

        Returns
        -------
        ts_t : pd.DataFrame, shape (n_samples, 1)
            The shifted version of the original ``ts``.

        """
        ts_t = ts.shift(self.shift)
        ts_t = self._rename_columns(ts_t)
        return ts_t


class MovingAverageFeature(IndexDependentFeature):
    """For each row in ``ts``, compute the moving average of the previous
     ``window_size`` rows. If there are not enough rows, the value is Nan.

    Parameters
    ----------
    window_size : int, optional, default: ``1``
        The number of previous points on which to compute the moving average

    output_name : str, optional, default: ``'MovingAverageFeature'``
        The name of the output column.

    """

    def __init__(self, window_size: int = 1, output_name: str = "MovingAverageFeature"):
        super().__init__(output_name)
        self.window_size = window_size

    def transform(self, ts: pd.DataFrame) -> pd.DataFrame:
        """Compute the moving average, for every row of ``ts``, of the previous
        ``window_size`` elements.

        Parameters
        ----------
        ts : pd.DataFrame, shape (n_samples, 1), required
            The DataFrame on which to compute the rolling moving average

        Returns
        -------
        ts_t : pd.DataFrame, shape (n_samples, 1)
            A DataFrame, with the same length as ``ts``, containing the rolling
            moving average for each element.

        """
        ts_t = ts.rolling(self.window_size).mean().shift(1)
        ts_t = self._rename_columns(ts_t)
        return ts_t


class PolynomialFeature(IndexDependentFeature):
    """Compute the polynomial feature_creation, of a degree equal to the input
    ``degree``.

    Parameters
    ----------
    degree: int, optional, default: ``2``
        The degree of the polynomial feature_creation.

    output_name : str, optional, default: ``'PolynomialFeature'``
        The name of the output column.

    """

    def __init__(self, degree: int = 2, output_name: str = "PolynomialFeature"):
        super().__init__(output_name)
        self.degree = degree

    def transform(self, ts: pd.DataFrame) -> pd.DataFrame:
        """Compute the polynomial feature_creation of ``ts``, up to a degree
        equal to ``degree``.

        Parameters
        ----------
        ts : pd.DataFrame, shape (n_samples, 1), required
            The input DataFrame. Used only for its index.

        Returns
        -------
        ts_t : pd.DataFrame, shape (n_samples, 1)
            The computed polynomial feature_creation.

        """
        pol_feature = PolynomialFeatures(self.degree)
        pol_of_X_array = pol_feature.fit_transform(ts)
        pol_of_X = pd.DataFrame(pol_of_X_array, index=ts.index)

        ts_t = self._rename_columns(pol_of_X)

        return ts_t


class ExogenousFeature(IndexDependentFeature):
    """Reindex ``exogenous_time_series`` with the index of ``ts``. To check the
    documentation of ``pandas.DataFrame.reindex`` and to see which type of
    ``method`` are available, please refer to the pandas `documentation
    <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reindex.html>`_.

    Parameters
    ----------
    exogenous_time_series : pd.DataFrame, shape (n_samples, 1), required
        The time series to reindex

    method : str, optional, default: ``None``
        The method used to re-index. This must be a method used by the
        ``pandas.DataFrame.reindex`` method.

    output_name : str, optional, default: ``'ExogenousFeature'``
        The name of the output column.

    """

    def __init__(
        self,
        exogenous_time_series: pd.DataFrame,
        method: Optional[str] = None,
        output_name: str = "ExogenousFeature",
    ):
        super().__init__(output_name)
        self.method = method
        self.exogenous_time_series = exogenous_time_series

    def transform(self, ts: pd.DataFrame) -> pd.DataFrame:
        """Reindex the ``exogenous_time_series`` with the index of ``ts``.

        Parameters
        ----------
        ts : pd.DataFrame, shape (n_samples, 1), required
            The input DataFrame. Used only for its index.

        Returns
        -------
        ts_t :  pd.DataFrame, shape (n_samples, 1)
            The original ``exogenous_time_series``, re-indexed with the index
            of ``ts``.

        """
        exog_feature = self.exogenous_time_series.reindex(
            index=ts.index, method=self.method
        )
        ts_t = self._rename_columns(exog_feature)
        return ts_t
