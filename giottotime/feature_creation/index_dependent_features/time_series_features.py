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
    shift : ``int``, required.
        How much to shift.

    output_name : ``str``, required.
        The name of the output column.

    """

    def __init__(self, shift: int, output_name: str):
        super().__init__(output_name)
        self.shift = shift

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create a shifted version of ``X``.

        Parameters
        ----------
        X : ``pd.DataFrame``, required.
            The DataFrame to shift.

        Returns
        -------
        X_shifted_renamed : ``pd.DataFrame``
            The shifted version of the original ``X``.

        """
        X_shifted = X.shift(self.shift)
        X_shifted_renamed = self._rename_columns(X_shifted)
        return X_shifted_renamed


class MovingAverageFeature(IndexDependentFeature):
    """For each row in ``X``, compute the moving average of the previous
     ``window_size`` rows. If there are not enough rows, the value is Nan.

    Parameters
    ----------
    window_size : ``int``, required.
        The number of previous points on which to compute the moving average

    output_name : ``str``, required.
        The name of the output column.

    """

    def __init__(self, window_size: int, output_name: str):
        super().__init__(output_name)
        self.window_size = window_size

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute the moving average, for every row of ``X``, of the previous
        ``window_size`` elements.

        Parameters
        ----------
        X : ``pd.DataFrame``, required.
            The DataFrame on which to compute the rolling moving average

        Returns
        -------
        X_mov_avg_renamed : ``pd.DataFrame``
            A DataFrame, with the same length as ``X``, containing the rolling
            moving average for each element.

        """
        X_mov_avg = X.rolling(self.window_size).mean().shift(1)
        X_mov_avg_renamed = self._rename_columns(X_mov_avg)
        return X_mov_avg_renamed


class PolynomialFeature(IndexDependentFeature):
    """Compute the polynomial feature_creation, of a degree equal to the input
    ``degree``.

    Parameters
    ----------
    degree: ``int``, required.
        The degree of the polynomial feature_creation.

    output_name : ``str``, required.
        The name of the output column.
    """

    def __init__(self, degree: int, output_name: str):
        super().__init__(output_name)
        self._degree = degree

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute the polynomial feature_creation of ``X``, up to a degree
        equal to ``degree``.

        Parameters
        ----------
        X : ``pd.DataFrame``, required.
            The input DataFrame. Used only for its index.

        Returns
        -------
        pol_of_X_renamed : ``pd.DataFrame``
            The computed polynomial feature_creation.

        """
        pol_feature = PolynomialFeatures(self._degree)
        pol_of_X_array = pol_feature.fit_transform(X)
        pol_of_X = pd.DataFrame(pol_of_X_array, index=X.index)

        pol_of_X_renamed = self._rename_columns(pol_of_X)

        return pol_of_X_renamed


class ExogenousFeature(IndexDependentFeature):
    """Reindex ``exogenous_time_series`` with the index of ``X``. To check the
    documentation of ``pandas.DataFrame.reindex`` and to see which type of
    ``method`` are available, please refer to the pandas `documentation
    <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reindex.html>`_.

    Parameters
    ----------
    exogenous_time_series : ``pd.DataFrame``, required.
        The time-series to reindex
        None, ‘backfill’/’bfill’, ‘pad’/’ffill’, ‘nearest’
    output_name : ``str``, required.
        The name of the output column.

    method : ``str``, optional, (default=``None``).
        The method used to re-index. This must be a method used by the
        ``pandas.DataFrame.reindex`` method.
    """

    def __init__(
        self,
        exogenous_time_series: pd.DataFrame,
        output_name: str,
        method: Optional[str] = None,
    ):
        super().__init__(output_name)
        self._method = method
        self.exogenous_time_series = exogenous_time_series

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reindex the ``exogenous_time_series`` with the index of ``X``.

        Parameters
        ----------
        X : ``pd.DataFrame``, required.
            The input DataFrame. Used only for its index.

        Returns
        -------
        exog_feature_renamed : ``pd.DataFrame``
            The original ``exogenous_time_series``, re-indexed with the index
            of ``X``.

        """
        exog_feature = self.exogenous_time_series.reindex(
            index=X.index, method=self._method
        )
        exog_feature_renamed = self._rename_columns(exog_feature)
        return exog_feature_renamed
