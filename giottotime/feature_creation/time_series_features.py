from typing import Callable, Optional

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from .base import TimeSeriesFeature

__all__ = [
    "ShiftFeature",
    "MovingAverageFeature",
    "ConstantFeature",
    "PolynomialFeature",
    "ExogenousFeature",
    "CustomFeature",
]


class ShiftFeature(TimeSeriesFeature):
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


class MovingAverageFeature(TimeSeriesFeature):
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


class ConstantFeature(TimeSeriesFeature):
    """Generate a ``pd.DataFrame`` with one column, of the same length as the
    input ``X`` and containing the value ``constant`` across the whole column.

    Parameters
    ----------
    constant : ``int``, required.
        The value to use to generate the constant column of the
        ``pd.DataFrame``.

    output_name : ``str``, required.
        The name of the output column.

    """

    def __init__(self, constant: int, output_name: str):
        super().__init__(output_name)
        self.constant = constant

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate a ``pd.DataFrame`` with one column with the same length as
        ``X`` and with the same index, containing a value equal to
        ``constant``.

        Parameters
        ----------
        X : ``pd.DataFrame``, required.
            The input DataFrame. It is used only for its index.

        Returns
        -------
        constant_series_renamed : ``pd.DataFrame``
            A constant series, with the same length of ``X`` and with the same
            index.

        """
        constant_series = pd.Series(data=self.constant, index=X.index).to_frame()
        constant_series_renamed = self._rename_columns(constant_series)
        return constant_series_renamed


class PolynomialFeature(TimeSeriesFeature):
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

    # TODO: finish the polynomial feature_creation
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


class ExogenousFeature(TimeSeriesFeature):
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


class CustomFeature(TimeSeriesFeature):
    """Given a custom function, apply it to a time series and generate a
    ``pd.Dataframe``.

    Parameters
    ----------
    custom_feature_function`: ``Callable`, required.
        The function to use to generate a ``pd.DataFrame`` containing the
        feature.

    output_name: ``str``, required.
        The name of the output column.

    kwargs : ``object``, optional.
        Optional arguments to pass to the function.

    """

    def __init__(
        self, custom_feature_function: Callable, output_name: str, **kwargs: object
    ):
        super().__init__(output_name)
        self.custom_feature_function = custom_feature_function
        self.kwargs = kwargs

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate a ``pd.DataFrame``, given ``X`` as input to the
        ``custom_feature_function``, as well as other optional arguments.

        Parameters
        ----------
        X : ``pd.DataFrame``, required.
            The DataFrame from which to generate the feature_creation

        Returns
        -------
        custom_feature_renamed : ``pd.DataFrame``
            A DataFrame containing the generated feature_creation.

        """
        custom_feature = self.custom_feature_function(X, **self.kwargs)
        custom_feature_renamed = self._rename_columns(custom_feature)
        return custom_feature_renamed
