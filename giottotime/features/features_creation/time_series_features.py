from typing import Callable

import pandas as pd

from giottotime.features.features_creation.base import TimeSeriesFeature


class ShiftFeature(TimeSeriesFeature):
    """Perform a shift of a DataFrame of size equal to ``shift``.

    Parameters
    ----------
    shift: int
        How much to shift.

    output_name: str
        The name of the output column.

    """
    def __init__(self, shift: int, output_name: str):
        super().__init__(output_name)
        self.shift = shift

    def transform(self, X: pd.DataFrame) -> pd.Series:
        """Create a shifted version of ``X``.

        Parameters
        ----------
        X: pd.DataFrame
            The DataFrame to shift.

        Returns
        -------
        X_shifted: pd.DataFrame
            The shifted version of the original ``X``.

        """
        return X.shift(self.shift)


class MovingAverageFeature(TimeSeriesFeature):
    """For each row in ``X``, compute the moving average of the previous
     ``window_size`` rows. If there are not enough rows, the value is Nan.

    Parameters
    ----------
    window_size: int
        The number of previous points on which to compute the moving average

    output_name: str
        The name of the output column.

    """
    def __init__(self, window_size: int, output_name: str):
        super().__init__(output_name)
        self.window_size = window_size

    def transform(self, X: pd.DataFrame) -> pd.Series:
        """Compute the moving average, for every row of ``X``, of the previous
        ``window_size`` elements.

        Parameters
        ----------
        X: pd.DataFrame
            The DataFrame on which to compute the rolling moving average

        Returns
        -------
        X_mov_avg: pd.Series
            A Series, with the same length as ``X``, containing the rolling
            moving average for each element.

        """
        X_mov_avg = X.rolling(self.window_size).mean().shift(1)
        X_mov_avg.name = self.output_name
        return X_mov_avg


class ConstantFeature(TimeSeriesFeature):
    """Generate a constant Series, of the same length as the input ``X`` and
    containing the value ``constant`` across the whole Series.

    Parameters
    ----------
    constant: int
        The value to use to generate the Series.

    output_name: str
        The name of the output column.

    """
    def __init__(self, constant: int, output_name: str):
        super().__init__(output_name)
        self.constant = constant

    def transform(self, X: pd.DataFrame) -> pd.Series:
        """Generate a constant Series, with the same length as ``X`` and with
        the same index.

        Parameters
        ----------
        X: pd.DataFrame
            The input DataFrame. It is used only for its index.

        Returns
        -------
        constant_series: pd.Series
            A constant series, with the same length of ``X`` and with the same
            index.

        """
        constant_series = pd.Series(data=self.constant, index=X.index)
        constant_series.name = self.output_name
        return constant_series


class PolynomialFeature(TimeSeriesFeature):
    def __init__(self, degree=2):
        self.degree = degree

    def fit_transform(self, time_series):
        pass


class ExogenousFeature(TimeSeriesFeature):
    """Reindex ``exogenous_time_series`` with the index of ``X``.

    Parameters
    ----------
    exogenous_time_series: pd.DataFrame
        The time-series to reindex

    output_name: str
        The name of the output column.

    """
    def __init__(self, exogenous_time_series: pd.DataFrame, output_name: str):
        super().__init__(output_name)
        self.exogenous_time_series = exogenous_time_series

    def transform(self, X: pd.DataFrame) -> pd.Series:
        """Reindex the ``exogenous_time_series`` with the index of ``X``.

        Parameters
        ----------
        X: pd.DataFrame
            The input DataFrame. Used only for its index.

        Returns
        -------
        exog_feature: pd.Series
            The original ``exogenous_time_series``, re-indexed with the index
            of ``X``.

        """
        exog_feature = self.exogenous_time_series.reindex(index=X.index)
        exog_feature.name = self.output_name
        return exog_feature


class CustomFeature(TimeSeriesFeature):
    """Given a custom function, generate a Series.

    Parameters
    ----------
    `custom_feature_function`: Callable
        The function to use to generate the Series containing the feature

    output_name: str
        The name of the output column.

    kwargs: dict
        Optional arguments to pass to the function.

    """
    def __init__(self, custom_feature_function: Callable, output_name: str,
                 **kwargs):
        super().__init__(output_name)
        self.custom_feature_function = custom_feature_function
        self.kwargs = kwargs

    def transform(self, X: pd.DataFrame) -> pd.Series:
        """Generate a Series, given ``X`` as input to the
        ``custom_feature_function``, as well as other optional arguments.

        Parameters
        ----------
        X: pd.DataFrame
            The DataFrame from which to generate the features

        Returns
        -------
        custom_feature: pd.Series
            A Series containing the generated features.

        """
        custom_feature = self.custom_feature_function(X, **self.kwargs)
        custom_feature.name = self.output_name
        return custom_feature
