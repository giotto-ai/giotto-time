from typing import Optional, Callable

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..base import FeatureMixin

__all__ = [
    "Shift",
    "MovingAverage",
    "MovingCustomFunction",
    "Polynomial",
    "Exogenous",
    "CustomFeature",
]


class Shift(BaseEstimator, TransformerMixin, FeatureMixin):
    """Perform a shift of a DataFrame of size equal to ``shift``.

    Parameters
    ----------
    shift : int, optional, default: ``1``
        How much to shift.

    Notes
    -----
    The ``shift`` parameter can also accept negative values. However, this should be
    used carefully, since if the resulting feature is used for training or testing it
    might generate a leak from the feature.

    Examples
    --------
    >>> import pandas as pd
    >>> from giottotime.feature_extraction import Shift
    >>> ts = pd.DataFrame([0, 1, 2, 3, 4, 5])
    >>> shift = Shift(shift=3)
    >>> shift.fit_transform(ts)
       0__Shift
    0       NaN
    1       NaN
    2       NaN
    3       0.0
    4       1.0
    5       2.0

    """

    def __init__(self, shift: int = 1):
        super().__init__()
        self.shift = shift

    def fit(self, X, y=None):
        """Fit the estimator.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Input data.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        self.columns_ = X.columns.values
        return self

    def transform(self, time_series: pd.DataFrame) -> pd.DataFrame:
        """Create a shifted version of ``time_series``.

        Parameters
        ----------
        time_series : pd.DataFrame, shape (n_samples, 1), required
            The DataFrame to shift.

        Returns
        -------
        time_series_t : pd.DataFrame, shape (n_samples, 1)
            The shifted version of the original ``time_series``.

        """
        check_is_fitted(self)

        time_series_shifted = time_series.shift(self.shift).add_suffix(
            "__" + self.__class__.__name__
        )
        return time_series_shifted


class MovingAverage(BaseEstimator, TransformerMixin, FeatureMixin):
    """For each row in ``time_series``, compute the moving average of the previous
     ``window_size`` rows. If there are not enough rows, the value is Nan.

    Parameters
    ----------
    window_size : int, optional, default: ``1``
        The number of previous points on which to compute the moving average

    Examples
    --------
    >>> import pandas as pd
    >>> from giottotime.feature_extraction import MovingAverage
    >>> ts = pd.DataFrame([0, 1, 2, 3, 4, 5])
    >>> mv_avg = MovingAverage(window_size=2)
    >>> mv_avg.fit_transform(ts)
       0__MovingAverage
    0               NaN
    1               0.5
    2               1.5
    3               2.5
    4               3.5
    5               4.5

    """

    def __init__(self, window_size: int = 1):
        super().__init__()
        self.window_size = window_size

    def fit(self, X, y=None):
        """Fit the estimator.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Input data.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        self.columns_ = X.columns.values
        return self

    def transform(self, time_series: pd.DataFrame) -> pd.DataFrame:
        """Compute the moving average, for every row of ``time_series``, of the previous
        ``window_size`` elements.

        Parameters
        ----------
        time_series : pd.DataFrame, shape (n_samples, 1), required
            The DataFrame on which to compute the rolling moving average

        Returns
        -------
        time_series_t : pd.DataFrame, shape (n_samples, 1)
            A DataFrame, with the same length as ``time_series``, containing the rolling
            moving average for each element.

        """
        check_is_fitted(self)

        time_series_mvg_avg = (
            time_series.rolling(self.window_size)
            .mean()
            .add_suffix("__" + self.__class__.__name__)
        )
        return time_series_mvg_avg


class MovingCustomFunction(BaseEstimator, TransformerMixin, FeatureMixin):
    """For each row in ``time_series``, compute the moving custom function of the
    previous ``window_size`` rows. If there are not enough rows, the value is Nan.

    Parameters
    ----------
    custom_feature_function : Callable, required.
        The function to use to generate a ``pd.DataFrame`` containing the feature.

    window_size : int, optional, default: ``1``
        The number of previous points on which to compute the custom function

    raw : bool, optional, default: ``True``
        - False : passes each row or column as a Series to the function.
        - True or None : the passed function will receive ndarray objects instead.
         If you are just applying a NumPy reduction function this will achieve much
         better performance. Credits: https://pandas.pydata.org/pandas-docs/stable/
         reference/api/pandas.core.window.Rolling.apply.html

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from giottotime.feature_extraction import MovingCustomFunction
    >>> ts = pd.DataFrame([0, 1, 2, 3, 4, 5])
    >>> mv_custom = MovingCustomFunction(np.max, window_size=2)
    >>> mv_custom.transform(ts)
       0__MovingCustomFunction
    0                      NaN
    1                      1.0
    2                      2.0
    3                      3.0
    4                      4.0
    5                      5.0

    """

    def __init__(
        self, custom_feature_function: Callable, window_size: int = 1, raw: bool = True,
    ):
        super().__init__()
        self.custom_feature_function = custom_feature_function
        self.window_size = window_size
        self.raw = raw

    def transform(self, time_series: pd.DataFrame) -> pd.DataFrame:
        """For every row of ``time_series``, compute the moving custom function of the
         previous ``window_size`` elements.

        Parameters
        ----------
        time_series : pd.DataFrame, shape (n_samples, 1), required
            The DataFrame on which to compute the rolling moving custom function.

        Returns
        -------
        time_series_t : pd.DataFrame, shape (n_samples, 1)
            A DataFrame, with the same length as ``time_series``, containing the rolling
            moving custom function for each element.

        """
        time_series_mvg_cust = time_series.rolling(self.window_size).apply(
            self.custom_feature_function, raw=self.raw
        )
        time_series_t = time_series_mvg_cust.add_suffix("__" + self.__class__.__name__)
        return time_series_t


# TODO: use make_column_transformer instead
class Polynomial(BaseEstimator, TransformerMixin, FeatureMixin):
    """Compute the polynomial feature_extraction, of a degree equal to the input
    ``degree``.

    Parameters
    ----------
    degree: int, optional, default: ``2``
        The degree of the polynomial feature_extraction.

    Examples
    --------
    >>> import pandas as pd
    >>> from giottotime.feature_extraction import Polynomial
    >>> ts = pd.DataFrame([0, 1, 2, 3, 4, 5])
    >>> pol = Polynomial(degree=3)
    >>> pol.fit_transform(ts)
       0__Polynomial  1__Polynomial  2__Polynomial  3__Polynomial
    0            1.0            0.0            0.0            0.0
    1            1.0            1.0            1.0            1.0
    2            1.0            2.0            4.0            8.0
    3            1.0            3.0            9.0           27.0
    4            1.0            4.0           16.0           64.0
    5            1.0            5.0           25.0          125.0

    """

    def __init__(self, degree: int = 2):
        super().__init__()
        self.degree = degree

    def get_feature_names(self):
        """Return feature names for output features.

        Returns
        -------
        output_feature_names : ndarray, shape (n_output_features,)
            Array of feature names.

        """
        return [
            f"_{index}" + self.__class__.__name__ for index in range(self.degree + 1)
        ]

    def fit(self, time_series: pd.DataFrame, y=None):
        """Fit the estimator.

        Parameters
        ----------
        time_series : pd.DataFrame, shape (n_samples, n_features)
            Input data.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        self.columns_ = time_series.columns.values
        return self

    def transform(self, time_series: pd.DataFrame) -> pd.DataFrame:
        """Compute the polynomial feature_extraction of ``time_series``, up to a degree
        equal to ``degree``.

        Parameters
        ----------
        time_series : pd.DataFrame, shape (n_samples, 1), required
            The input DataFrame. Used only for its index.

        Returns
        -------
        time_series_t : pd.DataFrame, shape (n_samples, 1)
            The computed polynomial feature_extraction.

        """
        check_is_fitted(self)

        pol_feature = PolynomialFeatures(self.degree)
        pol_of_X_array = pol_feature.fit_transform(time_series)
        pol_of_X = pd.DataFrame(pol_of_X_array, index=time_series.index).add_suffix(
            "__" + self.__class__.__name__
        )

        return pol_of_X


class Exogenous(BaseEstimator, TransformerMixin, FeatureMixin):
    """Reindex ``exogenous_time_series`` with the index of ``time_series``. To check the
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

    Examples
    --------
    >>> import pandas as pd
    >>> from giottotime.feature_extraction import Exogenous
    >>> ts = pd.DataFrame([0, 1, 2, 3, 4, 5], index=[3, 4, 5, 6, 7, 8])
    >>> exog_ts = pd.DataFrame([10, 8, 1, 3, 2, 7])
    >>> exog = Exogenous(exog_ts)
    >>> exog.fit_transform(ts)
       0__Exogenous
    3           3.0
    4           2.0
    5           7.0
    6           NaN
    7           NaN
    8           NaN

    >>> exog = Exogenous(exog_ts, method="nearest")
    >>> exog.fit_transform(ts)
       0__Exogenous
    3             3
    4             2
    5             7
    6             7
    7             7
    8             7

    """

    def __init__(
        self, exogenous_time_series: pd.DataFrame, method: Optional[str] = None,
    ):
        super().__init__()
        self.method = method
        self.exogenous_time_series = exogenous_time_series

    def fit(self, time_series, y=None):
        """Fit the estimator.

        Parameters
        ----------
        time_series : pd.DataFrame, shape (n_samples, n_features)
            Input data.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        self.columns_ = time_series.columns.values
        return self

    def transform(self, time_series: pd.DataFrame) -> pd.DataFrame:
        """Reindex the ``exogenous_time_series`` with the index of ``time_series``.

        Parameters
        ----------
        time_series : pd.DataFrame, shape (n_samples, 1), required
            The input DataFrame. Used only for its index.

        Returns
        -------
        time_series_t :  pd.DataFrame, shape (n_samples, 1)
            The original ``exogenous_time_series``, re-indexed with the index
            of ``time_series``.

        """
        check_is_fitted(self)

        exog_feature = self.exogenous_time_series.reindex(
            index=time_series.index, method=self.method
        ).add_suffix("__" + self.__class__.__name__)

        return exog_feature


class CustomFeature(BaseEstimator, TransformerMixin, FeatureMixin):
    """Given a custom function, apply it to a time series and generate a
    ``pd.Dataframe``.

    Parameters
    ----------
    custom_feature_function : Callable, required.
        The function to use to generate a ``pd.DataFrame`` containing the feature.

    kwargs : ``object``, optional.
        Optional arguments to pass to the function.

    Examples
    --------
    >>> import pandas as pd
    >>> from giottotime.feature_extraction import CustomFeature
    >>> def custom_function(X, power):
    ...     return X**power
    >>> X = pd.DataFrame([0, 1, 2, 3, 4, 5])
    >>> custom_feature = CustomFeature(custom_function, power=3)
    >>> custom_feature.fit_transform(X)
       0__CustomFeature
    0                 0
    1                 1
    2                 8
    3                27
    4                64
    5               125

    """

    def __init__(
        self, custom_feature_function: Callable, **kwargs: object,
    ):
        super().__init__()
        self.custom_feature_function = custom_feature_function
        self.kwargs = kwargs

    def fit(self, time_series, y=None):
        """Fit the estimator.

        Parameters
        ----------
        time_series : pd.DataFrame, shape (n_samples, n_features)
            Input data.

        y : None
            There is no need of a target in a transformer, yet the pipeline API requires
            this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        self.columns_ = time_series.columns.values
        return self

    def transform(self, time_series: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate a ``pd.DataFrame``, given ``time_series`` as input to the
        ``custom_feature_function``, as well as other optional arguments.

        Parameters
        ----------
        time_series : pd.DataFrame, shape (n_samples, 1), optional, default: ``None``
            The DataFrame on which to apply the the custom function.

        Returns
        -------
        custom_feature_renamed : pd.DataFrame, shape (length, 1)
            A DataFrame containing the generated feature_creation.

        Notes
        -----
        In order to use the ``CustomFeature`` class inside a
        ``giottotime.feature_creation.FeatureCreation`` class, the output of  the custom
         function should be a ``pd.DataFrame`` and have the same index as
         ``time_series``.
        """
        check_is_fitted(self)

        custom_feature = self.custom_feature_function(time_series, **self.kwargs)
        return custom_feature.add_suffix("__" + self.__class__.__name__)
