import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class NaiveModel(BaseEstimator, RegressorMixin):

    """Naïve model, all predicted values are equal to the most recent available observation.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gtime.model_selection import horizon_shift, FeatureSplitter
    >>> from gtime.forecasting import NaiveModel
    >>> idx = pd.period_range(start='2011-01-01', end='2012-01-01')
    >>> np.random.seed(1)
    >>> df = pd.DataFrame(np.random.random((len(idx), 1)), index=idx, columns=['1'])
    >>> y = horizon_shift(df, horizon=3)
    >>> X_train, y_train, X_test, y_test = FeatureSplitter().transform(df, y)
    >>> m = NaiveModel()
    >>> m.fit(X_train, y_train).predict(X_test)
                         y_1       y_2       y_3
        2011-12-30  0.541559  0.541559  0.541559
        2011-12-31  0.974740  0.974740  0.974740
        2012-01-01  0.636604  0.636604  0.636604
    """

    def fit(self, X: pd.DataFrame, y=None):

        """Fit the estimator.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), train sample, required for compatibility, not used for a naive model.

        y : None
            Used to store the predict feature names and prediction horizon.

        Returns
        -------
        self : NaiveModel
            Returns self.
        """

        self._y_columns = y.columns
        self._horizon = len(y.columns)

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:

        """Using the value of each element in ``X`` predicts the rest of the forecast to be equal to it.

        Parameters
        ----------
        X: pd.DataFrame, shape (n_samples, 1), required
            The time series on which to predict.

        Returns
        -------
        predictions : pd.DataFrame, shape (n_samples, self._horizon)
            The output predictions.

        Raises
        ------
        NotFittedError
            Raised if the model is not fitted yet.

        """

        check_is_fitted(self)
        y_pred = np.broadcast_to(X, (len(X), self._horizon))
        predictions = pd.DataFrame(data=y_pred, columns=self._y_columns, index=X.index)

        return predictions


class SeasonalNaiveModel(NaiveModel):
    """Seasonal naïve model. The forecast is expected to follow a seasonal pattern of ``seasonal_length`` data points, which is determined by the last ``seasonal_length`` observations of a training dataset available.

    Parameters
    ----------
    seasonal_length: int, required
        Length of a full seasonal cycle in number of periods. Period length is inferred from the training data.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gtime.model_selection import horizon_shift, FeatureSplitter
    >>> from gtime.forecasting import SeasonalNaiveModel
    >>> idx = pd.period_range(start='2011-01-01', end='2012-01-01')
    >>> np.random.seed(1)
    >>> df = pd.DataFrame(np.random.random((len(idx), 1)), index=idx, columns=['1'])
    >>> y = horizon_shift(df, horizon=5)
    >>> X_train, y_train, X_test, y_test = FeatureSplitter().transform(df, y)
    >>> m = SeasonalNaiveModel(seasonal_length=3)
    >>> m.fit(X_train, y_train).predict(X_test)
                         y_1       y_2       y_3       y_4       y_5
        2011-12-28  0.990472  0.300248  0.782749  0.990472  0.300248
        2011-12-29  0.300248  0.782749  0.990472  0.300248  0.782749
        2011-12-30  0.782749  0.990472  0.300248  0.782749  0.990472
        2011-12-31  0.990472  0.300248  0.782749  0.990472  0.300248
        2012-01-01  0.300248  0.782749  0.990472  0.300248  0.782749
    """


    def __init__(self, seasonal_length: int):

        super().__init__()
        self.seasonal_length = seasonal_length

    def fit(self, X: pd.DataFrame, y=None):
        """Stores the seasonal pattern from the last ``self.seasonal_length`` observations

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), train sample, required for compatibility, not used for a naive model.

        y : None
            Used to store the predict feature names and prediction horizon.

        Returns
        -------
        self : SeasonalNaiveModel
            Returns self.

        """

        x_freq = X.index.freq
        seasonal_td = pd.Timedelta(self.seasonal_length * x_freq.n, x_freq.name)
        self.season_ = X.loc[X.index.max()-seasonal_td:]
        self.season_ = self.season_.iloc[-self.seasonal_length:] # TODO think of a better way to get a non-inclusive index
        super().fit(X, y)

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:

        """Using stored ``self._season`` predicts a seasonal time series starting from ``X``

        Parameters
        ----------
        X: pd.DataFrame, shape (n_samples, 1), required
            The time series on which to predict.

        Returns
        -------
        predictions : pd.DataFrame, shape (n_samples, self._horizon)
            The output predictions.

        Raises
        ------
        NotFittedError
            Raised if the model is not fitted yet.

        """

        check_is_fitted(self)
        time_diff = X.index.to_timestamp() - self.season_.index.max().to_timestamp()
        len_s = len(self.season_)
        seasonal_pos = time_diff.days.values % len_s
        y_pred = np.squeeze([self._season_roll_(x, self._horizon) for x in seasonal_pos], axis=2)
        predictions = pd.DataFrame(data=y_pred, index=X.index, columns=self._y_columns)

        return predictions

    def _season_roll_(self, start: int, horizon: int) -> np.array:
        """
        Generates a seasonal time series of length ``horizon``, repeating ``self.season_`` starting from ``start`` position.

        Parameters
        ----------
        start : int, starting position of a series within the season
        horizon : lenth of the series

        Returns
        -------

        np.array, seasonal time series to be used in prediction

        """

        season = self.season_
        len_s = len(season)
        cycles = np.maximum(horizon + start - len_s, 0) // len_s
        tail = horizon - len_s * (cycles + 1) + start
        tail = tail % len_s if tail >= len_s else tail
        if tail <= 0 and cycles == 0:
            return season.iloc[start:start + horizon].to_numpy()
        else:
            return np.concatenate(
                (season.iloc[start:, :], np.tile(season, (cycles, 1)), season.iloc[:tail, :]))


class DriftModel(NaiveModel):

    """Simple drift model, calculates drift as the difference between the first and the last elements of the train series, divided by the number of periods.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gtime.model_selection import horizon_shift, FeatureSplitter
    >>> from gtime.forecasting import DriftModel
    >>> idx = pd.period_range(start='2011-01-01', end='2012-01-01')
    >>> np.random.seed(1)
    >>> df = pd.DataFrame(np.random.random((len(idx), 1)), index=idx, columns=['1'])
    >>> y = horizon_shift(df, horizon=5)
    >>> X_train, y_train, X_test, y_test = FeatureSplitter().transform(df, y)
    >>> m = DriftModel()
    >>> m.fit(X_train, y_train).predict(X_test)
                         y_1       y_2       y_3       y_4       y_5
        2011-12-28  0.143006  0.142682  0.142359  0.142035  0.141712
        2011-12-29  0.901308  0.900985  0.900661  0.900338  0.900015
        2011-12-30  0.541559  0.541236  0.540912  0.540589  0.540265
        2011-12-31  0.974740  0.974417  0.974093  0.973770  0.973446
        2012-01-01  0.636604  0.636281  0.635957  0.635634  0.635311

    """

    def fit(self, X: pd.DataFrame, y=None):

        """Calculates and stores the drift as a difference between the first and the last observations of the train set divided by number of observations.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), train sample, required for compatibility, not used for a naive model.

        y : None
            Used to store the predict feature names and prediction horizon.

        Returns
        -------
        self : SeasonalNaiveModel
            Returns self.

        """

        self.drift_ = (X.iloc[-1] - X.iloc[0]) / len(X)
        super().fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:

        """Using fitted ``self.drift_`` builds a linear time series starting from each point of ``X``

        Parameters
        ----------
        X: pd.DataFrame, shape (n_samples, 1), required
            The time series on which to predict.

        Returns
        -------
        predictions : pd.DataFrame, shape (n_samples, self._horizon)
            The output predictions.

        Raises
        ------
        NotFittedError
            Raised if the model is not fitted yet.

        """

        check_is_fitted(self)
        y_pred = np.transpose(np.squeeze([X.values + i * self.drift_.values for i in range(self._horizon)], axis=2))
        predictions = pd.DataFrame(data=y_pred, index=X.index, columns=self._y_columns)

        return predictions


class AverageModel(NaiveModel):

    """ Predicts all future data points as an average of all train items and all test items prior to is

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gtime.model_selection import horizon_shift, FeatureSplitter
    >>> from gtime.forecasting import AverageModel
    >>> idx = pd.period_range(start='2011-01-01', end='2012-01-01')
    >>> np.random.seed(1)
    >>> df = pd.DataFrame(np.random.random((len(idx), 1)), index=idx, columns=['1'])
    >>> y = horizon_shift(df, horizon=5)
    >>> X_train, y_train, X_test, y_test = FeatureSplitter().transform(df, y)
    >>> m = AverageModel()
    >>> m.fit(X_train, y_train).predict(X_test)
                         y_1       y_2       y_3       y_4       y_5
        2011-12-28  0.510285  0.510285  0.510285  0.510285  0.510285
        2011-12-29  0.511362  0.511362  0.511362  0.511362  0.511362
        2011-12-30  0.511445  0.511445  0.511445  0.511445  0.511445
        2011-12-31  0.512714  0.512714  0.512714  0.512714  0.512714
        2012-01-01  0.513053  0.513053  0.513053  0.513053  0.513053


    """

    def fit(self, X: pd.DataFrame, y=None):

        """Stores average of all train data points

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), train sample, required for compatibility, not used for a naive model.

        y : None
            Used to store the predict feature names and prediction horizon.

        Returns
        -------
        self : SeasonalNaiveModel
            Returns self.

        """

        self.avg_train_ = X.mean(axis=0)
        super().fit(X, y)

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:

        """ Updates stored average of the train dataset adding all test points prior to it and using it as an estimate for the forecast.

        Parameters
        ----------
        X: pd.DataFrame, shape (n_samples, 1), required
            The time series on which to predict.

        Returns
        -------
        predictions : pd.DataFrame, shape (n_samples, self._horizon)
            The output predictions.

        Raises
        ------
        NotFittedError
            Raised if the model is not fitted yet.

        """

        check_is_fitted(self)
        sum_train = (self.avg_train_ * self._horizon).to_numpy()
        predictions = pd.DataFrame(data=np.nan, columns=self._y_columns, index=X.index)

        for i in range(len(X)):
            predictions.iloc[i, :] = (sum_train + X.iloc[i].to_numpy()) / (self._horizon + i + 1)
            sum_train += X.iloc[i].values

        return predictions

