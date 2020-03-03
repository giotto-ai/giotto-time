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
    >>> idx = pd.date_range(start='2011-01-01', end='2012-01-01')
    >>> df = pd.DataFrame(np.random.random((len(idx), 1)), index=idx, columns=['1'])
    >>> y = horizon_shift(df, horizon=3)
    >>> X_train, y_train, X_test, y_test = FeatureSplitter().transform(df, y)
    >>> m = NaiveModel()
    >>> m.fit(X_train, y_train).predict(X_test)
                         y_1       y_2       y_3
        2011-12-30  0.664035  0.664035  0.664035
        2011-12-31  0.912093  0.912093  0.912093
        2012-01-01  0.622069  0.622069  0.622069

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

        """Using the fitted polynomial, predict the values starting from ``X``.

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


    def __init__(self, seasonal_length: int):
        super().__init__()
        self.seasonal_length = seasonal_length

    def fit(self, X: pd.DataFrame, y=None):

        x_freq = X.index.freq
        seasonal_td = pd.Timedelta(self.seasonal_length * x_freq.n, x_freq.name)
        self.season_ = X.loc[X.index.max()-seasonal_td:]
        self.season_ = self.season_.iloc[-self.seasonal_length:] # TODO think of a better way to get a non-inclusive index
        super().fit(X, y)

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:

        check_is_fitted(self)
        time_diff = X.index.to_timestamp() - self.season_.index.max().to_timestamp()
        len_s = len(self.season_)
        len_y = len(self._y_columns) # TODO to add horizon?
        len_x = len(X)
        seasonal_pos = time_diff.days.values % len_s
        y_pred = np.squeeze([self._season_roll_(x, len_y) for x in seasonal_pos], axis=2)
        predictions = pd.DataFrame(data=y_pred, index=X.index, columns=self._y_columns)

        return predictions

    def _season_roll_(self, start: int, horizon: int) -> np.array:

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


def season_roll(start: int, horizon: int, season: pd.DataFrame) -> np.array:

    len_s = len(season)
    cycles = np.maximum(horizon + start - len_s, 0) // len_s
    tail = horizon - len_s * (cycles + 1) + start
    tail = tail % len_s if tail >= len_s else tail
    if tail <= 0 and cycles == 0:
        return season.iloc[start:start + horizon].to_numpy()
    else:
        # if start == 0:
        #     return np.concatenate(
        #         (np.tile(season, (cycles, 1)), season.iloc[:tail, :]))
        # else:
        return np.concatenate(
            (season.iloc[start:, :], np.tile(season, (cycles, 1)), season.iloc[:tail, :]))




class DriftModel(BaseEstimator, RegressorMixin):

    def fit(self, X: pd.DataFrame, y=None):

        self.drift_ = (X.iloc[-1] - X.iloc[0]) / len(X)
        self._y_columns = y.columns
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:

        check_is_fitted(self)
        len_y = len(self._y_columns) # TODO to add horizon?
        y_pred = np.squeeze([X.values + i * self.drift_.values for i in range(len_y)], axis=2)
        predictions = pd.DataFrame(data=y_pred, index=X.index, columns=self._y_columns)

        return predictions


class AverageModel(BaseEstimator, RegressorMixin):


    def fit(self, X: pd.DataFrame, y=None):

        self.avg_train_ = X.mean(axis=0)
        self.n_avg_ = len(X)
        self._y_columns = y.columns

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:

        check_is_fitted(self)

        sum_train = (self.avg_train_ * self.n_avg_).to_numpy()

        predictions = pd.DataFrame(data=np.nan, columns=self._y_columns, index=X.index)

        for i in range(len(X)):
            predictions.iloc[i, :] = (sum_train + X.iloc[i].to_numpy()) / (self.n_avg_ + i + 1)
            sum_train += X.iloc[i].values

        return predictions



if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from gtime.model_selection import horizon_shift, FeatureSplitter
    from gtime.forecasting import NaiveModel
    idx = pd.date_range(start='2011-01-01', end='2012-01-01')
    df = pd.DataFrame(np.random.random((len(idx), 1)), index=idx, columns=['1'])
    y = horizon_shift(df, horizon=3)
    X_train, y_train, X_test, y_test = FeatureSplitter().transform(df, y)
    m = NaiveModel()
    print(m.fit(X_train, y_train).predict(X_test))

# m = SeasonalNaiveModel(seasonal_length=pd.Timedelta(15, unit='d'))
# m = AverageModel()