import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class NaiveModel(BaseEstimator, RegressorMixin):

    def fit(self, X: pd.DataFrame, y=None):

        self._y_columns = y.columns

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:

        check_is_fitted(self)
        y_pred = np.broadcast_to(X, (len(X), len(self._y_columns)))
        predictions = pd.DataFrame(data=y_pred, columns=self._y_columns, index=X.index)

        return predictions


class SeasonalNaiveModel(BaseEstimator, RegressorMixin):


    def __init__(self, seasonal_length: pd.Timedelta):
        super().__init__()
        self.seasonal_length = seasonal_length

    def fit(self, X: pd.DataFrame, y=None):

        self.season_ = X.loc[X.index.max()-self.seasonal_length:]
        self._y_columns = y.columns

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:

        check_is_fitted(self)
        time_diff = X.index - self.season_.index.max()
        len_s = len(self.season_)
        len_y = len(self._y_columns) # TODO to add horizon?
        seasonal_pos = time_diff.days.values % len_s
        y_pred = np.squeeze([self._season_roll_(x, len_y) for x in seasonal_pos])
        predictions = pd.DataFrame(data=y_pred, index=X.index, columns=self._y_columns)

        return predictions

    def _season_roll_(self, start: int, horizon: int) -> np.array:

        len_s = len(self.season_)
        cycles = np.maximum(horizon - start, 0) // len_s
        tail = horizon - len_s * (cycles + 1) + start
        if tail >= 0:
            return np.concatenate(
                (self.season_.iloc[start:, :], np.tile(self.season_, (cycles, 1)), self.season_.iloc[:tail, :]))
        else:
            return self.season_.iloc[start:tail].to_numpy()


class DriftModel(BaseEstimator, RegressorMixin):

    def fit(self, X: pd.DataFrame, y=None):

        self.drift_ = (X.iloc[-1] - X.iloc[0]) / len(X)
        self._y_columns = y.columns
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:

        check_is_fitted(self)
        len_y = len(self._y_columns) # TODO to add horizon?
        y_pred = np.squeeze([X.values + i * self.drift_ for i in range(len_y)])
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



# idx = pd.date_range(start='2011-01-01', end='2012-01-01')
# df = pd.DataFrame(np.random.random((len(idx), 1)), index=idx, columns=['1'])
# train_cut = pd.to_datetime('2011-06-05')
# test_cut = pd.to_datetime('2011-08-10')
# test_end = pd.to_datetime('2011-11-10')
# train = df.loc[:train_cut]
# test = df.loc[test_cut:test_end]
# m = AverageModel()
# y = pd.DataFrame(np.nan, index=test.index, columns=['y1', 'y2', 'y3'])
# m.fit(train, y)
# mm = m.predict(test)
# print(mm)