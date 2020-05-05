import numpy as np
import pandas as pd
from gtime.forecasting import BaseForecaster


class NaiveForecaster(BaseForecaster):

    """Naïve model, all predicted values are equal to the most recent available observation.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gtime.model_selection import horizon_shift, FeatureSplitter
    >>> from gtime.forecasting import NaiveForecaster
    >>> idx = pd.period_range(start='2011-01-01', end='2012-01-01')
    >>> np.random.seed(1)
    >>> df = pd.DataFrame(np.random.random((len(idx), 1)), index=idx, columns=['1'])
    >>> y = horizon_shift(df, horizon=5)
    >>> X_train, y_train, X_test, y_test = FeatureSplitter().transform(df, y)
    >>> m = NaiveForecaster()
    >>> m.fit(X_train, y_train).predict(X_test)
                         y_1       y_2       y_3       y_4       y_5
        2011-12-28  0.143006  0.143006  0.143006  0.143006  0.143006
        2011-12-29  0.901308  0.901308  0.901308  0.901308  0.901308
        2011-12-30  0.541559  0.541559  0.541559  0.541559  0.541559
        2011-12-31  0.974740  0.974740  0.974740  0.974740  0.974740
        2012-01-01  0.636604  0.636604  0.636604  0.636604  0.636604

    """

    def _predict(self, X: pd.DataFrame) -> np.array:

        """Using the value of each element in ``X`` predicts the rest of the forecast to be equal to it.

        Parameters
        ----------
        X: pd.DataFrame, shape (n_samples, 1), required
            The time series on which to predict.

        Returns
        -------
        predictions : np.array, shape (n_samples, self.horizon_)
            The output predictions.

        Raises
        ------
        NotFittedError
            Raised if the model is not fitted yet.

        """

        y_pred = np.broadcast_to(X, (len(X), self.horizon_))
        predictions = y_pred

        return predictions


class SeasonalNaiveForecaster(BaseForecaster):
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
    >>> from gtime.forecasting import SeasonalNaiveForecaster
    >>> idx = pd.period_range(start='2011-01-01', end='2012-01-01')
    >>> np.random.seed(1)
    >>> df = pd.DataFrame(np.random.random((len(idx), 1)), index=idx, columns=['1'])
    >>> y = horizon_shift(df, horizon=5)
    >>> X_train, y_train, X_test, y_test = FeatureSplitter().transform(df, y)
    >>> m = SeasonalNaiveForecaster(seasonal_length=3)
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
        self.season_length = seasonal_length

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Stores the seasonal pattern from the last ``self.seasonal_length`` observations

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), train sample.

        y : None
            Used to store the predict feature names and prediction horizon.

        Returns
        -------
        self : SeasonalNaiveForecaster
            Returns self.

        """

        if self.season_length > len(X):
            raise ValueError(
                f"Only {len(X)} data points are available, at least {self.season_length} for this seasonal model"
            )
        self.season_ = X.iloc[-self.season_length :]
        super().fit(X, y)

        return self

    def _predict(self, X: pd.DataFrame) -> np.array:

        """Using stored ``self._season`` predicts a seasonal time series starting from ``X``

        Parameters
        ----------
        X: pd.DataFrame, shape (n_samples, 1), required
            The time series on which to predict.

        Returns
        -------
        predictions : np.array, shape (n_samples, self._horizon)
            The output predictions.

        Raises
        ------
        NotFittedError
            Raised if the model is not fitted yet.

        """

        seasonal_pos = [
            x.n % self.season_length for x in X.index - self.season_.index.max()
        ]
        predictions = np.concatenate(
            [self._season_roll(x, self.horizon_) for x in seasonal_pos], axis=1
        )

        return predictions

    def _season_roll(self, start: int, horizon: int) -> np.array:
        """
        Generates a seasonal time series of length ``horizon``, repeating ``self.season_`` starting from ``start`` position.

        Parameters
        ----------
        start : int, starting position of a series within the season

        horizon : int, length of the series

        Returns
        -------

        np.array, seasonal time series to be used in prediction

        """

        season = self.season_
        season_length = self.season_length
        cycles = np.maximum(horizon + start - season_length, 0) // season_length
        tail = horizon - season_length * (cycles + 1) + start
        tail = tail % season_length if tail >= season_length else tail
        if tail <= 0 and cycles == 0:
            return season.iloc[start : start + horizon].to_numpy()
        else:
            return np.concatenate(
                (
                    season.iloc[start:, :],
                    np.tile(season, (cycles, 1)),
                    season.iloc[:tail, :],
                )
            )


class DriftForecaster(BaseForecaster):

    """Simple drift model, calculates drift as the difference between the first and the last elements of the train series, divided by the number of periods.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gtime.model_selection import horizon_shift, FeatureSplitter
    >>> from gtime.forecasting import DriftForecaster
    >>> idx = pd.period_range(start='2011-01-01', end='2012-01-01')
    >>> np.random.seed(1)
    >>> df = pd.DataFrame(np.random.random((len(idx), 1)), index=idx, columns=['1'])
    >>> y = horizon_shift(df, horizon=5)
    >>> X_train, y_train, X_test, y_test = FeatureSplitter().transform(df, y)
    >>> m = DriftForecaster()
    >>> m.fit(X_train, y_train).predict(X_test)
                         y_1       y_2       y_3       y_4       y_5
        2011-12-28  0.143006  0.142682  0.142359  0.142035  0.141712
        2011-12-29  0.901308  0.900985  0.900661  0.900338  0.900015
        2011-12-30  0.541559  0.541236  0.540912  0.540589  0.540265
        2011-12-31  0.974740  0.974417  0.974093  0.973770  0.973446
        2012-01-01  0.636604  0.636281  0.635957  0.635634  0.635311

    """

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):

        """Calculates and stores the drift as a difference between the first and the last observations of the train set divided by number of observations.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), train sample.

        y : pd.DataFrame
            Used to store the predict feature names and prediction horizon.

        Returns
        -------
        self : DriftForecaster
            Returns self.

        """

        self.drift_ = (X.iloc[-1] - X.iloc[0]) / len(X)
        super().fit(X, y)
        return self

    def _predict(self, X: pd.DataFrame) -> np.array:

        """Using fitted ``self.drift_`` builds a linear time series starting from each point of ``X``

        Parameters
        ----------
        X: pd.DataFrame, shape (n_samples, 1), required
            The time series on which to predict.

        Returns
        -------
        predictions : np.array, shape (n_samples, self._horizon)
            The output predictions.

        Raises
        ------
        NotFittedError
            Raised if the model is not fitted yet.

        """

        predictions = np.transpose(
            np.squeeze(
                [X.values + i * self.drift_.values for i in range(self.horizon_)],
                axis=2,
            )
        )

        return predictions


class AverageForecaster(BaseForecaster):

    """ Predicts all future data points as an average of all train items and all test items prior to is

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gtime.model_selection import horizon_shift, FeatureSplitter
    >>> from gtime.forecasting import AverageForecaster
    >>> idx = pd.period_range(start='2011-01-01', end='2012-01-01')
    >>> np.random.seed(1)
    >>> df = pd.DataFrame(np.random.random((len(idx), 1)), index=idx, columns=['1'])
    >>> y = horizon_shift(df, horizon=5)
    >>> X_train, y_train, X_test, y_test = FeatureSplitter().transform(df, y)
    >>> m = AverageForecaster()
    >>> m.fit(X_train, y_train).predict(X_test)
                         y_1       y_2       y_3       y_4       y_5
        2011-12-28  0.510285  0.510285  0.510285  0.510285  0.510285
        2011-12-29  0.511362  0.511362  0.511362  0.511362  0.511362
        2011-12-30  0.511445  0.511445  0.511445  0.511445  0.511445
        2011-12-31  0.512714  0.512714  0.512714  0.512714  0.512714
        2012-01-01  0.513053  0.513053  0.513053  0.513053  0.513053


    """

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):

        """Stores average of all train data points

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), train sample, required for compatibility, not used for a naive model.

        y : None
            Used to store the predict feature names and prediction horizon.

        Returns
        -------
        self : SeasonalNaiveForecaster
            Returns self.

        """

        self.avg_train_ = X.mean(axis=0)
        self.train_lenth_ = len(X)
        super().fit(X, y)

        return self

    def _predict(self, X: pd.DataFrame) -> np.array:

        """Updates stored average of the train dataset adding all test points prior to it and using it as an estimate for the forecast.

        Parameters
        ----------
        X: pd.DataFrame, shape (n_samples, 1), required
            The time series on which to predict.

        Returns
        -------
        predictions : np.array, shape (n_samples, self._horizon)
            The output predictions.

        Raises
        ------
        NotFittedError
            Raised if the model is not fitted yet.

        """

        sum_train = (self.avg_train_ * self.train_lenth_).to_numpy()
        predictions = np.empty((len(X), self.horizon_)) * np.nan
        for i in range(len(X)):
            predictions[i, :] = (sum_train + X.iloc[i].to_numpy()) / (
                self.train_lenth_ + i + 1
            )
            sum_train += X.iloc[i].values
        return predictions
