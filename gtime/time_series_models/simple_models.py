from gtime.compose import FeatureCreation
from sklearn.compose import make_column_selector
from gtime.feature_extraction import Shift, MovingAverage, MovingCustomFunction
from gtime.time_series_models import TimeSeriesForecastingModel
from gtime.forecasting import (
    NaiveForecaster,
    SeasonalNaiveForecaster,
    DriftForecaster,
    AverageForecaster,
)


class Naive(TimeSeriesForecastingModel):
    """ Naive model pipeline, no feature creation and ``NaiveModel()`` as a model

        Parameters
        ----------
        horizon: int - prediction horizon, in time series periods

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from gtime.time_series_models import Naive
        >>> idx = pd.period_range(start='2011-01-01', end='2012-01-01')
        >>> np.random.seed(0)
        >>> df = pd.DataFrame(np.random.random((len(idx), 1)), index=idx, columns=['1'])
        >>> model = Naive(horizon=4)
        >>> model.fit(df)
        >>> model.predict()
                         y_1       y_2       y_3       y_4
        2011-12-29  0.543806  0.543806  0.543806  0.543806
        2011-12-30  0.456911  0.456911  0.456911  0.456911
        2011-12-31  0.882041  0.882041  0.882041  0.882041
        2012-01-01  0.458604  0.458604  0.458604  0.458604
    """

    def __init__(self, horizon: int):
        features = [
            ("s1", Shift(0), make_column_selector()),
        ]
        super().__init__(features=features, horizon=horizon, model=NaiveForecaster())


class Average(TimeSeriesForecastingModel):
    """ Average model pipeline, no feature creation and ``AverageModel()`` as a model

        Parameters
        ----------
        horizon: int - prediction horizon, in time series periods

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from gtime.time_series_models import Average
        >>> idx = pd.period_range(start='2011-01-01', end='2012-01-01')
        >>> np.random.seed(0)
        >>> df = pd.DataFrame(np.random.random((len(idx), 1)), index=idx, columns=['1'])
        >>> model = Average(horizon=5)
        >>> model.fit(df)
        >>> model.predict()
                         y_1       y_2       y_3       y_4       y_5
        2011-12-28  0.558475  0.558475  0.558475  0.558475  0.558475
        2011-12-29  0.556379  0.556379  0.556379  0.556379  0.556379
        2011-12-30  0.543946  0.543946  0.543946  0.543946  0.543946
        2011-12-31  0.581512  0.581512  0.581512  0.581512  0.581512
        2012-01-01  0.569221  0.569221  0.569221  0.569221  0.569221

    """

    def __init__(self, horizon: int):
        features = [
            ("s1", Shift(0), make_column_selector()),
        ]
        super().__init__(features=features, horizon=horizon, model=AverageForecaster())


class SeasonalNaive(TimeSeriesForecastingModel):
    """ Seasonal naive model pipeline, no feature creation and ``SeasonalNaiveModel()`` as a model

        Parameters
        ----------
        horizon: int - prediction horizon, in time series periods
        seasonal_length: int - full season cycle length, in time series periods

        Examples
        --------

        >>> import pandas as pd
        >>> import numpy as np
        >>> from gtime.time_series_models import SeasonalNaive
        >>> idx = pd.period_range(start='2011-01-01', end='2012-01-01')
        >>> np.random.seed(0)
        >>> df = pd.DataFrame(np.random.random((len(idx), 1)), index=idx, columns=['1'])
        >>> model = SeasonalNaive(horizon=5, seasonal_length=4)
        >>> model.fit(df)
        >>> model.predict()

                         y_1       y_2       y_3       y_4       y_5
        2011-12-28  0.392676  0.956406  0.187131  0.128861  0.392676
        2011-12-29  0.956406  0.187131  0.128861  0.392676  0.956406
        2011-12-30  0.187131  0.128861  0.392676  0.956406  0.187131
        2011-12-31  0.128861  0.392676  0.956406  0.187131  0.128861
        2012-01-01  0.392676  0.956406  0.187131  0.128861  0.392676
    """

    def __init__(self, horizon: int, seasonal_length: int):
        features = [
            ("s1", Shift(0), make_column_selector()),
        ]
        self.seasonal_length = seasonal_length
        self.horizon = horizon
        super().__init__(
            features=features,
            horizon=horizon,
            model=SeasonalNaiveForecaster(seasonal_length),
        )


class Drift(TimeSeriesForecastingModel):
    """ Simple drift model pipeline, no feature creation and ``DriftModel()`` as a model

        Parameters
        ----------
        horizon: int - prediction horizon, in time series periods

        Examples
        --------

        >>> import pandas as pd
        >>> import numpy as np
        >>> from gtime.time_series_models import Drift
        >>> idx = pd.period_range(start='2011-01-01', end='2012-01-01')
        >>> np.random.seed(0)
        >>> df = pd.DataFrame(np.random.random((len(idx), 1)), index=idx, columns=['1'])
        >>> model = Drift(horizon=5)
        >>> model.fit(df)
        >>> model.predict()

                         y_1       y_2       y_3       y_4       y_5
        2011-12-28  0.903984  0.902982  0.901980  0.900978  0.899976
        2011-12-29  0.543806  0.542804  0.541802  0.540800  0.539798
        2011-12-30  0.456911  0.455910  0.454908  0.453906  0.452904
        2011-12-31  0.882041  0.881040  0.880038  0.879036  0.878034
        2012-01-01  0.458604  0.457602  0.456600  0.455598  0.454596

    """

    def __init__(self, horizon: int):
        features = [
            ("s1", Shift(0), make_column_selector()),
        ]
        super().__init__(features=features, horizon=horizon, model=DriftForecaster())
