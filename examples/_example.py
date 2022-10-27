import pandas as pd
from sklearn.linear_model import LinearRegression

from gtime.feature_extraction import Shift
from gtime.time_series_models.base import TimeSeriesForecastingModel


class TimeSeries(pd.DataFrame):
    def plot(self):
        pass


time_series = TimeSeries()

# You can plot
time_series.plot()

# Decomposition
## Un peu bizarre le plot_stl() et deux fois stl_decomposition
time_series = time_series.stl_decomposition()
time_series.plot_stl()
time_series = time_series.recompose()  # Choose a good name

# Box-Cox
time_series = time_series.box_cox(lambda_=0.3)

# Feature forecasting
features = [("shift", Shift(1), "time_series")]
automatic_features = get_features()  # Similar to fast.ai get_transforms()
gar_forecaster = LinearRegression()
# This object TimeSeriesForecastingModel keeps into account all the intermediate steps.
# You don't need to manually deal with train/test split, etc..
forecasting_model = TimeSeriesForecastingModel(
    features=features, horizon=3, model=gar_forecaster
)
forecasting_model = forecasting_model.fit(time_series)
forecasting_model.predict()
forecasting_model.cross_validate()  # Is cross validation also on multiple time series?

# Residuals analysis
forecasting_model.residuals_.acf()

# Questions
"""
How to implement ARIMA? I think that a GAR forecaster with MA should work, but we should check. 
It helps that the user can't customize the feature matrix.

Exponential Smoothing? Maybe it could work also? Not clear if it is possible with additional
features

Add a learner object?
"""

time_series = TimeSeries(pandas_dataframe)

arima = ARIMA(time_series)
arima.fit(time_series,,
preds = arima.predict(time_series)


time_series.to_pandas()
