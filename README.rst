.. image:: https://www.giotto.ai/static/vector/logo.svg
   :width: 850

|Azure|_ 

.. |Azure| image:: https://dev.azure.com/maintainers/Giotto/_apis/build/status/giotto-ai.giotto-time?branchName=master
.. _Azure: https://dev.azure.com/maintainers/Giotto/_build/latest?definitionId=4&branchName=master

giotto-time
===========

Machine learning based time series forecasting tools for python.

Overview
========

giotto-time is a time series forecasting library in Python. The main novelties compared to traditional time series libraries are the following:

* Feature creation, model selection, model assessment and prediction pipeline for time series models.

* Plug-and-play availability of any scikit-learn-compatible (i.e., in the fit-transform framework) regression or classification models for forecasting.

* Minimization of standard and custom loss functions for time series (SMAPE, max error, etc..).

* Easy-to-use scikit-learn-familiar and pandas-familiar API.

* Additionally we provide a causality tests with a scikit-learn-like transformer interface.

Time Series Forecasting Model
=============================

Giotto-time provide the GAR class (Generalize Auto Regressive model). It operates in a similar way to the standard AR, but with an arbitrary number of features and with an arbitrary underlying regression model.

.. raw:: html

   <p align="center">
     <img width="460" src="https://storage.googleapis.com/l2f-open-models/giotto-time/images/gar.png">
   </p>

This model allows the full force of machine learning regressors (compatible with the fit-transform framework ok scikit-learn) to be combined with advanced feature creation stratagies to forecast time series in a convienent api.

>>> from giottotime.feature_creation import FeaturesCreation
>>> from giottotime.feature_creation.index_independent_features import ShiftFeature, MovingAverageFeature
>>> from giottotime.model_selection.train_test_splitter import TrainTestSplitter
>>> from giottotime.regressors import LinearRegressor
>>> from giottotime.models.time_series_models import GAR
>>> 
>>> time_series = get_time_series()
>>> 
>>> features_creation = FeaturesCreation(
>>>     horizon=4,
>>>     features = [ShiftFeature(1), ShiftFeature(2), MovingAverageFeature(5)]
>>> )
>>>
>>> train_test_splitter = TrainTestSplitter()
>>> time_series_model = GAR(base_model=LinearRegressor())
>>> 
>>> X, y = features_creation.transform(time_series)
>>> X_train, y_train, X_test, y_test = train_test_splitter.transform(X, y)
>>> 
>>> time_series_model.fit(X_train, y_train)
>>> predictions = time_series_model.predict(X_test)


Time Series Preparation
=======================

To transform an input array-like structure into a DataFrame with a PeriodIndex we provide the classes:

To transform an input array-like structure into a DataFrame with a PeriodIndex we provide the classes:

* TimeSeriesPreparation
* TimeSeriesConversion
* SequenceToTimeIndexSeries
* PandasSeriesToTimeIndexSeries
* TimeIndexSeriesToPeriodIndexSeries


Feature Creation
================

The following time series features are currently supported:

CalendarFeature
PeriodicSeasonalFeature
ShiftFeature
MovingAverageFeature
ConstantFeature
PolynomialFeature
ExogenousFeature
CustomFeature
These features all have a scikit-learn-like interface and behave as transformers.

The class FeatureCreation wraps a list of features together and returns the X and y matrices from a time series given as input.

Time Series Trend Model
=======================

We provide main classes to analyze and remove trends from time series in order to create trend stationary time series.

Specifically, giotto-time includes ExponentialTrend, PolynomialTrend model classes and de-trending transformers.

>>> import numpy as np
>>> import pandas as pd
>>>
>>> import matplotlib.pyplot as plt
>>>
>>> from giottotime.models.regressors.linear_regressor import LinearRegressor
>>> from giottotime.loss_functions.loss_functions import max_error, smape
>>>
>>> from giottotime.models.trend_models.polynomial_trend import PolynomialTrend
>>>
>>> from math import pi
>>>
>>> d = pd.read_csv('trend.csv', index_col=0, parse_dates=True)
>>> tm = PolynomialTrend(order=3)
>>>
>>> tm.fit(d)
>>>
>>> d.plot(figsize=(10, 10))
>>> plt.show()
>>>
>>> detrended = tm.transform(d)
>>>
>>> detrended.plot(figsize=(10, 10))
>>> plt.show()

.. raw:: html

   <table style="width:100%">
     <tr>
       <th><img width="460" src="https://storage.googleapis.com/l2f-open-models/giotto-time/images/trend.png"></th>
       <th><img width="460" src="https://storage.googleapis.com/l2f-open-models/giotto-time/images/no_trend.png"></th> 
     </tr>
   </table>

Before the detrending tranformer, a clear quadratic trend is present in the data. For additional information on trend stationarity, see: Trend stationarity: Wikipedia - https://en.wikipedia.org/wiki/Trend_stationary.



