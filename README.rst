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

.. image:: https://storage.googleapis.com/l2f-open-models/giotto-time/images/gar.png
   :height: 100px
   :width: 200 px
   :scale: 50 %
   :alt: alternate text
   :align: center

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


