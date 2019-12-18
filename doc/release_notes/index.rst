Release Notes
================

Release 0.1.0 (2019-12-20)
--------------------------
This is the first release of the library.

Overview
~~~~~~~~
`giotto-time` is a time series forecasting library in Python. The main novelties
compared to traditional time series libraries are the following:

- feature creation, model selection, model assessment and prediction pipeline for time series models.
- plug-and-play availability of any scikit-learn-compatible (i.e., in the fit-transform framework) regression or classification models for forecasting.
- minimization of standard and custom loss functions for time series (SMAPE, max error, etc..).
- easy-to-use scikit-learn-familiar and pandas-familiar API.

Additionally we provide a causality tests with a scikit-learn-like transformer interface.


Input-Output Specifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Input:** `pd.Series`, `pd.DataFrame` (single column), `np.array`, `list`

**Output:** `pd.DataFrame`

**Additional input parameters:** the user can pass a list of features and a scikit-learn
compatible model to giotto-time.


Time Series Preparation
~~~~~~~~~~~~~~~~~~~~~~~~
To transform an input array-like structure into a DataFrame with a PeriodIndex
we provide the classes:

- `TimeSeriesPreparation`
- `TimeSeriesConversion`
- `SequenceToTimeIndexSeries`
- `PandasSeriesToTimeIndexSeries`
- `TimeIndexSeriesToPeriodIndexSeries`


Feature Creation
~~~~~~~~~~~~~~~~
We support the following features:

- `CalendarFeature`
- `PeriodicSeasonalFeature`
- `ShiftFeature`
- `MovingAverageFeature`
- `ConstantFeature`
- `PolynomialFeature`
- `ExogenousFeature`
- `CustomFeature`

These features all have a scikit-learn-like interface and behave as transformers.

The class FeatureCreation wraps a list of features together and returns the X and y
matrices from a time series given as input.

Time Series Model
~~~~~~~~~~~~~~~~~
Giotto-time provide the `GAR` class (Generalize Auto Regressive model).
It operates in a similar way to the standard AR, but with an arbitrary number of
features and with an arbitrary underlying regression model.

.. image:: ../../../../images/gar.png
    :width: 60%
    :align: center

.. code-block:: python

    from giottotime.feature_creation import FeaturesCreation
    from giottotime.feature_creation.index_independent_features import ShiftFeature, MovingAverageFeature
    from giottotime.model_selection.train_test_splitter import TrainTestSplitter
    from giottotime.regressors import LinearRegressor
    from giottotime.models.time_series_models import GAR

    time_series = get_time_series()

    features_creation = FeaturesCreation(
        horizon=4,
        features = [ShiftFeature(1), ShiftFeature(2), MovingAverageFeature(5)]
    )
    train_test_splitter = TrainTestSplitter()
    time_series_model = GAR(base_model=LinearRegressor())

    X, y = features_creation.transform(time_series)
    X_train, y_train, X_test, y_test = train_test_splitter.transform(X, y)

    time_series_model.fit(X_train, y_train)
    predictions = time_series_model.predict(X_test)

Time Series Trend Model
~~~~~~~~~~~~~~~~~~~~~~~
We provide main classes to analyze and remove trends from time series in order to create trend stationary time series.

Specifically, giotto-time includes `ExponentialTrend`, `PolynomialTrend` model classes and de-trending transformers.

Example of Usage
~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import pandas as pd

    import matplotlib.pyplot as plt

    from giottotime.models.regressors.linear_regressor import LinearRegressor
    from giottotime.loss_functions.loss_functions import max_error, smape

    from giottotime.models.trend_models.polynomial_trend import PolynomialTrend

    from math import pi

    d = pd.read_csv('trend.csv', index_col=0, parse_dates=True)
    tm = PolynomialTrend(order=3)

    tm.fit(d)

    d.plot(figsize=(10, 10))
    plt.show()

    detrended = tm.transform(d)

    detrended.plot(figsize=(10, 10))
    plt.show()

Before the detrending tranformer, a clear quadratic trend is present in the data:

.. image:: ../../../../images/trend.png
    :width: 60%
    :align: center

After fitting and applying the detrending tranformer, a the transformed data is 'trend stationary':

.. image:: ../../../../images/no_trend.png
    :width: 60%
    :align: center

For additional information on trend stationarity, see:
Trend stationarity: `Wikipedia - Trend stationarity <https://en.wikipedia.org/wiki/Trend_stationary />`_.


Model Selection and Cross Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- `trim_feature_nans`

.. image:: ../../../../images/trimmer.png
    :width: 60%
    :align: center

- `TrainTestSplitter`



Custom Regressors
~~~~~~~~~~~~~~~~~

`LinearRegressor` is a linear regressor that minimizes a custom loss functions.

Causality Tests
~~~~~~~~~~~~~~~
We provide two tests: `ShiftedLinearCoefficient` and `ShiftedPearsonCorrelation`.

.. code-block:: python

    import numpy as np
    import pandas as pd

    import matplotlib.pyplot as plt

    from giottotime.causality_tests import ShiftedPearsonCorrelation

    #TODO


Release 0.2.0 (to be discussed)
-------------------------------
To be discussed.
