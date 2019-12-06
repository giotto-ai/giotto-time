Roadmap
========

Release 0.1.0 (2019-12-20)
--------------------------
This is the first release of the library.

Overview
~~~~~~~~
`giotto-time` is a time series forecasting library in Python. The main novelties
compared to traditional time series libraries are the following:

- feature creation, model selection, model assessment and prediction pipeline for time series models.
- plug-and-play availability of any scikit-learn-compatible regression or classification model for forecasting.
- minimization of standard custom loss functions for time series (SMAPE, max error, etc..)
- easy-to-use scikit-learn-familiar API.

Additionally we provide standard causality tests with a scikit-learn-like interface.


Input-Output specifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Input:** `pd.Series`, `pd.DataFrame` (single column), `np.array`, `list`
**Output:** the same format as the input

**Additional input parameters:** the user can pass a list of features and a scikit-learn
compatible model to giotto-time.

Example of Usage
~~~~~~~~~~~~~~~~

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

The features have a scikit-learn-like interface.

The class FeatureCreation wraps a list of features together and returns the X and y
matrices from a time series given as input.

Time Series Model
~~~~~~~~~~~~~~~~~
We provide the `GAR` class (Generalize Auto Regressive).
It operates in a similar way to the standard AR, but with an arbitrary number of
features and with an arbitrary regression model.

Custom Regressors
~~~~~~~~~~~~~~~~~

`LinearModel` is a linear regressor that minimizes a custom loss functions.

Causality Tests
~~~~~~~~~~~~~~~
We provide two tests: `ShiftedLinearCoefficient` and `ShiftedPearsonCorrelation`.

Others
~~~~~~
- `TrainTestSplitter`
- `FunctionTrend`
- `ExponentialTrend`
- `PolynomialTrend`

Release 0.2.0 (to be discussed)
-------------------------------
To be discussed.
