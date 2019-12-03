.. giottotime documentation master file, created by
   sphinx-quickstart on Mon Jun  3 11:56:46 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to giottotime's API reference!
======================================

:mod:`giottotime.causality_tests`: Causality Tests
====================================================

.. automodule:: giottotime.causality_tests
   :no-members:
   :no-inherited-members:

.. currentmodule:: giottotime

.. autosummary::
   :toctree: generated/
   :template: class.rst

   causality_tests.ShiftedLinearCoefficient
   causality_tests.ShiftedPearsonCorrelation

:mod:`giottotime.feature_creation`: Feature Creation
====================================================

.. automodule:: giottotime.feature_creation
   :no-members:
   :no-inherited-members:

.. currentmodule:: giottotime

.. autosummary::
   :toctree: generated/
   :template: class.rst

   feature_creation.FeaturesCreation
   feature_creation.ShiftFeature
   feature_creation.MovingAverageFeature
   feature_creation.ConstantFeature
   feature_creation.PolynomialFeature
   feature_creation.ExogenousFeature
   feature_creation.CustomFeature
   feature_creation.CalendarFeature
   feature_creation.PeriodicSeasonalFeature
   feature_creation.DetrendedFeature
   feature_creation.RemovePolynomialTrend
   feature_creation.RemoveExponentialTrend
   feature_creation.RemoveFunctionTrend

:mod:`giottotime.feature_creation.tda_features`: TDA Features
=============================================================

.. automodule:: giottotime.feature_creation.tda_features
   :no-members:
   :no-inherited-members:

.. currentmodule:: giottotime

.. autosummary::
   :toctree: generated/
   :template: class.rst

   feature_creation.tda_features.AmplitudeFeature
   feature_creation.tda_features.AvgLifeTimeFeature
   feature_creation.tda_features.BettiCurvesFeature
   feature_creation.tda_features.NumberOfRelevantHolesFeature

:mod:`giottotime.models.time_series_models`: Time Series Models
====================================================

.. automodule:: giottotime.models.time_series_models
   :no-members:
   :no-inherited-members:

.. currentmodule:: giottotime

.. autosummary::
   :toctree: generated/
   :template: class.rst

   models.time_series_models.GAR

:mod:`giottotime.models.trend_models`: Trend Models
====================================================

.. automodule:: giottotime.models.trend_models
   :no-members:
   :no-inherited-members:

.. currentmodule:: giottotime

.. autosummary::
   :toctree: generated/
   :template: class.rst

   models.trend_models.TrendModel
   models.trend_models.CustomTrendForm_ts
   models.trend_models.ExponentialTrend
   models.trend_models.FunctionTrend
   models.trend_models.PolynomialTrend

:mod:`giottotime.models.regressors`: Regressor Models
====================================================

.. automodule:: giottotime.models.regressors
   :no-members:
   :no-inherited-members:

.. currentmodule:: giottotime

.. autosummary::
   :toctree: generated/
   :template: class.rst

   models.regressors.LinearRegressor