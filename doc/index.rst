.. giotto documentation master file, created by
   sphinx-quickstart on Mon Jun  3 11:56:46 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to giotto-time's API reference!
========================================

:mod:`giottotime.causality_tests`: Causality Tests
==================================================

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

   feature_creation.FeatureCreation
   feature_creation.Shift
   feature_creation.MovingAverage
   feature_creation.ConstantFeature
   feature_creation.Polynomial
   feature_creation.Exogenous
   feature_creation.CustomFeature
   feature_creation.Calendar
   feature_creation.PeriodicSeasonalFeature
   feature_creation.DetrendedFeature
   feature_creation.RemovePolynomialTrend
   feature_creation.RemoveExponentialTrend

:mod:`giottotime.loss_functions`: Loss Functions
====================================================

.. automodule:: giottotime.loss_functions
   :no-members:
   :no-inherited-members:

.. currentmodule:: giottotime

.. autosummary::
   :toctree: generated/
   :template: function.rst

   loss_functions.smape
   loss_functions.max_error


:mod:`giottotime.model_selection`: Model Selection
====================================================

.. automodule:: giottotime.model_selection
   :no-members:
   :no-inherited-members:

.. currentmodule:: giottotime

.. autosummary::
   :toctree: generated/
   :template: class.rst

   model_selection.FeatureSplitter


:mod:`giottotime.models`: Models
====================================================

.. automodule:: giottotime.models
   :no-members:
   :no-inherited-members:

.. currentmodule:: giottotime

.. autosummary::
   :toctree: generated/
   :template: class.rst

   models.GAR
   models.LinearRegressor


:mod:`giottotime.time_series_preparation`: Time Series Preparation
====================================================================

.. automodule:: giottotime.time_series_preparation
   :no-members:
   :no-inherited-members:

.. currentmodule:: giottotime

.. autosummary::
   :toctree: generated/
   :template: class.rst

   time_series_preparation.SequenceToTimeIndexSeries
   time_series_preparation.PandasSeriesToTimeIndexSeries
   time_series_preparation.TimeIndexSeriesToPeriodIndexSeries
   time_series_preparation.TimeSeriesPreparation
