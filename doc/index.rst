.. giotto documentation master file, created by
   sphinx-quickstart on Mon Jun  3 11:56:46 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to giotto-time's API reference!
========================================

:mod:`gtime.causality`: Causality Tests
============================================

.. automodule:: gtime.causality
   :no-members:
   :no-inherited-members:

.. currentmodule:: gtime

.. autosummary::
   :toctree: generated/
   :template: class.rst

   causality.ShiftedLinearCoefficient
   causality.ShiftedPearsonCorrelation


:mod:`gtime.compose`: Feature Creation
===========================================

.. automodule:: gtime.compose
   :no-members:
   :no-inherited-members:

.. currentmodule:: gtime

.. autosummary::
   :toctree: generated/
   :template: class.rst

   compose.FeatureCreation


:mod:`gtime.feature_extraction`: Feature Extraction
========================================================

.. automodule:: gtime.feature_extraction
   :no-members:
   :no-inherited-members:

.. currentmodule:: gtime

.. autosummary::
   :toctree: generated/
   :template: class.rst

    feature_extraction.Shift
    feature_extraction.MovingAverage
    feature_extraction.MovingCustomFunction
    feature_extraction.Polynomial
    feature_extraction.Exogenous
    feature_extraction.CustomFeature


:mod:`gtime.feature_generation`: Feature Generation
========================================================

.. automodule:: gtime.feature_generation
   :no-members:
   :no-inherited-members:

.. currentmodule:: gtime

.. autosummary::
   :toctree: generated/
   :template: class.rst


   feature_generation.PeriodicSeasonal
   feature_generation.Constant
   feature_generation.Calendar


:mod:`gtime.forecasting`: Forecasting
==========================================

.. automodule:: gtime.forecasting
   :no-members:
   :no-inherited-members:

.. currentmodule:: gtime

.. autosummary::
   :toctree: generated/
   :template: function.rst

   forecasting.GAR
   forecasting.GARFF
   forecasting.TrendForecaster


:mod:`gtime.forecasting`: Regressors
=========================================

.. automodule:: gtime.regressors
   :no-members:
   :no-inherited-members:

.. currentmodule:: gtime

.. autosummary::
   :toctree: generated/
   :template: function.rst

   regressors.LinearRegressor


:mod:`gtime.metrics`: Metrics
================gtime==================

.. automodule:: gtime.metrics
   :no-members:
   :no-inherited-members:

.. currentmodule:: gtime

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.smape
   metrics.max_error


:mod:`gtime.model_selection`: Model Selection
==================================================

.. automodule:: gtime.model_selection
   :no-members:
   :no-inherited-members:

.. currentmodule:: gtime

.. autosummary::
   :toctree: generated/
   :template: class.rst

   model_selection.FeatureSplitter

.. autosummary::
   :toctree: generated/
   :template: function.rst

   model_selection.horizon_shift


:mod:`gtime.preprocessing`: Preprocessing
==============================================

.. automodule:: gtime.preprocessing
   :no-members:
   :no-inherited-members:

.. currentmodule:: gtime

.. autosummary::
   :toctree: generated/
   :template: class.rst

   preprocessing.SequenceToTimeIndexSeries
   preprocessing.PandasSeriesToTimeIndexSeries
   preprocessing.TimeIndexSeriesToPeriodIndexSeries
   preprocessing.TimeSeriesPreparation
