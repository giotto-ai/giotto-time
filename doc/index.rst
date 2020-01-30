.. giotto documentation master file, created by
   sphinx-quickstart on Mon Jun  3 11:56:46 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to giotto-time's API reference!
========================================

:mod:`giottotime.causality`: Causality Tests
============================================

.. automodule:: giottotime.causality
   :no-members:
   :no-inherited-members:

.. currentmodule:: giottotime

.. autosummary::
   :toctree: generated/
   :template: class.rst

   causality.ShiftedLinearCoefficient
   causality.ShiftedPearsonCorrelation


:mod:`giottotime.compose`: Feature Creation
===========================================

.. automodule:: giottotime.compose
   :no-members:
   :no-inherited-members:

.. currentmodule:: giottotime

.. autosummary::
   :toctree: generated/
   :template: class.rst

   compose.FeatureCreation


:mod:`giottotime.feature_extraction`: Feature Extraction
========================================================

.. automodule:: giottotime.feature_extraction
   :no-members:
   :no-inherited-members:

.. currentmodule:: giottotime

.. autosummary::
   :toctree: generated/
   :template: class.rst

    feature_extraction.Shift
    feature_extraction.MovingAverage
    feature_extraction.MovingCustomFunction
    feature_extraction.Polynomial
    feature_extraction.Exogenous
    feature_extraction.CustomFeature


:mod:`giottotime.feature_generation`: Feature Generation
========================================================

.. automodule:: giottotime.feature_generation
   :no-members:
   :no-inherited-members:

.. currentmodule:: giottotime

.. autosummary::
   :toctree: generated/
   :template: class.rst


   feature_generation.PeriodicSeasonal
   feature_generation.Constant
   feature_generation.Calendar


:mod:`giottotime.forecasting`: Forecasting
==========================================

.. automodule:: giottotime.forecasting
   :no-members:
   :no-inherited-members:

.. currentmodule:: giottotime

.. autosummary::
   :toctree: generated/
   :template: function.rst

   forecasting.GAR
   forecasting.GARFF
   forecasting.TrendForecaster


:mod:`giottotime.forecasting`: Regressors
=========================================

.. automodule:: giottotime.regressors
   :no-members:
   :no-inherited-members:

.. currentmodule:: giottotime

.. autosummary::
   :toctree: generated/
   :template: function.rst

   regressors.LinearRegressor


:mod:`giottotime.metrics`: Metrics
==================================

.. automodule:: giottotime.metrics
   :no-members:
   :no-inherited-members:

.. currentmodule:: giottotime

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.smape
   metrics.max_error


:mod:`giottotime.model_selection`: Model Selection
==================================================

.. automodule:: giottotime.model_selection
   :no-members:
   :no-inherited-members:

.. currentmodule:: giottotime

.. autosummary::
   :toctree: generated/
   :template: class.rst

   model_selection.FeatureSplitter

.. autosummary::
   :toctree: generated/
   :template: function.rst

   model_selection.horizon_shift


:mod:`giottotime.preprocessing`: Preprocessing
==============================================

.. automodule:: giottotime.preprocessing
   :no-members:
   :no-inherited-members:

.. currentmodule:: giottotime

.. autosummary::
   :toctree: generated/
   :template: class.rst

   preprocessing.SequenceToTimeIndexSeries
   preprocessing.PandasSeriesToTimeIndexSeries
   preprocessing.TimeIndexSeriesToPeriodIndexSeries
   preprocessing.TimeSeriesPreparation
