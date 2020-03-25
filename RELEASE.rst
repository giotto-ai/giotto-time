Release 0.2.1
==============

- Granger causality test added
- Hedge online algorithm added
- Update data for the tda example notebook
- Fix imports

Release 0.2.0
==============

Import name change
------------------
The main module has been renamed from ``giottotime`` to ``gtime``. Therefore, now all the imports
should be changed in order to match the new name of the package.

Renaming
--------
- ``causality_tests`` to ``causality``
- ``shifted_linear_coefficient.py`` to ``linear_coefficient.py``
- ``shifted_pearson_correlation.py`` to ``pearson_correlation.py``
- ``feature_creation`` to ``feature_extraction``
- ``models`` to ``forecasting``
- ``standard_features.py`` to ``external.py``
- ``calendar_features.py`` to ``calendar.py``
- ``trend_features.py`` to ``trend.py``
- ``time_series_features.py`` to ``standard.py``

Moving
------
- ``calendar.py`` and ``external.py`` to ``feature_generation``

Major Features and Improvements
-------------------------------
- Split ``TrendForecaster`` and ``Detrender``. Before, the ``TrendForecaster`` was both
  a predictor and a transformer. To have a more sklearn-like interface, the classes are
  now splitted and the ``PolynomialDetrender`` and ``ExponentialDetrender`` classes have
  been grouped into the ``Detrender`` class.
- The 'y' matrix is now created from the ``horizon_shit`` function, instead of being
  created by the ``FeatureCreation`` class.
- The ``ShiftedPearsonCorrelation`` and ``ShitedLinearCoefficient`` classes now have the
  possibility to test shifts starting from a specified value, instead of always testing
  the shifts starting from 1.
- The interface for creating new features through the ``FeatureCreation`` class has now
  changed and supports multiple columns as input.
- Implementation of ``MovingCustomFunction`` class, in order to allow the user to apply
  custom functions with a rolling window.
- General improvements in the documentation.
- The topology features are no longer part of giotto-time, but will instead be
  implemented in ``giotto-tda``.
- Removed ``giotto-tda`` from the requirements.

Bug Fixes
----------
- Fixed a bug that was causing the ``transform`` method of the ``ShitedLinearCoefficient``
  and of the ``ShiftedPearsonCorrelation`` classes to shift the time series in the wrong
  direction.
- Fixed some bugs in the documentation.


Release 0.1.2
==============

- Documentation fixes.
- The ``ShiftedLinearCoefficient`` and ``ShiftedPearsonCorrelation`` now can compute the
  p-values of the test through bootstrapping, in order to check for the significant of
  the test.


Release 0.1.1
==============

Documentation fixes.

Release 0.1.0
==============

Initial release of giotto-time.

