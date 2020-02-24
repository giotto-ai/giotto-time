.. image:: https://www.giotto.ai/static/vector/logo-time.svg
   :width: 500

|Version| |Azure-build| |PyPI download month| |Codecov| |PyPI pyversions| |Slack-join| |Black|

.. |Version| image:: https://badge.fury.io/py/giotto-time.svg
   :target: https://pypi.python.org/pypi/giotto-time/

.. |Azure-build| image:: https://dev.azure.com/maintainers/Giotto/_apis/build/status/giotto-ai.giotto-time?branchName=master
   :target: https://dev.azure.com/maintainers/Giotto/_build/latest?definitionId=4&branchName=master

.. |PyPI download month| image:: https://img.shields.io/pypi/dm/giotto-time.svg
   :target: https://pypi.python.org/pypi/giotto-time/

.. |Codecov| image:: https://codecov.io/gh/giotto-ai/giotto-time/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/giotto-ai/giotto-time

.. |PyPI pyversions| image:: https://img.shields.io/pypi/pyversions/giotto-time.svg
   :target: https://pypi.python.org/pypi/giotto-time/

.. |Slack-join| image:: https://img.shields.io/badge/Slack-Join-blue
   :target: https://slack.giotto.ai/

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black

giotto-time
===========

giotto-time is a machine learning based time series forecasting toolbox in Python.
It is part of the `Giotto <https://github.com/giotto-ai>`_ collection of open-source projects and aims to provide
feature extraction, analysis, causality testing and forecasting models based on
`scikit-learn <https://scikit-learn.org/stable/>`_ API.

License
-------

giotto-time is distributed under the AGPLv3 `license <https://github.com/giotto-ai/giotto-time/blob/master/LICENSE>`_.
If you need a different distribution license, please contact the L2F team at business@l2f.ch.

Documentation
-------------

- API reference (stable release): https://docs-time.giotto.ai

Getting started
---------------

Get started with giotto-time by following the installation steps below.
Simple tutorials and real-world use cases can be found in example folder as notebooks.

Installation
------------

User installation
~~~~~~~~~~~~~~~~~

Run this command in your favourite python environment  ::

    pip install giotto-time

Developer installation
~~~~~~~~~~~~~~~~~~~~~~

Get the latest state of the source code with the command

.. code-block:: bash

    git clone https://github.com/giotto-ai/giotto-time.git
    cd giotto-time
    pip install -e ".[tests, doc]"

Examples
--------

.. code-block:: python
    import numpy as np
    import pandas as pd
    from gtime import *
    from sklearn.linear_model import LinearRegression

    # Create random DataFrame with DatetimeIndex
    X_dt = pd.DataFrame(np.random.randint(4, size=(20)),
                        index=pd.date_range("2019-12-20", "2020-01-08"),
                        columns=['time_series'])

    # Convert the DatetimeIndex to PeriodIndex and y matrix
    X = TimeSeriesPreparation().transform(X_dt)
    y = horizon_shift(X, horizon=3)

    # Create some features
    cal = Calendar(region="europe", country="Switzerland", kernel=np.array([1, 2]))
    fc = FeatureCreation(
        [('s_2', Shift(2), ['time_series']),
         ('ma_3', MovingAverage(window_size=3), ['time_series']),
         ('cal', cal, ['time_series']),
        ]).fit_transform(X)

    # Train test split
    X_train, y_train, X_test, y_test = FeatureSplitter().transform(X, y)

    # Try some forecasting models
    tf = TrendForecaster(trend='polynomial', trend_x0=np.zeros(3))
    tf.fit(X_train).predict(X_test)

    gar = GAR(LinearRegression())
    gar.fit(X_train, y_train).predict(X_test)


Changelog
---------

See the `RELEASE.rst <https://github.com/giotto-ai/giotto-time/blob/master/RELEASE.rst>`__ file
for a history of notable changes to giotto-time.

Contributing
------------

We welcome new contributors of all experience levels. The Giotto
community goals are to be helpful, welcoming, and effective. To learn more about
making a contribution to giotto-time, please see the `CONTRIBUTING.rst
<https://github.com/giotto-ai/giotto-time/blob/master/CONTRIBUTING.rst>`_ file.

Links
-----

- Official source code repo: https://github.com/giotto-ai/giotto-time
- Download releases: https://pypi.org/project/giotto-time/
- Issue tracker: https://github.com/giotto-ai/giotto-time/issues

Community
---------

Giotto Slack workspace: https://slack.giotto.ai/

Contacts
--------

maintainers@giotto.ai
