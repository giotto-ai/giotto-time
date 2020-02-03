.. image:: https://www.giotto.ai/static/vector/logo.svg
   :width: 500

|Version| |Azure-build| |PyPI download month| |PyPI pyversions| |Slack-join| |Black|

.. |Version| image:: https://badge.fury.io/py/giotto-time.svg
   :target: https://pypi.python.org/pypi/giotto-time/

.. |Azure-build| image:: https://dev.azure.com/maintainers/Giotto/_apis/build/status/giotto-ai.giotto-time?branchName=master
   :target: https://dev.azure.com/maintainers/Giotto/_build/latest?definitionId=4&branchName=master

.. |PyPI download month| image:: https://img.shields.io/pypi/dm/giotto-time.svg
   :target: https://pypi.python.org/pypi/giotto-time/

.. |PyPI pyversions| image:: https://img.shields.io/pypi/pyversions/giotto-time.svg
   :target: https://pypi.python.org/pypi/giotto-time/

.. |Slack-join| image:: https://img.shields.io/badge/Slack-Join-blue
   :target: https://slack.giotto.ai/

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black

giotto-time
===========

giotto-time is a machine learning based time series forecasting toolbox in Python.
It is part of the `Giotto <https://github.com/giotto-ai>`_ family of open-source projects.

Project genesis
---------------

giotto-time was created to provide time series feature extraction, analysis and
forecasting tools based on scikit-learn API.

Documentation
-------------

- API reference (stable release): https://docs-time.giotto.ai

Getting started
---------------

Get started with giotto-time by following the installation steps below.
Simple tutorials and real-world use cases can be found in example folder as notebooks.

Installation
------------

Dependencies
~~~~~~~~~~~~

The latest stable version of giotto-time requires:

- Python (>= 3.6)
- scikit-learn (>= 0.22.0)
- pandas>=0.25.3
- workalendar>=7.1.1

To run the examples, jupyter is required.

User installation
~~~~~~~~~~~~~~~~~

Linux, MacOS and Windows
''''''''''''''''''''''''
Run this command in your favourite python environment  ::

    pip install giotto-time

Contributing
------------

We welcome new contributors of all experience levels. The Giotto
community goals are to be helpful, welcoming, and effective. To learn more about
making a contribution to giotto-time, please see the `CONTRIBUTING.rst
<https://github.com/giotto-ai/giotto-time/blob/master/CONTRIBUTING.rst>`_ file.

Developer installation
~~~~~~~~~~~~~~~~~~~~~~

Source code
'''''''''''

You can obtain the latest state of the source code with the command  ::

    git clone https://github.com/giotto-ai/giotto-time.git


then run

.. code-block:: bash

   cd giotto-time
   pip install -e ".[tests, doc]"

This way, you can pull the library's latest changes and make them immediately available on your machine.
Note: we recommend upgrading ``pip`` and ``setuptools`` to recent versions before installing in this way.

Testing
~~~~~~~

After installation, you can launch the test suite from outside the
source directory::

    pytest gtime


Changelog
---------

See the `RELEASE.rst <https://github.com/giotto-ai/giotto-time/blob/master/RELEASE.rst>`__ file
for a history of notable changes to giotto-time.

Important links
~~~~~~~~~~~~~~~

- Official source code repo: https://github.com/giotto-ai/giotto-time
- Download releases: https://pypi.org/project/giotto-time/
- Issue tracker: https://github.com/giotto-ai/giotto-time/issues

Community
---------

Giotto Slack workspace: https://slack.giotto.ai/

Contacts
--------

maintainers@giotto.ai
