#!/bin/bash

set -x

# upgrading pip and setuptools
PYTHON_PATH=$(eval find "/opt/python/*${python_ver}*" -print)
export PATH=${PYTHON_PATH}/bin:${PATH}
pip install --upgrade pip setuptools

# installing and uninstalling giotto-time
cd /io
pip install -e ".[doc, tests]"
pip uninstall -y giotto-time

# testing
pytest --cov . --cov-report xml --hypothesis-profile dev

# building wheels
pip install wheel twine
python setup.py sdist bdist_wheel