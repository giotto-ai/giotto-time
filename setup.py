#! /usr/bin/env python
"""Toolbox for Machine Learning using Topological Data Analysis."""

import os
import codecs
import re
import sys
import platform
import subprocess

from distutils.version import LooseVersion
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

version_file = os.path.join('giottotime', '_version.py')
with open(version_file) as f:
    exec(f.read())

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

DISTNAME = 'giotto-time'
DESCRIPTION = 'Toolbox for Time Series analysis and integration with Machine Learning.'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
LONG_DESCRIPTION_TYPE = 'text/x-rst'
MAINTAINER = 'Alessio Baccelli'
MAINTAINER_EMAIL = 'maintainers@giotto.ai'
URL = 'https://github.com/giotto-ai/giotto-time'
LICENSE = 'GPLv3'
DOWNLOAD_URL = 'https://github.com/giotto-ai/giotto-time/tarball/v0.0a0'
VERSION = __version__ # noqa
CLASSIFIERS = ['Intended Audience :: Science/Corporate',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7']
KEYWORDS = 'machine learning time series data analysis ' + \
    'topology, persistence diagrams'
INSTALL_REQUIRES = requirements
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov',
        'pytest-azurepipelines',
        'pytest-benchmark',
        'jupyter_contrib_nbextensions',
        'flake8'],
    'doc': [
        'sphinx',
        'sphinx-gallery',
        'sphinx-issues',
        'sphinx_rtd_theme',
        'numpydoc'],
    'examples': [
        'jupyter',
        'matplotlib',
        'plotly']
}


setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESCRIPTION_TYPE,
      zip_safe=False,
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      keywords=KEYWORDS,
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)
