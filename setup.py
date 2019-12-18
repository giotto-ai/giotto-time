#! /usr/bin/env python
"""Toolbox for Machine Learning using Topological Data Analysis."""

import os
import codecs

from setuptools import setup, find_packages

from giottotime import __version__

version_file = os.path.join("giottotime", "_version.py")
with open(version_file) as f:
    exec(f.read())

with open("requirements.txt") as f:
    requirements = f.read().splitlines()
with open("doc-requirements.txt") as f:
    doc_requirements = f.read().splitlines()
with open("dev-requirements.txt") as f:
    dev_requirements = f.read().splitlines()

DISTNAME = "giotto-time"
DESCRIPTION = "Toolbox for Time Series analysis and integration with Machine Learning."
with codecs.open("README.rst", encoding="utf-8-sig") as f:
    LONG_DESCRIPTION = f.read()
LONG_DESCRIPTION_TYPE = "text/x-rst"
MAINTAINER = "Alessio Baccelli"
MAINTAINER_EMAIL = "maintainers@giotto.ai"
URL = "https://github.com/giotto-ai/giotto-time"
LICENSE = "GPLv3"
DOWNLOAD_URL = "https://github.com/giotto-ai/giotto-time/tarball/v0.0a0"
VERSION = __version__
CLASSIFIERS = [
    "Intended Audience :: Information Technology",
    "Intended Audience :: Developers",
    "License :: OSI Approved",
    "Programming Language :: Python",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
]
KEYWORDS = (
    "machine learning time series data analysis " + "topology, persistence diagrams"
)
INSTALL_REQUIRES = requirements
EXTRAS_REQUIRE = {
    "tests": dev_requirements,
    "doc": doc_requirements,
    "examples": [],
}


setup(
    name=DISTNAME,
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
    extras_require=EXTRAS_REQUIRE,
)
