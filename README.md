
[![Deploy to gh-pages](https://github.com/giotto-ai/giotto-time/actions/workflows/deploy_github_pages.yml/badge.svg)](https://github.com/giotto-ai/giotto-time/actions/workflows/deploy_github_pages.yml)
[![Upload Python Package](https://github.com/giotto-ai/giotto-time/actions/workflows/build_and_publish.yml/badge.svg)](https://github.com/giotto-ai/giotto-time/actions/workflows/build_and_publish.yml)
[![CI](https://github.com/giotto-ai/giotto-time/actions/workflows/ci.yml/badge.svg)](https://github.com/giotto-ai/giotto-time/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/giotto-time.svg)](https://badge.fury.io/py/giotto-time)
[![Slack-join](https://img.shields.io/badge/Slack-Join-blue)](https://slack.giotto.ai/)

# giotto-time

giotto-time is a machine learning based time series forecasting toolbox in Python.
It is part of the [Giotto](https://github.com/giotto-ai) collection of open-source projects and aims to provide
feature extraction, analysis, causality testing and forecasting models based on
[scikit-learn](https://scikit-learn.org/stable/) API.

## License

giotto-time is distributed under the AGPLv3 [license](https://github.com/giotto-ai/giotto-time/blob/master/LICENSE).
If you need a different distribution license, please contact the L2F team at business@l2f.ch.

## Documentation

- API reference (stable release): https://giotto-ai.github.io/giotto-time/

## Getting started

Get started with giotto-time by following the installation steps below.
Simple tutorials and real-world use cases can be found in example folder as notebooks.

## Installation

### User installation

Run this command in your favourite python environment
```
pip install giotto-time
```

### Developer installation

Get the latest state of the source code with the command

```
git clone https://github.com/giotto-ai/giotto-time.git
cd giotto-time
pip install -e ".[tests, doc]"
```

## Example

```python
from gtime import *
from gtime.feature_extraction import *
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Create random DataFrame with DatetimeIndex
X_dt = pd.DataFrame(np.random.randint(4, size=(20)),
                    index=pd.date_range("2019-12-20", "2020-01-08"),
                    columns=['time_series'])

# Convert the DatetimeIndex to PeriodIndex and create y matrix
X = preprocessing.TimeSeriesPreparation().transform(X_dt)
y = model_selection.horizon_shift(X, horizon=2)

# Create some features
cal = feature_generation.Calendar(region="europe", country="Switzerland", kernel=np.array([1, 2]))
X_f = compose.FeatureCreation(
    [('s_2', Shift(2), ['time_series']),
     ('ma_3', MovingAverage(window_size=3), ['time_series']),
     ('cal', cal, ['time_series'])]).fit_transform(X)

# Train/test split
X_train, y_train, X_test, y_test = model_selection.FeatureSplitter().transform(X_f, y)

# Try sklearn's MultiOutputRegressor as time-series forecasting model
gar = forecasting.GAR(LinearRegression())
gar.fit(X_train, y_train).predict(X_test)

```


## Contributing

We welcome new contributors of all experience levels. The Giotto
community goals are to be helpful, welcoming, and effective. To learn more about
making a contribution to giotto-time, please see the [CONTRIBUTING.rst](https://github.com/giotto-ai/giotto-time/blob/master/CONTRIBUTING.rst) 
file.

## Links

- Official source code repo: https://github.com/giotto-ai/giotto-time
- Download releases: https://pypi.org/project/giotto-time/
- Issue tracker: https://github.com/giotto-ai/giotto-time/issues

## Community

Giotto Slack workspace: https://slack.giotto.ai/

## Contacts

maintainers@giotto.ai
