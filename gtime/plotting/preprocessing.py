import pandas as pd
import numpy as np
from scipy.linalg import toeplitz


def seasonal_split(df: pd.DataFrame, cycle='year', freq=None, agg='mean'):
    if freq is None:
        freq = df.index.freqstr
    df = df.resample(freq).agg(agg)

    if isinstance(cycle, str):
        if cycle == 'year':
            df['_Series'] = df.index.start_time.year
            if freq == 'D':
                df['_Season'] = df.index.dayofyear
            elif freq == 'M':
                df['_Season'] = df.index.month
            elif freq in ['Q', 'Q-DEC']:
                df['_Season'] = df.index.quarter
            else:
                df['_Season'] = df.resample('Y').apply(lambda x: pd.Series(np.arange(1, len(x) + 1))).values

        elif cycle == 'quarter':
            df['_Series'] = list(map(lambda x: '_'.join([str(x.year), str(x.quarter)]), df.index))
            df['_Season'] = df.resample('Q').apply(lambda x: pd.Series(np.arange(1, len(x) + 1))).values

        elif cycle == 'month':
            df['_Series'] = list(map(lambda x: '_'.join([str(x.year), str(x.month)]), df.index))
            if freq == 'D':
                df['_Season'] = df.index.day
            else:
                df['_Season'] = df.resample('M').apply(lambda x: pd.Series(np.arange(1, len(x) + 1))).values

        elif cycle == 'week':
            df['_Series'] = list(map(lambda x: '_'.join([str(x.year), str(x.weekofyear)]), df.index))
            if freq == 'D':
                df['_Season'] = df.index.day
            else:
                df['_Season'] = df.resample('W').apply(lambda x: pd.Series(np.arange(1, len(x) + 1))).values
        else:
            raise ValueError("Incorrect cycle period name")
    else:
        df['_Series'] = df.resample
        s = []
        for i, j in df.resample(freq):
            s += [i.__str__()] * len(j)
        df['_Season'] = s

    return df.set_index(['_Series', '_Season']).unstack(level=0)


def acf(x, max_lags=None):
    n = len(x)
    if max_lags is None or max_lags > n:
        max_lags = n
    x = (x - np.mean(x)) / (np.std(x) * np.sqrt(n))

    if max_lags == n:
        result = np.correlate(x, x, mode='full')[-n:]
    else:
        result = np.correlate(x, x, mode='full')[-n:-n + max_lags + 1]
    return result


def yw(x: np.array, order=1, unbiased=False):
    n = len(x)
    r = np.zeros(order + 1, np.float64)
    r[0] = (x ** 2).sum() / n
    for k in range(1, order + 1):
        r[k] = (x[0:-k] * x[k:]).sum() / (n - k * unbiased)
    R = toeplitz(r[:-1])
    rho = np.linalg.solve(R, r[1:])
    return rho


def pacf(x, max_lags=1):
    n = len(x)
    x = (x - np.mean(x)) / (np.std(x) * np.sqrt(n))
    pacf = [1.]
    for k in range(1, max_lags + 1):
        pacf.append(yw(x, k)[-1])
    return np.array(pacf)
