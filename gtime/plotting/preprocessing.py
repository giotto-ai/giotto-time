import pandas as pd
import numpy as np


def seasonal_split(df: pd.DataFrame, cycle='year', freq=None, agg='mean'):
    if freq is None:
        freq = df.index.freqstr
    df = df.resample(freq).agg(agg)

    if isinstance(cycle, str):
        if cycle == 'year':
            df['_Series'] = df.index.start_time.year
            if freq == 'D':
                df['_Season'] = df.index.dayofyear
            #             elif freq in ['W-SUN', 'W']:
            #                 df['_Season'] = df.index.start_time.weekofyear
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


def acf(df, max_lag=10):
    s = pd.DataFrame(np.nan, index=range(1, max_lag + 1), columns=df.columns)
    for i, col in df.iteritems():
        for j in s.index:
            s.loc[j, i] = col.autocorr(j)
    return s
