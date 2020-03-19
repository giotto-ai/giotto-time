import pandas as pd
import numpy as np


def seasonal_split(df: pd.DataFrame, season):

    df = df.copy()

    if isinstance(season, str):
        if season == 'year':
            df['_Period'] = ''
            df['_Season'] = df.index.year
            df['_Idx'] = df.index.dayofyear
        elif season == 'month':
            df['_Period'] = df.index.year
            df['_Season'] = df.index.month
            #             df['Season_name'] = list(map(lambda x: '_'.join([str(x.year), str(x.month)]), df.index))
            df['_Idx'] = df.index.day
        elif season == 'week':
            df['_Period'] = df.index.year
            df['_Season'] = df.index.week
            #             df['Season_name'] = list(map(lambda x: '_'.join([str(x.year), str(x.weekofyear)]), df.index))
            df['_Idx'] = df.index.dayofweek
        else:
            raise ValueError("Incorrect period name")

        return df.set_index(['_Period', '_Season', '_Idx']).unstack(level=[0, 1])

    elif isinstance(season, pd.Timedelta):
        cols = []
        series = []
        for i, col in df.resample(season):
            cols.append(i)
            series.append(col.reset_index(drop=True))
        return pd.concat(series, keys=cols, axis=1)


def subplot_split(df: pd.DataFrame, season, agg):

    res = seasonal_split(df, season)
    res = res.agg(agg, axis=0).unstack()
    return res


def acf(df, max_lag=10):
    s = pd.DataFrame(np.nan, index=range(1, max_lag + 1), columns=df.columns)
    for i, col in df.iteritems():
        for j in s.index:
            s.loc[j, i] = col.autocorr(j)
    return s
