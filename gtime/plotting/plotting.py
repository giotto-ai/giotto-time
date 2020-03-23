import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from gtime.plotting.preprocessing import seasonal_split, acf, pacf
from scipy.stats import norm


def lagplot(df: pd.DataFrame, lags):



    if isinstance(lags, int):
        lags = list(range(1, lags + 1))

    fig, ax = plt.subplots(df.shape[1], len(lags),
                           sharey=True, sharex=True, figsize=(18, 6), squeeze=False,
                           gridspec_kw={'wspace': 0.1})
    i = 0

    for col_name, col in df.iteritems():
        j = 0
        x_lim = [df.iloc[:, i].min(), df.iloc[:, i].max()]
        for l in lags:
            axes = ax[i, j]
            axes.scatter(df.iloc[l:, i], df.iloc[:-l, i])
            axes.set(title='Lag ' + str(l), ylabel=col_name)
            axes.plot(x_lim, x_lim, ls="--", c=".7")
            axes.set(xlim=x_lim, ylim=x_lim)
            axes.label_outer()
            j += 1
        i += 1
    return axes
    # plt.show()


def subplots(df: pd.DataFrame, cycle, freq=None, agg='mean', box=False):
    ss = seasonal_split(df, cycle, freq, agg)
    fig, ax = plt.subplots(ss.columns.levshape[0], ss.shape[0],
                           sharey=True, figsize=(15, 6), squeeze=False,
                           gridspec_kw={'wspace':0})
    i = 0
    for _, table in ss.groupby(level=0, axis=1):
        j = 0
        for _, col in table.iterrows():
            axes = ax[i, j]
            if box:
                axes.boxplot(col.dropna())
            else:
                col.plot(ax=axes)
                mean = col.mean()
                axes.axhline(mean, color='gray', linestyle='--')
            axes.set(xlabel=col.name)
            axes.set_xticklabels([])
            j += 1
        i += 1
    return axes


def plot_fun(df, ax=None):
    if ax is None:
        ax = df.plot(legend=False)
    else:
        df.plot(ax=ax, legend=False)
    return ax


def basic_ts(df, ax=None):
    ax = plot_fun(df, ax)
    return ax

def polar_ts(df, ax=None):
    df = df.append(df.iloc[0])
    if ax is None:
        ax = plt.subplot(111, projection='polar')
    angles = [x * 360 / (len(df) - 1) for x in range(len(df))]
    theta = [x / 360 * 2 * np.pi for x in angles]
    for col in df.columns:
        plt.polar(theta, df[col], scalex=False)
    ax.set_thetagrids(angles=angles)
    ax.set_xticklabels(df.index)
    return ax

def seasonal_plot(df: pd.DataFrame, cycle, freq=None, polar=False, ax=None):
    df_seas = seasonal_split(df, cycle, freq)
    if polar:
        ax = polar_ts(df_seas, ax=ax)
    else:
        ax = basic_ts(df_seas, ax=ax)
    return ax


def acf_plot(df: pd.DataFrame, max_lags: int = 10, ci: float = 0.05, partial=False, ax=None):
    x = np.squeeze(df.values)
    if partial:
        acfs = pacf(x, max_lags)
    else:
        acfs = acf(x, max_lags)
    print(acfs)

    if ax is None:
        ax = plt.subplot(111)
    ax.bar(range(1, max_lags + 1), acfs, 0.05)
    ci = norm.ppf(1 - ci / 2) / np.sqrt(len(x))
    print(ci)
    ax.axhline(ci, color='gray', linestyle='--')
    ax.axhline(0.0, color='black', linestyle='-')
    ax.axhline(-ci, color='gray', linestyle='--')
    return ax