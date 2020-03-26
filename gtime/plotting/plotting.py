import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from gtime.plotting.preprocessing import seasonal_split, acf, pacf
from scipy.stats import norm


def lagplot(df: pd.DataFrame, lags):
    """
    Lag scatter plots.

    Parameters
    ----------
    df : pd.DataFrame, time series to plot
    lags : int or list of ints, lags to plot

    Returns
    -------
    axes : plot axes

    """

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
    return ax


def subplots(df: pd.DataFrame, cycle, freq=None, agg='mean', box=False):
    """
    Seasonal subplots

    Parameters
    ----------
    df : pd.DataFrame, time series to plot
    cycle : str, cycle, calendar term ('year', 'quarter', 'month', 'week') or pandas offset string
    freq : frequency, if specified, time series is resampled to it
    agg : aggregation function used in resampling
    box : bool, use box plots

    Returns
    -------

    """
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
    return ax


def seasonal_line_plot(df, ax=None):
    """
    Basic seasonal line plot.

    Parameters
    ----------
    df : pd.DataFrame, input dataframe reformated by seasons
    ax : matplotlib axes to plot on

    Returns
    -------
    ax : matplotlib axes

    """
    if ax is None:
        ax = df.plot(legend=False)
    else:
        df.plot(ax=ax, legend=False)
    return ax


def seasonal_polar_ts(df, ax=None):
    """
    Seasonal polar plot.

    Parameters
    ----------
    df : pd.DataFrame, input dataframe reformated by seasons
    ax : matplotlib axes to plot on

    Returns
    -------
    ax : matplotlib axes

    """
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
    """
    Seasonal plot function

    Parameters
    ----------
    df : pd.DataFrame, input time series
    cycle : str, cycle, calendar term ('year', 'quarter', 'month', 'week') or pandas offset string
    freq : frequency, if specified, time series is resampled to it
    polar : bool, polar format
    ax : matplotlib axes to plot on

    Returns
    -------
    ax : matplotlib axes

    """
    df_seas = seasonal_split(df, cycle, freq)

    if polar:
        ax = seasonal_polar_ts(df_seas, ax=ax)
    else:
        ax = seasonal_line_plot(df_seas, ax=ax)
    return ax


def acf_plot(df: pd.DataFrame, max_lags: int = 10, ci: float = 0.05, partial: bool = False, ax=None):
    """
    ACF plot function

    Parameters
    ----------
    df : pd.DataFrame, input time series
    max_lags : int, maximum number of lags to be calculated
    ci : float, confidence interval for the estimate
    partial : bool, whether to calculate partial autocorrelation instead of regular one
    ax : matplotlib axes to plot on

    Returns
    -------
    ax : matplotlib axes

    """
    x = np.squeeze(df.values)
    if partial:
        acfs = pacf(x, max_lags)
    else:
        acfs = acf(x, max_lags)

    if ax is None:
        ax = plt.subplot(111)

    ax.bar(range(0, max_lags), acfs, 0.05)
    ci = norm.ppf(1 - ci / 2) / np.sqrt(len(x))
    ax.axhline(ci, color='gray', linestyle='--')
    ax.axhline(0.0, color='black', linestyle='-')
    ax.axhline(-ci, color='gray', linestyle='--')
    ax.set_xlabel('Lags')

    return ax

