import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Union, List, Callable, Optional
from gtime.plotting.preprocessing import seasonal_split
from gtime.stat_tools import acf, pacf
from scipy.stats import norm


def lag_plot(df: pd.DataFrame, lags: Union[int, List[int]], plots_per_row: int = 4):
    """
    Lag plots, scatter plot of x_i against x_i-k for every k in ``lags``.
    https://www.itl.nist.gov/div898/handbook/eda/section3/lagplot.htm

    Parameters
    ----------
    df : pd.DataFrame, time series to plot
    lags : int or list of ints, lags to plot, if int n is given, first n lags are used
    plots_per_row : int, number of lag plots per one row of seasonal_subplots

    Returns
    -------
    axes : plot axes

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from gtime.plotting import lag_plot
    >>> from gtime.forecasting import NaiveForecaster
    >>> idx = pd.period_range(start='2011-01-01', end='2012-01-01')
    >>> np.random.seed(1)
    >>> df = pd.DataFrame(np.random.random((len(idx), 1)), index=idx, columns=['1'])
    >>> lag_plot(df, lags=[1, 5, 10])
    >>> plt.show()

    """

    if isinstance(lags, int):
        lags = list(range(1, lags + 1))

    if len(lags) > plots_per_row:
        rows = (len(lags) - 1) // plots_per_row + 1
        cols = plots_per_row
    else:
        rows = 1
        cols = len(lags)

    fig, ax = plt.subplots(
        rows,
        cols,
        sharey=True,
        sharex=True,
        figsize=(20, 5 * rows),
        squeeze=False,
        gridspec_kw={"wspace": 0.05},
    )
    x_lim = df.agg(["min", "max"]).values

    for i, l in enumerate(lags):
        axes = ax[i // plots_per_row, i % plots_per_row]
        axes.scatter(df.iloc[l:], df.iloc[:-l])
        axes.set(title="Lag " + str(l))
        axes.plot(x_lim, x_lim, ls="--", c=".7")
        axes.set(xlim=x_lim, ylim=x_lim)
        axes.label_outer()

    return ax


def seasonal_subplots(
    df: pd.DataFrame,
    cycle: str = "year",
    freq: Optional[str] = None,
    agg: Union[str, Callable] = "mean",
    box: bool = False,
):
    """
    Seasonal subplots, a series of subplots representing average values and cycle-over-cycle dynamics or box plots for each season.
    https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4431.htm

    Parameters
    ----------
    df : pd.DataFrame, time series to plot
    cycle : str, cycle, calendar term ('year', 'quarter', 'month', 'week') or pandas offset string
    freq : str, series frequency to serample to
    agg : str or function, aggregation function used in resampling
    box : bool, use box plots

    Returns
    -------
    axes : plot axes

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from gtime.plotting import seasonal_subplots
    >>> idx = pd.period_range(start='2011-01-01', end='2014-01-01')
    >>> np.random.seed(1)
    >>> df = pd.DataFrame(np.random.random((len(idx), 1)), index=idx, columns=['1'])
    >>> seasonal_subplots(df, 'year', freq='1M', agg='last')
    >>> plt.show()

    """
    ss = seasonal_split(df, cycle, freq, agg)
    fig, ax = plt.subplots(
        ss.columns.levshape[0],
        ss.shape[0],
        sharey=True,
        figsize=(15, 6),
        squeeze=False,
        gridspec_kw={"wspace": 0},
    )
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
                axes.axhline(mean, color="gray", linestyle="--")
            axes.set(xlabel=col.name)
            axes.set_xticklabels([])
            j += 1
        i += 1
    return ax


def _seasonal_line_plot(df: pd.DataFrame, ax: plt.Axes) -> plt.Axes:
    """
    Basic seasonal line plot.

    Parameters
    ----------
    df : pd.DataFrame, input dataframe reformated by seasons
    ax : plt.Axes, matplotlib axes to plot on

    Returns
    -------
    ax : matplotlib axes

    """
    if ax is None:
        ax = plt.subplot(111)
    df.plot(ax=ax)
    return ax


def _seasonal_polar_plot(df: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Seasonal polar plot.

    Parameters
    ----------
    df : pd.DataFrame, input dataframe reformated by seasons
    ax : plt.Axes, matplotlib axes to plot on

    Returns
    -------
    ax : matplotlib axes

    """
    df = df.append(df.iloc[0])
    if ax is None:
        ax = plt.subplot(111, projection="polar")
    angles = [x * 360 / (len(df) - 1) for x in range(len(df))]
    theta = [x / 360 * 2 * np.pi for x in angles]
    for col in df.columns:
        plt.polar(theta, df[col], scalex=False, label=col)
    ax.set_thetagrids(angles=angles)
    ax.set_xticklabels(df.index)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    return ax


def seasonal_plot(
    df: pd.DataFrame,
    cycle: str,
    freq: Optional[str] = None,
    agg: Union[str, Callable] = "mean",
    polar: bool = False,
    ax: plt.Axes = None,
):
    """
    Seasonal plot function

    Parameters
    ----------
    df : pd.DataFrame, time series to plot
    cycle : str, cycle, calendar term ('year', 'quarter', 'month', 'week') or pandas offset string
    freq : frequency, if specified, time series is resampled to it
    agg : str or function, aggregation function used in resampling
    polar : bool, polar format
    ax : plt.Axes, matplotlib axes to plot on

    Returns
    -------
    ax : matplotlib axes

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from gtime.plotting import seasonal_plot
    >>> idx = pd.period_range(start='2011-01-01', end='2014-01-01')
    >>> np.random.seed(1)
    >>> df = pd.DataFrame(np.random.random((len(idx), 1)), index=idx, columns=['1'])
    >>> seasonal_plot(df, 'year', freq='1M', agg='last')
    >>> plt.show()

    """
    df_seas = seasonal_split(df, cycle, freq, agg=agg).droplevel(0, axis=1)

    if polar:
        ax = _seasonal_polar_plot(df_seas, ax=ax)
    else:
        ax = _seasonal_line_plot(df_seas, ax=ax)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)
    ax.set_xlabel(freq)
    return ax


def acf_plot(
    df: pd.DataFrame,
    max_lags: int = 10,
    ci: float = 0.05,
    partial: bool = False,
    ax=None,
):
    """
    ACF plot function, showing autocorrelation or partial autucorrelation for lags up to ``max_lags``.
    https://www.itl.nist.gov/div898/handbook/eda/section3/autocopl.htm

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

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from gtime.plotting import acf_plot
    >>> idx = pd.period_range(start='2011-01-01', end='2014-01-01')
    >>> np.random.seed(1)
    >>> df = pd.DataFrame(np.random.random((len(idx), 1)), index=idx, columns=['1'])
    >>> acf_plot(df, max_lags=20)
    >>> plt.show()

    """
    x = np.squeeze(df.values)
    if partial:
        acfs = pacf(x, max_lags)
    else:
        acfs = acf(x, max_lags)

    if ax is None:
        ax = plt.subplot(111)

    ax.bar(range(len(acfs)), acfs, 0.05)
    ci = norm.ppf(1 - ci / 2) / np.sqrt(len(x))
    ax.axhline(ci, color="gray", linestyle="--")
    ax.axhline(0.0, color="black", linestyle="-")
    ax.axhline(-ci, color="gray", linestyle="--")
    ax.set_xlabel("Lags")

    return ax
