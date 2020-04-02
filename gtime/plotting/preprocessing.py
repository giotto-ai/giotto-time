import pandas as pd
import numpy as np
from typing import Optional, Union, Callable
from scipy.linalg import toeplitz


def _week_of_year(t: pd.Period) -> str:

    """
    Week of the year, according to ISO week date format, returned as `year_week`

    Parameters
    ----------
    t : pd.Period, input period

    Returns
    -------
    str, week of the year

    """

    if t.start_time.weekofyear == 1 and t.start_time.month == 12:
        year = t.end_time.year
        week = t.start_time.weekofyear
    elif t.start_time.weekofyear == 52 and t.end_time.month == 1:
        year = t.end_time.year - 1
        week = t.start_time.weekofyear
    else:
        year = t.start_time.year
        week = t.start_time.weekofyear

    return "_".join([str(year), str(week)])


def _get_season_names(df: pd.DataFrame, cycle: str, freq: Optional[str] = None) -> pd.DataFrame:

    """
    Gets names of each season (days of week, months of year etc.) of ``df`` period index

    Parameters
    ----------
    df : pd.DataFrame
    cycle : str, cycle, calendar term ('year', 'quarter', 'month', 'week') or pandas offset string
    freq : str, series frequency to resample to

    Returns
    -------
    pd.DataFrame, a column of season names

    """

    calendar_map = {
        ('year', 'D'): 'index.start_time.dayofyear',
        ('year', 'W'): 'index.start_time.weekofyear',
        ('year', 'W-SUN'): 'index.start_time.weekofyear',
        ('year', 'M'): 'index.start_time.dayofyear',
        ('year', 'Q'): 'index.start_time.quarter',
        ('year', 'Q-DEC'): 'index.start_time.quarter',
        ('month', 'D'): 'index.start_time.day',
        ('week', 'D'): 'index.start_time.dayofweek',
    }

    freq_map = {
        'year': 'Y',
        'quarter': 'Q',
        'month': 'M',
        'week': 'W'
    }

    if (cycle, freq) in calendar_map.keys():
        return getattr(df, calendar_map[(cycle, freq)])
    else:
        col_name = df.columns[0]
        if cycle in freq_map.keys():
            cycle = freq_map[cycle]
        return df[col_name].groupby(pd.Grouper(freq=cycle, convention="s")).cumcount()+ 1


def _get_cycle_names(df: pd.DataFrame, cycle: str):
    """
    Gets names of each cycle (year, week etc.) of ``df`` period index

    Parameters
    ----------
    df : pd.DataFrame
    cycle : str, cycle, calendar term ('year', 'quarter', 'month', 'week') or pandas offset string

    Returns
    -------
    pd.DataFrame, a column of cycle names

    """

    if cycle == 'year':
        return df.index.start_time.year
    elif cycle == 'quarter':
        return list(map(lambda x: "_".join([str(x.start_time.year), str(x.start_time.quarter)]), df.index))
    elif cycle == 'month':
        return list(map(lambda x: "_".join([str(x.start_time.year), str(x.start_time.month)]), df.index))
    elif cycle == 'quarter':
        return list(map(_week_of_year, df.index))
    else:
        col_name = df.columns[0]
        return df[col_name].groupby(pd.Grouper(freq=cycle, convention="s")).ngroup()


def seasonal_split(df: pd.DataFrame, cycle: str, freq: Optional[str] = None, agg: Union[str, Callable] = "mean") -> pd.DataFrame:
    """
    Converts time series to a DataFrame with columns for each ``cycle`` period.

    Parameters
    ----------
    df : pd.DataFrame
    cycle : str, cycle, calendar term ('year', 'quarter', 'month', 'week') or pandas offset string
    freq : str, series frequency to resample to
    agg : str or function, aggregation function used in resampling

    Returns
    -------
    pd.DataFrame with seasonal columns

    """

    if freq is None:
        freq = df.index.freqstr
    df = df.resample(freq).agg(agg)

    df["_Season"] = _get_cycle_names(df, cycle)
    df["_Series"] = _get_season_names(df, cycle, freq)

    return df.set_index(["_Series", "_Season"]).unstack(level=0)


def _scale(x: np.array) -> np.array:
    """
    Scales x to mean(x) == 0 and std(x) == 1

    Parameters
    ----------
    x: np.array, array of float to be scaled

    Returns
    -------
    np.array, scaled array

    """

    if np.std(x) == 0:
        return x - np.mean(x)
    else:
        return (x - np.mean(x)) / (np.std(x) * np.sqrt(x.size))


def _autocorrelation(x: np.array) -> np.array:
    """
    Autocorrelation via np.correlate for a scaled array `x`

    Parameters
    ----------
    x: np.array, input array

    Returns
    -------
    np.array, autocorrelation for all lags

    """

    n = len(x)
    return np.correlate(x, x, mode="full")[-n:] / n


def _solve_yw_equation(r: np.array) -> np.array:
    """
    Solution to Yule-Walker equations via Töplitz matrix

    Parameters
    ----------
    r: autocorrelation coefficients

    Returns
    -------
    np.array: partial autocorrelation function

    """

    R = toeplitz(r[:-1])
    try:
        return np.linalg.solve(R, r[1:])
    except np.linalg.LinAlgError:
        print("Solution is not defined for singular matrices")
        return [np.nan]


def yule_walker(x: np.array, order=1) -> np.array:

    """ Estimate ``order`` parameters from a sequence using the Yule-Walker equations.
    http://www-stat.wharton.upenn.edu/~steele/Courses/956/Resource/YWSourceFiles/YW-Eshel.pdf

    Parameters
    ----------
    x : np.array, input time series
    order : order of the autoregressive process
    unbiased : bool, debiasing correction, False by default

    Returns
    -------
    rho : np.array, autoregressive coefficients
    """

    if order == 0:
        return np.array([1.0])

    r = _autocorrelation(x)
    r = r[:order]
    rho = _solve_yw_equation(r)
    return rho


def pacf(x: np.array, max_lags: Optional[int] = None) -> np.array:

    """Partial autocorrelation estimate based on Yule-Walker equations

    Parameters
    ----------
    x : np.array, a time series
    max_lags : int, maximum number of lags to be calculated

    Returns
    -------
    pacf : np.array, partial autocorrelations for min(max_lags, len(x)) lags, including lag 0
    """

    n = x.size
    if max_lags is None or max_lags > n:
        max_lags = n
    x = _scale(x)
    pacf = np.array([yule_walker(x, i)[-1] for i in range(max_lags)])
    return pacf


def acf(x: np.array, max_lags: Optional[int] = None) -> np.array:

    """ Autocorrelation estimate function

    Parameters
    ----------
    x : np.array, a time series
    max_lags : int, maximum number of lags to be calculated

    Returns
    -------
    acf : np.array, partial autocorrelations for min(max_lags, len(x)) lags, including lag 0
    """

    n = x.size
    if max_lags is None or max_lags > n:
        max_lags = n
    x = _scale(x)
    acf = _autocorrelation(x)
    return acf[:max_lags]
