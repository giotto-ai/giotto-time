import pandas as pd
import numpy as np
from scipy.linalg import toeplitz


def seasonal_split(df: pd.DataFrame, cycle: str = "year", freq=None, agg="mean"):
    """
    Converts time series to a DataFrame with columns for each ``cycle`` period.

    Parameters
    ----------
    df : pd.DataFrame
    cycle : str, cycle, calendar term ('year', 'quarter', 'month', 'week') or pandas offset string
    freq : frequency, if specified, time series is resampled to it
    agg : aggregation function used in resampling

    Returns
    -------
    pd.DataFrame with seasonal columns

    """

    def _week_of_year(t: pd.Period):
        # pandas weekofyear inconsistency fix, according to ISO week date
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

    if freq is None:
        freq = df.index.freqstr
    df = df.resample(freq).agg(agg)
    col_name = df.columns[0]

    if cycle in ["year", "quarter", "month", "week"]:
        if cycle == "year":
            df["_Series"] = df.index.start_time.year
            if freq == "D":
                df["_Season"] = df.index.dayofyear
            elif freq in ["W", "W-SUN"]:
                df["_Season"] = df.start_time.weekofyear
            elif freq == "M":
                df["_Season"] = df.index.start_time.month
            elif freq in ["Q", "Q-DEC"]:
                df["_Season"] = df.index.start_time.quarter
            else:
                df["_Season"] = (
                    df[col_name]
                    .groupby(pd.Grouper(freq="Y", convention="s"))
                    .cumcount()
                    + 1
                )

        elif cycle == "quarter":
            df["_Series"] = list(
                map(
                    lambda x: "_".join(
                        [str(x.start_time.year), str(x.start_time.quarter)]
                    ),
                    df.index,
                )
            )
            df["_Season"] = (
                df[col_name].groupby(pd.Grouper(freq="Q", convention="s")).cumcount()
                + 1
            )

        elif cycle == "month":
            df["_Series"] = list(
                map(
                    lambda x: "_".join(
                        [str(x.start_time.year), str(x.start_time.month)]
                    ),
                    df.index,
                )
            )
            if freq == "D":
                df["_Season"] = df.index.day
            else:
                df["_Season"] = (
                    df[col_name]
                    .groupby(pd.Grouper(freq="M", convention="s"))
                    .cumcount()
                    + 1
                )

        elif cycle == "week":
            df["_Series"] = list(map(_week_of_year, df.index))
            if freq == "D":
                df["_Season"] = df.index.dayofweek
            else:
                df["_Season"] = (
                    df[col_name]
                    .groupby(pd.Grouper(freq="W", convention="s"))
                    .cumcount()
                    + 1
                )
        else:
            raise ValueError("Incorrect cycle period name")
    else:
        df["_Series"] = (
            df[col_name].groupby(pd.Grouper(freq=cycle, convention="s")).ngroup()
        )
        df["_Season"] = (
            df[col_name].groupby(pd.Grouper(freq=cycle, convention="s")).cumcount() + 1
        )

    return df.set_index(["_Series", "_Season"]).unstack(level=0)


def acf(x, max_lags=None):

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
    if np.std(x) == 0:
        x = x - np.mean(x)
    else:
        x = (x - np.mean(x)) / (np.std(x) * np.sqrt(n))
    if max_lags == n:
        acf = np.correlate(x, x, mode="full")[-n:]
    else:
        acf = np.correlate(x, x, mode="full")[-n : -n + max_lags]
    return acf


def yw(x: np.array, order=1, unbiased=False):

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

    n = len(x)
    r = np.zeros(order + 1, np.float64)
    r[0] = (x ** 2).sum() / n
    for k in range(1, order + 1):
        r[k] = (x[0:-k] * x[k:]).sum() / (n - k * unbiased)
    R = toeplitz(r[:-1])

    try:
        rho = np.linalg.solve(R, r[1:])
    except np.linalg.LinAlgError:
        print("Solution is not defined for singular matrices")
        rho = [np.nan]
    return rho


def pacf(x, max_lags: int = 1):

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
    if np.std(x) == 0:
        x = x - np.mean(x)
    else:
        x = (x - np.mean(x)) / (np.std(x) * np.sqrt(n))
    pacf = np.array([yw(x, i)[-1] for i in range(min(n, max_lags))])
    return pacf
