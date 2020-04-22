import pandas as pd
from typing import Optional, Union, Callable


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
    elif t.start_time.weekofyear == 52 and t.end_time.month == 1:
        year = t.end_time.year - 1
    else:
        year = t.start_time.year
    week = t.start_time.weekofyear

    return "_".join([str(year), str(week)])


def _get_season_names(
    df: pd.DataFrame, cycle: str, freq: Optional[str] = None
) -> pd.DataFrame:

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
        ("year", "D"): "dayofyear",
        ("year", "M"): "month",
        ("year", "Q"): "quarter",
        ("year", "Q-DEC"): "quarter",
        ("month", "D"): "day",
        ("week", "D"): "dayofweek",
    }

    freq_map = {"year": "Y", "quarter": "Q", "month": "M", "week": "W"}

    if (cycle, freq) in calendar_map.keys():
        return getattr(df.index.start_time, calendar_map[(cycle, freq)])
    else:
        col_name = df.columns[0]
        if cycle in freq_map.keys():
            cycle = freq_map[cycle]
        return (
            df[col_name].groupby(pd.Grouper(freq=cycle, convention="s")).cumcount() + 1
        )


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

    if cycle == "year":
        return df.index.start_time.year
    elif cycle == "quarter":
        return list(
            map(
                lambda x: "_".join([str(x.start_time.year), str(x.start_time.quarter)]),
                df.index,
            )
        )
    elif cycle == "month":
        return list(
            map(
                lambda x: "_".join([str(x.start_time.year), str(x.start_time.month)]),
                df.index,
            )
        )
    elif cycle == "week":
        return list(map(_week_of_year, df.index))
    else:
        col_name = df.columns[0]
        return df[col_name].groupby(pd.Grouper(freq=cycle, convention="s")).ngroup()


def seasonal_split(
    df: pd.DataFrame,
    cycle: str,
    freq: Optional[str] = None,
    agg: Union[str, Callable] = "mean",
) -> pd.DataFrame:
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

    return df.set_index(["_Season", "_Series"]).unstack(level=0)



