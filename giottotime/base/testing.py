from hypothesis import assume
import hypothesis._strategies as st

import pandas as pd


@st.defines_strategy
def double_dates(
        min_start_date: pd.datetime = None,
        max_start_date: pd.datetime = None,
        min_end_date: pd.datetime = None,
        max_end_date: pd.datetime = None,
):
    start_date = st.datetimes(min_start_date, max_start_date)
    end_date = st.datetimes(min_end_date, max_end_date)
    assume(start_date < end_date)
    return start_date, end_date


@st.defines_strategy
def start_date_end_date(
        min_start_date: pd.datetime = None,
        max_start_date: pd.datetime = None,
        min_end_date: pd.datetime = None,
        max_end_date: pd.datetime = None,
):
    return double_dates(
        min_start_date, max_start_date,
        min_end_date, max_end_date
    ).filter(lambda x, y: y < y)


@st.defines_strategy
def period_ranges(
        min_start_date: pd.datetime,
        max_start_date: pd.datetime,
        min_end_date: pd.datetime,
        max_end_date: pd.datetime,
        min_periods: int,
        max_periods: int,
):
    start_date = st.datetimes(min_start_date, max_start_date)
    end_date = st.datetimes(min_end_date, max_end_date)
    periods = st.integers(min_periods, max_periods)



@st.defines_strategy
def period_indexes(
        min_start_date: pd.datetime,
        max_start_date: pd.datetime,
        min_end_date: pd.datetime,
        max_end_date: pd.datetime,
        min_periods: int,
        max_periods: int,
):
    pass
