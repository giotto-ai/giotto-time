from typing import List

import pandas as pd

import pandas.util.testing as testing


def make_df_from_expected_shifts(expected_shifts: List[int]) -> pd.DataFrame:
    testing.N, testing.K = 500, 1

    df = testing.makeTimeDataFrame(freq="D")
    for sh, k in zip(expected_shifts, range(3)):
        df[f"shift_{k}"] = df["A"].shift(-sh)
    df = df.dropna()

    return df


def shift_df_from_expected_shifts(
    df: pd.DataFrame, expected_shifts: List[int]
) -> pd.DataFrame:
    for sh, k in zip(expected_shifts, range(3)):
        df[f"shift_{k}"] = df[f"shift_{k}"].shift(-sh)
    return df.dropna()
