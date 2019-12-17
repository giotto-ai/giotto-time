from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from giottotime.causality_tests.base import CausalityTest
from giottotime.models.utils import check_is_fitted


class ShiftedLinearCoefficient(CausalityTest):
    """Class responsible for assessing the shifted linear fit coefficients between two
    or more series.

    Parameters
    ----------
    max_shift : ``int``, optional, (default=``10``).

    target_col : ``str``, optional, (default='y').
            The column to use as the a reference (i.e., the columns which is not shifted).

    dropna : ``bool``, optional, (default=False).
        Determines if the Nan values created by shifting are retained or dropped.

    """

    def __init__(
        self, max_shift: int = 10, target_col: str = "y", dropna: bool = False
    ):
        self._max_shift = max_shift
        self._target_col = target_col
        self._dropna = dropna

    def fit(self, data: pd.DataFrame) -> "ShiftedLinearCoefficient":
        """Create the dataframe of shifts of each time series which maximize
        the shifted linear fit coefficients.

        Parameters
        ----------
        data : ``pd.DataFrame``, required.
            The time-series on which to compute the shifted linear fit coefficients.

        max_shift : ``int``, optional, (default=10).

        Returns
        -------
        self : ``ShiftedLinearCoefficient``

        """
        best_shifts = pd.DataFrame(columns=["x", "y", "shift", "max_corr"])
        best_shifts = best_shifts.astype(
            {"x": np.float64, "y": np.float64, "shift": np.int64, "max_corr": np.int64}
        )

        for x, y in product(data.columns, repeat=2):
            res = self._get_max_coeff_shift(data, self._max_shift, x=x, y=y)

            best_shift = res[1]
            max_corr = res[0]
            # N = data.shape[0] - max_shift

            best_shifts = best_shifts.append(
                {"x": x, "y": y, "shift": best_shift, "max_corr": max_corr},
                ignore_index=True,
            )

        pivot_best_shifts = pd.pivot_table(
            best_shifts, index=["x"], columns=["y"], values="shift"
        )
        max_corrs = pd.pivot_table(
            best_shifts, index=["x"], columns=["y"], values="max_corr"
        )

        self.best_shifts_ = pivot_best_shifts
        self.max_corrs_ = max_corrs

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Shifts each input timeseries but the amount which maximizes
        shifted linear fit coefficients with the selected 'y' colums.

        Parameters
        ----------
        data : ``pd.DataFrame``, required.
            The time-series on which to perform the transformation.

        Returns
        -------
        shifted_data : ``pd.DataFrame``
            The dataframe (Pivot table) of the shifts which maximize the shifted linear
            fit coefficients between each timeseries. The shift is indicated in rows.

        """
        check_is_fitted(self)
        shifted_data = data.copy()

        for col in shifted_data:
            if col != self._target_col:
                shifted_data[col] = shifted_data[col].shift(
                    self.best_shifts_[col][self._target_col]
                )

        if self._dropna:
            shifted_data = shifted_data.dropna()

        return shifted_data

    def _get_max_coeff_shift(
        self, data: pd.DataFrame, max_shift: int, x: str = "x", y: str = "y"
    ) -> (float, int):
        shifts = pd.DataFrame()

        shifts[x] = data[x]
        shifts[y] = data[y]

        for shift in range(1, max_shift):
            shifts[shift] = data[x].shift(shift)

        shifts = shifts.dropna()
        lf = LinearRegression().fit(
            shifts[range(1, max_shift)].values, shifts[y].values
        )

        q = lf.coef_.max(), np.argmax(lf.coef_) + 1
        return q
