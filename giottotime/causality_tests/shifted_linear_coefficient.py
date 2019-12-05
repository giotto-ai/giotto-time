from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from giottotime.causality_tests.base import CausalityTest


class ShiftedLinearCoefficient(CausalityTest):
    """Class responsible for assessing the shifted linear fit coefficients between two or more series.

    Parameters
    ----------
    """

    def __init__(self):
        pass

    def fit(self, data, max_shift=10):
        """Create the dataframe of shifts of each time series which maximize
        the shifted linear fit coefficients.

        Parameters
        ----------
        data : ``pd.DataFrame``, required.
            The time-series on which to compute the shifted linear fit coefficients.

        max_shift : ``int``, optional, (default=10).

        Returns
        -------
        best_shifts, max_corrs: ``pd.DataFrame``, ``pd.DataFrame``.
            best_shifts: The dataframe (Pivot table) of the shifts which
            maximize the shifted linear fit coefficients between each
            timeseries. The shift is indicated in rows.

            max_corrs: The dataframe (Pivot table) of the maximum (over all
            shifts) shifted linear fit coefficients between each pair of
            input timeseries.
        """
        self.best_shifts_ = pd.DataFrame()

        for x, y in product(data.columns, repeat=2):
            res = self._get_max_coeff_shift(data, max_shift, x=x, y=y)

            best_shift = res[1]
            max_corr = res[0]
            N = data.shape[0] - max_shift

            self.best_shifts_ = self.best_shifts_.append(
                {"x": x, "y": y, "shift": best_shift, "max_corr": max_corr},
                ignore_index=True,
            )

        best_shifts = pd.pivot_table(
            self.best_shifts_, index=["x"], columns=["y"], values="shift"
        )
        max_corrs = pd.pivot_table(
            self.best_shifts_, index=["x"], columns=["y"], values="max_corr"
        )

        self.best_shifts_ = best_shifts
        self.max_corrs_ = max_corrs

        return best_shifts, max_corrs

    def transform(self, data, target_col="y", dropna=False):
        """Shifts each input timeseries but the amount which maximizes
        shifted linear fit coefficients with the selected 'y' colums.

        Parameters
        ----------
        data : ``pd.DataFrame``, required.
            The time-series on which to perform the transformation.

        target_col : optional, (default='y').
            The column to use as the a reference (i.e., the columns which is not shifted).

        dropna : ``bool``, optional, (default=False).
            Determins if the Nan values created by shifting are retained or dropped.

        Returns
        -------
        best_shifts, max_corrs: ``pd.DataFrame``, ``pd.DataFrame``.
            best_shifts: The dataframe (Pivot table) of the shifts which
            maximize the shifted linear fit coefficients between each
            timeseries. The shift is indicated in rows.

            max_corrs: The dataframe (Pivot table) of the maximum (over all
            shifts) shifted linear fit coefficients between each pair of input
            timeseries.
        """
        for col in data:
            data[col] = data[col].shift(self.best_shifts_[col][target_col])
        if dropna:
            data = data.dropna()

        return data

    def _all_best_coeff_shifts(self, max_shift):
        self.best_shifts_ = pd.DataFrame()

        for x, y in product(self.data.columns, repeat=2):
            res = self._get_max_coeff_shift(max_shift, x=x, y=y)

            best_shift = res[1]
            max_corr = res[0]
            N = self.data.shape[0] - max_shift

            self.best_shifts_ = self.best_shifts_.append(
                {"x": x, "y": y, "shift": best_shift, "max_corr": max_corr},
                ignore_index=True,
            )

        best_shifts = pd.pivot_table(
            self.best_shifts_, index=["x"], columns=["y"], values="shift"
        )
        max_corrs = pd.pivot_table(
            self.best_shifts_, index=["x"], columns=["y"], values="max_corr"
        )

        return best_shifts, max_corrs

    def _get_max_coeff_shift(self, data, max_shift, x="x", y="y"):
        shifts = pd.DataFrame()

        shifts[x] = self.data[x]
        shifts[y] = self.data[y]

        for shift in range(max_shift):
            shifts[shift] = self.data[x].shift(shift)

        shifts = shifts.dropna()
        lf = LinearRegression().fit(shifts[range(max_shift)].values, shifts[y].values)

        q = lf.coef_.max(), np.argmax(lf.coef_)
        return q
