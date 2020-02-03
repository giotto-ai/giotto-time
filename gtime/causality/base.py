from itertools import product

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.utils.validation import check_is_fitted


class CausalityMixin:
    """ Base class for causality tests. """

    def __init__(self, bootstrap_iterations):
        self.bootstrap_iterations = bootstrap_iterations

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Shifts each input time series by the amount which optimizes correlation with
        the selected 'y' column.

        Parameters
        ----------
        data : pd.DataFrame, shape (n_samples, n_time_series), required
            The DataFrame containing the time series on which to perform the
            transformation.

        Returns
        -------
        data_t : pd.DataFrame, shape (n_samples, n_time_series)
            The DataFrame (Pivot table) of the shifts which maximize the correlation
            between each time series. The shift is indicated in rows.

        """
        check_is_fitted(self)
        data_t = data.copy()

        for col in data_t:
            if col != self.target_col:
                data_t[col] = data_t[col].shift(
                    -self.best_shifts_[col][self.target_col]
                )

        if self.dropna:
            data_t = data_t.dropna()

        return data_t

    def _initialize_table(self):
        best_shifts = pd.DataFrame(columns=["x", "y", "shift", "max_corr"])
        column_types = {
            "x": np.float64,
            "y": np.float64,
            "shift": np.int64,
            "max_corr": np.int64,
        }

        if self.bootstrap_iterations:
            best_shifts = best_shifts.reindex(
                best_shifts.columns.tolist() + ["p_values"], axis=1
            )
            column_types["p_values"] = np.float64

        best_shifts = best_shifts.astype(column_types)
        return best_shifts

    def _compute_best_shifts(self, data, shift_func):
        best_shifts = self._initialize_table()

        for x, y in product(data.columns, repeat=2):
            res = shift_func(data, x=x, y=y)
            best_shift = res[1]
            max_corr = res[0]
            tables = {
                "x": x,
                "y": y,
                "shift": best_shift,
                "max_corr": max_corr,
            }
            if self.bootstrap_iterations:
                p_value = self._compute_is_test_significant(data, x, y, best_shift)
                tables["p_values"] = p_value

            best_shifts = best_shifts.append(tables, ignore_index=True,)

        return best_shifts

    def _compute_is_test_significant(self, data, x, y, best_shift):
        bootstrap_matrix = data.copy()
        bootstrap_matrix[y] = bootstrap_matrix.shift(best_shift)[y]
        bootstrap_matrix.dropna(axis=0, inplace=True)
        rhos = []

        for k in range(self.bootstrap_iterations):
            bootstraps = bootstrap_matrix.sample(n=len(data), replace=True)
            rhos.append(stats.pearsonr(bootstraps[x], bootstraps[y])[0])
        rhos = pd.DataFrame(rhos)

        percentile = stats.percentileofscore(rhos, 0) / 100
        p_value = 2 * (percentile if percentile < 0.5 else 1 - percentile)

        return p_value

    def _create_pivot_tables(self, best_shifts):
        pivot_best_shifts = pd.pivot_table(
            best_shifts, index=["x"], columns=["y"], values="shift"
        )
        max_corrs = pd.pivot_table(
            best_shifts, index=["x"], columns=["y"], values="max_corr"
        )

        pivot_tables = {"best_shifts": pivot_best_shifts, "max_corrs": max_corrs}

        if self.bootstrap_iterations:
            p_values = pd.pivot_table(
                best_shifts, index=["x"], columns=["y"], values="p_values"
            )
            pivot_tables["p_values"] = p_values

        return pivot_tables
