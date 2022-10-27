import warnings
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.utils.validation import check_is_fitted


class CausalityMixin:
    """ Base class for causality tests. """

    def __init__(self, bootstrap_iterations, permutation_iterations):
        self.bootstrap_iterations = bootstrap_iterations
        self.permutation_iterations = permutation_iterations

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Shifts each input time series by the amount which optimizes correlation with
        the selected 'target_col' column. If no target column is specified, the first
        column of the DataFrame is taken as the target.

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

        if self.target_col is None:
            self.target_col = data_t.columns[0]
            warnings.warn(
                "The target column was not specified. Therefore, the first "
                f"column {self.target_col } of the DataFrame was taken as "
                "target column. If you want to transform with respect to "
                "another column, please use it as a target column."
            )

        for col in data_t:
            if col != self.target_col:
                data_t[col] = data_t[col].shift(self.best_shifts_[self.target_col][col])
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

        if self.target_col is None:
            columns_to_shift = [(x, y) for x, y in product(data.columns, repeat=2)]

        else:
            columns_to_shift = [(col, self.target_col) for col in data.columns]

        for (x, y) in columns_to_shift:
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
                bootstrap_p_value = self._compute_p_values(
                    data=data, x=x, y=y, shift=best_shift, test_type="bootstrap"
                )
                tables["bootstrap_p_values"] = bootstrap_p_value

            if self.permutation_iterations:
                bootstrap_p_value = self._compute_p_values(
                    data=data, x=x, y=y, shift=best_shift, test_type="permutation"
                )
                tables["permutation_p_values"] = bootstrap_p_value

            best_shifts = best_shifts.append(tables, ignore_index=True,)

        return best_shifts

    def _compute_p_values(self, data, x, y, shift, test_type):
        data_t = data.copy()
        data_t[x] = data_t.shift(shift)[x]
        data_t.dropna(axis=0, inplace=True)
        rhos = []
        n_iterations = (
            self.permutation_iterations
            if test_type == "permutation"
            else self.bootstrap_iterations
        )

        for k in range(n_iterations):
            if test_type == "permutation":
                samples = data_t.sample(frac=1)
            else:
                samples = data_t.sample(n=len(data), replace=True)

            rhos.append(stats.pearsonr(samples[x], samples[y])[0])
        rhos = pd.DataFrame(rhos)
        percentiles = stats.percentileofscore(rhos, 0) / 100
        # print("percentile: ", percentiles)
        p_values = [2*percentile if percentile < 0.5 else 1 - percentile for percentile in percentiles]

        return p_values

    def _create_pivot_tables(self, best_shifts):
        pivot_best_shifts = pd.pivot_table(
            best_shifts, index=["x"], columns=["y"], values="shift"
        )
        max_corrs = pd.pivot_table(
            best_shifts, index=["x"], columns=["y"], values="max_corr"
        )

        pivot_tables = {"best_shifts": pivot_best_shifts, "max_corrs": max_corrs}

        if self.bootstrap_iterations:
            bootstrap_p_values = pd.pivot_table(
                best_shifts, index=["x"], columns=["y"], values="bootstrap_p_values"
            )
            pivot_tables["bootstrap_p_values"] = bootstrap_p_values

        if self.permutation_iterations:
            permutation_p_values = pd.pivot_table(
                best_shifts, index=["x"], columns=["y"], values="permutation_p_values"
            )
            pivot_tables["permutation_p_values"] = permutation_p_values

        return pivot_tables
