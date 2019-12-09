from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class GrangerTest:
    def __init__(self, data):
        self.data = data

    def shifted_partial_corr_matrix(self, max_shift, x="x", y="y"):
        shifts = pd.DataFrame()

        for shift in range(max_shift):
            shifts[shift] = self.data[x].shift(shift)

        shifts = shifts.dropna()
        self.shifted_corrs = shifts.corrwith(self.data[y])

        return self.shifted_corrs

    def all_best_corr_shifts(self, max_shift):
        self.best_shifts = pd.DataFrame()

        for x, y in product(self.data.columns, repeat=2):
            res = self.get_max_corr_shift(max_shift, x=x, y=y)

            best_shift = res[1]
            max_corr = res[0]
            N = self.data.shape[0] - max_shift

            self.best_shifts = self.best_shifts.append(
                {"x": x, "y": y, "shift": best_shift, "max_corr": max_corr},
                ignore_index=True,
            )

        best_shifts = pd.pivot_table(
            self.best_shifts, index=["x"], columns=["y"], values="shift"
        )
        max_corrs = pd.pivot_table(
            self.best_shifts, index=["x"], columns=["y"], values="max_corr"
        )

        return best_shifts, max_corrs

    def get_max_corr_shift(self, max_shift, x="x", y="y"):
        shifts = pd.DataFrame()

        for shift in range(max_shift):
            shifts[shift] = self.data[x].shift(shift)

        shifts = shifts.dropna()
        self.shifted_corrs = shifts.corrwith(self.data[y])

        q = self.shifted_corrs.max(), self.shifted_corrs.idxmax()
        return q

    def all_best_coeff_shifts(self, max_shift):
        self.best_shifts = pd.DataFrame()

        for x, y in product(self.data.columns, repeat=2):
            res = self.get_max_coeff_shift(max_shift, x=x, y=y)

            best_shift = res[1]
            max_corr = res[0]
            N = self.data.shape[0] - max_shift

            self.best_shifts = self.best_shifts.append(
                {"x": x, "y": y, "shift": best_shift, "max_corr": max_corr},
                ignore_index=True,
            )

        best_shifts = pd.pivot_table(
            self.best_shifts, index=["x"], columns=["y"], values="shift"
        )
        max_corrs = pd.pivot_table(
            self.best_shifts, index=["x"], columns=["y"], values="max_corr"
        )

        return best_shifts, max_corrs

    def get_max_coeff_shift(self, max_shift, x="x", y="y"):
        shifts = pd.DataFrame()

        shifts[x] = self.data[x]
        shifts[y] = self.data[y]

        for shift in range(max_shift):
            shifts[shift] = self.data[x].shift(shift)

        shifts = shifts.dropna()
        lf = LinearRegression().fit(shifts[range(max_shift)].values, shifts[y].values)

        q = lf.coef_.max(), np.argmax(lf.coef_)
        return q


if __name__ == "__main__":
    import pandas.util.testing as testing
    from random import randint

    testing.N, testing.K = 2000, 1

    df = testing.makeTimeDataFrame(freq="MS")
    df["B"] = df["A"].shift(8) + pd.Series(
        data=np.random.normal(0, 0.3, df.shape[0]), index=df.index
    )

    for k in range(10):
        df[f"C_{k}"] = df["A"].shift(8 + randint(10, 20)) + pd.Series(
            data=np.random.normal(0, 0.3, df.shape[0]), index=df.index
        )

    df = df.dropna()

    gt = GrangerTest(df)
    # q = gt.shifted_partial_corr_matrix(20, 'A', 'B')

    w = gt.all_best_corr_shifts(max_shift=20)
    print(w[0])
    print("\n\n±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±\n\n")
    print(w[1])

    print(
        "\n\n±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±\n\n"
    )

    q = gt.all_best_coeff_shifts(max_shift=20)

    print(q[0])
    print("\n\n±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±\n\n")
    print(q[1])

    print(q[1] - w[1])
    print(q[0] == w[0])

#


#
