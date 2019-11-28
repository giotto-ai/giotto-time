from sklearn.metrics import mean_squared_error

import pandas.util.testing as testing

import numpy as np
import pandas as pd

from itertools import product

from math import sqrt

from sklearn.linear_model import LinearRegression

################################################################################################

class ShiftedPearsonCorrelation(CausalityTest):
    def __init__(self):
        pass

    def fit(self, data, max_shift=10):
        self.best_shifts = pd.DataFrame()

        for x, y in product( data.columns, repeat=2 ):
            res = self._get_max_corr_shift( data, max_shift, x=x, y=y )

            best_shift = res[1]
            max_corr = res[0]
            N = data.shape[0] - max_shift

            self.best_shifts = self.best_shifts.append( {   'x' : x ,
                                                            'y' : y,
                                                            'shift' : best_shift,
                                                            'max_corr' : max_corr
                                                            }, ignore_index = True )

        best_shifts = pd.pivot_table( self.best_shifts, index=['x'], columns=['y'], values='shift' )
        max_corrs = pd.pivot_table( self.best_shifts, index=['x'], columns=['y'], values='max_corr' )

        self.best_shifts = best_shifts
        self.max_corrs = max_corrs

        return best_shifts, max_corrs

    def transform(self, data, target_col='y', dropna=False):
        for col in data:
            data[col] = data[col].shift( self.best_shifts[col][target_col] )
        if drop_na:
            data = data.dropna()

        return data

    def _shifted_partial_corr_matrix(self, data, max_shift, x='x', y='y'):
        shifts = pd.DataFrame()

        for shift in range(max_shift):
            shifts[shift] = data[x].shift(shift)

        shifts = shifts.dropna()
        self.shifted_corrs = shifts.corrwith(data[y])

        return self.shifted_corrs

    def _get_max_corr_shift(self, data, max_shift, x='x', y='y'):
        shifts = pd.DataFrame()

        for shift in range(max_shift):
            shifts[shift] = data[x].shift(shift)

        shifts = shifts.dropna()
        self.shifted_corrs = shifts.corrwith(data[y])

        q = self.shifted_corrs.max(), self.shifted_corrs.idxmax()
        return q
