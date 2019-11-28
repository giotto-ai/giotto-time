from sklearn.metrics import mean_squared_error

import pandas.util.testing as testing

import numpy as np
import pandas as pd

from itertools import product

from math import sqrt

from sklearn.linear_model import LinearRegression

################################################################################################

class ShiftedLinearCoefficient(CausalityTest):
    def __init__(self):
        pass

    def fit(self, data, max_shift=10):
        self.best_shifts = pd.DataFrame()

        for x, y in product( data.columns, repeat=2 ):
            res = self._get_max_coeff_shift( data, max_shift, x=x, y=y )

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

    def _all_best_coeff_shifts(self, max_shift):
        self.best_shifts = pd.DataFrame()

        for x, y in product( self.data.columns, repeat=2 ):
            res = self._get_max_coeff_shift(max_shift, x=x, y=y)

            best_shift = res[1]
            max_corr = res[0]
            N = self.data.shape[0] - max_shift

            self.best_shifts = self.best_shifts.append( {   'x' : x ,
                                                            'y' : y,
                                                            'shift' : best_shift,
                                                            'max_corr' : max_corr
                                                            }, ignore_index = True )

        best_shifts = pd.pivot_table( self.best_shifts, index=['x'], columns=['y'], values='shift' )
        max_corrs = pd.pivot_table( self.best_shifts, index=['x'], columns=['y'], values='max_corr' )

        return best_shifts, max_corrs

    def transform(self, data, target_col='y', dropna=False):
        for col in data:
            data[col] = data[col].shift( self.best_shifts[col][target_col] )
        if dropna:
            data = data.dropna()

        return data
    
    def _get_max_coeff_shift(self, data, max_shift, x='x', y='y'):
        shifts = pd.DataFrame()

        shifts[x] = self.data[x]
        shifts[y] = self.data[y]

        for shift in range(max_shift):
            shifts[shift] = self.data[x].shift(shift)

        shifts = shifts.dropna()
        lf = LinearRegression().fit( shifts[range(max_shift)].values, shifts[y].values )

        q = lf.coef_.max(), np.argmax(lf.coef_)
        return q
