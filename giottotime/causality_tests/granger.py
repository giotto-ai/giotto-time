from sklearn.metrics import mean_squared_error

import pandas.util.testing as testing

import numpy as np
import pandas as pd

#

class GrangerTest():
    def __init__(self, data):
        self.data = data

    def shifted_partial_corr_matrix(self, max_shift, x='x', y='y'):
        shifts = pd.DataFrame()
        for shift in range(max_shift):
            shifts[shift] = self.data[x].shift(shift)

        shifts = shifts.dropna()
        self.shifted_corrs = shifts.corrwith(self.data[y])

        return self.shifted_corrs

    def get_max_corr_shift(self):
        #check fitted
        return self.shifted_corrs.max(), self.shifted_corrs.idxmax()

#

if __name__ == "__main__":
    import pandas.util.testing as testing
    import matplotlib.pyplot as plt

    testing.N, testing.K = 200, 1

    df = testing.makeTimeDataFrame(freq='MS')
    df['B'] = df['A'].shift(5) + pd.Series(data=np.random.normal( 0, 1 , df.shape[0] ), index=df.index)

    df = df.dropna()

    df.plot()
    plt.show()

    gt = GrangerTest(df)
    q = gt.shifted_partial_corr_matrix(50, 'A', 'B')

    q.plot()
    plt.show()

    print(gt.get_max_corr_shift())

#


#
