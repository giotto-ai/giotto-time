

class ColumnsProduct:

    def __init__(self, column1, column2):
        self.column1 = column1
        self.column2 = column2

    def transform(self, X):
        return X[self.column1] * X[self.column2]
