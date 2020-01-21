import numpy as np


def polynomial(X, weights):
    """TODO: doc"""
    return np.poly1d(weights)(X)


def exponential(X, exponent):
    """TODO: doc"""
    return np.exp(X * exponent)


TRENDS = {'polynomial': polynomial, 'exponential': exponential}
