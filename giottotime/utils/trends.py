import numpy as np


def polynomial(X, weights):
    return np.poly1d(weights)(X)


def exponential(X, exponent):
    return np.exp(X * exponent)


TRENDS = {"polynomial": polynomial, "exponential": exponential}
