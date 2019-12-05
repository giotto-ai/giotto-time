from abc import ABCMeta, abstractmethod


class Splitter(metaclass=ABCMeta):
    """ Base class for splitters. The children classes must implement
    the transform method.

    Both the X and y matrices must be passed in the transform methods
    """

    @abstractmethod
    def transform(self, X, y, **kwargs):
        pass

    def fit(self, X, y=None, **kwargs):
        return self

    def fit_transform(self, X, y, **kwargs):
        return self.transform(X, y, **kwargs)
