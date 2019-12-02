from abc import ABCMeta, abstractmethod


class Splitter(metaclass=ABCMeta):

    @abstractmethod
    def transform(self, X, y, **kwargs):
        pass
