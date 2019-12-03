from .base import TrendModel
from .custom_trend import CustomTrend
from .exponential_trend import ExponentialTrend
from .function_trend import FunctionTrend
from .polynomial_trend import PolynomialTrend

__all__ = [
    'TrendModel',
    'CustomTrend',
    'ExponentialTrend',
    'FunctionTrend',
    'PolynomialTrend'
]
