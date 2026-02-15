"""Safe numpy functions"""

import numpy as np
from numpy.typing import NDArray


def inverse(denominator: NDArray[np.float64], fill_value: float = 0) -> NDArray[np.float64]:
    """Safely calculates `1/denominator`, filling cells with `denominator=0` with `fill_value`"""
    return np.divide(1, denominator, out=np.full_like(denominator, fill_value), where=denominator != 0)


def sqrt(x: NDArray[np.float64], fill_value: float = 0) -> NDArray[np.float64]:
    """Safely calculates `np.sqrt(x)`, filling cells with `x<0` with `fill_value`"""
    return np.sqrt(x, out=np.full_like(x, fill_value), where=x >= 0)


def log(x: NDArray[np.float64], fill_value: float = 0) -> NDArray[np.float64]:
    """Safely calculates `np.log(x)`, filling cells with `x<=0` with `fill_value`"""
    return np.log(x, out=np.full_like(x, fill_value), where=x > 0)
