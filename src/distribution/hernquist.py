from numba import njit
import numpy as np
from typing import Any
from ..types import FloatOrArray
from .distribution import Distribution


class Hernquist(Distribution):
    """Hernquist density profile."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.title = 'Hernquist'

    @staticmethod
    @njit
    def calculate_rho(r: FloatOrArray, rho_s: float = 1, Rs: float = 1, Rvir: float = 1) -> FloatOrArray:
        rho = rho_s / ((r / Rs) * (1 + (r / Rs)) ** 3) / np.exp((r / Rvir) ** 2)
        if isinstance(r, float):
            return rho[0]
        return rho
