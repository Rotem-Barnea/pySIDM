import numpy as np
from numba import njit
from typing import Any
from astropy.units import Quantity
from .distribution import Distribution
from ..types import FloatOrArray


class NFW(Distribution):
    """NFW density profile."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.title = 'NFW'

    @staticmethod
    @njit
    def calculate_rho(r: FloatOrArray, rho_s: float = 1, Rs: float = 1, Rvir: float = 1) -> FloatOrArray:
        rho = rho_s / ((r / Rs) * (1 + (r / Rs)) ** 2) / np.exp((r / Rvir) ** 2)
        if isinstance(r, float):
            return rho[0]
        return rho

    def calculate_theoretical_M(self, r: Quantity['length']) -> Quantity['mass']:
        x = self.to_scale(r)
        return 4 * np.pi * self.rho_s * self.Rs**3 * (np.log(1 + x) - x / (1 + x))
