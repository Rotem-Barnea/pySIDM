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
        """Calculate the density (`rho`) at a given radius.

        This method is meant to be overwritten by subclasses. The function gets called by njit parallelized functions and must be njit compatible.

        Parameters:
            r: The radius at which to calculate the density.
            rho_s: The scale density.
            Rs: The scale radius.
            Rvir: The virial radius.

        Returns:
            The density at the given radius.
        """
        rho = rho_s / ((r / Rs) * (1 + (r / Rs)) ** 3) / np.exp((r / Rvir) ** 2)
        if isinstance(r, float):
            return rho[0]
        return rho
