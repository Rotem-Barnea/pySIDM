from typing import Any

import numpy as np
from numba import njit
from astropy.units import Quantity

from ..types import FloatOrArray
from .distribution import Distribution


class NFW(Distribution):
    """NFW density profile."""

    def __init__(
        self,
        rho_s: Quantity['mass density'] | None = Quantity(2.73e7, 'Msun/kpc**3'),
        Rs: Quantity['length'] | None = Quantity(1.18, 'kpc'),
        c: int | float | None = 19,
        **kwargs: Any,
    ) -> None:
        super().__init__(rho_s=rho_s, Rs=Rs, c=c, **kwargs)
        self.title = 'NFW'

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
        return rho_s / ((r / Rs) * (1 + (r / Rs)) ** 2) / (1 + (r / Rvir) ** 4)

    def calculate_theoretical_M(self, r: Quantity['length']) -> Quantity['mass']:
        """Calculate the enclosed mass based on the theoretical density profile (without truncation)."""
        x = self.to_scale(r)
        return 4 * np.pi * self.rho_s * self.Rs**3 * (np.log(1 + x) - x / (1 + x))
