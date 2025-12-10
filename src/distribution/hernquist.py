from typing import Any

from numba import njit
from astropy.units import Quantity

from ..types import FloatOrArray
from .distribution import Distribution


class Hernquist(Distribution):
    """Hernquist density profile."""

    def __init__(
        self,
        rho_s: Quantity['mass density'] | None = Quantity(1.1e4, 'Msun/kpc**3'),
        Rs: Quantity['length'] | None = Quantity(1.18, 'kpc'),
        c: int | float | None = 19,
        **kwargs: Any,
    ) -> None:
        super().__init__(rho_s=rho_s, Rs=Rs, c=c, **kwargs)
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
        return rho_s / ((r / Rs) * (1 + (r / Rs)) ** 3) / (1 + (r / Rvir) ** 4)
