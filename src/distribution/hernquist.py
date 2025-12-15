from typing import Any, Self, Literal

import numpy as np
from numba import njit
from astropy.units import Quantity

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
        return rho_s / ((r / Rs) * (1 + (r / Rs)) ** 3) / (1 + (r / Rvir) ** 4)

    @staticmethod
    def r_half_light_to_Rs(r: Quantity['length']) -> Quantity['length']:
        """Calculates the scale radius (`Rs`) from the half-light radius."""
        return r / (1 + np.sqrt(2))

    @classmethod
    def from_examples(cls, name: Literal['Sague-1', 'Draco', 'Fornax', 'default'] = 'default', **kwargs: Any) -> Self:
        """Create a Hernquist distribution from a predefined list of examples matching real galaxies."""
        if name == 'Sague-1':
            return cls(
                Rs=cls.r_half_light_to_Rs(Quantity(30, 'pc')),
                Mtot=Quantity(5.8e2, 'Msun'),
                c=100,
                particle_type='baryon',
                **kwargs,
            )
        elif name == 'Draco':
            return cls(
                Rs=cls.r_half_light_to_Rs(Quantity(200, 'pc')),
                Mtot=Quantity(2e5, 'Msun'),
                c=100,
                particle_type='baryon',
                **kwargs,
            )
        elif name == 'Fornax':
            return cls(
                Rs=cls.r_half_light_to_Rs(Quantity(700, 'pc')),
                Mtot=Quantity(3e7, 'Msun'),
                c=100,
                particle_type='baryon',
                **kwargs,
            )
        return cls(
            Rs=Quantity(1.18, 'kpc'),
            rho_s=Quantity(1.1e4, 'Msun/kpc**3'),
            c=19,
            particle_type='baryon',
            **kwargs,
        )
