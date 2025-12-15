from typing import Any, Self, Literal

import numpy as np
from numba import njit
from astropy import cosmology
from astropy.units import Quantity

from .. import run_units
from ..types import FloatOrArray
from .distribution import Distribution


class NFW(Distribution):
    """NFW density profile."""

    def __init__(self, Rs: Quantity['length'] | None | Literal['From mass'] = None, **kwargs: Any) -> None:
        if Rs == 'From mass':
            assert 'Mtot' in kwargs, 'Mtot must be provided when calculating Rs from the total mass'
            Rs = self.calculate_theoretical_Rvir(kwargs['Mtot'])

        super().__init__(Rs=Rs, **kwargs)
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

    @staticmethod
    def calculate_theoretical_Rvir(M: Quantity['mass']) -> Quantity['length']:
        """Calculate the virial radius based on the theoretical density profile (without truncation)."""
        return ((3 * M / (4 * np.pi * 200 * cosmology.Planck18.critical_density0)) ** (1 / 3)).to(run_units.length)

    @classmethod
    def from_examples(cls, name: Literal['Sague-1', 'Draco', 'Fornax', 'default'] = 'default', **kwargs: Any) -> Self:
        """Create an NFW distribution from a predefined list of examples matching real galaxies."""
        if name == 'Sague-1':
            return cls(
                Mtot=(Mtot := Quantity(1e8, 'Msun')),
                Rvir=cls.calculate_theoretical_Rvir(Mtot),
                c=cls.c_from_M_Dutton14(Mtot),
                particle_type='dm',
                name=name,
                **kwargs,
            )
        elif name == 'Draco':
            return cls(
                Mtot=(Mtot := Quantity(1e9, 'Msun')),
                Rvir=cls.calculate_theoretical_Rvir(Mtot),
                c=cls.c_from_M_Dutton14(Mtot),
                particle_type='dm',
                name=name,
                **kwargs,
            )
        elif name == 'Fornax':
            return cls(
                Mtot=(Mtot := Quantity(1e10, 'Msun')),
                Rvir=cls.calculate_theoretical_Rvir(Mtot),
                c=cls.c_from_M_Dutton14(Mtot),
                particle_type='dm',
                name=name,
                **kwargs,
            )
        return cls(
            rho_s=Quantity(2.73e7, 'Msun/kpc**3'),
            Rs=Quantity(1.18, 'kpc'),
            c=19,
            particle_type='dm',
            name=name,
            **kwargs,
        )
