from typing import TYPE_CHECKING, Any, Self, Literal

import numpy as np
from numba import njit
from astropy import cosmology
from astropy.units import Quantity

from . import agama_wrappers
from .. import run_units
from ..types import FloatOrArray
from .distribution import Distribution

if TYPE_CHECKING:
    from .physical_examples import physical_examples


class NFW(Distribution):
    """NFW density profile."""

    def __init__(
        self,
        Rvir: Quantity['length'] | None | Literal['From mass'] = None,
        truncate: bool = True,
        **kwargs: Any,
    ) -> None:
        if Rvir == 'From mass':
            assert 'Mtot' in kwargs, 'Mtot must be provided when calculating Rvir from the total mass'
            Rvir = self.calculate_theoretical_Rvir(kwargs['Mtot'])

        super().__init__(Rvir=Rvir, truncate=truncate, **kwargs)
        self.title = 'NFW'

    @staticmethod
    @njit
    def calculate_rho(
        r: FloatOrArray,
        rho_s: float = 1,
        Rs: float = 1,
        Rvir: float = 1,
        truncate: bool = True,
        truncate_power: int = 4,
    ) -> FloatOrArray:
        """Calculate the density (`rho`) at a given radius.

        This method is meant to be overwritten by subclasses. The function gets called by njit parallelized functions and must be njit compatible.

        Parameters:
            r: The radius at which to calculate the density.
            rho_s: The scale density.
            Rs: The scale radius.
            Rvir: The virial radius.
            truncate: Whether to truncate the density at the virial radius.
            truncate_power: The power law used for truncation.

        Returns:
            The density at the given radius.
        """
        rho = rho_s / ((r / Rs) * (1 + (r / Rs)) ** 2)
        if truncate:
            return rho / (1 + (r / Rvir) ** truncate_power)
        return rho

    def calculate_theoretical_M(self, r: Quantity['length']) -> Quantity['mass']:
        """Calculate the enclosed mass based on the theoretical density profile (without truncation)."""
        x = self.to_scale(r)
        return 4 * np.pi * self.rho_s * self.Rs**3 * (np.log(1 + x) - x / (1 + x))

    @staticmethod
    def calculate_theoretical_Rvir(M: Quantity['mass']) -> Quantity['length']:
        """Calculate the virial radius based on the theoretical density profile (without truncation)."""
        return ((3 * M / (4 * np.pi * 200 * cosmology.Planck18.critical_density0)) ** (1 / 3)).to(run_units.length)

    def to_agama_potential(
        self, type: str | None = 'Spheroid', gamma: int | None = 1, beta: int | None = 3, **kwargs: Any
    ) -> agama_wrappers.Potential:
        """Generate an agama potential from the distribution. NFW is a `Spheroid` potential with `gamma=1` and `beta=3`."""
        return super().to_agama_potential(type=type, gamma=gamma, beta=beta, **kwargs)

    @classmethod
    def from_example(cls, name: 'physical_examples' = 'default', **kwargs: Any) -> Self:
        """Create an NFW distribution from a predefined list of examples matching real galaxies."""
        if name == 'Sague-1':  # Numbers taken from arXiv:0809.2781
            return cls(
                Mtot=Quantity(4.5e5, 'Msun'),
                Rvir='From mass',
                c='Dutton14',
                name=name,
                **kwargs,
            )
        elif name == 'Draco':  # Numbers taken from arXiv:2407.07769
            return cls(
                Mtot=Quantity(0.80e8, 'Msun'),
                Rs=Quantity(2.47e2, 'pc'),
                Rvir='From mass',
                name=name,
                **kwargs,
            )
        elif name == 'Fornax dSph':  # Numbers taken from doi:10.1093/mnrasl/sls031
            return cls(
                Mtot=Quantity((9 * 2 - 1) * (1e8 / 1.5) + 1e8, 'Msun'),
                Rs=Quantity(2, 'kpc'),
                c=9,
                name=name,
                **kwargs,
            )
        return cls(
            rho_s=Quantity(2.73e7, 'Msun/kpc**3'),
            Rs=Quantity(1.18, 'kpc'),
            c=19,
            name=name,
            **kwargs,
        )
