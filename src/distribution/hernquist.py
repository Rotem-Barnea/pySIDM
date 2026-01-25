"""Hernquist profile distribution class"""

from typing import TYPE_CHECKING, Any, Self, Literal

import numpy as np
from numba import njit
from astropy.units import Quantity

from src import agama_wrappers
from src.types import FloatOrArray

from . import example_db
from .distribution import Distribution

if TYPE_CHECKING:
    from .bundle import PhysicalExample


class Hernquist(Distribution):
    """Hernquist density profile."""

    def __init__(
        self,
        total_mass: Quantity['mass'] | None = None,
        mass_stellar: Quantity['mass'] | None = None,
        c: float | None = 1,
        truncate: bool = False,
        **kwargs: Any,
    ) -> None:
        if mass_stellar is not None and total_mass is None:
            total_mass = mass_stellar
        super().__init__(total_mass=total_mass, c=c, truncate=truncate, **kwargs)
        self.title = 'Hernquist'

    @staticmethod
    @njit
    def calculate_density(
        r: FloatOrArray,
        rho_s: float = 1,
        r_s: float = 1,
        r_vir: float = 1,
        truncate: bool = False,
        truncate_power: int = 4,
    ) -> FloatOrArray:
        """Calculate the density (`rho`) at a given radius.

        This method is meant to be overwritten by subclasses. The function gets called by njit parallelized functions and must be njit compatible.

        Parameters:
            r: The radius at which to calculate the density.
            rho_s: The scale density.
            r_s: The scale radius.
            r_vir: The virial radius.
            truncate: Whether to truncate the density at the virial radius.
            truncate_power: The power law used for truncation.

        Returns:
            The density at the given radius.
        """
        rho = rho_s / ((r / r_s) * (1 + (r / r_s)) ** 3)
        if truncate:
            return rho / (1 + (r / r_vir) ** truncate_power)
        return rho

    def to_agama_potential(
        self, type: str | None = 'Dehnen', gamma: int | None = 1, beta: int | None = None, **kwargs: Any
    ) -> agama_wrappers.Potential:
        """Generate an agama potential from the distribution. Hernquist is a `Dehnen` potential with `gamma=1`."""
        return super().to_agama_potential(type=type, gamma=gamma, beta=beta, **kwargs)

    @classmethod
    def from_example(
        cls,
        name: 'PhysicalExample' = 'default',
        on_unknown: Literal['error', 'warning', 'suppress'] = 'suppress',
        **kwargs: Any,
    ) -> Self:
        """Create a Hernquist distribution from a predefined list of examples matching real galaxies."""
        if name == 'Sague-1':
            return cls(
                r_half_light=Quantity(30, 'pc'),  # doi:10.1111/j.1365-2966.2009.15287.x
                total_mass=Quantity(1e3, 'Msun'),  # arXiv:0809.2781
                name=name,
                **kwargs,
            )
        elif name == 'Daneng2024:DM11+baryon':
            return cls(
                total_mass=Quantity(1e11, 'Msun') * 0.01,
                r_s=Quantity(9.1, 'kpc') * 0.0844,
                name=name,
                **kwargs,
            )
        return cls(
            **example_db.get_db_parameters(
                name=name,
                on_unknown=on_unknown,
                default=example_db.ExampleParameters(
                    mass_stellar=Quantity(1.11e5, 'Msun'),
                    mass_half_light=Quantity(np.nan, 'Msun'),
                    r_half_light=Quantity(1.18, 'kpc'),
                ),
            ),
            name=name,
            **kwargs,
        )
