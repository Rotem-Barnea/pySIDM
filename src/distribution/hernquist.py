from typing import TYPE_CHECKING, Any, Self

from numba import njit
from astropy.units import Quantity

from . import example_db, agama_wrappers
from ..types import FloatOrArray
from .distribution import Distribution

if TYPE_CHECKING:
    from .physical_examples import physical_examples


class Hernquist(Distribution):
    """Hernquist density profile."""

    def __init__(
        self,
        Rs: Quantity['length'] | None = None,
        Mtot: Quantity['mass'] | None = None,
        mass_stellar: Quantity['mass'] | None = None,
        r_half_light: Quantity['length'] | None = None,
        c: float | None = 1,
        truncate: bool = False,
        **kwargs: Any,
    ) -> None:
        if mass_stellar is not None and Mtot is None:
            Mtot = mass_stellar
        if r_half_light is not None and Rs is None:
            Rs = self.r_half_light_to_Rs(r_half_light)
        super().__init__(Rs=Rs, Mtot=Mtot, c=c, truncate=truncate, **kwargs)
        self.title = 'Hernquist'

    @staticmethod
    @njit
    def calculate_rho(
        r: FloatOrArray,
        rho_s: float = 1,
        Rs: float = 1,
        Rvir: float = 1,
        truncate: bool = False,
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
        rho = rho_s / ((r / Rs) * (1 + (r / Rs)) ** 3)
        if truncate:
            return rho / (1 + (r / Rvir) ** truncate_power)
        return rho

    def to_agama_potential(
        self, type: str | None = 'Dehnen', gamma: int | None = 1, beta: int | None = None, **kwargs: Any
    ) -> agama_wrappers.Potential:
        """Generate an agama potential from the distribution. Hernquist is a `Dehnen` potential with `gamma=1`."""
        return super().to_agama_potential(type=type, gamma=gamma, beta=beta, **kwargs)

    @classmethod
    def from_example(cls, name: 'physical_examples' = 'default', **kwargs: Any) -> Self:
        """Create a Hernquist distribution from a predefined list of examples matching real galaxies."""
        if name == 'Sague-1':
            return cls(
                R_half_light=Quantity(30, 'pc'),  # doi:10.1111/j.1365-2966.2009.15287.x
                Mtot=Quantity(1e3, 'Msun'),  # arXiv:0809.2781
                name=name,
                **kwargs,
            )
        elif name == 'Draco':
            raise NotImplementedError('Draco example not implemented for Hernquist (try Plummer for baryons).')
        # elif name == 'Fornax dSph':
        #     return cls(
        #         R_half_light=Quantity(668, 'pc'),  # doi:10.1111/j.1365-2966.2012.21885.x
        #         Mtot=Quantity(3e7, 'Msun'),  # Why?
        #         name=name,
        #         **kwargs,
        #     )
        elif name == 'Fornax dSph':  # Calculation.
            return cls(
                **example_db.get_db_parameters(name=name),
                name=name,
                **kwargs,
            )
        return cls(
            Rs=Quantity(1.18, 'kpc'),
            rho_s=Quantity(1.1e4, 'Msun/kpc**3'),
            name=name,
            **kwargs,
        )
