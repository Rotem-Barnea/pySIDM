"""Cored Plummer-like density profile.

Density profile: ρ(r) = ρ₀ / (1 + (r/rc)²)³

Constant-density core with ρ(r→0) = ρ₀ and asymptotic falloff ρ ∝ r⁻⁶.
"""

from typing import TYPE_CHECKING, Any, Self

from numba import njit
from astropy.units import Quantity

from src import agama_wrappers
from src.types import FloatOrArray

from .distribution import Distribution

if TYPE_CHECKING:
    from .bundle import PhysicalExample


class Cored(Distribution):
    """Cored Plummer-like density profile.

    Density: ρ(r) = ρ₀ / (1 + (r/rc)²)³

    Parameters:
        rc: Core radius (scale radius)
        M_total: Total halo mass (total_mass)
        r_max_factor: Maximum radius as multiple of rc (default 85.0, maps to c parameter)
        particle_type: Particle type ('dm' or 'baryon')
    """

    def __init__(
        self,
        r_s: Quantity['length'] | None = None,
        r_vir: Quantity['length'] | None = None,
        rc: Quantity['length'] | None = None,
        r_max_factor: Quantity['length'] | None = None,
        c: float | None = 1,
        truncate: bool = False,
        **kwargs: Any,
    ):
        # r_s: core radius rc
        # c: cutoff parameter (r_max = c × rc)
        # total_mass: total halo mass
        # rho_s: central density ρ₀ (normalized from M_total)

        if rc is not None:
            r_s = rc
        if r_max_factor is not None:
            r_vir = r_max_factor
        super().__init__(r_s=r_s, r_vir=r_vir, c=c, truncate=truncate, **kwargs)
        self.title = 'Cored'

        # Physical parameters
        self.rc = self.r_s  # Core radius
        self.rho_0 = self.rho_s  # Central density ρ₀

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
        """Cored Plummer density profile.

        ρ(r) = ρ₀ / (1 + (r/rc)²)³

        Parameters:
            r: Radius
            rho_s: Central density ρ₀
            r_s: Core radius rc
            r_vir: Not used

        Returns:
            Density at radius r

        Note:
            Parameter names (rho_s, r_s) follow Distribution class convention.
            For Cored: rho_s represents ρ₀ (actual density at r=0).
        """
        rho = rho_s / (1 + (r / r_s) ** 2) ** 3
        if truncate:
            return rho / (1 + (r / r_vir) ** truncate_power)
        return rho

    def to_agama_potential(
        self, type: str | None = 'Plummer', gamma: int | None = None, beta: int | None = None, **kwargs: Any
    ) -> agama_wrappers.Potential:
        """Generate an agama potential from the distribution. Hernquist is a `Dehnen` potential with `gamma=1`."""
        return super().to_agama_potential(type=type, gamma=gamma, beta=beta, **kwargs)

    @classmethod
    def from_example(cls, name: 'PhysicalExample' = 'default', **kwargs: Any) -> Self:
        """Create a Plummer distribution from a predefined list of examples matching real galaxies."""
        if name == 'Draco':
            return cls(
                r_s=Quantity(1.75e2, 'pc'),  # arXiv:2407.07769
                total_mass=Quantity(5.07e5, 'Msun'),  # arXiv:2407.07769
                name=name,
                **kwargs,
            )
        elif name == 'default':
            return cls(
                r_s=Quantity(23, 'kpc'),
                rho_s=Quantity(3.52e07, 'Msun/kpc**3'),
                r_vir=Quantity(85, 'kpc'),
                c=None,
                name=name,
                **kwargs,
            )
        else:
            raise NotImplementedError(f'{name} example not implemented for Plummer.')
