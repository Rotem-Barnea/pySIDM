"""Cored Plummer-like density profile.

Density profile: ρ(r) = ρ₀ / (1 + (r/rc)²)³

Constant-density core with ρ(r→0) = ρ₀ and asymptotic falloff ρ ∝ r⁻⁶.
"""

from typing import Any

from numba import njit

from ..types import FloatOrArray
from .distribution import Distribution


class Cored(Distribution):
    """Cored Plummer-like density profile.

    Density: ρ(r) = ρ₀ / (1 + (r/rc)²)³

    Parameters:
        rc: Core radius (scale radius)
        M_total: Total halo mass (Mtot)
        r_max_factor: Maximum radius as multiple of rc (default 85.0, maps to c parameter)
        particle_type: Particle type ('dm' or 'baryon')
    """

    def __init__(self, c: float = 85, **kwargs: Any):
        # Rs: core radius rc
        # c: cutoff parameter (r_max = c × rc)
        # Mtot: total halo mass
        # rho_s: central density ρ₀ (normalized from M_total)

        super().__init__(c=c, **kwargs)
        self.title = 'Cored'

        # Physical parameters
        self.rc = self.Rs  # Core radius
        self.rho_0 = self.rho_s  # Central density ρ₀

    @staticmethod
    @njit
    def calculate_rho(r: FloatOrArray, rho_s: float = 1, Rs: float = 1, Rvir: float = 1) -> FloatOrArray:
        """Cored Plummer density profile.

        ρ(r) = ρ₀ / (1 + (r/rc)²)³

        Parameters:
            r: Radius
            rho_s: Central density ρ₀
            Rs: Core radius rc
            Rvir: Not used

        Returns:
            Density at radius r

        Note:
            Parameter names (rho_s, Rs) follow Distribution class convention.
            For Cored: rho_s represents ρ₀ (actual density at r=0).
        """
        rho = rho_s / (1 + (r / Rs) ** 2) ** 3
        return rho
