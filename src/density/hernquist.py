from numba import njit
from typing import Any
from ..types import FloatOrArray
from .density import Density


class Hernquist(Density):
    """Hernquist density profile."""

    def __init__(self, **kwargs: Any) -> None:
        """Hernquist density profile.

        Parameters:
            kwargs: Additional keyword arguments passed to the Density parent class.

        Returns:
            Hernquist distribution object.
        """
        super().__init__(**kwargs)
        self.title = 'Hernquist'
        self.rho_s = self.calculate_rho_scale()

    def __repr__(self):
        return f"""Hernquist density
  - Rs = {self.Rs:.4f}
  - Rvir = {self.Rvir:.4f}
  - Mtot = {self.Mtot:.3e}
  - rho_s = {self.rho_s:.4f}
  - Tdyn = {self.Tdyn:.4f}

  - Rmin = {self.Rmin:.4f}
  - Rmax = {self.Rmax:.4f}
  - space_steps = {self.space_steps:.0e}"""

    @staticmethod
    @njit
    def calculate_rho(r: FloatOrArray, rho_s: float = 1, Rs: float = 1, Rvir: float = 1) -> FloatOrArray:
        return rho_s / ((r / Rs) * (1 + (r / Rs)) ** 3) / (1 + (r / Rvir) ** 10)
