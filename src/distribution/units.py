"""Internal module for managing distribution-derived units"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from astropy import constants
from astropy.units import Unit, Quantity, def_unit

from src import units

if TYPE_CHECKING:
    from . import Distribution


class DistributionUnits:
    """Unit management for distribution-derived quantities"""

    def __init__(self, distribution: Distribution):
        self.distribution: Distribution = distribution

    @property
    def r_s(self) -> Unit:
        """Scale length."""
        return def_unit('r_s', self.distribution.r_s, doc=f'{self.title} length scale')

    @property
    def r_vir(self) -> Unit:
        """Virial radius."""
        return def_unit('r_vir', self.distribution.r_vir, doc=f'{self.title} virial radius')

    @property
    def rho_s(self) -> Unit:
        """Scale density."""
        return def_unit('rho_s', self.distribution.rho_s, doc=f'{self.title} density scale')

    @property
    def dynamical_time(self) -> Unit:
        """Dynamic time of the profile."""
        return def_unit(
            't_dyn',
            np.sqrt(self.distribution.r_s**3 / (constants.G * self.distribution.total_mass)).decompose(units.system),
            doc=f'{self.distribution.title} dynamic time',
        )

    def t_s(self, sigma: Quantity[units.cross_section], C: float = 0.9, a: float = 4 / np.sqrt(np.pi)) -> Unit:
        """Time scale between scatter events."""
        return def_unit(
            't_s',
            1
            / (
                C * a * 2 * self.distribution.r_s * sigma * np.sqrt((np.pi * constants.G * self.distribution.rho_s**3))
            ).decompose(units.system),
            doc=f'{self.distribution.title} scatter time scale',
        )

    def t_c(self, **kwargs: Any) -> Unit:
        """Base estimation for the core collapse time"""
        return def_unit(
            't_c',
            Quantity(340, self.t_s(**kwargs)),
            doc=f'{self.distribution.title} collapse time',
        )
