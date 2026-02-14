"""Internal module for managing halo-derived units"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from astropy.units import Unit, def_unit

from src import units

if TYPE_CHECKING:
    from .halo import Halo


class HaloUnits:
    """Unit management for distribution-derived quantities"""

    def __init__(self, halo: Halo):
        self.halo: Halo = halo

    @property
    def time_step(self) -> Unit:
        """time-step size"""
        return def_unit(
            'time step',
            self.halo.dt.decompose(units.system),
            format={'latex': r'time\ step'},
            doc='Single simulation time step (`dt`)',
        )

    @property
    def dynamical_time(self) -> Unit:
        """Dynamic time of the halo. Taken from the first distribution."""
        if len(self.halo.distributions) > 0:
            return self.halo.distributions[0].units.dynamical_time
        return cast(Unit, units.time)

    @property
    def core_collapse(self) -> Unit:
        """Real core collapse time of the halo. Only calculated after the halo reached core collapse during its run."""
        raise NotImplementedError('Core collapse time is not implemented yet')

    @property
    def t_c(self) -> Unit:
        """Estimated core collapse time of the halo. Taken from the first distribution."""
        if len(self.halo.distributions) > 0:
            return self.halo.distributions[0].units.t_c(sigma=self.halo.scatter_params['sigma'])
        return cast(Unit, units.time)
