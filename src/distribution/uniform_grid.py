from typing import Any, cast

import numpy as np
from astropy.units import Quantity

from .. import utils
from .distribution import Distribution


class UniformGrid(Distribution):
    """Uniform grid dummy density profile.

    NOT MEANT FOR PHYSICAL ANALYSIS, AND NOT GRAVITATIONALLY STABLE.
    This is a dummy density profile for tracking particles to create a map from the initial conditions to the collapse phase.
    """

    def __init__(
        self,
        rmin: Quantity['length'] = Quantity(1e-2, 'kpc'),
        rmax: Quantity['length'] = Quantity(1e2, 'kpc'),
        vmin: Quantity['velocity'] = Quantity(0, 'km/s'),
        vmax: Quantity['velocity'] = Quantity(80, 'km/s'),
        r_resolution: int = 100,
        v_resolution: int = 100,
        Rs: Quantity['length'] | None = Quantity(1, 'kpc'),
        c: float | None = 19,
        Mtot: Quantity['mass'] = Quantity(1e-5, 'Msun'),
        **kwargs: Any,
    ) -> None:
        super().__init__(Rs=Rs, c=c, Mtot=Mtot, **kwargs)
        self.title = 'Uniform grid'
        self.rmin, self.rmax, self.vmin, self.vmax = rmin, rmax, vmin, vmax
        self.r_resolution, self.v_resolution = r_resolution, v_resolution

    @property
    def physical(self) -> bool:
        """Return whether the profile is physical."""
        return False

    def sample(
        self,
        n_particles: int | float,
        radius_min_value: Quantity['length'] | None = None,
        radius_max_value: Quantity['length'] | None = None,
        velocity_min_value: Quantity['velocity'] = Quantity(0, 'km/second'),
        velocity_max_value: Quantity['velocity'] = Quantity(100, 'km/second'),
        radius_resolution: int | float = 10000,
        velocity_resolution: int | float = 10000,
        radius_range: Quantity['length'] | None = None,
        velocity_range: Quantity['velocity'] | None = None,
        radius_noise: float = 1,
        velocity_noise: float = 1,
        generator: np.random.Generator | None = None,
    ) -> tuple[Quantity['length'], Quantity['velocity']]:
        """Samples particles from a uniform grid. All parameters are ignored."""
        r, v = np.meshgrid(
            np.geomspace(self.rmin, self.rmax, self.r_resolution),
            np.linspace(self.vmin, self.vmax, self.v_resolution),
        )
        return cast(Quantity, r.ravel()), cast(Quantity, np.vstack(utils.split_3d(v.ravel(), generator=generator)).T)
