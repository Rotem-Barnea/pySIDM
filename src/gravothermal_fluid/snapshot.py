"""module handling the snapshot saving of the system"""

from typing import Any
from collections import deque

import numpy as np
from numpy.typing import NDArray

saved_attributes = [
    'radius',
    'shell_center',
    'velocity_dispersion',
    'density',
    'enclosed_mass',
    'shell_mass',
    'pressure',
    'internal_energy',
    'internal_energy_gradient',
    'luminosity',
    'luminosity_gradient',
    'time',
]


class Snapshot:
    """Snapshot of the state of the system, saved over time."""

    def __init__(
        self,
        radius: NDArray[np.float64],
        shell_center: NDArray[np.float64],
        velocity_dispersion: NDArray[np.float64],
        density: NDArray[np.float64],
        enclosed_mass: NDArray[np.float64],
        shell_mass: NDArray[np.float64],
        pressure: NDArray[np.float64],
        internal_energy: NDArray[np.float64],
        internal_energy_gradient: NDArray[np.float64],
        luminosity: NDArray[np.float64],
        luminosity_gradient: NDArray[np.float64],
        time: float,
    ):
        self._radius: deque[NDArray[np.float64]] = deque()
        self._shell_center: deque[NDArray[np.float64]] = deque()
        self._velocity_dispersion: deque[NDArray[np.float64]] = deque()
        self._density: deque[NDArray[np.float64]] = deque()
        self._enclosed_mass: deque[NDArray[np.float64]] = deque()
        self._shell_mass: deque[NDArray[np.float64]] = deque()
        self._pressure: deque[NDArray[np.float64]] = deque()
        self._internal_energy: deque[NDArray[np.float64]] = deque()
        self._internal_energy_gradient: deque[NDArray[np.float64]] = deque()
        self._luminosity: deque[NDArray[np.float64]] = deque()
        self._luminosity_gradient: deque[NDArray[np.float64]] = deque()
        self._time: deque[NDArray[np.float64]] = deque()

        self.update(
            radius=radius,
            shell_center=shell_center,
            velocity_dispersion=velocity_dispersion,
            density=density,
            enclosed_mass=enclosed_mass,
            shell_mass=shell_mass,
            pressure=pressure,
            internal_energy=internal_energy,
            internal_energy_gradient=internal_energy_gradient,
            luminosity=luminosity,
            luminosity_gradient=luminosity_gradient,
            time=time,
        )

    def update(self, **kwargs: Any) -> None:
        """Update the snapshot with new data."""
        for key, value in kwargs.items():
            setattr(self, f'_{key}', getattr(self, f'_{key}') + deque([np.array(value).copy()[np.newaxis, ...]]))

    def rollback(self, to_index: int) -> None:
        """Rollback the snapshot to a previous state."""
        for key in saved_attributes:
            setattr(self, f'_{key}', deque(list(getattr(self, f'_{key}'))[:to_index]))

    @property
    def radius(self):
        """The radius of each shell (radius of the outer shell edge). Edge-aligned array (shape (N,))."""
        return np.vstack(self._radius)

    @property
    def shell_center(self):
        """The radius of each shell's center. Center-aligned array (shape (N-1,))."""
        return np.vstack(self._shell_center)

    @property
    def velocity_dispersion(self):
        """The velocity dispersion. Center-aligned array (shape (N-1,))."""
        return np.vstack(self._velocity_dispersion)

    @property
    def density(self):
        """The therodynamical density. Center-aligned array (shape (N-1,))."""
        return np.vstack(self._density)

    @property
    def enclosed_mass(self):
        """The mass enclosed by each shell's outer radius. Edge-aligned array (shape (N,))."""
        return np.vstack(self._enclosed_mass)

    @property
    def shell_mass(self):
        """The mass contained within each shell. Center-aligned array (shape (N-1,))."""
        return np.vstack(self._shell_mass)

    @property
    def pressure(self):
        """The pressure in each shell. Center-aligned array (shape (N-1,))."""
        return np.vstack(self._pressure)

    @property
    def internal_energy(self):
        """The internal energy in each shell. Center-aligned array (shape (N-1,))."""
        return np.vstack(self._internal_energy)

    @property
    def internal_energy_gradient(self):
        """The energy gradient in mass coordinates. Edge-aligned array (shape (N,))."""
        return np.vstack(self._internal_energy_gradient)

    @property
    def luminosity(self):
        """The luminosity at each point. Edge-aligned array (shape (N,))."""
        return np.vstack(self._luminosity)

    @property
    def luminosity_gradient(self):
        """The luminosity gradient in mass coordinates. Center-aligned array (shape (N-1,))."""
        return np.vstack(self._luminosity_gradient)

    @property
    def time(self):
        """The time of each snapshot."""
        return np.hstack(self._time)
