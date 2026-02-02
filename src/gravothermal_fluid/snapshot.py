"""module handling the snapshot saving of the system"""

from typing import Any, Literal, cast, get_args
from collections import deque

import numpy as np
from numpy.typing import NDArray
from astropy.units import Quantity

from .scale import ScaleType

EdgeAlignedAttributes = Literal[
    'radius',
    'enclosed mass',
    'internal energy gradient',
    'luminosity',
]
CenterAlignedAttributes = Literal[
    'shell center',
    'velocity dispersion',
    'density',
    'shell mass',
    'pressure',
    'internal energy',
    'luminosity gradient',
]

SavedAttributes = Literal[
    'radius',
    'shell center',
    'velocity dispersion',
    'density',
    'enclosed mass',
    'shell mass',
    'pressure',
    'internal energy',
    'internal energy gradient',
    'luminosity',
    'luminosity gradient',
    'time',
]

physical_type: dict[SavedAttributes, ScaleType] = {
    'radius': 'length',
    'shell center': 'length',
    'velocity dispersion': 'velocity',
    'density': 'density',
    'enclosed mass': 'mass',
    'shell mass': 'mass',
    'pressure': 'pressure',
    'internal energy': 'internal energy',
    'internal energy gradient': 'internal energy gradient',
    'luminosity': 'luminosity',
    'luminosity gradient': 'luminosity gradient',
    'time': 'time',
}


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
    ) -> None:
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

    def __call__(self, key: SavedAttributes) -> NDArray[np.float64]:
        """Returns the snapshot for the specified attribute."""
        return getattr(self, key.replace(' ', '_'))

    def __len__(self) -> int:
        """The number of snapshots saved"""
        return len(self.radius)

    def update(self, **kwargs: Any) -> None:
        """Update the snapshot with new data."""
        for key, value in kwargs.items():
            setattr(self, f'_{key}', getattr(self, f'_{key}') + deque([np.array(value).copy()[np.newaxis, ...]]))

    def rollback(self, to_index: int) -> None:
        """Rollback the snapshot to a previous state."""
        for key in map(lambda x: '_' + x.replace(' ', '_'), self.saved_properties):
            setattr(self, key, deque(list(getattr(self, key))[:to_index]))

    @property
    def saved_properties(self) -> list[str]:
        """Total saved physical properties"""
        return list(get_args(SavedAttributes))

    @property
    def edge_aligned(self) -> list[str]:
        """Edge-aligned physical properties"""
        return list(get_args(EdgeAlignedAttributes))

    @property
    def center_aligned(self) -> list[str]:
        """Center-aligned physical properties"""
        return list(get_args(CenterAlignedAttributes))

    @staticmethod
    def physical_type(key: SavedAttributes) -> ScaleType:
        """Return the physical type of a saved attribute."""
        return physical_type[key]

    @property
    def radius(self) -> NDArray[np.float64]:
        """The radius of each shell (radius of the outer shell edge). Edge-aligned array (shape (N,))."""
        return np.vstack(self._radius)

    @property
    def shell_center(self) -> NDArray[np.float64]:
        """The radius of each shell's center. Center-aligned array (shape (N-1,))."""
        return np.vstack(self._shell_center)

    @property
    def velocity_dispersion(self) -> NDArray[np.float64]:
        """The velocity dispersion. Center-aligned array (shape (N-1,))."""
        return np.vstack(self._velocity_dispersion)

    @property
    def density(self) -> NDArray[np.float64]:
        """The therodynamical density. Center-aligned array (shape (N-1,))."""
        return np.vstack(self._density)

    @property
    def enclosed_mass(self) -> NDArray[np.float64]:
        """The mass enclosed by each shell's outer radius. Edge-aligned array (shape (N,))."""
        return np.vstack(self._enclosed_mass)

    @property
    def shell_mass(self) -> NDArray[np.float64]:
        """The mass contained within each shell. Center-aligned array (shape (N-1,))."""
        return np.vstack(self._shell_mass)

    @property
    def pressure(self) -> NDArray[np.float64]:
        """The pressure in each shell. Center-aligned array (shape (N-1,))."""
        return np.vstack(self._pressure)

    @property
    def internal_energy(self) -> NDArray[np.float64]:
        """The internal energy in each shell. Center-aligned array (shape (N-1,))."""
        return np.vstack(self._internal_energy)

    @property
    def internal_energy_gradient(self) -> NDArray[np.float64]:
        """The energy gradient in mass coordinates. Edge-aligned array (shape (N,))."""
        return np.vstack(self._internal_energy_gradient)

    @property
    def luminosity(self) -> NDArray[np.float64]:
        """The luminosity at each point. Edge-aligned array (shape (N,))."""
        return np.vstack(self._luminosity)

    @property
    def luminosity_gradient(self) -> NDArray[np.float64]:
        """The luminosity gradient in mass coordinates. Center-aligned array (shape (N-1,))."""
        return np.vstack(self._luminosity_gradient)

    @property
    def time(self) -> NDArray[np.float64]:
        """The time of each snapshot."""
        return np.hstack(self._time)

    @staticmethod
    def skip_snapshot(
        current_index: int,
        current_time: Quantity['time'],
        specific_snapshots: int | list[int] | NDArray[np.int64] | Quantity['time'] | None = None,
        undersample_snapshots: int | Quantity['time'] | None = None,
    ) -> bool:
        """Checks whether the current snapshot should be ignored, used for plotting selective snapshots.

        Parameters:
            current_index: The index of the currently considered snapshot.
            current_time: The time of the currently considered snapshot.
            specific_snapshots: Only plot the specified snapshots (if they exist). A Quantity input will filter by time, and an int (or array of ints) will filter by steps. Ignored if `None`.
            undersample_snapshots: Only plot every `undersample_snapshots` snapshots. Ignored if `None` or if `specific_snapshots` is provided.

        Returns:
            `True` if it should be skipped, `False` otherwise.
        """
        if specific_snapshots is not None:
            if isinstance(specific_snapshots, Quantity):
                specific_snapshots = cast(Quantity, specific_snapshots.copy().to(current_time))
                return current_time != specific_snapshots and current_time not in specific_snapshots
            return current_index not in np.array(specific_snapshots)
        elif undersample_snapshots is not None:
            if isinstance(undersample_snapshots, int):
                return current_index % undersample_snapshots != 0
            return current_time % undersample_snapshots != 0  # TODO - fix
        return False
