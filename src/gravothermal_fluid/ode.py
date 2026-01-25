"""Solver management class for the gravothermal fluid ODE equations"""

from copy import deepcopy
from typing import Any, cast
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from astropy.units import Quantity

from src import utils, run_units
from src.tqdm import tqdm
from src.distribution.distribution import Distribution

from . import types, snapshot
from .scale import Scale


class GravothermalFluid:
    """Solver management class for the gravothermal fluid ODE equations"""

    def __init__(
        self,
        radius: Quantity['length'],
        velocity_dispersion: Quantity['velocity'],
        density: Quantity['mass density'],
        enclosed_mass: Quantity['mass'],
        sigma: Quantity[run_units.cross_section],
        dt: Quantity['time'],
        scale: Scale,
        time: Quantity['time'] = Quantity(0, run_units.time),
        gamma: float = 5 / 3,
        a: float = 4 / np.sqrt(np.pi),
        b: float = 25 * np.sqrt(np.pi) / 32,
        C: float = 0.9,
        CFL: float = 0.01,
        relaxation_params: types.RelaxationParams | dict[str, Any] | None = None,
        savgol_window_length: int = 11,
        eos_factor_from_data: bool = False,
    ):
        radius, velocity_dispersion, density, enclosed_mass = self.regulate_input(
            radius, velocity_dispersion, density, enclosed_mass
        )

        self.scale = scale

        # density[0] = 3 * enclosed_mass[0] / radius[0] ** 3
        _density, _velocity_dispersion = self.scale(density), self.scale(velocity_dispersion)

        self._pressure = _density * _velocity_dispersion**2
        # self._pressure[0] = utils.to_edge(_density)[0] * utils.to_edge(_velocity_dispersion)[0] ** 2
        # _velocity_dispersion[0] = np.sqrt(
        #     (utils.to_edge(_density)[0] * utils.to_edge(_velocity_dispersion)[0] ** 2) / _density[0]
        # )
        self._internal_energy = 3 / 2 * _velocity_dispersion**2

        self.base_dt: float
        self.cross_section: float
        self._radius, self.enclosed_mass, self.base_dt, self.cross_section = map(
            self.scale, [radius, enclosed_mass, dt, sigma]
        )

        self.time = float(self.scale(time))

        self.a, self.b, self.C = a, b, C
        self.CFL = CFL
        self.gamma = gamma
        self.relaxation_params = types.normalize_params(cast(types.RelaxationParams | None, relaxation_params))
        self.savgol_window_length = savgol_window_length

        if eos_factor_from_data:
            self.eos_factor = (self.initial_pressure[0] / (self.shell_density[0] * self.initial_internal_energy[0])) / (
                2 / 3
            )
        else:
            self.eos_factor = float(self.scale.mass / (self.scale.density * self.scale.volume))

        self.snapshots = snapshot.Snapshot(
            radius=self.radius,
            shell_center=self.shell_center,
            velocity_dispersion=_velocity_dispersion,
            density=_density,
            enclosed_mass=self.enclosed_mass,
            shell_mass=self.shell_mass,
            pressure=self.pressure,
            internal_energy=self.internal_energy,
            internal_energy_gradient=self.internal_energy_gradient,
            luminosity=self.luminosity,
            luminosity_gradient=self.luminosity_gradient,
            time=self.time,
        )

    @staticmethod
    def regulate_input(
        radius: Quantity['length'],
        velocity_dispersion: Quantity['velocity'],
        density: Quantity['mass density'],
        enclosed_mass: Quantity['mass'],
        min_velocity_dispersion: Quantity['velocity'] = Quantity(1e-10, 'kpc/Myr'),
        min_density: Quantity['mass density'] = Quantity(1e-10, 'Msun/kpc^3'),
    ):
        """Regulates the input to ensure the mass is monotonically increasing, and that velocity dispersion and density are not zero."""
        edge_mask = np.gradient(enclosed_mass) > 0
        center_mask = edge_mask[:-1] & edge_mask[1:]
        radius, enclosed_mass = map(lambda x: cast(Quantity, x[edge_mask]), [radius, enclosed_mass])
        velocity_dispersion, density = map(lambda x: cast(Quantity, x[center_mask]), [velocity_dispersion, density])
        velocity_dispersion[np.isnan(velocity_dispersion)] = min_velocity_dispersion
        density = cast(Quantity, density.clip(min=min_density))
        return radius, velocity_dispersion, density, enclosed_mass

    @classmethod
    def from_distribution(
        cls,
        distribution: Distribution,
        sigma: Quantity[run_units.cross_section],
        dt: Quantity['time'],
        a: float = 4 / np.sqrt(np.pi),
        radius: Quantity['length'] | None = None,
        **kwargs: Any,
    ):
        """Create an object from a given distribution."""
        if radius is None:
            radius = distribution.geomspace_grid
        cell_center = utils.to_center(radius, method='geometric')
        return cls(
            radius=radius,
            velocity_dispersion=distribution.calculate_velocity_dispersion(cell_center),
            density=distribution.rho(cell_center),
            enclosed_mass=distribution.M(radius),
            sigma=sigma,
            dt=dt,
            scale=Scale.from_distribution(distribution, sigma=sigma, a=a),
            a=a,
            **kwargs,
        )

    def invalidate(self, *properties: str):
        """Invalidates the cache for the specified properties, so they are recalculated."""
        for property in properties:
            if property == 'density':
                continue
            try:
                delattr(self, property)
            except AttributeError:
                pass

    def save_snapshot(self) -> None:
        """Saves a snapshot of the current state of the system."""
        self.snapshots.update(
            radius=self.radius,
            shell_center=self.shell_center,
            velocity_dispersion=self.velocity_dispersion,
            density=self.density,
            enclosed_mass=self.enclosed_mass,
            shell_mass=self.shell_mass,
            pressure=self.pressure,
            internal_energy=self.internal_energy,
            internal_energy_gradient=self.internal_energy_gradient,
            luminosity=self.luminosity,
            luminosity_gradient=self.luminosity_gradient,
            time=self.time,
        )

    def rollback(self, to_index: int) -> None:
        """Rollback the system to a previous state."""
        self.snapshots.rollback(to_index)
        self.pressure = self.snapshots.pressure[-1]
        self.internal_energy = self.snapshots.internal_energy[-1]
        self.radius = self.snapshots.radius[-1]
        self.enclosed_mass = self.snapshots.enclosed_mass[-1]
        self.time = self.snapshots.time[-1]

    def reset(self) -> None:
        """Resets the system to its initial state."""
        self.rollback(to_index=1)

    @property
    def radius(self) -> NDArray[np.float64]:
        """The radius of each shell (radius of the outer shell edge). Edge-aligned array (shape (N,))."""
        return self._radius

    @radius.setter
    def radius(self, value: NDArray[np.float64]):
        self._radius = value
        self.invalidate(
            'shell_center',
            'shell_width',
            'shell_volume',
            'shell_density',
            'luminosity',
            'luminosity_gradient',
            'dt',
        )

    @property
    def internal_energy(self) -> NDArray[np.float64]:
        """The internal energy in each shell. Center-aligned array (shape (N-1,))."""
        return self._internal_energy

    @internal_energy.setter
    def internal_energy(self, value: NDArray[np.float64]):
        self._internal_energy = value
        self.invalidate(
            'internal_energy_gradient', 'velocity_dispersion', 'density', 'luminosity', 'dt', 'luminosity_gradient'
        )

    @property
    def pressure(self) -> NDArray[np.float64]:
        """The pressure in each shell. Center-aligned array (shape (N-1,))."""
        return self._pressure

    @pressure.setter
    def pressure(self, value: NDArray[np.float64]):
        self._pressure = value
        self.invalidate('heat_conduction', 'density', 'luminosity', 'luminosity_gradient', 'dt')

    @property  # @cached_property
    def density(self) -> NDArray[np.float64]:
        """Calculate the density from the energy and pressure. Center-aligned array (shape (N-1,))."""
        return 3 / 2 * self.pressure * utils.safe_inverse(self.internal_energy)

    @property  # @cached_property
    def velocity_dispersion(self) -> NDArray[np.float64]:
        """Calculate the velocity dispersion from the energy. Center-aligned array (shape (N-1,))."""
        return np.sqrt(2 / 3 * np.abs(self.internal_energy))

    @property  # @cached_property
    def heat_conduction(self) -> NDArray[np.float64]:
        """The heat conduction (kappa) at each point. Center-aligned array (shape (N-1,))."""
        return self.a / self.b * self.cross_section + utils.safe_inverse(self.C * self.pressure)

    @property  # @cached_property
    def internal_energy_gradient(self) -> NDArray[np.float64]:
        """The energy gradient in mass coordinates. Edge-aligned array (shape (N,))."""
        gradient = utils.to_edge(np.diff(utils.to_edge(self.internal_energy)) / np.diff(self.enclosed_mass))
        gradient[0] = 0
        return gradient

        # return utils.to_center(
        #     edges=utils.differentiate_savgol(
        #         x=np.pad(self.enclosed_mass[1:], (1, 1), mode='edge'),
        #         y=np.pad(self.internal_energy, (1, 1), mode='edge'),
        #         window_length=self.savgol_window_length,
        #     ),
        #     method='algebric',
        # )

        # gradient = utils.differentiate_savgol(
        #     x=np.pad(self.enclosed_mass[1:], (1, 1), mode='edge'),
        #     y=np.pad(self.internal_energy, (1, 1), mode='edge'),
        #     window_length=self.savgol_window_length,
        # )

        # # Linear interpolate the inner core gradient to avoid savgol filter boundary errors
        # gradient[: self.savgol_window_length] = (
        #     (gradient[self.savgol_window_length + 1] - gradient[self.savgol_window_length])
        #     / (self.enclosed_mass[self.savgol_window_length + 1] - self.enclosed_mass[self.savgol_window_length])
        # ) * (
        #     self.enclosed_mass[: self.savgol_window_length] - self.enclosed_mass[self.savgol_window_length]
        # ) + gradient[self.savgol_window_length]

        # return utils.to_center(
        #     edges=gradient,
        #     method='algebric',
        # )

    @property  # @cached_property
    def luminosity(self) -> NDArray[np.float64]:
        """The luminosity at each point. Edge-aligned array (shape (N,))."""
        self.invalidate('luminosity_gradient')
        luminosity = -(
            self.radius**4 * utils.to_edge(self.pressure) * self.internal_energy_gradient
        ) * utils.safe_inverse(utils.to_edge(self.heat_conduction) * utils.to_edge(self.velocity_dispersion))
        # luminosity[:11] = 0  # No luminosity in the first cell
        luminosity[0] = 0
        return luminosity

    @property  # @cached_property
    def luminosity_gradient(self) -> NDArray[np.float64]:
        """The luminosity gradient in mass coordinates. Center-aligned array (shape (N-1,))."""
        self.invalidate('dt')
        gradient = np.diff(self.luminosity) / np.diff(self.enclosed_mass)
        gradient[0] = 0
        gradient[1] = 0
        # gradient = np.diff(self.luminosity) / np.diff(self.radius)
        # gradient[(np.abs(gradient) < 1e-5)] = 0
        # gradient /= (self.shell_center**2) * utils.safe_inverse(self.density)
        return gradient
        # gradient = utils.differentiate_savgol(
        #     x=self.enclosed_mass, y=self.luminosity, window_length=self.savgol_window_length
        # )

        # # Linear interpolate the inner core gradient to avoid savgol filter boundary errors
        # gradient[: self.savgol_window_length] = (
        #     (gradient[self.savgol_window_length + 1] - gradient[self.savgol_window_length])
        #     / (self.enclosed_mass[self.savgol_window_length + 1] - self.enclosed_mass[self.savgol_window_length])
        # ) * (
        #     self.enclosed_mass[: self.savgol_window_length] - self.enclosed_mass[self.savgol_window_length]
        # ) + gradient[self.savgol_window_length]

        # # finite_gradient = np.gradient(self.luminosity, self.enclosed_mass)

        # # n_blend = min(2 * self.savgol_window_length, len(gradient))
        # # weights = np.linspace(1, 0, n_blend)
        # # gradient[:n_blend] = (weights * finite_gradient[:n_blend]) + ((1 - weights) * gradient[:n_blend])

        # return utils.to_center(gradient, method='algebric')

    @property  # @cached_property
    def dt(self) -> float:
        """Calculate the size of the time step `dt` that conforms with the CFL condition."""
        return min(
            self.base_dt,
            self.CFL
            * np.min(np.abs(self.internal_energy * utils.safe_inverse(self.luminosity_gradient, fill_value=np.nan))),
        )

    def transfer_heat(self) -> float:
        """Transfer heat in the system. Returns the time step used (to account for the CFL condition)."""
        dt = self.dt.copy()
        heat = -self.luminosity_gradient * dt
        self.internal_energy += heat
        self.pressure[:-1] += (self.pressure * utils.safe_inverse(self.internal_energy) * heat)[:-1]
        return dt

    @property  # @cached_property
    def shell_center(self) -> NDArray[np.float64]:
        """The radius of each shell's center. Center-aligned array (shape (N-1,))."""
        return utils.to_center(self.radius, method='geometric')

    @property  # @cached_property
    def shell_width(self) -> NDArray[np.float64]:
        """The width the shells. Center-aligned array (shape (N-1,))."""
        return np.diff(self.radius)

    @property  # @cached_property
    def shell_volume(self) -> NDArray[np.float64]:
        """The volume of the shells. Center-aligned array (shape (N-1,))."""
        return (4 / 3) * np.pi * np.diff(self.radius**3)

    @property  # @cached_property
    def shell_mass(self) -> NDArray[np.float64]:
        """The mass contained within each shell. Center-aligned array (shape (N-1,))."""
        return np.diff(self.enclosed_mass)

    @property  # @cached_property
    def shell_density(self) -> NDArray[np.float64]:
        """The mass density of each shell (contained mass / shell volume). Corrected by `eos_factor` to account for the parameter scale and line up with the thermodynamical calculation. Center-aligned array (shape (N-1,))."""
        return self.shell_mass / self.shell_volume * self.eos_factor

    @property  # @cached_property
    def n_shells(self) -> int:
        """The number of shells in the simulation (number of grid points)"""
        return len(self.radius)

    @property
    def gravitational_force(self) -> NDArray[np.float64]:
        """The gravitational force applied to each shell. Edge-aligned array without the first and final cells (shape (N-2,))."""
        return -(self.enclosed_mass[1:-1] / (self.radius[1:-1] ** 2))

    @property
    def pressure_force(self) -> NDArray[np.float64]:
        """The pressure force applied to each shell. Edge-aligned array without the first and final cells (shape (N-2,))."""
        return -utils.safe_inverse(np.sqrt(self.shell_density[:-1] * self.shell_density[1:])) * (
            np.diff(self.pressure) / np.diff(self.shell_center)
        )
        # return -utils.safe_inverse(utils.to_edge(self.shell_density)[1:-1]) * (
        #     np.diff(self.pressure) / np.diff(self.shell_center)
        # )

    @property
    def net_force(self) -> NDArray[np.float64]:
        """The net hydrostatic force applied to each shell. Edge-aligned array without the first and final cells (shape (N-2,))."""
        return self.pressure_force + self.gravitational_force

    @property
    def force_ratio(self) -> NDArray[np.float64]:
        """The hydrostatic force ratio applied to each shell (pressure/gravitation). Edge-aligned array without the first and final cells (shape (N-2,))."""
        return self.pressure_force / self.gravitational_force

    def relax(self, dt: float, verbose: bool = False) -> None:
        """
        Adjusts the radius of shells to move the system toward HSE
        while keeping the newly calculated internal energy (u) fixed.
        """
        # print(f'Pre-Relax R[0:3]: {self.radius[:3]}')
        # print(f'Pre-Relax Vol[0:3]: {self.shell_volume[:3]}')
        # print(f'Pre-Relax GeoDensity[0:3]: {self.shell_density[:3]}')
        # print(f'Pre-Relax InternalEnergy[0:3]: {self.internal_energy[:3]}')
        # print(f'Pre-Relax Pressure[0:3]: {self.pressure[:3]}')

        for iteration in tqdm(
            range(self.relaxation_params['max_relaxation_iterations']), desc='Relaxing back to HSE', disable=not verbose
        ):
            r_old = self.radius.copy()
            self.radius[1:-1] += (
                self.radius[1:-1]
                * utils.safe_log(
                    np.abs(self.force_ratio).clip(
                        min=1 / self.relaxation_params['driving_force_limit'],
                        max=self.relaxation_params['driving_force_limit'],
                    )
                )
                * self.dt
                * self.relaxation_params['relaxation_dt_factor']
            )

            self.pressure[:-1] = (2 / 3 * self.shell_density[:] * self.internal_energy)[:-1]

            if np.max(
                (np.abs(r_old - self.radius) / r_old)[
                    : int(self.n_shells * self.relaxation_params['relaxation_core_fraction'])
                ]
                < self.relaxation_params['relaxation_threshold']
            ):
                if verbose:
                    print(f'Finished early after {iteration + 1} iterations')
                break
        # print(f'Post-Relax R[0:3]: {self.radius[:3]}')
        # print(f'Post-Relax Vol[0:3]: {self.shell_volume[:3]}')
        # print(f'Post-Relax GeoDensity[0:3]: {self.shell_density[:3]}')
        # print(f'Post-Relax Pressure[0:3]: {self.pressure[:3]}')

    def evolve(self, n_steps: int | float = 1, verbose: bool = True) -> None:
        """Evolve the system."""
        for _ in tqdm(range(int(n_steps))):
            dt = self.transfer_heat()
            self.relax(dt=dt)
            self.time += dt
            self.save_snapshot()
