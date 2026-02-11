"""Solver management class for the gravothermal fluid ODE equations"""

from typing import Any, Literal, cast

# from functools import cached_property
import numpy as np
import seaborn as sns
from numpy.typing import NDArray
from astropy.units import Quantity
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from astropy.units.typing import UnitLike

from src import plot, utils, run_units
from src.tqdm import tqdm
from src.distribution.distribution import Distribution

# from src.distribution.distribution import PhysicalProperty as BasePhysicalProperty
from . import types, snapshot
from .scale import Scale

# PhysicalProperty = BasePhysicalProperty | Literal['luminosity', 'luminosity gradient', 'internal energy gradient']


class GravothermalFluid:
    """Solver management class for the gravothermal fluid ODE equations"""

    def __init__(
        self,
        distribution: Distribution,
        sigma: Quantity[run_units.cross_section],
        dt: Quantity['time'],
        radius: Quantity['length'] | None = None,
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
        if radius is None:
            radius = distribution.geomspace_grid
        self.distribution = distribution
        cell_center = utils.to_center(radius, method='geometric')
        velocity_dispersion = distribution.calculate_velocity_dispersion(cell_center)
        density = distribution.density(cell_center)
        enclosed_mass = distribution.enclosed_mass(radius)

        radius, velocity_dispersion, density, enclosed_mass = self.regulate_input(
            radius, velocity_dispersion, density, enclosed_mass
        )

        self.scale = Scale.from_distribution(distribution, sigma=sigma, a=a)

        density[0] = enclosed_mass[0] / (4 / 3 * np.pi * radius[0] ** 3)
        _density, _velocity_dispersion = self.scale(density), self.scale(velocity_dispersion)

        self._pressure = self.scale(self.distribution.calculate_pressure(cell_center))
        self._internal_energy = self.scale(self.distribution.calculate_internal_energy(cell_center))

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
        grad = np.diff(self.internal_energy) / ((self.shell_mass[:-1] + self.shell_mass[1:]) / 2)
        return np.hstack([(self.internal_energy[1] - self.internal_energy[0]) / (self.shell_mass[0]), grad, 0])

    @property  # @cached_property
    def luminosity(self) -> NDArray[np.float64]:
        """The luminosity at each point. Edge-aligned array (shape (N,))."""
        self.invalidate('luminosity_gradient')
        return -(self.radius**4 * utils.to_edge(self.pressure) * self.internal_energy_gradient) * utils.safe_inverse(
            utils.to_edge(self.heat_conduction) * utils.to_edge(self.velocity_dispersion)
        )

    @property  # @cached_property
    def luminosity_gradient(self) -> NDArray[np.float64]:
        """The luminosity gradient in mass coordinates. Center-aligned array (shape (N-1,))."""
        self.invalidate('dt')

        grad = np.diff(self.luminosity) / self.shell_mass
        return np.hstack([self.luminosity[0] / self.enclosed_mass[0], grad[1:]])

        # return np.diff(self.luminosity) / np.diff(self.enclosed_mass)

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
        self.pressure += self.pressure * utils.safe_inverse(self.internal_energy) * heat
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

            self.pressure = 2 / 3 * self.shell_density * self.internal_energy

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

    def evolve(self, n_steps: int | float = 1, verbose: bool = True, save_every_n_steps: int | None = None) -> None:
        """Evolve the system."""
        for step in tqdm(range(int(n_steps))):
            dt = self.transfer_heat()
            self.relax(dt=dt)
            self.time += dt
            if save_every_n_steps is None or step % save_every_n_steps == 0:
                self.save_snapshot()

    def plot(
        self,
        y: snapshot.SavedAttributes,
        x: Literal['radius', 'enclosed mass'] = 'radius',
        specific_snapshots: int | list[int] | NDArray[np.int64] | Quantity['time'] | None = None,
        undersample_snapshots: int | Quantity['time'] | None = None,
        xlabel: str | None | Literal['auto'] = 'auto',
        ylabel: str | None | Literal['auto'] = 'auto',
        x_unit: UnitLike | None | Literal['auto'] = 'auto',
        y_unit: UnitLike | None | Literal['auto'] = 'auto',
        time_unit: UnitLike = run_units.time,
        time_format: str | None = '.1f',
        xscale: plot.Scale = 'log',
        yscale: plot.Scale | None = None,
        label: str | None | Literal['auto step', 'auto time'] = 'auto',
        lineplot_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot a physical property's evolution over time.

        Parameters:
            y: Value to plot on the y-axis.
            x: Value to plot on the x-axis.
            specific_snapshots: Only plot the specified snapshots (if they exist). A Quantity input will filter by time, and an int (or array of ints) will filter by steps. Ignored if `None`.
            undersample_snapshots: Only plot every `undersample_snapshots` snapshots. Ignored if `None` or if `specific_snapshots` is provided.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            x_unit: The units of the x-axis.
            y_unit: The units of the y-axis.
            xscale: The scale of the x-axis.
            yscale: The scale of the y-axis.
            label: The label for each plot. If `auto` plots `initial` for t=0 and `{i} steps` for the i-th step.
            lineplot_kwargs: Additional keyword arguments to pass to the lineplot function (`sns.lineplot()`).
            **kwargs: Additional keyword arguments passed to `plot.setup_kwargs()`.

        Returns:
            fig, ax.
        """
        if x_unit == 'auto':
            x_unit = str(self.scale.get_unit('length') if x == 'radius' else self.scale.get_unit('mass'))
        if y_unit == 'auto':
            y_unit = self.scale.get_unit(self.snapshots.physical_type(y))
        if xlabel == 'auto':
            xlabel = x.title()
        if ylabel == 'auto':
            ylabel = y.title()
        if yscale is None:
            yscale = 'log' if y in ['density', 'enclosed mass', 'shell mass', 'radius', 'shell center'] else 'linear'

        fig, ax = plot.setup(
            xlabel=xlabel, ylabel=ylabel, x_unit=x_unit, y_unit=y_unit, xscale=xscale, yscale=yscale, **kwargs
        )

        if x == 'radius':
            x_snapshots = self.snapshots.shell_center if y in self.snapshots.center_aligned else self.snapshots.radius
        else:
            x_snapshots = (
                self.snapshots.enclosed_mass[..., 1:]
                if y in self.snapshots.center_aligned
                else self.snapshots.enclosed_mass
            )

        for i, (x_values, y_values, t) in enumerate(zip(x_snapshots, self.snapshots(y), self.snapshots('time'))):
            t = self.scale(cast(NDArray[np.float64], t), 'time')
            if self.snapshots.skip_snapshot(i, t, specific_snapshots, undersample_snapshots):
                continue
            if x_unit is not None:
                x_values = self.scale(x_values, x_unit)
            if y_unit is not None:
                y_values = self.scale(y_values, y_unit)
            if label == 'auto step':
                _label = 'initial' if i == 0 else f'{i} steps'
            elif label == 'auto time':
                t_label = f'{t.to(time_unit):{time_format}}' if time_format is not None else t.to(time_unit)
                _label = f't={t_label} time'
            else:
                _label = label
            sns.lineplot(
                x=x_values,
                y=y_values,
                ax=ax,
                label=_label,
                **lineplot_kwargs,
            )
        return fig, ax
