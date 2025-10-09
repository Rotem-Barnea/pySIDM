import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
import pickle
import scipy
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy.typing import NDArray
from typing import Any, Self, Callable, cast, Literal
from astropy import table
from astropy.units import Quantity, Unit, def_unit
from astropy.units.typing import UnitLike
from .spatial_approximation import Lattice
from .density.density import Density
from .background import Mass_Distribution
from . import utils, run_units, physics, plot
from .physics import sidm, leapfrog
from .types import ParticleType


class Halo:
    """Halo class for SIDM simulations"""

    def __init__(
        self,
        dt: Quantity['time'],
        r: Quantity['length'],
        v: Quantity['velocity'],
        m: Quantity['mass'],
        particle_type: list[ParticleType] | NDArray[np.str_] | None = None,
        Tdyn: Quantity['time'] | None = None,
        Phi0: Quantity['energy'] | None = None,
        densities: list[Density] = [],
        n_interactions: int = 0,
        scatter_rounds: list[int] = [],
        interactions_track: list[NDArray[np.float64]] = [],
        time: Quantity['time'] = 0 * run_units.time,
        background: Mass_Distribution | None = None,
        last_saved_time: Quantity['time'] = 0 * run_units.time,
        save_every_time: Quantity['time'] | None = None,
        save_every_n_steps: int | None = None,
        dynamics_params: leapfrog.Params = {},
        scatter_params: sidm.Params = {},
        snapshots: table.QTable = table.QTable(),
        scatter_every_n_steps: int = 1,
        hard_save: str | None = 'halo state',
    ) -> None:
        """Initialize a Halo object.

        Parameters:
            r: Radius of the halo particles.
            v: Velocity of the halo particles, of shape (n_particles, 3), (vx,vy,vr) with vx,vy the two perpendicular components of the off-radial plane.
            m: Mass of the halo particles.
            particle_type: Type of the halo particles. Should comply with ParticleType (i.e. 'dm'/'baryon').
            Tdyn: Dynamical time of the halo. If None, calculates from the first density.
            Phi0: Potential at infinity of the halo. If None, calculates from the first density.
            densities: List of densities of the halo.
            n_interactions: Number of interactions the halo had.
            scatter_rounds: Number of scatter rounds the halo had every time step.
            interactions_track: The interacting particles (particle index) at every time step.
            time: Time of the halo.
            background: Background mass distribution of the halo.
            last_saved_time: Last time a snapshot was saved.
            save_every_time: How often should a snapshot be saved, in time units.
            save_every_n_steps: How often should a snapshot be saved, in time-step units (integer).
            dynamics_params: Dynamics parameters of the halo, sent to the leapfrog integrator.
            scatter_params: Scatter parameters of the halo, used in the SIDM calculation.
            snapshots: Snapshots of the halo.
            scatter_every_n_steps: How often should a scattering event be conducted, in time-step units (integer).
            hard_save: Whether to save the halo to db at every snapshot save, or just keep in RAM.

        Returns:
            Halo object.
        """
        self._particles = self.to_dataframe(r, v, m, particle_type)
        self._particles.sort_values('r', inplace=True)
        self.time: Quantity['time'] = time.to(run_units.time)
        self.dt: Quantity['time'] = dt.to(run_units.time)
        self.densities: list[Density] = densities
        self.Tdyn: Quantity['time']
        if Tdyn is not None:
            self.Tdyn = Tdyn
        elif len(self.densities) > 0:
            self.Tdyn = self.densities[0].Tdyn
        elif len(self.densities) == 0:
            self.Tdyn = Quantity(1, run_units.time)
        self.background: Mass_Distribution | None = background
        self.Phi0: Quantity['energy'] = Phi0 if Phi0 is not None else physics.utils.Phi(self.r, self.M, self.m)[-1]
        self.n_interactions = n_interactions
        self.snapshots: table.QTable = snapshots
        self.save_every_n_steps = save_every_n_steps
        self.save_every_time: Quantity['time'] | None = save_every_time if save_every_time is None else save_every_time.to(run_units.time)
        self._dynamics_params: leapfrog.Params = leapfrog.normalize_params(dynamics_params, add_defaults=True)
        self._scatter_params: sidm.Params = sidm.normalize_params(scatter_params, add_defaults=True)
        self.interactions_track = interactions_track
        self._initial_particles = self._particles.copy()
        self.initial_particles = self.particles.copy()
        self.last_saved_time = last_saved_time
        self.scatter_rounds = scatter_rounds
        self.scatter_every_n_steps: int = scatter_every_n_steps
        self.hard_save: str | None = hard_save

    @classmethod
    def setup(cls, densities: list[Density], particle_types: list[ParticleType], n_particles: list[int | float], **kwargs: Any) -> Self:
        """Initialize a Halo object from a given set of densities.

        Parameters:
            densities: List of densities for each particle type.
            particle_types: List of particle types.
            n_particles: List of number of particles for each particle type.
            kwargs: Additional keyword arguments, passed to the constructor.

        Returns:
            Halo object.
        """
        r, v, particle_type, m = [], [], [], []
        for density, p_type, n in zip(densities, particle_types, n_particles):
            r_sub = density.roll_r(int(n)).to(run_units.length)
            v_sub = density.roll_v_3d(r_sub).to(run_units.velocity)
            r += [r_sub]
            v += [v_sub]
            particle_type += [[p_type] * int(n)]
            m += [[1] * int(n) * density.unit_mass]

        return cls(
            r=cast(Quantity['length'], np.hstack(r)),
            v=cast(Quantity['length'], np.vstack(v)),
            m=cast(Quantity['length'], np.hstack(m)),
            particle_type=np.hstack(particle_type),
            densities=densities,
            **kwargs,
        )

    @staticmethod
    def to_dataframe(
        r: Quantity['length'],
        v: Quantity['velocity'],
        m: Quantity['mass'],
        particle_type: list[ParticleType] | NDArray[np.str_] | None = None,
        particle_index: NDArray[np.int_] | None = None,
    ) -> pd.DataFrame:
        """Convert particle data to a DataFrame."""
        vx, vy, vr = v.to(run_units.velocity).T
        data = pd.DataFrame(
            {
                'r': r.to(run_units.length),
                'vx': vx,
                'vy': vy,
                'vr': vr,
                'm': m.to(run_units.mass),
                'particle_type': particle_type if particle_type is not None else np.full(len(r), 'dm'),
                'particle_index': particle_index if particle_index is not None else np.arange(len(r)),
            }
        )
        data.set_index('particle_index', inplace=True)
        return data

    def add_background(self, background: Mass_Distribution) -> None:
        """Adds a background mass distribution to the halo."""
        self.background = background

    def reset(self) -> None:
        """Resets the halo to its initial state (no interactions, time=0, cleared snapshots, particles at initial positions)."""
        self.time = 0 * run_units.time
        self.n_interactions = 0
        self._particles = self._initial_particles.copy()
        self.interactions_track = []
        self.snapshots = table.QTable()

    @property
    def particles(self) -> table.QTable:
        """Particle data QTable.

        Has the following columns:
            r: Radius.
            vx: The first pernpendicular component (to the radial direction) of the velocity.
            vy: The second pernpendicular component (to the radial direction) of the velocity.
            vr: The radial velocity.
            vp: Tangential velocity (np.sqrt(vx**2 + vy**2)).
            m: Mass.
            v_norm: Velocity norm (np.sqrt(vx**2 + vy**2 + vr**2)).
            time: Current simulation time.
            E: Relative energy (Psi-1/2*m*v_norm**2).
            particle_type: Type of particle.
            particle_index: Index of particle.
        """
        self._particles.sort_values('r', inplace=True)
        data = table.QTable(
            {
                'r': self.r,
                'vx': self.vx,
                'vy': self.vy,
                'vr': self.vr,
                'vp': self.vp,
                'm': self.m,
                'v_norm': self.v_norm,
                'time': [self.time] * len(self.r),
                'E': self.E,
                'particle_type': self._particles['particle_type'],
                'particle_index': self._particles.index,
            }
        )
        return data

    @property
    def dynamics_params(self) -> leapfrog.Params:
        """Dynamics parameters of the halo, sent to the leapfrog integrator."""
        return self._dynamics_params

    @dynamics_params.setter
    def dynamics_params(self, value: leapfrog.Params) -> None:
        """Normalize and set the dynamics parameters of the halo."""
        self._dynamics_params = leapfrog.normalize_params(value)

    @property
    def scatter_params(self) -> sidm.Params:
        """Scatter parameters of the halo, used in the SIDM calculation."""
        return self._scatter_params

    @scatter_params.setter
    def scatter_params(self, value: sidm.Params) -> None:
        """Normalize and set the scatter parameters of the halo."""
        self._scatter_params = sidm.normalize_params(value)

    #####################
    ##Physical properties
    #####################

    @property
    def r(self) -> Quantity['length']:
        """Particle radius."""
        return Quantity(self._particles['r'], run_units.length)

    @property
    def vx(self) -> Quantity['velocity']:
        """The first pernpendicular component (to the radial direction) of the particle velocity."""
        return Quantity(self._particles['vx'], run_units.velocity)

    @property
    def vy(self) -> Quantity['velocity']:
        """The second pernpendicular component (to the radial direction) of the particle velocity."""
        return Quantity(self._particles['vy'], run_units.velocity)

    @property
    def vr(self) -> Quantity['velocity']:
        """The radial component of the particle velocity."""
        return Quantity(self._particles['vr'], run_units.velocity)

    @property
    def v(self) -> Quantity['velocity']:
        """The velocity of the particle, as a 3-vector (vx, vy, vr)."""
        return Quantity(self._particles[['vx', 'vy', 'vr']], run_units.velocity)

    @property
    def time_step(self) -> Unit:
        """Calculate the time step size, returning it as a Unit object"""
        return def_unit('time step', self.dt.to(run_units.time), format={'latex': r'time\ step'})

    @property
    def M(self) -> Quantity['mass']:
        """The enclosed mass below the particle."""
        halo_mass = physics.utils.M(r=self.r, m=self.m)
        if self.background is not None:
            background_mass = self.background.M_at_time(self.r, self.time)
            return cast(Quantity['mass'], halo_mass + background_mass)
        return halo_mass

    @property
    def vp(self) -> Quantity['velocity']:
        """The tangential velocity of the particle."""
        return utils.fast_quantity_norm(cast(Quantity['velocity'], self.v[:, :2]))

    @property
    def v_norm(self) -> Quantity['velocity']:
        """The velocity norm of the particle."""
        return utils.fast_quantity_norm(self.v)

    @property
    def m(self) -> Quantity['mass']:
        """The mass of the particle."""
        return Quantity(self._particles['m'], run_units.mass)

    @property
    def kinetic_energy(self) -> Quantity['energy']:
        """The kinetic energy of the particle."""
        return 0.5 * self.m * self.v_norm**2

    @property
    def Phi(self) -> Quantity['energy']:
        """The gravitational potential energy of the particle."""
        return cast(Quantity['energy'], physics.utils.Phi(self.r, self.M, self.m))

    @property
    def Psi(self) -> Quantity['specific energy']:
        """The relative gravitational potential energy of the particle."""
        return cast(Quantity['energy'], physics.utils.Psi(self.r, self.M, self.m)).to(run_units.energy)
        # return (self.Phi0 - self.Phi).to(run_units.energy)

    @property
    def E(self) -> Quantity['specific energy']:
        """The energy of the particle."""
        return (self.Psi - self.kinetic_energy).to(run_units.energy)

    @property
    def local_density(self) -> Quantity['mass density']:
        """The local mass density around the particle."""
        return cast(
            Quantity['mass density'],
            physics.utils.local_density(
                self.r,
                self.m,
                self.scatter_params.get('max_radius_j', sidm.default_params.get('max_radius_j', 10)),
            ),
        )

    @property
    def dt_scatter(self) -> Quantity['time']:
        """The time step for scattering."""
        return self.scatter_every_n_steps * self.dt

    #####################
    ##Dynamic evolution
    #####################

    def to_step(self, time: Quantity['time']) -> int:
        """Calculate the number of steps required to reach the given time."""
        return int(time / self.dt)

    @property
    def current_step(self) -> int:
        """The current simulation step count (calculated based on the simulation time)."""
        return self.to_step(self.time)

    def save_snapshot(self) -> None:
        """Save the current state of the simulation."""
        data = self.particles.copy()
        data['step'] = self.current_step
        self.snapshots = table.vstack([self.snapshots, data])
        self.last_saved_time = self.time.copy()
        if self.hard_save is not None:
            self.save(self.hard_save)

    def is_save_round(self) -> bool:
        """Check if it's time to save the simulation state."""
        if self.save_every_time is not None:
            next_save_time = self.last_saved_time + self.save_every_time
            if self.time <= next_save_time and self.time + self.dt > next_save_time:
                return True
        elif self.save_every_n_steps is not None and self.current_step % self.save_every_n_steps == 0:
            return True
        return False

    def step(self) -> None:
        """Perform a single time step of the simulation.

        Every step:
            - Sort particles by radius.
            - Save a snapshot if it's time.
            - Perform scattering if necessary. This is done before the leapfrog integration since it doesn't modify the particle positions and thus doesn't require resorting.
            - Perform leapfrog integration.
            - Update simulation time.
        """
        self._particles.sort_values('r', inplace=True)
        if self.is_save_round():
            self.save_snapshot()
        if self.scatter_params.get('sigma', 0) > 0 and self.current_step % self.scatter_every_n_steps == 0:
            mask = self._particles['particle_type'] == 'dm'
            (
                self._particles.loc[mask, 'vx'],
                self._particles.loc[mask, 'vy'],
                self._particles.loc[mask, 'vr'],
                n_interactions,
                indices,
                scatter_rounds,
            ) = sidm.scatter(
                r=self._particles.loc[mask, 'r'],
                vx=self._particles.loc[mask, 'vx'],
                vy=self._particles.loc[mask, 'vy'],
                vr=self._particles.loc[mask, 'vr'],
                dt=self.dt_scatter,
                m=self._particles.loc[mask, 'm'],
                **self.scatter_params,
            )
            self.n_interactions += n_interactions
            self.interactions_track += [self.r[indices]]
            self.scatter_rounds += [scatter_rounds]
        self._particles['r'], self._particles['vx'], self._particles['vy'], self._particles['vr'] = leapfrog.step(
            r=self._particles['r'],
            vx=self._particles['vx'],
            vy=self._particles['vy'],
            vr=self._particles['vr'],
            m=self._particles['m'],
            M=self.M,
            dt=self.dt,
            **self.dynamics_params,
        )
        self.time += self.dt

    def evolve(self, n_steps: int | None = None, t: Quantity['time'] | None = None, disable_tqdm: bool = False) -> None:
        """Evolve the simulation for a given number of steps or time."""
        if n_steps is None:
            if t is not None:
                n_steps = self.to_step(t)
            else:
                raise ValueError('Either n_steps or t must be specified')
        for _ in tqdm(range(int(n_steps)), disable=disable_tqdm):
            self.step()

    #####################
    ##Save/Load
    #####################

    def to_payload(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Convert the simulation state to a payload for saving."""
        payload = {
            'time': self.time,
            'dt': self.dt,
            'densities': self.densities,
            'n_interactions': self.n_interactions,
            'save_every_n_steps': self.save_every_n_steps,
            'save_every_time': self.save_every_time,
            'dynamics_params': self.dynamics_params,
            'scatter_params': self.scatter_params,
            'interactions_track': self.interactions_track,
            'background': self.background,
            'last_saved_time': self.last_saved_time,
            'scatter_rounds': self.scatter_rounds,
            'scatter_every_n_steps': self.scatter_every_n_steps,
            'hard_save': self.hard_save,
        }
        tables = {
            'particles': self.particles,
            'initial_particles': self.initial_particles,
            'snapshots': self.snapshots,
        }
        return payload, tables

    def save(self, path: str | Path = Path('halo_state')) -> None:
        """Save the simulation state to a directory."""
        path = Path(path)
        path.mkdir(exist_ok=True)
        payload, tables = self.to_payload()
        with open(path / 'halo_payload.pkl', 'wb') as f:
            pickle.dump(payload, f)
        for name, data in tables.items():
            data[[column for column in data.colnames if data[column].dtype != np.dtype('O')]].write(path / f'{name}.fits', overwrite=True)
            data[[column for column in data.colnames if data[column].dtype == np.dtype('O')]].write(path / f'{name}_strings.csv', overwrite=True)

    @classmethod
    def load(cls, path: str | Path = Path('halo_state')) -> Self:
        """Load the simulation state from a directory."""
        path = Path(path)
        with open(path / 'halo_payload.pkl', 'rb') as f:
            payload = pickle.load(f)
        tables = {}
        for name in ['particles', 'initial_particles', 'snapshots']:
            tables[name] = table.hstack([table.QTable.read(path / f'{name}.fits'), table.QTable.read(path / f'{name}_strings.csv')])
        particles = tables['particles']
        particles.sort('particle_index')
        output = cls(
            r=particles['r'],
            v=cast(Quantity['velocity'], np.vstack([particles['vx'], particles['vy'], particles['vr']]).T),
            particle_type=particles['particle_type'],
            m=particles['m'],
            **payload,
            snapshots=tables['snapshots'],
        )
        output.initial_particles = tables['initial_particles']
        return output

    #####################
    ##Plots
    #####################

    def fill_time_unit(self, unit: UnitLike) -> UnitLike:
        """If the unit is `Tdyn` return self.Tdyn. If it's `time step` return self.time_step, otherwise return unit."""
        if unit == 'Tdyn':
            return self.Tdyn
        elif unit == 'time step':
            return self.time_step
        return unit

    def print_energy_change_summary(self, filter_particle_type: ParticleType | None = None) -> str:
        """Print a summary of the energy change during the simulation."""
        initial = self.initial_particles.copy()
        final = self.particles.copy()
        if filter_particle_type is not None:
            initial = initial[initial['particle_type'] == filter_particle_type].copy()
            final = final[final['particle_type'] == filter_particle_type].copy()
        return f"""After {self.current_step} steps with dt={self.dt:.4f} | {self.time:.1f}
Total energy at the start:        {initial['E'].sum():.1f}
Total energy at the end:          {final['E'].sum():.1f}
Energy change:                    {np.abs(final['E'].sum() - initial['E'].sum()):.1f}
Energy change per step:           {np.abs(final['E'].sum() - initial['E'].sum()) / self.current_step:.1e}
Energy change per dt:             {np.abs(final['E'].sum() - initial['E'].sum()) / self.dt:.1e}
Relative energy change:           {np.abs(final['E'].sum() - initial['E'].sum()) / initial['E'].sum():.3%}
Relative energy change per step:  {np.abs(final['E'].sum() - initial['E'].sum()) / initial['E'].sum() / self.current_step:.1e}
Relative energy change per dt:    {np.abs(final['E'].sum() - initial['E'].sum()) / initial['E'].sum() / self.dt:.3%}
Mean velocity change:             {np.abs(final['v_norm'].mean() - initial['v_norm'].mean()).to('km/second'):.1f}
Mean velocity change per step:    {np.abs(final['v_norm'].mean() - initial['v_norm'].mean()).to('km/second') / self.current_step:.1e}
Mean velocity change per dt:      {np.abs(final['v_norm'].mean() - initial['v_norm'].mean()).to('km/second') / self.dt:.1e}
Relative Mean velocity change:    {np.abs(final['v_norm'].mean() - initial['v_norm'].mean()) / initial['v_norm'].mean():.3%}"""

    def plot_r_kde_over_time(
        self,
        include_start: bool = True,
        include_now: bool = True,
        filter_particle_type: ParticleType | None = None,
        x_range: Quantity['length'] | None = None,
        x_units: UnitLike = 'kpc',
        time_units: UnitLike = 'Tdyn',
        time_format: str = '.1f',
        title: str | None = 'Density progression over time',
        xlabel: str | None = 'Radius',
        indices: list[int] | None = None,
        color_palette: str | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the density progression over time.

        Parameters:
            include_start: Whether to include the initial particle distribution in the plot.
            include_now: Whether to include the current particle distribution in the plot.
            filter_particle_type: Whether to filter to only plot the specified particle type.
            x_range: The radius range to clip the data to. If None, ignores.
            time_units: The time units to use in the plot.
            time_format: Format string for time.
            title: The title of the plot.
            xlabel: The label for the x-axis.
            indices: The snapshot indices to plot. If None plots everything.
            color_palette: The color palette to use for the halos. If None, uses the default color palette.
            kwargs: Additional keyword arguments to pass to the plot function (plot.setup_plot).

        Returns:
            fig, ax.
        """

        data_tables = [] if not include_now else [self.initial_particles]
        data_tables += [self.snapshots]
        if include_now:
            data_tables += [self.particles]
        data = table.QTable(table.vstack(data_tables))
        if filter_particle_type is not None:
            data = data[data['particle_type'] == filter_particle_type].copy()

        indices = indices if indices is not None else list(range(len(np.unique(np.array(data['time'])))))

        if color_palette is not None:
            colors = sns.color_palette(color_palette, len(indices))
        else:
            colors = None

        time_units = self.fill_time_unit(time_units)
        fig, ax = plot.setup_plot(**utils.drop_None(title=title, xlabel=utils.add_label_unit(xlabel, x_units)), **kwargs)
        clip = tuple(x_range.to(x_units).value) if x_range is not None else None
        for i, group in enumerate(self.snapshots.group_by('time').groups):
            if i in indices:
                sns.kdeplot(
                    group['r'].to(x_units),
                    ax=ax,
                    clip=clip,
                    color=colors[indices.index(i)] if colors is not None else None,
                    label=group['time'][0].to(time_units).to_string(format='latex', formatter=time_format),
                )
        fig.legend(loc='outside center right')
        return fig, ax

    def plot_distribution(
        self,
        key: str,
        data: table.QTable,
        filter_particle_type: ParticleType | None = None,
        cumulative: bool = False,
        absolute: bool = False,
        title: str | None = None,
        xlabel: str | None = None,
        x_range: Quantity | None = None,
        x_plot_range: Quantity | None = None,
        stat: str = 'density',
        plot_type: Literal['hist', 'kde'] = 'hist',
        x_units: UnitLike | None = None,
        ylabel: str | None = None,
        label: str | None = None,
        fig: Figure | None = None,
        ax: Axes | None = None,
        plt_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the distribution of a given key in the data.

        Parameters:
            key: The key to plot.
            data: The data to plot.
            filter_particle_type: Whether to filter to only plot the specified particle type.
            cumulative: Whether to plot the cumulative distribution.
            absolute: Whether to plot the absolute values.
            title: The title of the plot.
            xlabel: The label for the x-axis.
            x_range: The radius range to clip the data to. If None, ignores.
            x_plot_range: The range to plot on the x-axis. If None, uses the data range.
            stat: The type of statistic to plot. Gets passed to sns.histplot. Only used if `plot_type` is 'hist'.
            plot_type: The type of plot to create.
            x_units: The x-axis units to use in the plot.
            ylabel: The label for the y-axis.
            label: The label for the histogram (legend).
            fig: The figure to plot on.
            ax: The axes to plot on.
            plt_kwargs: Additional keyword arguments to pass to the sns plotting function (sns.histplot or sns.kdeplot).
            kwargs: Additional keyword arguments to pass to plot.setup_plot().

        Returns:
            fig, ax.
        """
        x_units = plot.default_plot_unit_type(key, x_units)
        if filter_particle_type is not None:
            data = cast(table.QTable, data[data['particle_type'] == filter_particle_type].copy())
        x = data[key].to(x_units)
        if x_range is not None:
            x = x[(x > x_range[0]) * (x < x_range[1])]
        if absolute:
            x = np.abs(x)
        params = {**plot.default_plot_text(key, x_units=x_units), **utils.drop_None(title=title, xlabel=xlabel, ylabel=ylabel)}
        fig, ax = plot.setup_plot(fig, ax, **params, **kwargs)
        if plot_type == 'kde':
            sns.kdeplot(x, cumulative=cumulative, ax=ax, label=label, **plt_kwargs)
        else:
            sns.histplot(x, cumulative=cumulative, ax=ax, stat=stat, label=label, **plt_kwargs)
        if x_plot_range is not None:
            ax.set_xlim(*x_plot_range.to(x_units).value)
        return fig, ax

    def plot_r_distribution(
        self,
        data: table.QTable,
        cumulative: bool = False,
        add_density: bool = True,
        x_units: UnitLike = 'kpc',
        x_range: Quantity | None = None,
        hist_label: str | None = None,
        density_label: str | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the radial distribution of the halo. Wraps plot_distribution() with additional options.

        Args:
            data: The data to plot.
            cumulative: Whether to plot the cumulative distribution.
            add_density: Whether to add the density distribution from the first density in the densities list.
            x_units: The units to plot the x-axis in.
            x_range: The range of the x-axis.
            hist_label: The label for the histogram (legend).
            density_label: The label for the density distribution (legend).
            kwargs: Additional keyword arguments to pass to plot.setup_plot().

        Returns:
            fig, ax.
        """
        fig, ax = self.plot_distribution(key='r', data=data, cumulative=cumulative, x_units=x_units, x_range=x_range, label=hist_label, **kwargs)
        if add_density:
            params = {'r_start': cast(Quantity, x_range[0]), 'r_end': cast(Quantity, x_range[1])} if x_range is not None else {}
            return self.densities[0].plot_radius_distribution(
                cumulative=cumulative, length_units=x_units, fig=fig, ax=ax, label=density_label, **params
            )
        return fig, ax

    def plot_phase_space(
        self,
        data: table.QTable,
        filter_particle_type: ParticleType | None = None,
        r_range: Quantity['length'] = Quantity(np.linspace(1e-2, 50, 200), 'kpc'),
        v_range: Quantity['velocity'] = Quantity(np.linspace(0, 100, 200), 'km/second'),
        length_units: UnitLike = 'kpc',
        velocity_units: UnitLike = 'km/second',
        fig: Figure | None = None,
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot the phase space distribution of the particles.

        Parameters:
            data: The data to plot.
            filter_particle_type: Whether to filter to only plot the specified particle type.
            r_range: Range of radial distances to plot.
            v_range: Range of velocities to plot.
            length_units: Units to use for the length axis.
            velocity_units: Units to use for the velocity axis. Set to default to 'km/second' and not 'kpc/Myr' and thus explicitly mentioned (the length unit use the default of the plotting function, and can be passed on as optional keyword arguments)
            fig: Figure to use for the plot.
            ax: Axes to use for the plot.

        Returns:
            fig, ax.
        """
        r_lattice = Lattice(len(r_range), r_range.min().to(length_units).value, r_range.max().to(length_units).value, log=False)
        v_lattice = Lattice(len(v_range), v_range.min().to(velocity_units).value, v_range.max().to(velocity_units).value, log=False)
        grid = np.zeros((len(v_range), len(r_range)))

        if filter_particle_type is not None:
            data = cast(table.QTable, data[data['particle_type'] == filter_particle_type].copy())
        r = data['r'].to(length_units).value
        v_norm = data['v_norm'].to(velocity_units).value

        mask = r_lattice.in_lattice(r) * v_lattice.in_lattice(v_norm)
        data_table = pd.DataFrame({'r': r_lattice(r[mask]), 'v_norm': v_lattice(v_norm[mask])})
        data_table['count'] = 1
        data_table = data_table.groupby(['r', 'v_norm']).agg('count').reset_index()
        grid[data_table['v_norm'].to_numpy(), data_table['r'].to_numpy()] = data_table['count']

        return plot.plot_phase_space(grid, r_range, v_range, length_units, velocity_units, fig=fig, ax=ax)

    def plot_inner_core_density(
        self,
        include_start: bool = True,
        include_now: bool = True,
        radius: Quantity['length'] = Quantity(0.2, 'kpc'),
        filter_particle_type: ParticleType | None = None,
        time_units: UnitLike = 'Tdyn',
        xlabel: str | None = 'time',
        ylabel: str | None = '#particles',
        title: str | None = '',
        fig: Figure | None = None,
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot the number of particles in the inner core as a function of time.

        Parameters:
            include_start: Whether to include the initial particle distribution in the plot.
            include_now: Whether to include the current particle distribution in the plot.
            radius: Radius of the inner core.
            filter_particle_type: Whether to filter to only plot the specified particle type.
            time_units: The time units to use in the plot.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            title: Title for the plot.
            fig: Figure to use for the plot.
            ax: Axes to use for the plot.

        Returns:
            fig, ax.
        """
        data_tables = [] if not include_now else [self.initial_particles]
        data_tables += [self.snapshots]
        if include_now:
            data_tables += [self.particles]
        data = table.QTable(table.vstack(data_tables))
        if filter_particle_type is not None:
            data = cast(table.QTable, data[data['particle_type'] == filter_particle_type].copy())
        time_units = self.fill_time_unit(time_units)
        data['time'] = data['time'].to(time_units)
        data['in_radius'] = data['r'] <= radius

        agg_data = utils.aggregate_QTable(data, groupby='time', keys=['in_radius'], agg_fn='sum', final_units={'time': time_units})

        if title == '':
            title = f'#particles in inner core ({radius.to_string(format="latex", formatter=".2f")})'

        xlabel = f'{xlabel} [{time_units}]' if xlabel is not None else None
        fig, ax = plot.setup_plot(fig, ax, **utils.drop_None(title=title, xlabel=xlabel, ylabel=ylabel))
        sns.lineplot(agg_data.to_pandas(index='time'), ax=ax)
        return fig, ax

    @staticmethod
    def prep_2d_data(
        data: table.QTable,
        radius_range: tuple[Quantity['length'], Quantity['length']],
        time_range: tuple[Quantity['time'], Quantity['time']] | None = None,
        length_units: UnitLike = 'kpc',
        time_units: UnitLike = 'Myr',
        agg_fn: str | Callable[[Any], Any] = 'count',
        n_posts: int = 100,
    ) -> tuple[NDArray[Any], tuple[Quantity['length'], Quantity['length'], Quantity['time'], Quantity['time']]]:
        """Prepare data for 2D plotting.

        Parameters:
            data: Data to prepare.
            radius_range: Range of radius to consider (filters the data).
            time_range: Range of times to consider (filters the data).
            length_units: Units to use for the radius axis.
            time_units: Units to use for the time axis.
            agg_fn: Function to aggregate data.
            n_posts: Number of posts to use in the discretization lattice (affects resolution).

        Returns:
            data, extent
        """
        data = data.copy()
        data['r'] = data['r'].to(length_units)
        data['time'] = data['time'].to(time_units)
        radius_mask = (data['r'] >= radius_range[0]) * (data['r'] <= radius_range[1])
        time_mask = (data['time'] >= time_range[0]) * (data['time'] <= time_range[1]) if time_range is not None else np.full_like(radius_mask, True)
        data = cast(table.QTable, data[radius_mask * time_mask])
        lattice = Lattice(n_posts=n_posts, start=data['r'].min().value, end=data['r'].max().value * 1.1, log=False)
        data['bin'] = lattice.posts[lattice(data['r'].value)]
        agg_data = pd.DataFrame(data.to_pandas().groupby(['time', 'bin'])['output'].agg(agg_fn)).reset_index()
        r, time = np.meshgrid(lattice.posts, np.unique(data['time'].value))
        pad = pd.DataFrame({'time': time.ravel(), 'bin': r.ravel()})
        pad['output'] = np.nan
        agg_data = pd.concat([agg_data, pad]).drop_duplicates(['time', 'bin']).sort_values(['time', 'bin'])
        extent = (
            Quantity(r.min(), length_units),
            Quantity(r.max(), length_units),
            Quantity(time.min(), time_units),
            Quantity(time.max(), time_units),
        )
        return agg_data.output.to_numpy().reshape(r.shape), extent

    def plot_density_evolution(
        self,
        include_start: bool = True,
        include_now: bool = True,
        filter_particle_type: ParticleType | None = None,
        radius_range: tuple[Quantity['length'], Quantity['length']] = (Quantity(0, 'kpc'), Quantity(40, 'kpc')),
        time_range: tuple[Quantity['time'], Quantity['time']] | None = None,
        length_units: UnitLike = 'kpc',
        time_units: UnitLike = 'Tdyn',
        xlabel: str | None = 'Radius',
        ylabel: str | None = 'Time',
        cbar_label: str | None = '#Particles',
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the density evolution of the halo. Wraps prep_2d_data().

        Parameters:
            include_start: Whether to include the initial particle distribution in the plot.
            include_now: Whether to include the current particle distribution in the plot.
            filter_particle_type: Whether to filter to only plot the specified particle type.
            radius_range: Range of radius to consider (filters the data).
            time_range: Range of times to consider (filters the data).
            length_units: Units to use for the radius axis.
            time_units: Units to use for the time axis.
            xlabel: Label for the radius axis.
            ylabel: Label for the time axis.
            cbar_label: Label for the colorbar.
            kwargs: Additional keyword arguments to pass to the plot function (utils.plot_2d()).

        Returns:
            fig, ax.
        """
        time_units = self.fill_time_unit(time_units)
        data_tables = [] if not include_now else [self.initial_particles]
        data_tables += [self.snapshots]
        if include_now:
            data_tables += [self.particles]
        data = table.QTable(table.vstack(data_tables))
        data['output'] = data['r']
        if filter_particle_type is not None:
            data = cast(table.QTable, data[data['particle_type'] == filter_particle_type].copy())
        grid, extent = self.prep_2d_data(data, radius_range, time_range, length_units, time_units, agg_fn='count')

        return plot.plot_2d(
            grid=grid,
            extent=extent,
            x_units=length_units,
            y_units=time_units,
            xlabel=utils.add_label_unit(xlabel, length_units),
            ylabel=utils.add_label_unit(ylabel, time_units),
            cbar_label=cbar_label,
            **kwargs,
        )

    def plot_temperature(
        self,
        include_start: bool = True,
        include_now: bool = True,
        filter_particle_type: ParticleType | None = None,
        radius_range: tuple[Quantity['length'], Quantity['length']] = (Quantity(0, 'kpc'), Quantity(40, 'kpc')),
        time_range: tuple[Quantity['time'], Quantity['time']] | None = None,
        velocity_units: UnitLike = 'km/second',
        length_units: UnitLike = 'kpc',
        time_units: UnitLike = 'Tdyn',
        xlabel: str | None = 'Radius',
        ylabel: str | None = 'Time',
        cbar_label: str | None = 'Temperature (velocity std)',
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the temperature evolution of the halo. Wraps prep_2d_data().

        Parameters:
            include_start: Whether to include the initial particle distribution in the plot.
            include_now: Whether to include the current particle distribution in the plot.
            filter_particle_type: Whether to filter to only plot the specified particle type.
            radius_range: Range of radius to consider (filters the data).
            time_range: Range of times to consider (filters the data).
            velocity_units: Units to use for the velocity axis.
            length_units: Units to use for the radius axis.
            time_units: Units to use for the time axis.
            xlabel: Label for the radius axis.
            ylabel: Label for the time axis.
            cbar_label: Label for the colorbar.
            kwargs: Additional keyword arguments to pass to the plot function (utils.plot_2d()).

        Returns:
            fig, ax.
        """
        time_units = self.fill_time_unit(time_units)
        data_tables = [] if not include_now else [self.initial_particles]
        data_tables += [self.snapshots]
        if include_now:
            data_tables += [self.particles]
        data = table.QTable(table.vstack(data_tables))
        data['output'] = data['v_norm']
        if filter_particle_type is not None:
            data = cast(table.QTable, data[data['particle_type'] == filter_particle_type].copy())
        grid, extent = self.prep_2d_data(data, radius_range, time_range, length_units, time_units, agg_fn='std')

        return plot.plot_2d(
            grid=grid,
            extent=extent,
            x_units=length_units,
            y_units=time_units,
            xlabel=utils.add_label_unit(xlabel, velocity_units),
            ylabel=utils.add_label_unit(ylabel, time_units),
            cbar_label=utils.add_label_unit(cbar_label, velocity_units),
            **kwargs,
        )

    def plot_start_end_distribution(
        self,
        key: str = 'r',
        time_units: UnitLike = 'Tdyn',
        time_format: str = '.1f',
        label_start: str = 'start',
        label_end: str = 'after {t}',
        fig: Figure | None = None,
        ax: Axes | None = None,
        start_kwargs: dict[str, Any] = {},
        end_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the distribution comparison between the current state and the initial state.

        Parameters:
            key: The key to plot.
            time_units: The time units to use in the plot.
            time_format: Format string for time.
            label_start: Label for the start distribution.
            label_end: Label for the end distribution.
            fig: Figure to use for the plot.
            ax: Axes to use for the plot.
            start_kwargs: Additional keyword arguments to pass to the distribution plotting function (self.plot_distribution), for the start distribution only.
            end_kwargs: Additional keyword arguments to pass to the distribution plotting function (self.plot_distribution), for the end distribution only.
            **kwargs: Additional keyword arguments to pass to the distribution plotting function (self.plot_distribution), for *both* distributions. Overwritten by start_kwargs/end_kwargs as needed.

        Returns:
            fig, ax.
        """
        time_units = self.fill_time_unit(time_units)
        fig, ax = self.plot_distribution(key=key, data=self.initial_particles, fig=fig, ax=ax, label=label_start, **{**kwargs, **start_kwargs})
        fig, ax = self.plot_distribution(
            key=key,
            data=self.particles,
            fig=fig,
            ax=ax,
            label=label_end.format(t=self.time.to(time_units).to_string(format='latex', formatter=time_format)),
            **{
                **kwargs,
                **end_kwargs,
            },
        )
        ax.legend()
        return fig, ax

    def plot_scattering_location(
        self,
        title: str | None = 'Scattering location distribution within the first {time}, total of {n_interactions} events',
        xlabel: str | None = 'Radius',
        length_units: UnitLike = 'kpc',
        time_units: UnitLike = 'Gyr',
        time_format: str = '.1f',
        figsize: tuple[int, int] = (12, 6),
        fig: Figure | None = None,
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot the histogram of scattering event locations.

        Flattens all the events (ignores time), and plots the location of the particles at each scattering event.

        Parameters:
            title: Title for the plot.
            xlabel: Label for the x-axis.
            length_units: Units to use for the radius axis.
            time_units: Units to use for time.
            time_format: Format string for time.
            figsize: Size of the figure.
            fig: Figure to use for the plot.
            ax: Axes to use for the plot.

        Returns:
            fig, ax.

        """
        xlabel = utils.add_label_unit(xlabel, length_units)
        time_units = self.fill_time_unit(time_units)
        if title is not None:
            title = title.format(time=self.time.to(time_units).to_string(format='latex', formatter=time_format), n_interactions=self.n_interactions)
        fig, ax = plot.setup_plot(fig, ax, figsize=figsize, minorticks=True, **utils.drop_None(title=title, xlabel=xlabel))
        sns.histplot(Quantity(np.hstack(self.interactions_track), run_units.length).to(length_units), ax=ax)
        return fig, ax

    def plot_scattering_density(
        self,
        num: int = 500,
        xlabel: str | None = 'Radius',
        ylabel: str | None = 'Density',
        title: str | None = 'Scattering density within the first {time}, total of {n_interactions} events',
        length_units: UnitLike = 'kpc',
        time_units: UnitLike = 'Gyr',
        time_format: str = '.1f',
        smooth_sigma: float = 5,
        smooth_interpolate_kind: str = 'linear',
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the location of scattering events location densities (number of events per bin volume).

        Flattens all the events (ignores time), and plots the location of the particles at each scattering event.
        Bins are linearly spaced between 0 and max(r).

        Parameters:
            num: Number of bins to use for the radius axis.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            title: Title for the plot.
            length_units: Units to use for the radius axis.
            time_units: Units to use for time.
            time_format: Format string for time.
            smooth_sigma: Smoothing factor for the density plot (sigma for a 1d Gaussian kernel).
            smooth_interpolate_kind: Interpolation kind for the density plot. Applied after the gaussian smoothing to further smooth the plot data.
            kwargs: Additional keyword arguments to pass to the plot function (plot.setup_plot).

        Returns:
            fig, ax.
        """
        r = np.hstack(self.interactions_track).to(length_units)
        r_bins = np.linspace(0, r.max(), num=num)
        dr = r_bins[1] - r_bins[0]
        density = Quantity(
            [((r >= low) * (r < high)).sum() / (4 * np.pi * dr * ((low + high) / 2) ** 2) for low, high in zip(r_bins[:-1], r_bins[1:])]
        )
        r_bins = r_bins[:-1]
        density_units = str(density.unit)
        interpolated_density = scipy.interpolate.interp1d(r_bins[density != 0], density[density != 0], kind=smooth_interpolate_kind)(r_bins)
        smoothed_density = scipy.ndimage.gaussian_filter1d(interpolated_density, sigma=smooth_sigma)

        xlabel = utils.add_label_unit(xlabel, length_units)
        ylabel = utils.add_label_unit(ylabel, density_units)
        time_units = self.fill_time_unit(time_units)
        if title is not None:
            title = title.format(time=self.time.to(time_units).to_string(format='latex', formatter=time_format), n_interactions=self.n_interactions)
        fig, ax = plot.setup_plot(**kwargs, ax_set={'yscale': 'log'}, **utils.drop_None(title=title, xlabel=xlabel))
        sns.lineplot(x=r_bins, y=smoothed_density, ax=ax)
        return fig, ax

    def plot_local_density_by_range(
        self,
        x_range: Quantity['length'] | None = None,
        xlabel: str | None = 'Radius',
        ylabel: str | None = 'local density',
        title: str | None = 'Local density ({nn} nearest neighbors) after t={time}',
        x_units: UnitLike = 'kpc',
        density_units: UnitLike = 'Msun/kpc**3',
        time_units: UnitLike = 'Gyr',
        time_format: str = '.1f',
        smooth_sigma: float = 50,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the local density profile of the halo as a function of the radius.

        Parameters:
            radius_range: Range of radius to consider (filters the data).
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            title: Title for the plot.
            x_units: Units to use for the x-axis.
            density_units: Units to use for the y-axis.
            time_units: Units to use for time.
            time_format: Format string for time.
            smooth_sigma: Smoothing factor for the density plot (sigma for a 1d Gaussian kernel).
            kwargs: Additional keyword arguments to pass to the plot function (plot.setup_plot).

        Returns:
            fig, ax.
        """
        x = self.r
        time_units = self.fill_time_unit(time_units)
        local_density = self.local_density.to(density_units)
        smoothed_local_density = scipy.ndimage.gaussian_filter1d(local_density, sigma=smooth_sigma) if smooth_sigma > 0 else local_density
        if x_range is not None:
            smoothed_local_density = smoothed_local_density[(x > x_range[0]) * (x < x_range[1])]
            x = x[(x > x_range[0]) * (x < x_range[1])]
        if title is not None:
            title = title.format(
                nn=self.scatter_params.get('max_radius_j', None),
                time=self.time.to(time_units).to_string(format='latex', formatter=time_format),
            )
        xlabel = utils.add_label_unit(xlabel, x_units)
        ylabel = utils.add_label_unit(ylabel, density_units)
        fig, ax = plot.setup_plot(**kwargs, **utils.drop_None(title=title, xlabel=xlabel, ylabel=ylabel))
        sns.lineplot(x=x, y=smoothed_local_density, ax=ax)
        return fig, ax

    def plot_local_density_distribution(
        self,
        xlabel: str | None = 'local density',
        title: str | None = 'Local density ({nn} nearest neighbors) after t={time}',
        density_units: UnitLike = 'Msun/kpc**3',
        time_units: UnitLike = 'Gyr',
        time_format: str = '.1f',
        log_scale: bool = True,
        stat: str = 'density',
        cumulative: bool = False,
        hist_kwargs: Any = {},
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the local density distribution of the halo (histogram).

        Parameters:
            xlabel: Label for the x-axis.
            title: Title for the plot.
            density_units: Units to use for the x-axis.
            time_units: Units to use for time.
            time_format: Format string for time.
            log_scale: Whether to use a logarithmic scale for the x-axis.
            stat: The type of statistic to plot. Gets passed to sns.histplot.
            cumulative: Whether to plot the cumulative distribution.
            hist_kwargs: Additional keyword arguments to pass to sns.histogram().
            kwargs: Additional keyword arguments to pass to the plot function (plot.setup_plot).

        Returns:
            fig, ax.
        """
        time_units = self.fill_time_unit(time_units)
        xlabel = utils.add_label_unit(xlabel, density_units)
        if title is not None:
            title = title.format(
                nn=self.scatter_params.get('max_radius_j', None),
                time=self.time.to(time_units).to_string(format='latex', formatter=time_format),
            )
        fig, ax = plot.setup_plot(**kwargs, **utils.drop_None(title=title, xlabel=xlabel))
        sns.histplot(self.local_density.to(density_units), log_scale=log_scale, stat=stat, cumulative=cumulative, **hist_kwargs)
        return fig, ax

    def plot_trace(
        self,
        particle_index: int,
        key: str,
        data: table.QTable | None = None,
        relative: Literal['relative change', 'change', 'absolute'] = 'absolute',
        xlabel: str | None = 'Time',
        ylabel: str | None = None,
        title: str | None = 'Trace of particle id={particle_index}, initial position={r}',
        label: str | None = 'particle id={particle_index}, initial position={r}',
        time_units: UnitLike = 'Tdyn',
        y_units: UnitLike | None = None,
        length_units: UnitLike = 'kpc',
        length_format: str = '.1f',
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the trace of a particle's property over time.

        Parameters:
            key: The property to plot. Must be a valid column name in the data table.
            data: The data table to plot (i.e. halo.snapshots, or an external table from an NSphere run). If None, use the halo's snapshots + initial and current states.
            particle_index: The index of the particle to trace.
            relative: If `absolute`, plot the property as is. If `relative`, plot the change in the property relative to the initial value. If `relative change`, plot the change in the property relative to the initial value divided by the initial value.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            title: Title for the plot.
            time_units: Units for the x-axis.
            y_units: Units for the y-axis.
            length_units: Units for the length.
            length_format: Format string for length.
            kwargs: Additional keyword arguments to pass to the plot function (plot.setup_plot).

        Returns:
            fig, ax.
        """
        return plot.plot_trace(
            key=key,
            data=data if data is not None else table.QTable(table.vstack([self.initial_particles, self.snapshots, self.particles])).copy(),
            particle_index=particle_index,
            relative=relative,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            label=label,
            time_units=self.fill_time_unit(time_units),
            y_units=y_units,
            length_units=length_units,
            length_format=length_format,
            **kwargs,
        )
