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
from typing import Any, Self, Callable, cast
from astropy import table, constants
from astropy.units import Quantity, Unit, def_unit
from astropy.units.typing import UnitLike
from .spatial_approximation import Lattice
from .density.density import Density
from .background import Mass_Distribution
from . import utils, run_units, physics
from .physics import sidm, leapfrog
from .types import ParticleType


class Halo:
    def __init__(
        self,
        dt: Quantity['time'],
        r: Quantity['length'],
        v: Quantity['velocity'],
        m: Quantity['mass'],
        particle_type: NDArray[np.str_] | None = None,
        Tdyn: Quantity['time'] | None = None,
        Phi0: Quantity['energy'] = Quantity(0, run_units.energy),
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
    ) -> None:
        self.time: Quantity['time'] = time.to(run_units.time)
        self.dt: Quantity['time'] = dt.to(run_units.time)
        self.densities: list[Density] = densities
        self.r: Quantity['length'] = r.to(run_units.length)
        self.v: Quantity['velocity'] = v.to(run_units.velocity)
        self.m: Quantity['mass'] = m.to(run_units.mass)
        self.particle_type: NDArray[np.str_] = particle_type if particle_type is not None else np.full(len(r), 'dm')
        self.particle_index = np.arange(len(r))
        self.Tdyn: Quantity['time']
        if Tdyn is not None:
            self.Tdyn = Tdyn.to(run_units.time)
        elif len(self.densities) > 0:
            self.Tdyn = self.densities[0].Tdyn.to(run_units.time)
        elif len(self.densities) == 0:
            self.Tdyn = Quantity(1, run_units.time)
        self.Phi0: Quantity['energy'] = Phi0
        self.n_interactions = n_interactions
        self.snapshots: table.QTable = snapshots
        self.save_every_n_steps = save_every_n_steps
        self.save_every_time: Quantity['time'] | None = save_every_time if save_every_time is None else save_every_time.to(run_units.time)
        self._dynamics_params: leapfrog.Params = leapfrog.normalize_params(dynamics_params, add_defaults=True)
        self._scatter_params: sidm.Params = sidm.normalize_params(scatter_params, add_defaults=True)
        self.interactions_track = interactions_track
        self.background: Mass_Distribution | None = background
        self.initial_particles = self.particles.copy()
        self.last_saved_time = last_saved_time
        self.scatter_rounds = scatter_rounds
        self.scatter_every_n_steps: int = scatter_every_n_steps

    @classmethod
    def setup(cls, densities: list[Density], particle_types: list[ParticleType], n_particles: list[int | float], **kwargs: Any) -> Self:
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
            v=cast(Quantity['length'], np.hstack(v)),
            m=cast(Quantity['length'], np.hstack(m)),
            particle_type=np.hstack(particle_type),
            densities=densities,
            **kwargs,
        )

    def add_background(self, background: Mass_Distribution) -> None:
        self.background = background

    def reset(self) -> None:
        self.time = 0 * run_units.time
        self.n_interactions = 0
        self.particle_index = np.array(self.initial_particles['particle_index'])
        self.r = self.initial_particles['r'].to(run_units.length)
        self.v = np.vstack([cast(Quantity, self.initial_particles[i]) for i in ['vx', 'vy', 'vr']]).T.to(run_units.velocity)
        self.particle_type = self.initial_particles['particle_type']
        self.interactions_track = []
        self.snapshots = table.QTable()

    @property
    def particles(self) -> table.QTable:
        self.sort_particles()
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
                'particle_type': self.particle_type,
                'particle_index': self.particle_index,
            }
        )
        data.add_index('particle_index')
        return data

    @property
    def dynamics_params(self) -> leapfrog.Params:
        return self._dynamics_params

    @dynamics_params.setter
    def dynamics_params(self, value: leapfrog.Params) -> None:
        self._dynamics_params = leapfrog.normalize_params(value)

    @property
    def scatter_params(self) -> sidm.Params:
        return self._scatter_params

    @scatter_params.setter
    def scatter_params(self, value: sidm.Params) -> None:
        self._scatter_params = sidm.normalize_params(value)

    ##Physical properties

    @property
    def time_step(self) -> Unit:
        return def_unit('time step', self.dt.to(run_units.time), format={'latex': r'time\ step'})

    @property
    def M(self) -> Quantity['mass']:
        halo_mass = physics.utils.M(r=self.r, m=self.m, method='rank presorted')
        if self.background is not None:
            background_mass = self.background.M_at_time(self.r, self.time)
            return cast(Quantity['mass'], halo_mass + background_mass)
        return halo_mass

    @property
    def vx(self) -> Quantity['velocity']:
        return cast(Quantity['velocity'], self.v[:, 0])

    @property
    def vy(self) -> Quantity['velocity']:
        return cast(Quantity['velocity'], self.v[:, 1])

    @property
    def vr(self) -> Quantity['velocity']:
        return cast(Quantity['velocity'], self.v[:, 2])

    @property
    def vp(self) -> Quantity['velocity']:
        return utils.fast_quantity_norm(cast(Quantity['velocity'], self.v[:, :2]))

    @property
    def v_norm(self) -> Quantity['velocity']:
        return utils.fast_quantity_norm(self.v)

    @property
    def kinetic_energy(self) -> Quantity['energy']:
        return 0.5 * self.m * self.v_norm**2

    @property
    def Phi(self) -> Quantity['energy']:
        return -(constants.G * self.M * self.m / self.r).to(run_units.energy)

    @property
    def Psi(self) -> Quantity['specific energy']:
        return (self.Phi0 - self.Phi).to(run_units.energy)

    @property
    def E(self) -> Quantity['specific energy']:
        return (self.Psi - self.kinetic_energy).to(run_units.energy)

    @property
    def local_density(self) -> Quantity['mass density']:
        return physics.utils.local_density(self.r, self.m, self.scatter_params.get('max_radius_j', sidm.default_params.get('max_radius_j', 10)))

    @property
    def ranks(self) -> NDArray[np.int64]:
        return utils.rank_array(self.r)

    def sort_particles(self) -> None:
        indices = np.argsort(self.r)
        self.r = self.r[indices]
        self.v = self.v[indices]
        self.m = self.m[indices]
        self.particle_type = self.particle_type[indices]
        self.particle_index = self.particle_index[indices]

    ##Scatter-related properties

    @property
    def dt_scatter(self) -> Quantity['time']:
        return self.scatter_every_n_steps * self.dt

    @property
    def scatter_timescale(self) -> Quantity['time']:
        return sidm.t_scatter(local_density=self.local_density, sigma=self.scatter_params.get('sigma', None), v_norm=self.v_norm)

    @property
    def required_scatter_rounds(self) -> NDArray[np.int64]:
        return sidm.calculate_scatter_rounds(
            v=self.v,
            dt=self.dt_scatter,
            local_density=self.local_density,
            sigma=self.scatter_params.get('sigma', None),
            kappa=self.scatter_params.get('kappa', sidm.default_params.get('kappa', 1)),
            max_allowed_rounds=self.scatter_params.get('max_allowed_rounds', sidm.default_params.get('max_allowed_rounds', None)),
            random_round_rounding=self.scatter_params.get('random_round_rounding', sidm.default_params.get('random_round_rounding', False)),
        )

    @property
    def required_scatter_rounds_uncapped(self) -> NDArray[np.int64]:
        return sidm.calculate_scatter_rounds(
            v=self.v,
            dt=self.dt_scatter,
            local_density=self.local_density,
            sigma=self.scatter_params.get('sigma', None),
            kappa=self.scatter_params.get('kappa', sidm.default_params.get('kappa', 1)),
            max_allowed_rounds=None,
            random_round_rounding=self.scatter_params.get('random_round_rounding', sidm.default_params.get('random_round_rounding', False)),
        )

    ##Dynamic evolution

    def to_step(self, time: Quantity['time']) -> int:
        return int(time / self.dt)

    @property
    def current_step(self) -> int:
        return self.to_step(self.time)

    def save_snapshot(self) -> None:
        data = self.particles.copy()
        data['step'] = self.current_step
        self.snapshots = table.vstack([self.snapshots, data])
        self.last_saved_time = self.time.copy()

    def is_save_round(self) -> bool:
        if self.save_every_time is not None:
            next_save_time = self.last_saved_time + self.save_every_time
            if self.time <= next_save_time and self.time + self.dt > next_save_time:
                return True
        elif self.save_every_n_steps is not None and self.current_step % self.save_every_n_steps == 0:
            return True
        return False

    def step(self) -> None:
        self.sort_particles()
        if self.is_save_round():
            self.save_snapshot()
        if self.scatter_params.get('sigma', 0) > 0 and self.current_step % self.scatter_every_n_steps == 0:
            mask = self.particle_type == 'dm'
            self.v[mask], n_interactions, indices, scatter_rounds = sidm.scatter(
                r=self.r, v=self.v, dt=self.dt_scatter, m=self.m, scattering_mask=mask, **self.scatter_params
            )
            self.n_interactions += n_interactions
            self.interactions_track += [self.r[indices]]
            self.scatter_rounds += [scatter_rounds]
        self.r, self.v = leapfrog.step(r=self.r, v=self.v, M=self.M, dt=self.dt, **self.dynamics_params)
        self.time += self.dt

    def evolve(self, n_steps: int | None = None, t: Quantity['time'] | None = None, disable_tqdm: bool = False) -> None:
        if n_steps is None:
            if t is not None:
                n_steps = self.to_step(t)
            else:
                raise ValueError('Either n_steps or t must be specified')
        for _ in tqdm(range(int(n_steps)), disable=disable_tqdm):
            self.step()

    ##Save/Load

    def to_payload(self) -> tuple[dict[str, Any], dict[str, Any]]:
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
        }
        tables = {
            'particles': self.particles,
            'initial_particles': self.initial_particles,
            'snapshots': self.snapshots,
        }
        return payload, tables

    def save(self, path: str | Path = Path('halo_state')) -> None:
        path = Path(path)
        path.mkdir(exist_ok=True)
        payload, tables = self.to_payload()
        with open(path / 'halo_payload.pkl', 'wb') as f:
            pickle.dump(payload, f)
        for name, data in tables.items():
            data.write(path / f'{name}.fits', overwrite=True)

    @classmethod
    def load(cls, path: str | Path = Path('halo_state')) -> Self:
        path = Path('halo_state')
        with open(path / 'halo_payload.pkl', 'rb') as f:
            payload = pickle.load(f)
        tables = {name: table.QTable.read(path / f'{name}.fits') for name in ['particles', 'initial_particles', 'snapshots']}
        r = tables['particles']['r']
        v = cast(Quantity['velocity'], np.vstack([tables['particles']['vx'], tables['particles']['vy'], tables['particles']['vr']]).T)
        output = cls(r=r, v=v, **payload, snapshots=tables['snapshots'])
        output.initial_particles = tables['initial_particles']
        return output

    ##Plots

    def default_plot_text(self, key: str, x_units: UnitLike) -> dict[str, str | None]:
        return {
            'vr': {'title': 'Radial velocity distribution', 'xlabel': utils.add_label_unit('Radial velocity', x_units), 'ylabel': 'Density'},
            'vx': {'title': 'Pendicular velocity distribution', 'xlabel': utils.add_label_unit('Pendicular velocity', x_units), 'ylabel': 'Density'},
            'vy': {'title': 'Pendicular velocity distribution', 'xlabel': utils.add_label_unit('Pendicular velocity', x_units), 'ylabel': 'Density'},
            'vp': {'title': 'Pendicular velocity distribution', 'xlabel': utils.add_label_unit('Pendicular velocity', x_units), 'ylabel': 'Density'},
            'v_norm': {'title': 'Velocity distribution', 'xlabel': utils.add_label_unit('Velocity', x_units), 'ylabel': 'Density'},
            'r': {'title': 'Radius distribution', 'xlabel': utils.add_label_unit('Radius', x_units), 'ylabel': 'Density'},
        }.get(key, {})

    def plot_unit_type(self, key: str, plot_unit: UnitLike | None = None) -> UnitLike:
        if plot_unit is not None:
            return plot_unit
        if key == 'r':
            return Unit('kpc')
        elif key in ['vr', 'vx', 'vy', 'vp', 'v_norm']:
            return Unit('km/second')
        return ''

    def fill_time_unit(self, unit: UnitLike) -> UnitLike:
        if unit == 'Tdyn':
            return self.Tdyn
        elif unit == 'time step':
            return self.time_step
        return unit

    def print_energy_change_summary(self) -> str:
        initial = self.initial_particles.copy()
        final = self.particles.copy()
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

    def plot_r_density_over_time(
        self,
        clip: Quantity['length'] | None = None,
        x_units: UnitLike = 'kpc',
        time_units: UnitLike = 'Tdyn',
        title: str | None = 'Density progression over time',
        xlabel: str | None = 'Radius',
        ylabel: str | None = None,
        fig: Figure | None = None,
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        time_units = self.fill_time_unit(time_units)
        fig, ax = utils.setup_plot(fig, ax, **utils.drop_None(title=title, xlabel=utils.add_label_unit(xlabel, x_units), ylabel=ylabel))
        legend = []
        clip_tuple = tuple(clip.to(x_units).value) if clip is not None else None
        for group in self.snapshots.group_by('time').groups:
            sns.kdeplot(group['r'].to(x_units).value, ax=ax, clip=clip_tuple)
            legend += [group['time'][0].to(time_units).to_string(format='latex', formatter='.1f')]
        fig.legend(legend, loc='outside center right')
        return fig, ax

    def plot_distribution(
        self,
        key: str,
        data: table.QTable,
        cumulative: bool = False,
        absolute: bool = False,
        title: str | None = None,
        xlabel: str | None = None,
        x_range: Quantity | None = None,
        x_plot_range: Quantity | None = None,
        stat: str = 'density',
        x_units: UnitLike | None = None,
        ylabel: str | None = None,
        fig: Figure | None = None,
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        x_units = self.plot_unit_type(key, x_units)
        x = data[key].to(x_units)
        if x_range is not None:
            x = x[(x > x_range[0]) * (x < x_range[1])]
        if absolute:
            x = np.abs(x)
        params = {**self.default_plot_text(key, x_units), **utils.drop_None(title=title, xlabel=xlabel, ylabel=ylabel)}
        fig, ax = utils.setup_plot(fig, ax, **params, **kwargs)
        sns.histplot(x, cumulative=cumulative, ax=ax, stat=stat)
        if x_plot_range is not None:
            ax.set_xlim(*x_plot_range.to(x_units).value)
        return fig, ax

    def plot_r_distribution(
        self,
        data: table.QTable,
        cumulative: bool = False,
        add_density: bool = True,
        x_units: UnitLike | None = None,
        x_range: Quantity | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        fig, ax = self.plot_distribution(key='r', data=data, cumulative=cumulative, x_units=x_units, x_range=x_range, **kwargs)
        if add_density:
            params = {'r_start': cast(Quantity, x_range[0]), 'r_end': cast(Quantity, x_range[1])} if x_range is not None else {}
            return self.densities[0].plot_radius_distribution(
                cumulative=cumulative, plot_units=self.plot_unit_type('r', x_units), fig=fig, ax=ax, **params
            )
        return fig, ax

    def plot_phase_space(
        self,
        data: table.QTable,
        r_range: Quantity['length'] = Quantity(np.linspace(1e-2, 50, 200), 'kpc'),
        v_range: Quantity['velocity'] = Quantity(np.linspace(0, 100, 200), 'km/second'),
        length_units: UnitLike = 'kpc',
        velocity_units: UnitLike = 'km/second',
        fig: Figure | None = None,
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        r_lattice = Lattice(len(r_range), r_range.min().to(length_units).value, r_range.max().to(length_units).value, log=False)
        v_lattice = Lattice(len(v_range), v_range.min().to(velocity_units).value, v_range.max().to(velocity_units).value, log=False)
        grid = np.zeros((len(v_range), len(r_range)))

        r = data['r'].to(length_units).value
        v_norm = data['v_norm'].to(velocity_units).value

        mask = r_lattice.in_lattice(r) * v_lattice.in_lattice(v_norm)
        data_table = pd.DataFrame({'r': r_lattice(r[mask]), 'v_norm': v_lattice(v_norm[mask])})
        data_table['count'] = 1
        data_table = data_table.groupby(['r', 'v_norm']).agg('count').reset_index()
        grid[data_table['v_norm'].to_numpy(), data_table['r'].to_numpy()] = data_table['count']

        return utils.plot_phase_space(grid, r_range, v_range, length_units, velocity_units, fig=fig, ax=ax)

    def plot_inner_core_density(
        self,
        radius: Quantity['length'] = Quantity(0.2, 'kpc'),
        time_units: UnitLike = 'Tdyn',
        xlabel: str | None = 'time',
        ylabel: str | None = '#particles',
        title: str | None = '',
        fig: Figure | None = None,
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        data = self.snapshots.copy()
        time_units = self.fill_time_unit(time_units)
        data['time'] = data['time'].to(time_units)
        data['in_radius'] = data['r'] <= radius

        agg_data = utils.aggregate_QTable(data, groupby='time', keys=['in_radius'], agg_fn='sum', final_units={'time': time_units})

        if title == '':
            title = f'#particles in inner core ({radius.to_string(format="latex", formatter=".2f")})'

        xlabel = f'{xlabel} [{time_units}]' if xlabel is not None else None
        fig, ax = utils.setup_plot(fig, ax, **utils.drop_None(title=title, xlabel=xlabel, ylabel=ylabel))
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
        radius_range: tuple[Quantity['length'], Quantity['length']] = (Quantity(0, 'kpc'), Quantity(40, 'kpc')),
        time_range: tuple[Quantity['time'], Quantity['time']] | None = None,
        length_units: UnitLike = 'kpc',
        time_units: UnitLike = 'Tdyn',
        xlabel: str | None = 'Radius',
        ylabel: str | None = 'Time',
        cbar_label: str | None = '#Particles',
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        time_units = self.fill_time_unit(time_units)
        data = self.snapshots.copy()
        data['output'] = data['r']
        grid, extent = self.prep_2d_data(data, radius_range, time_range, length_units, time_units, agg_fn='count')

        return utils.plot_2d(
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
        time_units = self.fill_time_unit(time_units)
        data = self.snapshots.copy()
        data['output'] = data['v_norm']
        grid, extent = self.prep_2d_data(data, radius_range, time_range, length_units, time_units, agg_fn='std')

        return utils.plot_2d(
            grid=grid,
            extent=extent,
            x_units=length_units,
            y_units=time_units,
            xlabel=utils.add_label_unit(xlabel, velocity_units),
            ylabel=utils.add_label_unit(ylabel, time_units),
            cbar_label=utils.add_label_unit(cbar_label, velocity_units),
            **kwargs,
        )

    def plot_before_after_histogram(self, time_units: UnitLike = 'Tdyn', time_format: str = '.1f', **kwargs: Any) -> tuple[Figure, Axes]:
        time_units = self.fill_time_unit(time_units)
        fig, ax = self.plot_distribution(data=self.initial_particles, **kwargs)
        fig, ax = self.plot_distribution(data=self.particles, fig=fig, ax=ax, **kwargs)
        ax.legend(['start', f'after {self.time.to(time_units).to_string(format="latex", formatter=time_format)}'])
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
        xlabel = utils.add_label_unit(xlabel, length_units)
        time_units = self.fill_time_unit(time_units)
        if title is not None:
            title = title.format(time=self.time.to(time_units).to_string(format='latex', formatter=time_format), n_interactions=self.n_interactions)
        fig, ax = utils.setup_plot(fig, ax, figsize=figsize, minorticks=True, **utils.drop_None(title=title, xlabel=xlabel))
        sns.histplot(Quantity(np.hstack(self.interactions_track), run_units.length).to(length_units), ax=ax)
        return fig, ax

    def plot_scattering_density(
        self,
        num: int = 500,
        xlabel: str | None = 'Radius',
        ylabel: str | None = 'Density',
        length_units: UnitLike = 'kpc',
        time_units: UnitLike = 'Gyr',
        title: str | None = 'Scattering density within the first {time}, total of {n_interactions} events',
        time_format: str = '.1f',
        smooth_sigma: float = 5,
        smooth_interpolate_kind: str = 'linear',
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
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
        fig, ax = utils.setup_plot(**kwargs, ax_set={'yscale': 'log'}, **utils.drop_None(title=title, xlabel=xlabel))
        sns.lineplot(x=r_bins, y=smoothed_density, ax=ax)
        return fig, ax

    def plot_scattering_rounds_amount(
        self,
        xlabel: str | None = 'Time',
        ylabel: str | None = 'Number of scattering rounds',
        time_units: UnitLike = 'Gyr',
        title: str | None = 'Number of scattering rounds per time step',
        smooth_sigma: float = 1,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        time_units = self.fill_time_unit(time_units)
        xlabel = utils.add_label_unit(xlabel, time_units)
        fig, ax = utils.setup_plot(**kwargs, **utils.drop_None(title=title, xlabel=xlabel, ylabel=ylabel))
        rounds = np.array(self.scatter_rounds)
        smoothed_rounds = scipy.ndimage.gaussian_filter1d(rounds, sigma=smooth_sigma) if smooth_sigma > 0 else rounds
        time = (np.arange(len(rounds)) * self.dt).to(time_units)
        sns.lineplot(x=time, y=smoothed_rounds, ax=ax)
        return fig, ax

    def plot_required_scatter_rounds_by_range(
        self,
        x_range: Quantity['length'] | None = None,
        xlabel: str | None = 'Radius',
        ylabel: str | None = 'Number of required scattering rounds',
        title: str | None = r'Number of required scattering rounds to match $\kappa$={kappa} rate after t={time}',
        x_units: UnitLike = 'kpc',
        time_units: UnitLike = 'Gyr',
        time_format: str = '.1f',
        smooth_sigma: float = 50,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        self.sort_particles()
        time_units = self.fill_time_unit(time_units)
        x = self.r
        rounds_uncapped = self.required_scatter_rounds_uncapped
        smoothed_rounds_uncapped = scipy.ndimage.gaussian_filter1d(rounds_uncapped, sigma=smooth_sigma) if smooth_sigma > 0 else rounds_uncapped
        rounds = self.required_scatter_rounds
        smoothed_rounds = scipy.ndimage.gaussian_filter1d(rounds, sigma=smooth_sigma) if smooth_sigma > 0 else rounds
        if x_range is not None:
            smoothed_rounds_uncapped = smoothed_rounds_uncapped[(x > x_range[0]) * (x < x_range[1])]
            smoothed_rounds = smoothed_rounds[(x > x_range[0]) * (x < x_range[1])]
            x = x[(x > x_range[0]) * (x < x_range[1])]
        if title is not None:
            title = title.format(
                kappa=self.scatter_params.get('kappa', None),
                time=self.time.to(time_units).to_string(format='latex', formatter=time_format),
            )
        xlabel = utils.add_label_unit(xlabel, x_units)
        fig, ax = utils.setup_plot(**kwargs, **utils.drop_None(title=title, xlabel=xlabel, ylabel=ylabel))
        sns.lineplot(x=x, y=smoothed_rounds_uncapped, label='Required', ax=ax)
        sns.lineplot(x=x, y=smoothed_rounds, label='After cap', ax=ax)
        ax.legend()
        return fig, ax

    def plot_scattering_timescale_by_range(
        self,
        x_range: Quantity['length'] | None = None,
        xlabel: str | None = 'Radius',
        ylabel: str | None = r'$t_{\mathrm{scatter}}$',
        title: str | None = r'Scattering timescale after t={time}',
        x_units: UnitLike = 'kpc',
        title_units: UnitLike = 'Gyr',
        time_units: UnitLike = 'Myr',
        title_time_format: str = '.1f',
        smooth_sigma: float = 50,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        self.sort_particles()
        x = self.r
        time_units = self.fill_time_unit(time_units)
        title_units = self.fill_time_unit(title_units)
        time = self.scatter_timescale.to(time_units)
        smoothed_time = scipy.ndimage.gaussian_filter1d(time, sigma=smooth_sigma) if smooth_sigma > 0 else time
        if x_range is not None:
            smoothed_time = smoothed_time[(x > x_range[0]) * (x < x_range[1])]
            x = x[(x > x_range[0]) * (x < x_range[1])]
        if title is not None:
            title = title.format(time=self.time.to(title_units).to_string(format='latex', formatter=title_time_format))
        xlabel = utils.add_label_unit(xlabel, x_units)
        ylabel = utils.add_label_unit(ylabel, time_units)
        fig, ax = utils.setup_plot(**kwargs, **utils.drop_None(title=title, xlabel=xlabel, ylabel=ylabel))
        sns.lineplot(x=x, y=smoothed_time, ax=ax)
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
        self.sort_particles()
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
        fig, ax = utils.setup_plot(**kwargs, **utils.drop_None(title=title, xlabel=xlabel, ylabel=ylabel))
        sns.lineplot(x=x, y=smoothed_local_density, ax=ax)
        return fig, ax

    def plot_required_scatter_rounds_distribution(
        self,
        xlabel: str | None = 'Number of required scattering rounds',
        title: str | None = r'Number of required scattering rounds to match $\kappa$={kappa} rate after t={time}',
        time_units: UnitLike = 'Gyr',
        time_format: str = '.1f',
        log_scale: bool = True,
        stat: str = 'density',
        cumulative: bool = False,
        hist_kwargs: Any = {},
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        self.sort_particles()
        time_units = self.fill_time_unit(time_units)
        if title is not None:
            title = title.format(
                kappa=self.scatter_params.get('kappa', None),
                time=self.time.to(time_units).to_string(format='latex', formatter=time_format),
            )
        fig, ax = utils.setup_plot(**kwargs, **utils.drop_None(title=title, xlabel=xlabel))
        sns.histplot(self.required_scatter_rounds, log_scale=log_scale, stat=stat, cumulative=cumulative, **hist_kwargs)
        return fig, ax

    def plot_scattering_timescale_distribution(
        self,
        xlabel: str | None = r'$t_{\mathrm{scatter}}$',
        title: str | None = r'Scattering timescale after t={time}',
        time_units: UnitLike = 'Myr',
        title_units: UnitLike = 'Gyr',
        title_time_format: str = '.1f',
        log_scale: bool = True,
        stat: str = 'density',
        cumulative: bool = False,
        hist_kwargs: Any = {},
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        self.sort_particles()
        time_units = self.fill_time_unit(time_units)
        title_units = self.fill_time_unit(title_units)
        xlabel = utils.add_label_unit(xlabel, time_units)
        if title is not None:
            title = title.format(time=self.time.to(title_units).to_string(format='latex', formatter=title_time_format))
        fig, ax = utils.setup_plot(**kwargs, **utils.drop_None(title=title, xlabel=xlabel))
        sns.histplot(self.scatter_timescale.to(time_units), log_scale=log_scale, stat=stat, cumulative=cumulative, **hist_kwargs)
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
        self.sort_particles()
        time_units = self.fill_time_unit(time_units)
        xlabel = utils.add_label_unit(xlabel, density_units)
        if title is not None:
            title = title.format(
                nn=self.scatter_params.get('max_radius_j', None),
                time=self.time.to(time_units).to_string(format='latex', formatter=time_format),
            )
        fig, ax = utils.setup_plot(**kwargs, **utils.drop_None(title=title, xlabel=xlabel))
        sns.histplot(self.local_density.to(density_units), log_scale=log_scale, stat=stat, cumulative=cumulative, **hist_kwargs)
        return fig, ax
