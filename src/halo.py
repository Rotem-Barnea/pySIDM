import itertools
import shutil
import time
from datetime import datetime
from collections import deque
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import pickle
import scipy
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from PIL import Image
from numpy.typing import NDArray
from typing import Any, Self, cast, Literal
from astropy import table
from astropy.units import Quantity, Unit, def_unit
from astropy.units.typing import UnitLike
from .distribution.distribution import Distribution
from .background import Mass_Distribution
from . import utils, run_units, physics, plot
from .physics import sidm, leapfrog
from .types import ParticleType
from .tqdm import tqdm


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
        distributions: list[Distribution] = [],
        scatter_rounds: list[int] = [],
        scatter_track: deque[NDArray[np.float64]] = deque(),
        time: Quantity['time'] = 0 * run_units.time,
        background: Mass_Distribution | None = None,
        last_saved_time: Quantity['time'] = 0 * run_units.time,
        save_every_time: Quantity['time'] | None = None,
        save_every_n_steps: int | None = None,
        dynamics_params: leapfrog.Params = {},
        scatter_params: sidm.Params = {},
        snapshots: table.QTable = table.QTable(),
        hard_save: bool = False,
        save_path: Path | str | None = None,
        Rmax: Quantity['length'] = Quantity(300, 'kpc'),
        scatters_to_collapse: int = 340,
        cleanup_nullish_particles: bool = False,
        cleanup_particles_by_radius: bool = False,
        runtime_realtime_track: deque[float] = deque(),
        runtime_track_sort: deque[float] = deque(),
        runtime_track_cleanup: deque[float] = deque(),
        runtime_track_sidm: deque[float] = deque(),
        runtime_track_leapfrog: deque[float] = deque(),
        runtime_track_full_step: deque[float] = deque(),
    ) -> None:
        """Initialize a Halo object.

        Parameters:
            r: Radius of the halo particles.
            v: Velocity of the halo particles, of shape `(n_particles, 3)`, `(vx,vy,vr)` with `vx`,`vy` the two perpendicular components of the off-radial plane.
            m: Mass of the halo particles.
            particle_type: Type of the halo particles. Should comply with ParticleType (i.e. `dm` or `baryon`).
            Tdyn: Dynamical time of the halo. If `None` calculates from the first density.
            Phi0: Potential at infinity of the halo. If `None` calculates from the first density.
            distributions: List of distributions of the halo.
            n_interactions: Number of interactions the halo had.
            scatter_rounds: Number of scatter rounds the halo had every time step.
            scatter_track: The interacting particles (particle index) at every time step.
            time: Time of the halo.
            background: Background mass distribution of the halo.
            last_saved_time: Last time a snapshot was saved.
            save_every_time: How often should a snapshot be saved, in time units.
            save_every_n_steps: How often should a snapshot be saved, in time-step units (integer).
            dynamics_params: Dynamics parameters of the halo, sent to the leapfrog integrator.
            scatter_params: Scatter parameters of the halo, used in the SIDM calculation.
            snapshots: Snapshots of the halo.
            hard_save: Whether to save the halo to memory at every snapshot save, or just keep in RAM.
            save_path: Path to save the halo to memory.
            Rmax: Maximum radius of the halo, particles outside of this radius get killed off. If `None` ignores.
            scatters_to_collapse: Number of scatters required on average for every dark matter particle to reach core collapse. Only used for estimating core collapse time for the early stopping mechanism, has no effect on the physical calculation (which will reach core-collapse on its own independently).
            cleanup_nullish_particles: Whether to remove particles from the halo after each interaction if they are nullish.
            cleanup_particles_by_radius: Whether to remove particles from the halo based on their radius (r >= `Rmax`).

        Returns:
            Halo object.
        """
        self._particles = self.to_dataframe(r, v, m, particle_type)
        self._particles.sort_values('r', inplace=True)
        self.time: Quantity['time'] = time.to(run_units.time)
        self.dt: Quantity['time'] = dt.to(run_units.time)
        self.distributions: list[Distribution] = distributions
        self.Tdyn: Quantity['time']
        if Tdyn is not None:
            self.Tdyn = Tdyn
        elif len(self.distributions) > 0:
            self.Tdyn = self.distributions[0].Tdyn
        elif len(self.distributions) == 0:
            self.Tdyn = Quantity(1, run_units.time)
        self.background: Mass_Distribution | None = background
        self.Phi0: Quantity['energy'] = Phi0 if Phi0 is not None else physics.utils.Phi(self.r, self.M, self.m)[-1]
        self.snapshots: table.QTable = snapshots
        self.save_every_n_steps = save_every_n_steps
        self.save_every_time: Quantity['time'] | None = save_every_time if save_every_time is None else save_every_time.to(run_units.time)
        self._dynamics_params: leapfrog.Params = leapfrog.normalize_params(dynamics_params, add_defaults=True)
        self._scatter_params: sidm.Params = sidm.normalize_params(scatter_params, add_defaults=True)
        self.scatter_track = scatter_track
        self._initial_particles = self._particles.copy()
        self.initial_particles = self.particles.copy()
        self.last_saved_time = last_saved_time
        self.scatter_rounds = scatter_rounds
        self.hard_save: bool = hard_save
        self.save_path: Path | str | None = save_path
        self.Rmax: Quantity['length'] = Rmax.to(run_units.length)
        self.scatters_to_collapse: int = scatters_to_collapse
        self.cleanup_nullish_particles = cleanup_nullish_particles
        self.cleanup_particles_by_radius = cleanup_particles_by_radius
        self.runtime_realtime_track = runtime_realtime_track
        self.runtime_track_sort = runtime_track_sort
        self.runtime_track_cleanup = runtime_track_cleanup
        self.runtime_track_sidm = runtime_track_sidm
        self.runtime_track_leapfrog = runtime_track_leapfrog
        self.runtime_track_full_step = runtime_track_full_step

    @classmethod
    def setup(cls, distributions: list[Distribution], n_particles: list[int | float], **kwargs: Any) -> Self:
        """Initialize a Halo object from a given set of distributions.

        Parameters:
            distributions: List of distributions for each particle type.
            n_particles: List of number of particles for each particle type.
            kwargs: Additional keyword arguments, passed to the constructor.

        Returns:
            Halo object.
        """
        r, v, particle_type, m = [], [], [], []
        Distribution.merge_distribution_grids(distributions)
        for distribution, n in zip(distributions, n_particles):
            r_sub = distribution.roll_r(int(n)).to(run_units.length)
            v_sub = distribution.roll_v_3d(r_sub).to(run_units.velocity)
            r += [r_sub]
            v += [v_sub]
            particle_type += [[distribution.particle_type] * int(n)]
            m += [np.ones(int(n)) * distribution.Mtot / n]

        return cls(
            r=cast(Quantity, np.hstack(r)),
            v=cast(Quantity, np.vstack(v)),
            m=cast(Quantity, np.hstack(m)),
            particle_type=np.hstack(particle_type),
            distributions=distributions,
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
        """Convert particle data to a `DataFrame`."""
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
        data['interacting'] = data['particle_type'] == 'dm'
        data.set_index('particle_index', inplace=True)
        return data

    def add_background(self, background: Mass_Distribution) -> None:
        """Adds a background mass distribution to the halo."""
        self.background = background

    def reset(self) -> None:
        """Resets the halo to its initial state (no interactions, `time`=0, cleared snapshots, particles at initial positions)."""
        self.time = 0 * run_units.time
        self._particles = self._initial_particles.copy()
        self.scatter_track = []
        self.snapshots = table.QTable()
        self.runtime_track_sort = deque()
        self.runtime_track_cleanup = deque()
        self.runtime_track_sidm = deque()
        self.runtime_track_leapfrog = deque()
        self.runtime_track_full_step = deque()

    @property
    def particles(self) -> table.QTable:
        """Particle data QTable.

        Has the following columns:
            r: Radius.
            vx: The first pernpendicular component (to the radial direction) of the velocity.
            vy: The second pernpendicular component (to the radial direction) of the velocity.
            vr: The radial velocity.
            vp: Tangential velocity (`np.sqrt(vx**2 + vy**2)`).
            m: Mass.
            v_norm: Velocity norm (`np.sqrt(vx**2 + vy**2 + vr**2)`).
            time: Current simulation time.
            E: Relative energy (`Psi-1/2*m*v_norm**2`).
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

    def get_particle_states(self, now: bool = True, snapshots: bool = True, initial: bool = True):
        """Return a table of particle snapshots, potentially including the initial and current states."""
        assert now or snapshots or initial, 'At least one of now, snapshots, or initial must be True'
        data_tables = []
        if now:
            data_tables += [self.particles]
        if snapshots:
            data_tables += [self.snapshots]
        if initial:
            data_tables += [self.initial_particles]
        return table.QTable(table.vstack(data_tables))

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

    def cleanup_particles(self, presorted: bool = True) -> None:
        """Cleanup the particles by dropping nullish values and particles outside the radius.

        Significantly faster if the dataframe is presorted.
        """
        if self.cleanup_nullish_particles or self.cleanup_particles_by_radius:
            drop_indices = pd.Series(data=np.zeros(len(self._particles), dtype=np.bool_), index=self._particles.index)
            if self.cleanup_nullish_particles:
                drop_indices += self._particles['r'].isna()
            if self.cleanup_particles_by_radius:
                drop_indices += self._particles['r'] > self.Rmax.value
            if drop_indices.any():
                if presorted:
                    end = drop_indices.argmax()
                    self._particles = self._particles.iloc[:end]
                else:
                    self._particles.drop(index=drop_indices[drop_indices].index, inplace=True)

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
        """The velocity of the particle, as a 3-vector `(vx, vy, vr)`."""
        return Quantity(self._particles[['vx', 'vy', 'vr']], run_units.velocity)

    @property
    def time_step(self) -> Unit:
        """Calculate the time step size, returning it as a `Unit` object"""
        return def_unit('time step', self.dt.to(run_units.time), format={'latex': r'time\ step'})

    @property
    def M(self) -> Quantity['mass']:
        """The enclosed mass below the particle."""
        halo_mass = physics.utils.M(r=self.r, m=self.m)
        if self.background is not None:
            background_mass = self.background.M_at_time(self.r, self.time)
            return cast(Quantity, halo_mass + background_mass)
        return halo_mass

    @property
    def vp(self) -> Quantity['velocity']:
        """The tangential velocity of the particle."""
        return utils.fast_quantity_norm(cast(Quantity, self.v[:, :2]))

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
        return cast(Quantity, physics.utils.Phi(self.r, self.M, self.m))

    @property
    def Psi(self) -> Quantity['specific energy']:
        """The relative gravitational potential energy of the particle."""
        return cast(Quantity, physics.utils.Psi(self.r, self.M, self.m)).to(run_units.energy)
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
    def n_scatters(self) -> NDArray[np.int64]:
        """The number of scatters every scattering round."""
        return np.array([len(x) / 2 for x in self.scatter_track], dtype=np.int64)

    @property
    def n_particles(self) -> dict[str, int]:
        """The total number of particles of every type in the halo."""
        return self._particles['particle_type'].value_counts().to_dict()

    @property
    def core_collapse_time(self) -> Quantity['time']:
        """Time at which the halo underwent core collapse, defined by on average every dm particle undergoing `scatters_to_collapse` events."""
        return (self.n_scatters.cumsum() < self.scatters_to_collapse * self.n_particles['dm']).argmin() * self.dt

    @property
    def runtime_track(self):
        """Runtime tracking of the simulation."""
        return pd.DataFrame(
            itertools.zip_longest(
                self.runtime_track_sort,
                self.runtime_track_cleanup,
                self.runtime_track_sidm,
                self.runtime_track_leapfrog,
                self.runtime_track_full_step,
                self.runtime_realtime_track,
                fillvalue=np.nan,
            ),
            columns=['sort', 'cleanup', 'sidm', 'leapfrog', 'full step', 'real timestep'],
        )

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

    def save_snapshot(self, **kwargs: Any) -> None:
        """Save the current state of the simulation."""
        data = self.particles.copy()
        data['step'] = self.current_step
        self.snapshots = table.vstack([self.snapshots, data])
        self.last_saved_time = self.time.copy()
        if self.hard_save:
            self.save(**kwargs)

    def is_save_round(self) -> bool:
        """Check if it's time to save the simulation state."""
        if self.save_every_time is not None:
            next_save_time = self.last_saved_time + self.save_every_time
            if self.time <= next_save_time and self.time + self.dt > next_save_time:
                return True
        elif self.save_every_n_steps is not None and self.current_step % self.save_every_n_steps == 0:
            return True
        return False

    def step(self, save_kwargs: dict[str, Any] = {}) -> None:
        """Perform a single time step of the simulation.

        Every step:
            - Sort particles by radius.
            - Cleanup erroneous particles.
            - Save a snapshot if it's time.
            - Perform scattering. This is done before the leapfrog integration since it doesn't modify the particle positions and thus doesn't require resorting.
            - Perform leapfrog integration.
            - Update simulation time.
        """
        self.runtime_realtime_track += [datetime.now().timestamp()]
        t_start = time.perf_counter()
        t0 = time.perf_counter()
        self._particles.sort_values('r', inplace=True)
        self.runtime_track_sort += [time.perf_counter() - t0]
        t0 = time.perf_counter()
        self.cleanup_particles()
        self.runtime_track_cleanup += [time.perf_counter() - t0]
        if self.is_save_round():
            self.save_snapshot(**save_kwargs)

        r, vx, vy, vr, m = self._particles[['r', 'vx', 'vy', 'vr', 'm']].values.T

        if self.scatter_params.get('sigma', Quantity(0, run_units.cross_section)) > Quantity(0, run_units.cross_section):
            t0 = time.perf_counter()
            mask = self._particles['interacting'].values
            (vx[mask], vy[mask], vr[mask], indices, scatter_rounds) = sidm.scatter(
                r=r[mask], vx=vx[mask], vy=vy[mask], vr=vr[mask], dt=self.dt, m=m[mask], **self.scatter_params
            )
            self.scatter_track += [self.r[mask][indices]]
            self.scatter_rounds += [scatter_rounds]
            self.runtime_track_sidm += [time.perf_counter() - t0]
        t0 = time.perf_counter()
        r, vx, vy, vr = leapfrog.step(r=r, vx=vx, vy=vy, vr=vr, m=m, M=self.M, dt=self.dt, **self.dynamics_params)

        self._particles['r'] = r
        self._particles['vx'] = vx
        self._particles['vy'] = vy
        self._particles['vr'] = vr

        self.runtime_track_leapfrog += [time.perf_counter() - t0]
        self.time += self.dt
        self.runtime_track_full_step += [time.perf_counter() - t_start]

    def evolve(
        self,
        n_steps: int | None = None,
        t: Quantity['time'] | None = None,
        until_t: Quantity['time'] | None = None,
        tqdm_kwargs: dict[str, Any] = {},
        save_kwargs: dict[str, Any] = {},
        t_after_core_collapse: Quantity['time'] = Quantity(-1, 'Myr'),
    ) -> None:
        """Evolve the simulation for a given number of steps or time.

        t_after_core_collapse IS DEPRECATED FOR NOW!!!

        Parameters:
            n_steps: Number of steps to evolve the simulation for. Takes precedence over `t`.
            t: Time to evolve the simulation for. Ignored if `n_steps` is specified, otherwise transformed into steps using `to_steps()`.
            until_t: Evolve the simulation until this time. Ignored if `n_steps` or `t` are specified, otherwise transformed into steps using `to_steps()`.
            tqdm_kwargs: Additional keyword arguments to pass to `tqdm` (NOTE this is the custom submodule defined in this project at `tqdm.py`).
            save_kwargs: Additional keyword arguments to pass to `save()`.
            t_after_core_collapse: Time after core collapse to evolve the simulation for, afterwhich the simulation will stop (early quit). If negative ignore the core collapse check.

        Returns:
            None
        """
        if n_steps is None:
            if t is not None:
                n_steps = self.to_step(t)
            elif until_t is not None:
                if self.time > until_t:
                    raise ValueError('current time is greater than the specified end time')
                n_steps = self.to_step(cast(Quantity, until_t - self.time))
            else:
                raise ValueError('Either `n_steps`, `t`, or `until_t` must be specified')
        for _ in tqdm(range(n_steps), start_time=self.time, dt=self.dt, **tqdm_kwargs):
            self.step()
            if np.sign(t_after_core_collapse) >= 0 and self.n_scatters.sum() > self.scatters_to_collapse * self.n_particles['dm']:
                if self.time > self.core_collapse_time + t_after_core_collapse:
                    print(f'Core collapse detected at time {self.time}')
                    break
        if self.hard_save:
            self.save(**save_kwargs)

    #####################
    ##Save/Load
    #####################

    @staticmethod
    def payload_keys() -> list[str]:
        """Return the keys of the payload dictionary, used for saving and loading halos. A `@staticmethod` and not a `@property` to allow getting it from an uninitialized cls during `@classmethod`."""
        return [
            'time',
            'dt',
            'distributions',
            'save_every_n_steps',
            'save_every_time',
            'dynamics_params',
            'scatter_params',
            'scatter_track',
            'background',
            'last_saved_time',
            'scatter_rounds',
            'hard_save',
            'save_path',
            'Rmax',
            'scatters_to_collapse',
            'cleanup_nullish_particles',
            'cleanup_particles_by_radius',
            'runtime_track_sort',
            'runtime_track_cleanup',
            'runtime_track_sidm',
            'runtime_track_leapfrog',
            'runtime_track_full_step',
        ]

    @staticmethod
    def save_table(data: table.QTable, path: str | Path, **kwargs: Any) -> None:
        """Save a QTable to a file, splitting the strings from the Quantity data, and saving into `{}_strings.csv` and `{}.fits`."""
        data[[column for column in data.colnames if data[column].dtype != np.dtype('O')]].write(path.with_name(f'{path.stem}.fits'), **kwargs)
        data[[column for column in data.colnames if data[column].dtype == np.dtype('O')]].write(path.with_name(f'strings_{path.stem}.csv'), **kwargs)

    @staticmethod
    def load_table(path: str | Path) -> table.QTable:
        """Load a QTable saved via `save_table()`."""
        fits_table = table.QTable.read(path.with_name(f'{path.stem}.fits'))
        csv_table = table.QTable.read(path.with_name(f'strings_{path.stem}.csv'))
        for col in fits_table.colnames:
            fits_table[col] = fits_table[col].astype(fits_table[col].dtype.newbyteorder('='), copy=False)
        for col in csv_table.colnames:
            csv_table[col] = np.array(csv_table[col]).astype('O')
        return cast(table.QTable, table.hstack([fits_table, csv_table]))

    def save(self, path: str | Path | None = None, two_steps: bool = True, keep_last_backup: bool = True, split_snapshots: bool = True) -> None:
        """Save the simulation state to a directory.

        Parameters:
            path: Save path. If `path` is None attempts to use the internal save path.
            two_steps: If `True` saves the simulation state in two steps, to avoid rewriting the existing file with data that can be stopped midway (leaving just the 1 corrupted file). This means that for the duration of the saving the disk size used is doubled.
            keep_last_backup: If `True` keeps a full backup of the previous save, otherwise overwrite it based on `two_steps` rules. This option _always_ uses twice the disk space.
            split_snapshots: If `True` saves the snapshots QTable as separate files.

        Returns:
            None
        """
        if path is None:
            path = self.save_path
        assert path is not None, 'Save path must be provided'
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        if keep_last_backup:
            for file in path.glob('*'):
                shutil.copyfile(file, file.with_stem(f'{file.stem}_backup'))
        payload = {key: getattr(self, key) for key in self.payload_keys()}
        tables = {'particles': self.particles, 'initial_particles': self.initial_particles}
        if not split_snapshots:
            tables['snapshots'] = self.snapshots
        tag = '_' if two_steps else ''
        with open(path / f'halo_payload{tag}.pkl', 'wb') as f:
            pickle.dump(payload, f)
        for name, data in tables.items():
            self.save_table(data, path / f'{name}{tag}.fits', overwrite=True)
        for file in path.glob('*_.*'):
            file.rename(file.with_stem(file.stem[:-1]))
        if split_snapshots:
            (path / 'split_snapshots').mkdir()
            for i, group in enumerate(self.snapshots.group_by('time').groups):
                self.save_table(group, path / f'split_snapshots/snapshot_{i}.fits', overwrite=True)

    @classmethod
    def load(cls, path: str | Path = Path('halo_state'), update_save_path: bool = True) -> Self:
        """Load the simulation state from a directory."""
        path = Path(path)
        with open(path / 'halo_payload.pkl', 'rb') as f:
            payload = {key: value for key, value in pickle.load(f).items() if key in cls.payload_keys()}

        tables = {}

        if (path / 'split_snapshots').exists():
            tables['snapshots'] = cast(table.QTable, table.vstack([cls.load_table(file) for file in (path / 'split_snapshots').glob('*.fits')]))
        else:
            tables['snapshots'] = cls.load_table(path / 'snapshots.fits')

        for name in ['particles', 'initial_particles']:
            tables[name] = cls.load_table(path / f'{name}.fits')
        particles = tables['particles']
        particles.sort('particle_index')
        output = cls(
            r=particles['r'],
            v=cast(Quantity, np.vstack([particles['vx'], particles['vy'], particles['vr']]).T),
            particle_type=particles['particle_type'],
            m=particles['m'],
            **payload,
            snapshots=tables['snapshots'],
        )
        output.initial_particles = tables['initial_particles']
        if update_save_path:
            output.save_path = path.resolve()
        return output

    #####################
    ##Plots
    #####################

    def fill_time_unit(self, unit: UnitLike) -> UnitLike:
        """If the `unit` is `Tdyn` return `self.Tdyn`. If it's `time step` return `self.time_step`, otherwise return `unit`."""
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
        x_clip: Quantity['length'] | None = None,
        x_units: UnitLike = 'kpc',
        time_units: UnitLike = 'Tdyn',
        time_format: str = '.1f',
        title: str | None = 'Density progression over time',
        xlabel: str | None = 'Radius',
        indices: list[int] | None = None,
        color_palette: str | None = None,
        legend_loc: str | None = 'outside center right',
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the density progression over time.

        Parameters:
            include_start: Whether to include the initial particle distribution in the plot.
            include_now: Whether to include the current particle distribution in the plot.
            filter_particle_type: Whether to filter to only plot the specified particle type.
            x_range: The radius range to clip the data to (hard slicing the source data prior to plotting). If `None`, ignores.
            x_clip: The radius range to clip the data to (using the `clip` keyword argument in `sns.kdeplot()`). If `None`, ignores.
            time_units: The time units to use in the plot.
            time_format: Format string for time.
            title: The title of the plot.
            xlabel: The label for the x-axis.
            indices: The snapshot indices to plot. If `None` plots everything.
            color_palette: The color palette to use for the halos. If `None`, uses the default color palette.
            legend_loc: The location of the legend. If `None` uses the default location.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.setup_plot()`).

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

        colors = sns.color_palette(color_palette, len(indices)) if color_palette is not None else None

        time_units = self.fill_time_unit(time_units)
        fig, ax = plot.setup_plot(**utils.drop_None(title=title, xlabel=utils.add_label_unit(xlabel, x_units)), **kwargs)
        clip = tuple(x_clip.to(x_units).value) if x_clip is not None else None
        for i, group in enumerate(data.group_by('time').groups):
            if i not in indices:
                continue
            sub = group['r'].to(x_units).value
            if x_range is not None:
                sub = sub[pd.Series(sub).between(*x_range.to(x_units).value)]
            sns.kdeplot(
                sub,
                ax=ax,
                clip=clip,
                color=colors[indices.index(i)] if colors is not None else None,
                label=group['time'][0].to(time_units).to_string(format='latex', formatter=time_format),
            )
        if legend_loc:
            fig.legend(loc=legend_loc)
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
            x_range: The radius range to clip the data to. If `None` ignores.
            x_plot_range: The range to plot on the x-axis. If `None` uses the data range.
            stat: The type of statistic to plot. Gets passed to `sns.histplot()`. Only used if `plot_type` is `hist`.
            plot_type: The type of plot to create.
            x_units: The x-axis units to use in the plot.
            ylabel: The label for the y-axis.
            label: The label for the histogram (legend).
            fig: The figure to plot on.
            ax: The axes to plot on.
            plt_kwargs: Additional keyword arguments to pass to the sns plotting function (`sns.histplot()` or `sns.kdeplot()`).
            kwargs: Additional keyword arguments to pass to `plot.setup_plot()`.

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
        add_density: int | None = 0,
        x_units: UnitLike = 'kpc',
        x_range: Quantity | None = None,
        hist_label: str | None = None,
        density_label: str | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the radial distribution of the halo. Wraps `plot_distribution()` with additional options.

        Parameters:
            data: The data to plot.
            cumulative: Whether to plot the cumulative distribution.
            add_density: Density distribution to plot on top of the plot (index from the distributions list). If `None` ignores.
            x_units: The units to plot the x-axis in.
            x_range: The range of the x-axis.
            hist_label: The label for the histogram (legend).
            density_label: The label for the density distribution (legend).
            kwargs: Additional keyword arguments to pass to `plot.setup_plot()`.

        Returns:
            fig, ax.
        """
        fig, ax = self.plot_distribution(key='r', data=data, cumulative=cumulative, x_units=x_units, x_range=x_range, label=hist_label, **kwargs)
        if add_density is not None:
            params: dict[str, Any] = {'r_start': cast(Quantity, x_range[0]), 'r_end': cast(Quantity, x_range[1])} if x_range is not None else {}
            return self.distributions[add_density].plot_radius_distribution(
                cumulative=cumulative,
                length_units=x_units,
                fig=fig,
                ax=ax,
                label=density_label,
                **params,
            )
        return fig, ax

    def plot_phase_space_evolution(
        self,
        include_start: bool = True,
        include_now: bool = True,
        filter_particle_type: ParticleType | None = None,
        save_path: str | Path | None = None,
        frame_plot_kwargs: dict[str, Any] = {},
        save_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> list[Image.Image]:
        """Plot the phase space evolution of the halo.

        Parameters:
            include_start: Whether to include the initial state.
            include_now: Whether to include the current state.
            filter_particle_type: Whether to filter to only plot the specified particle type.
            save_path: Path to save the images. If `None` images are not saved.
            frame_plot_kwargs: Additional keyword arguments for the frame plot (`plot_phase_space()`).
            save_kwargs: Additional keyword arguments for saving the images (`plot.save_images()`).
            **kwargs: Additional keyword arguments for transforming the frames to images (`plot.to_images()`).

        Returns:
            The list of frames.
        """
        data = self.get_particle_states(now=include_now, initial=include_start)
        if filter_particle_type is not None:
            data = cast(table.QTable, data[data['particle_type'] == filter_particle_type]).copy()

        images = plot.evolution_to_images(
            data=data,
            plot_fn=lambda x: self.plot_phase_space(
                data=x,
                texts=[{'s': f'{x["time"][0].to("Gyr"):.2f}', **plot.pretty_ax_text(x=0.05, y=0.95, transform='transAxes')}],
                **frame_plot_kwargs,
            ),
            **kwargs,
        )
        if save_path is not None:
            plot.save_images(images=images, save_path=save_path, **save_kwargs)
        return images

    def plot_phase_space(
        self,
        data: table.QTable,
        filter_particle_type: ParticleType | None = None,
        radius_bins: Quantity['length'] = Quantity(np.linspace(1e-2, 50, 200), 'kpc'),
        velocity_bins: Quantity['velocity'] = Quantity(np.linspace(0, 60, 200), 'km/second'),
        cmap: str = 'jet',
        transparent_value: float | None = 0,
        length_units: UnitLike = 'kpc',
        velocity_units: UnitLike = 'km/second',
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the phase space distribution of the data.

        Parameters:
            data: The data to plot.
            filter_particle_type: Whether to filter to only plot the specified particle type.
            radius_bins: The bins for the radius axis. Also used to define the radius range to consider.
            velocity_bins: The bins for the velocity axis. Also used to define the velocity range to consider.
            cmap: The colormap to use for the plot.
            transparent_value: Grid value to turn transparent (i.e. plot as `NaN`). If `None` ignores.
            length_units: Units to use for the radius axis.
            velocity_units: Units to use for the velocity axis.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.plot_2d()`).

        Returns:
            fig, ax.
        """

        if filter_particle_type is not None:
            data = cast(table.QTable, data[data['particle_type'] == filter_particle_type]).copy()
        grid, extent = plot.aggregate_phase_space_data(data=data, radius_bins=radius_bins, velocity_bins=velocity_bins)
        return plot.plot_2d(
            grid=grid,
            extent=extent,
            xlabel='Radius',
            ylabel='Velocity',
            x_units=length_units,
            y_units=velocity_units,
            cmap=cmap,
            transparent_value=transparent_value,
            **kwargs,
        )

    def plot_inner_core_density(
        self,
        include_start: bool = True,
        include_now: bool = True,
        radius: Quantity['length'] = Quantity(0.2, 'kpc'),
        filter_particle_type: ParticleType | None = None,
        time_units: UnitLike = 'Tdyn',
        xlabel: str | None = 'time',
        ylabel: str | None = 'particles',
        title: str | None = 'particles in inner core ({radius})',
        aggregation_type: Literal['amount', 'percent'] = 'amount',
        title_radius_format: str = '.1f',
        label: str | None = None,
        fig: Figure | None = None,
        ax: Axes | None = None,
        line_kwargs: dict[str, Any] = {},
        **kwargs: Any,
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
            aggregation_type: Type of aggregation to use for the plot.
            title_radius_format: Format string for the title radius.
            label: Label for the plot (legend).
            fig: Figure to use for the plot.
            ax: Axes to use for the plot.
            line_kwargs: Additional keyword arguments to pass to the lineplot function (`sns.lineplot()`).
            kwargs: Additional keyword arguments to pass to the plot function (`plot.setup_plot()`).

        Returns:
            fig, ax.
        """
        base_data = self.get_particle_states(now=include_now, initial=include_start, snapshots=True)
        data_time_units = base_data['time'].unit
        data_length_units = base_data['r'].unit
        data = base_data[['r', 'time', 'particle_type']].to_pandas()

        if filter_particle_type is not None:
            data = data[data['particle_type'] == filter_particle_type].copy()
        time_units = self.fill_time_unit(time_units)
        data['in_radius'] = data['r'] <= radius.to(data_length_units)

        if aggregation_type == 'amount':
            agg_func = 'sum'
            prefix = '#'
        else:
            agg_func = 'mean'
            prefix = '%'

        agg_data = data.groupby('time')['in_radius'].agg(agg_func).reset_index()
        if prefix == '%':
            agg_data['in_radius'] *= 100

        if title is not None:
            title = title.format(radius=radius.to_string(format='latex', formatter=title_radius_format))
            title = f'{prefix}{title}'

        if ylabel is not None:
            ylabel = f'{prefix}{ylabel}'

        fig, ax = plot.setup_plot(fig, ax, **utils.drop_None(title=title, xlabel=utils.add_label_unit(xlabel, time_units), ylabel=ylabel), **kwargs)
        sns.lineplot(
            x=np.array(Quantity(agg_data['time'], data_time_units).to(time_units)),
            y=agg_data['in_radius'],
            ax=ax,
            label=label,
            **line_kwargs,
        )
        return fig, ax

    def plot_particle_evolution(
        self,
        include_start: bool = True,
        include_now: bool = True,
        filter_particle_type: ParticleType | None = None,
        radius_bins: Quantity = Quantity(np.linspace(1e-3, 5, 100), 'kpc'),
        time_range: Quantity = Quantity([2, 17], 'Gyr'),
        length_units: UnitLike = 'kpc',
        time_units: UnitLike = 'Tdyn',
        xlabel: str | None = 'Radius',
        ylabel: str | None = 'Time',
        cbar_label: str | None = 'Particles',
        row_normalization: Literal['max', 'sum', 'integral'] | float | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the evolution of the particle position in the halo.

        Parameters:
            include_start: Whether to include the initial particle distribution in the plot.
            include_now: Whether to include the current particle distribution in the plot.
            filter_particle_type: Whether to filter to only plot the specified particle type.
            radius_bins: The bins for the radius axis. Also used to define the radius range to consider.
            time_range: Range of times to consider (filters the data).
            length_units: Units to use for the radius axis.
            time_units: Units to use for the time axis.
            xlabel: Label for the radius axis.
            ylabel: Label for the time axis.
            cbar_label: Label for the colorbar.
            row_normalization: The normalization to apply to each row. If `None` no normalization is applied. If `float` it must be a percentile value (between 0 and 1), and the normalization will be based on this quantile of each row.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.plot_2d()`).

        Returns:
            fig, ax.
        """

        time_units = self.fill_time_unit(time_units)
        data = self.get_particle_states(now=include_now, initial=include_start, snapshots=True)
        if filter_particle_type is not None:
            data = cast(table.QTable, data[data['particle_type'] == filter_particle_type].copy())
        grid, extent = plot.aggregate_evolution_data(
            data=data,
            radius_bins=radius_bins,
            time_range=time_range,
            output_type='counts',
            row_normalization=row_normalization,
        )

        return plot.plot_2d(
            grid=grid,
            extent=extent,
            x_units=length_units,
            y_units=time_units,
            xlabel=xlabel,
            ylabel=ylabel,
            cbar_label=cbar_label,
            grid_row_normalization=row_normalization,
            **kwargs,
        )

    def plot_temperature_evolution(
        self,
        include_start: bool = True,
        include_now: bool = True,
        filter_particle_type: ParticleType | None = None,
        radius_bins: Quantity = Quantity(np.linspace(1e-1, 5, 100), 'kpc'),
        time_range: Quantity = Quantity([2, 17], 'Gyr'),
        specific_energy_units: UnitLike = 'km^2/second^2',
        length_units: UnitLike = 'kpc',
        time_units: UnitLike = 'Tdyn',
        xlabel: str | None = 'Radius',
        ylabel: str | None = 'Time',
        cbar_label: str | None = r'$\propto$Temperature (velocity std)',
        row_normalization: Literal['max', 'sum', 'integral'] | float | None = None,
        cmap: str = 'jet',
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the temperature evolution of the halo. Wraps `prep_2d_data()`.

        Parameters:
            include_start: Whether to include the initial particle distribution in the plot.
            include_now: Whether to include the current particle distribution in the plot.
            filter_particle_type: Whether to filter to only plot the specified particle type.
            radius_bins: The bins for the radius axis. Also used to define the radius range to consider.
            time_range: Range of times to consider (filters the data).
            specific_energy_units: Units to use for the specific energy.
            length_units: Units to use for the radius axis.
            time_units: Units to use for the time axis.
            xlabel: Label for the radius axis.
            ylabel: Label for the time axis.
            cbar_label: Label for the colorbar.
            row_normalization: The normalization to apply to each row. If `None` no normalization is applied. If `float` it must be a percentile value (between 0 and 1), and the normalization will be based on this quantile of each row.
            cmap: The colormap to use for the plot.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.plot_2d()`).

        Returns:
            fig, ax.
        """
        time_units = self.fill_time_unit(time_units)
        data = self.get_particle_states(now=include_now, initial=include_start, snapshots=True)
        if filter_particle_type is not None:
            data = cast(table.QTable, data[data['particle_type'] == filter_particle_type].copy())
        grid, extent = plot.aggregate_evolution_data(
            data=data,
            radius_bins=radius_bins,
            time_range=time_range,
            output_type='temperature',
            row_normalization=row_normalization,
            output_grid_units=specific_energy_units,
        )

        return plot.plot_2d(
            grid=grid,
            extent=extent,
            x_units=length_units,
            y_units=time_units,
            xlabel=xlabel,
            ylabel=ylabel,
            cbar_label=cbar_label,
            grid_row_normalization=row_normalization,
            cbar_label_autosuffix=True if row_normalization is not None else False,
            cmap=cmap,
            **kwargs,
        )

    def plot_heat_flux_evolution(
        self,
        include_start: bool = True,
        include_now: bool = True,
        filter_particle_type: ParticleType | None = None,
        radius_bins: Quantity = Quantity(np.linspace(1e-3, 5, 100), 'kpc'),
        time_range: Quantity = Quantity([2, 17], 'Gyr'),
        v_axis: Literal['vx', 'vy', 'vr'] = 'vr',
        heat_units: UnitLike = '1/Myr^3',
        length_units: UnitLike = 'kpc',
        time_units: UnitLike = 'Tdyn',
        xlabel: str | None = 'Radius',
        ylabel: str | None = 'Time',
        cbar_label: str | None = 'Specific Heat flux',
        row_normalization: Literal['max', 'sum', 'integral'] | float | None = None,
        cmap: str = 'seismic',
        setup_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the temperature evolution of the halo. Wraps `prep_2d_data()`.

        Parameters:
            include_start: Whether to include the initial particle distribution in the plot.
            include_now: Whether to include the current particle distribution in the plot.
            filter_particle_type: Whether to filter to only plot the specified particle type.
            radius_bins: The bins for the radius axis. Also used to define the radius range to consider.
            time_range: Range of times to consider (filters the data).
            v_axis: The velocity to calculate the heat flux in.
            velocity_units: Units to use for the velocity.
            length_units: Units to use for the radius axis.
            time_units: Units to use for the time axis.
            xlabel: Label for the radius axis.
            ylabel: Label for the time axis.
            cbar_label: Label for the colorbar.
            row_normalization: The normalization to apply to each row. If `None` no normalization is applied. If `float` it must be a percentile value (between 0 and 1), and the normalization will be based on this quantile of each row.
            cmap: The colormap to use for the plot.
            setup_kwargs: Additional keyword arguments to pass to `utils.setup_plot()`.
            kwargs: Additional keyword arguments to pass to the plot function (`utils.plot_2d()`).

        Returns:
            fig, ax.
        """
        time_units = self.fill_time_unit(time_units)
        data = self.get_particle_states(now=include_now, initial=include_start, snapshots=True)
        if filter_particle_type is not None:
            data = cast(table.QTable, data[data['particle_type'] == filter_particle_type].copy())
        grid, extent = plot.aggregate_evolution_data(
            data=data,
            radius_bins=radius_bins,
            time_range=time_range,
            v_axis=v_axis,
            output_type='specific heat flux',
            row_normalization=row_normalization,
            output_grid_units=heat_units,
        )

        return plot.plot_2d(
            grid=grid,
            extent=extent,
            x_units=length_units,
            y_units=time_units,
            xlabel=xlabel,
            ylabel=ylabel,
            cbar_label=cbar_label,
            grid_row_normalization=row_normalization,
            cbar_label_autosuffix=True if row_normalization is not None else False,
            cmap=cmap,
            setup_kwargs=setup_kwargs,
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
            start_kwargs: Additional keyword arguments to pass to the distribution plotting function (`self.plot_distribution()`), for the start distribution only.
            end_kwargs: Additional keyword arguments to pass to the distribution plotting function (`self.plot_distribution()`), for the end distribution only.
            **kwargs: Additional keyword arguments to pass to the distribution plotting function (`self.plot_distribution()`), for *both* distributions. Overwritten by start_kwargs/end_kwargs as needed.

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
        title: str | None = 'Scattering location distribution within the first {time}, total of {n_scatters} events',
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
            title = title.format(time=self.time.to(time_units).to_string(format='latex', formatter=time_format), n_scatters=self.n_scatters.sum())
        fig, ax = plot.setup_plot(fig, ax, figsize=figsize, minorticks=True, **utils.drop_None(title=title, xlabel=xlabel))
        sns.histplot(Quantity(np.hstack(self.scatter_track), run_units.length).to(length_units), ax=ax)
        return fig, ax

    def plot_scattering_density(
        self,
        num: int = 500,
        xlabel: str | None = 'Radius',
        ylabel: str | None = 'Density',
        title: str | None = 'Scattering density within the first {time}, total of {n_scatters} events',
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
            kwargs: Additional keyword arguments to pass to the plot function (`plot.setup_plot()`).

        Returns:
            fig, ax.
        """
        r = np.hstack(self.scatter_track).to(length_units)
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
            title = title.format(time=self.time.to(time_units).to_string(format='latex', formatter=time_format), n_scatters=self.n_scatters.sum())
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
            kwargs: Additional keyword arguments to pass to the plot function (`plot.setup_plot()`).

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
            hist_kwargs: Additional keyword arguments to pass to `sns.histogram()`.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.setup_plot()`).

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
            data: The data table to plot (i.e. halo.snapshots, or an external table from an NSphere run). If `None` use the halo's snapshots + initial and current states.
            particle_index: The index of the particle to trace.
            relative: If `absolute` plot the property as is. If `relative` plot the change in the property relative to the initial value. If `relative change` plot the change in the property relative to the initial value divided by the initial value.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            title: Title for the plot.
            label: Label for the plot (legend).
            time_units: Units for the x-axis.
            y_units: Units for the y-axis.
            length_units: Units for the length.
            length_format: Format string for length.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.setup_plot()`).

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

    def plot_cumulative_scattering_amount_over_time(
        self,
        time_unit: UnitLike = 'Gyr',
        xlabel: str | None = 'Time',
        ylabel: str | None = 'Cumulative number of scattering events',
        title: str | None = 'Cumulative number of scattering events',
        label: str | None = None,
        ax_set: dict[str, Any] = {'yscale': 'log'},
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the cumulative number of scattering events over time.

        Parameters:
            time_unit: Units for the x-axis.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            title: The title of the plot.
            label: Label for the plot (legend).
            ax_set: Additional keyword arguments to pass to `Axes.set()`.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.setup_plot()`).

        Returns:
            fig, ax.
        """
        fig, ax = plot.setup_plot(xlabel=utils.add_label_unit(xlabel, time_unit), ylabel=ylabel, title=title, ax_set=ax_set, **kwargs)
        sns.lineplot(x=(np.arange(len(self.n_scatters)) * self.dt).to(time_unit), y=self.n_scatters.cumsum(), ax=ax, label=label)
        if label is not None:
            ax.legend()
        return fig, ax

    def plot_cumulative_scattering_amount_per_particle_over_time(
        self,
        time_unit: UnitLike = 'Gyr',
        xlabel: str | None = 'Time',
        ylabel: str | None = 'Cumulative number of scattering events',
        title: str | None = 'Mean cumulative number of scattering events per particle',
        label: str | None = None,
        per_dm_particle: bool = False,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the cumulative number of scattering events over time.

        Parameters:
            time_unit: Units for the x-axis.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            title: The title of the plot.
            label: Label for the plot (legend).
            per_dm_particle: If `True` plot the mean cumulative number of scattering events, i.e. devide by the number of dm particles.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.setup_plot()`).

        Returns:
            fig, ax.
        """
        fig, ax = plot.setup_plot(xlabel=utils.add_label_unit(xlabel, time_unit), ylabel=ylabel, title=title, **kwargs)
        sns.lineplot(x=(np.arange(len(self.n_scatters)) * self.dt).to(time_unit), y=self.n_scatters, ax=ax, label=label)
        if label is not None:
            ax.legend()
        return fig, ax

    def plot_binned_scattering_amount_over_time(
        self,
        time_binning: Quantity = Quantity(100, 'Myr'),
        time_unit: UnitLike = 'Gyr',
        xlabel: str | None = 'Time',
        ylabel: str | None = 'Number of scattering events',
        title: str | None = 'Number of scattering events over time per {time}',
        label: str | None = None,
        time_format: str | None = None,
        title_time_unit: str | None = 'Myr',
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the number of scattering events over time, binned.

        Parameters:
            time_unit: Units for the x-axis.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            label: Label for the plot (legend).
            kwargs: Additional keyword arguments to pass to the plot function (`plot.setup_plot()`).

        Returns:
            fig, ax.
        """
        n = int(time_binning / self.dt)
        x = (np.arange(len(self.n_scatters)) * self.dt).to(time_unit)[::n]
        scatters = np.add.reduceat(self.n_scatters, np.arange(0, len(self.n_scatters), n))

        if title is not None:
            title = title.format(time=time_binning.to(title_time_unit).to_string(format='latex', formatter=time_format))

        fig, ax = plot.setup_plot(xlabel=utils.add_label_unit(xlabel, time_unit), ylabel=ylabel, title=title, **kwargs)
        sns.lineplot(x=x, y=scatters, ax=ax, label=label)
        if label is not None:
            ax.legend()
        return fig, ax

    def plot_distributions_rho(self, markers_on_first_only: bool = False, **kwargs: Any) -> tuple[Figure, Axes]:
        """Plot the density profile (`rho`) of each of the provided distributions in the halo.

        Parameters:
            markers_on_first_only: If `True` only plot markers (`Rs` and `Rvir`) for the first density.
            kwargs: Additional keyword arguments are passed to every call to the plotting function.

        Returns:
            fig, ax.
        """
        fig, ax = None, None
        for i, distribution in enumerate(self.distributions):
            fig, ax = distribution.plot_rho(
                label=f'{distribution.label} ({distribution.title})', fig=fig, ax=ax, add_markers=(i == 0 or not markers_on_first_only), **kwargs
            )
        assert fig is not None and ax is not None
        return fig, ax

    def plot_distributions_over_time(
        self,
        times: Quantity['time'] = Quantity([0, 2, 10], 'Gyr'),
        labels: list[str] = ['start', 'max core', 'core collapse'],
        radius_bins: Quantity['length'] = Quantity(np.geomspace(1e-3, 1e3, 100), 'kpc'),
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the density profiles of the halo over time.

        Parameters:
            times (Quantity['time']): The times at which to plot the density profiles.
            labels (list[str]): The labels for the density profiles.
            radius_bins (Quantity['length']): The radius bins for the density profile calculations.
            kwargs: Additional keyword arguments are passed to every call to the plotting function.

        Returns:
            fig, ax.
        """
        data = self.get_particle_states()
        unique_times = np.unique(cast(NDArray[np.float64], data['time']))
        real_times = unique_times[np.argmin(np.abs(unique_times - np.expand_dims(times, 1)), axis=1)]
        fig, ax = None, None
        for name, particle_type, unit_mass in zip(['Baryonic matter', 'DM'], ['baryon', 'dm'], np.unique(self.m)):
            for label, t in zip(labels, real_times):
                fig, ax = plot.plot_density(
                    cast(Quantity, data[(data['time'] == t) * (data['particle_type'] == particle_type)]['r']),
                    unit_mass=unit_mass,
                    bins=radius_bins,
                    label=f'{name} {label}',
                    fig=fig,
                    ax=ax,
                    **kwargs,
                )
        assert fig is not None and ax is not None
        return fig, ax
