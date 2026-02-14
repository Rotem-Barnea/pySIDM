"""Halo class for SIDM simulations"""

from __future__ import annotations

import time
import itertools
from copy import deepcopy
from typing import Any, Self, Unpack, Literal, cast, overload
from pathlib import Path
from datetime import datetime
from functools import cached_property
from collections import deque
from collections.abc import Mapping

import numpy as np
import scipy
import pandas as pd
import seaborn as sns
from astropy import table
from numpy.typing import NDArray
from astropy.units import Unit, Quantity, def_unit
from pandas._typing import SortKind
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from astropy.units.typing import UnitLike

from src import plot, units, report, physics
from src.tqdm import tqdm
from src.types import ParticleType, TimeUnitLike
from src.utils import utils
from src.physics import sidm, leapfrog
from src.background import BackgroundDistribution
from src.phase_space import PhaseSpace
from src.distribution.distribution import Backends, Distribution

from . import io, types, run_optimization
from .plot import HaloPlotter
from .units import HaloUnits
from .animate import HaloAnimator


class Halo:
    """Halo class for SIDM simulations"""

    def __init__(
        self,
        r: Quantity['length'],
        v: Quantity['velocity'],
        m: Quantity['mass'],
        dt: Quantity['time'] | float = 1 / 1000,
        unoptimized_dt: Quantity['time'] | None = None,
        particle_type: list[ParticleType] | NDArray[np.str_] | None = None,
        distribution_id: list[int] | NDArray[np.int64] | None = None,
        leapfrog_convergence_rounds: NDArray[np.int64] | None = None,
        potential_reference: Quantity['energy'] | None = None,
        distributions: list[Distribution] | None = None,
        scatter_rounds: deque[int] | None = None,
        scatter_rounds_underestimated: deque[int] | None = None,
        ministep_size: deque[float] | None = None,
        scatter_track_time: deque[float] | None = None,
        scatter_track_index: deque[NDArray[np.int64]] | None = None,
        scatter_track_radius: deque[NDArray[np.float64]] | None = None,
        time: Quantity['time'] = 0 * units.time,
        steps: int | float = 0,
        background: BackgroundDistribution | Distribution | None = None,
        last_saved_time: Quantity['time'] = 0 * units.time,
        save_every_time: Quantity['time'] | float | None = 10,
        save_every_n_steps: int | None = None,
        dynamics_params: leapfrog.Params | dict[str, Any] | None = None,
        scatter_params: sidm.Params | dict[str, Any] | None = None,
        snapshots: table.QTable | None = None,
        hard_save: bool = True,
        save_path: Path | str | None = None,
        r_max: Quantity['length'] = Quantity(300, 'kpc'),
        inner_core_radius: Quantity['length'] | float = 0.2,
        critical_ratio: float = 7,
        bootstrap_steps: int = 100,
        cleanup_nullish_particles: bool = True,
        cleanup_particles_by_radius: bool = True,
        reached_core_collapse: bool = False,
        runtime_realtime_track: deque[float] | None = None,
        runtime_track_sort: deque[float] | None = None,
        runtime_track_cleanup: deque[float] | None = None,
        runtime_track_sidm: deque[float] | None = None,
        runtime_track_leapfrog: deque[float] | None = None,
        runtime_track_full_step: deque[float] | None = None,
        runtime_track_simulation_time: deque[float] | None = None,
        sort_kind: SortKind = 'stable',
        generator: np.random.Generator | None = None,
        seed: int | None = None,
        generator_state: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a Halo object.

        Parameters:
            dt: Time-step of the halo. If not a `Quantity`, assumed to be a factor multiplying the dynamical time of the first distribution in `distributions`.
            r: Radius of the halo particles.
            v: Velocity of the halo particles, of shape `(n_particles, 3)`, `(vx, vy, vr)` with `vx`,`vy` the two perpendicular components of the off-radial plane.
            m: Mass of the halo particles.
            particle_type: Type of the halo particles. Should comply with ParticleType (i.e. `dm` or `baryon`).
            distribution_id: ID of the relevant distribution that sourced the particles.
            leapfrog_convergence_rounds: Number of rounds each particle needs to converge the leapfrog integrator. Used to jump-start the next step for difficult particles.
            potential_reference: Potential at infinity of the halo. If `None` calculates from the first density.
            distributions: List of distributions of the halo.
            n_interactions: Number of interactions the halo had.
            scatter_rounds: Number of scatter rounds the halo had every time-step.
            scatter_rounds_underestimated: Number of underestimated scatter rounds the halo had every time-step (due to `max_allowed_rounds` in `physics.sidm.scatter()`).
            ministep_size: The size of the ministep used for each ministep (to track changes in them).
            scatter_track_time: The time for each scatter track round, must match `scatter_track_index` and `scatter_track_radius` in shape.
            scatter_track_index: The interacting particles (particle index) at every time-step.
            scatter_track_radius: The location of the interacting particles at every time-step.
            time: Time of the halo.
            steps: number of steps made in the simulation (should match `self.time/self.dt` but left as a sanity check).
            background: Background mass distribution of the halo.
            last_saved_time: Last time a snapshot was saved.
            save_every_time: How often should a snapshot be saved, in time units. If not a `Quantity`, assumed to be a factor multiplying the dynamical time of the first distribution in `distributions`.
            save_every_n_steps: How often should a snapshot be saved, in time-step units (integer).
            dynamics_params: Dynamics parameters of the halo, sent to the leapfrog integrator.
            scatter_params: Scatter parameters of the halo, used in the SIDM calculation.
            snapshots: Snapshots of the halo.
            hard_save: Whether to save the halo to memory at every snapshot save, or just keep in RAM.
            save_path: Path to save the halo to memory.
            r_max: Maximum radius of the halo, particles outside of this radius get killed off. If `None` ignores.
            inner_core_radius: Inner core radius of the halo, used for estimating the collapse. If a float is provided, assumed to be a factor multiplying the scale radius of the first distribution in `distributions`.
            critical_ratio: The critical ratio defining the core collapse.
            bootstrap_steps: Number of bootstrap rounds to perform before scattering begins. Time only begins counting after the bootstrap steps.
            cleanup_nullish_particles: Whether to remove particles from the halo after each interaction if they are nullish.
            cleanup_particles_by_radius: Whether to remove particles from the halo based on their radius (r >= `r_max`).
            reached_core_collapse: Whether the halo has reached core collapse.
            generator: Random number generator. If provided ignore `seed` and `generator_state`.
            seed: Seed for the random number generator.
            generator_state: State of the random number generator. If not provided, will be set by the `seed`.
            kwargs: Ignored.

        Returns:
            Halo object.
        """
        self.sort_kind: SortKind = sort_kind
        self._particles = self.to_dataframe(
            r=r,
            v=v,
            m=m,
            particle_type=particle_type,
            distribution_id=distribution_id,
            leapfrog_convergence_rounds=leapfrog_convergence_rounds,
        )
        self._particles.sort_values('r', kind=self.sort_kind, inplace=True)
        self.time: Quantity['time'] = time.to(units.time)
        self.steps: int = int(steps)
        self.distributions: list[Distribution] = utils.handle_default(distributions, [])
        self.dt: Quantity['time'] = (dt if isinstance(dt, Quantity) else Quantity(dt, self.units.dynamical_time)).to(
            units.time
        )
        self.unoptimized_dt: Quantity['time'] = utils.handle_default(unoptimized_dt, self.dt)
        if isinstance(background, Distribution):
            self.background: BackgroundDistribution | None = BackgroundDistribution(distribution=background)
        else:
            self.background = background
        self.potential_reference: Quantity['energy'] = (
            potential_reference
            if potential_reference is not None
            else physics.utils.poisson_potential(self.r, self.M, self.m)[-1]
        )
        self.snapshots: table.QTable = utils.handle_default(snapshots, table.QTable())
        self.save_every_n_steps = save_every_n_steps
        self.save_every_time: Quantity['time'] | None
        if save_every_time is None:
            self.save_every_time = None
        elif isinstance(save_every_time, Quantity):
            self.save_every_time = save_every_time.to(units.time)
        else:
            self.save_every_time = Quantity(save_every_time, self.units.dynamical_time).to(units.time)
        self._dynamics_params = leapfrog.normalize_params(cast(leapfrog.Params | None, dynamics_params))
        self._scatter_params = sidm.normalize_params(cast(sidm.Params | None, scatter_params))
        self.ministep_size: deque[float] = utils.handle_default(ministep_size, deque())
        self.scatter_track_time: deque[float] = utils.handle_default(scatter_track_time, deque())
        self.scatter_track_index: deque[NDArray[np.int64]] = utils.handle_default(scatter_track_index, deque())
        self.scatter_track_radius: deque[NDArray[np.float64]] = utils.handle_default(scatter_track_radius, deque())
        self._initial_particles = self._particles.copy()
        self.initial_particles = self.particles.copy()
        self.last_saved_time: Quantity['time'] = last_saved_time
        self.scatter_rounds: deque[int] = utils.handle_default(scatter_rounds, deque())
        self.scatter_rounds_underestimated: deque[int] = utils.handle_default(scatter_rounds_underestimated, deque())
        self.hard_save: bool = hard_save
        self.save_path: Path | str | None = Path(save_path) if isinstance(save_path, str) else save_path
        self.r_max: Quantity['length'] = r_max.to(units.length)
        if isinstance(inner_core_radius, Quantity):
            self.inner_core_radius: Quantity['length'] = inner_core_radius.to(units.length)
        else:
            self.inner_core_radius = self.distributions[0].r_s * inner_core_radius
        self.critical_ratio = critical_ratio
        self.bootstrap_steps = bootstrap_steps
        self.cleanup_nullish_particles = cleanup_nullish_particles
        self.cleanup_particles_by_radius = cleanup_particles_by_radius
        self.reached_core_collapse = reached_core_collapse
        self.runtime_realtime_track: deque[float] = utils.handle_default(runtime_realtime_track, deque())
        self.runtime_track_sort: deque[float] = utils.handle_default(runtime_track_sort, deque())
        self.runtime_track_cleanup: deque[float] = utils.handle_default(runtime_track_cleanup, deque())
        self.runtime_track_sidm: deque[float] = utils.handle_default(runtime_track_sidm, deque())
        self.runtime_track_leapfrog: deque[float] = utils.handle_default(runtime_track_leapfrog, deque())
        self.runtime_track_full_step: deque[float] = utils.handle_default(runtime_track_full_step, deque())
        self.runtime_track_simulation_time: deque[float] = utils.handle_default(runtime_track_simulation_time, deque())
        if generator is not None:
            self.rng = generator
            self.seed = generator.bit_generator.seed_seq.entropy
        else:
            self.seed = seed
            self.rng = np.random.default_rng(self.seed)
            if generator_state is not None:
                self.rng.bit_generator.state = generator_state

    @staticmethod
    def to_report_concise(
        metadata: io.Metadata,
        keys: list[str] = ['time', 'steps', 'save_path', 'name'],
        line_kwargs: dict[str, Any] | None = None,
    ) -> report.Report:
        """Concise descriptor from metadata"""
        return report.Report.from_dict(
            {**metadata},
            keys=keys,
            line_kwargs=line_kwargs
            or {
                '_global': {
                    'format_func': lambda x: (
                        '.1f' if isinstance(x, Quantity) and cast(Unit, x.unit).physical_type == 'time' else ''
                    )
                },
                'time': {'unit': 'Gyr'},
            },
        )

    def __str__(self) -> str:
        return str(self.to_report_concise(self.metadata))

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def setup(
        cls,
        distributions: list[Distribution],
        n_particles: NDArray[np.int64] | NDArray[np.float64] | list[int | float] | int | float = 1e5,
        seed: int | None = None,
        generator: np.random.Generator | None = None,
        sample_kwargs: dict[str, Any] = {},
        join_distributions: bool = False,
        distribution_as_background: int | None = None,
        **kwargs: Any,
    ) -> Self:
        """Initialize a Halo object from a given set of distributions.

        Parameters:
            distributions: List of distributions for each particle type.
            n_particles: List of number of particles for each particle type. If a number, use the same given amount for all distributions.
            seed: Seed for the random number generator. Ignore if `generator` is provided.
            generator: If `None` use the default generator from `rng.generator`.
            sample_kwargs: Additional keyword arguments to pass to the sampling function.
            join_distributions: If `True`, joining the distributions (`Distribution.merge_distributions`). Use `False` if the distributions already had Eddington inversion calculated elsewhere.
            distribution_as_background: If provided, treat as an index in the `distributions` list, pop it's value and treat it as a static background instead of sampling from it. If `background` is provided as a keyword argument, ignore this feature.
            kwargs: Additional keyword arguments, passed to the constructor.

        Returns:
            Halo object.
        """

        r, v, particle_type, m, distribution_id = [], [], [], [], []
        if generator is None:
            generator = np.random.default_rng(seed)
        distributions = deepcopy(distributions)
        if join_distributions:
            Distribution.merge_distributions(distributions)
        if 'background' in kwargs:
            kwargs = deepcopy(kwargs)
            background: BackgroundDistribution | Distribution | None = kwargs.pop('background')
        else:
            if distribution_as_background is not None:
                background = distributions.pop(distribution_as_background)
            else:
                background = None
        if isinstance(n_particles, int) or isinstance(n_particles, float):
            n_particles = [n_particles] * len(distributions)
        for distribution, n in zip(distributions, n_particles):
            r_, v_, m_, particle_type_, distribution_id_ = distribution.full_sample(
                n_particles=n, generator=generator, **sample_kwargs
            )
            r += [r_]
            v += [v_]
            m += [m_]
            particle_type += [particle_type_]
            distribution_id += [distribution_id_]

        return cls(
            r=cast(Quantity, np.hstack(r)),
            v=cast(Quantity, np.vstack(v)),
            m=cast(Quantity, np.hstack(m)),
            particle_type=np.hstack(particle_type),
            distribution_id=np.hstack(distribution_id),
            distributions=distributions,
            generator=generator,
            background=background,
            **kwargs,
        )

    @property
    def name(self) -> str | list[str]:
        """If the halo is made out of a physical example, return its name"""
        unique_names = np.unique([distribution.name for distribution in self.distributions])
        if len(unique_names) == 1 and unique_names[0] != '':
            return unique_names[0]
        return [distribution.name for distribution in self.distributions]

    @property
    def backend(self) -> Backends | list[Backends]:
        """If the halo is made out of a physical example, return its name"""
        unique_backends = np.unique([distribution.backend for distribution in self.distributions])
        if len(unique_backends) == 1 and unique_backends[0] != '':
            return self.distributions[0].backend
        return [distribution.backend for distribution in self.distributions]

    @staticmethod
    def to_dataframe(
        r: Quantity['length'] | None = None,
        v: Quantity['velocity'] | None = None,
        m: Quantity['mass'] | None = None,
        particle_type: list[ParticleType] | NDArray[np.str_] | None = None,
        distribution_id: list[int] | NDArray[np.int64] | None = None,
        particle_index: NDArray[np.int64] | None = None,
        leapfrog_convergence_rounds: NDArray[np.int64] | None = None,
        qtable: table.QTable | None = None,
    ) -> pd.DataFrame:
        """Convert particle data to a `DataFrame`."""
        assert qtable is not None or (r is not None and v is not None and m is not None), (
            'Either `qtable` must be provided, or `r`, `v` and `m` must be'
        )
        if qtable is not None:
            r, vx, vy, vr, m = utils.get_columns(qtable, columns=['r', 'vx', 'vy', 'vr', 'm'])
            return Halo.to_dataframe(
                r=r,
                v=cast(Quantity, np.vstack([vx, vy, vr]).T),
                m=m,
                particle_type=cast(list[ParticleType], qtable['particle_type']),
                distribution_id=(
                    cast(NDArray[np.int64], qtable['distribution_id']) if 'distribution_id' in qtable.columns else None
                ),
                particle_index=(
                    cast(NDArray[np.int64], qtable['particle_index']) if 'particle_index' in qtable.columns else None
                ),
                leapfrog_convergence_rounds=(
                    cast(NDArray[np.int64], qtable['leapfrog_convergence_rounds'])
                    if 'leapfrog_convergence_rounds' in qtable.columns
                    else None
                ),
            )
        assert r is not None and v is not None and m is not None
        vx, vy, vr = v.to(units.velocity).T
        data = pd.DataFrame(
            {
                'r': r.to(units.length),
                'vx': vx,
                'vy': vy,
                'vr': vr,
                'm': m.to(units.mass),
                'particle_type': particle_type if particle_type is not None else np.full(len(r), 'dm'),
                'particle_index': particle_index if particle_index is not None else np.arange(len(r)),
                'distribution_id': distribution_id if distribution_id is not None else np.full(len(r), 0),
                'leapfrog_convergence_rounds': leapfrog_convergence_rounds
                if leapfrog_convergence_rounds is not None
                else np.full(len(r), 0),
            }
        )
        data['interacting'] = data['particle_type'] == 'dm'
        data.set_index('particle_index', inplace=True)
        return data

    def add_background(self, background: BackgroundDistribution) -> None:
        """Adds a background mass distribution to the halo."""
        self.background = background

    def reset(self) -> None:
        """Resets the halo to its initial state (no interactions, `time`=0, cleared snapshots, particles at initial positions)."""
        self.time = Quantity(0, units.time)
        self.steps = 0
        self.last_saved_time = Quantity(0, units.time)
        self._particles = self._initial_particles.copy()
        self.scatter_rounds = deque()
        self.scatter_rounds_underestimated = deque()
        self.scatter_track_index = deque()
        self.scatter_track_radius = deque()
        self.scatter_track_time = deque()
        self.ministep_size = deque()
        self.snapshots = table.QTable()
        self.runtime_track_sort = deque()
        self.runtime_track_cleanup = deque()
        self.runtime_track_sidm = deque()
        self.runtime_track_leapfrog = deque()
        self.runtime_track_full_step = deque()
        self.runtime_track_simulation_time = deque()
        self.runtime_realtime_track = deque()
        self.rng = np.random.default_rng(self.seed)

    def copy(self) -> Self:
        """Returns a copy of the halo."""
        return deepcopy(self)

    @property
    def particles(self) -> table.QTable:
        """Particle data QTable.

        Has the following columns:
            r: Radius.
            vx: The first perpendicular component (to the radial direction) of the velocity.
            vy: The second perpendicular component (to the radial direction) of the velocity.
            vr: The radial velocity.
            vp: Tangential velocity (`np.sqrt(vx**2 + vy**2)`).
            m: Mass.
            v_norm: Velocity norm (`np.sqrt(vx**2 + vy**2 + vr**2)`).
            time: Current simulation time.
            E: Relative energy (`potential-1/2*m*v_norm**2`).
            particle_type: Type of particle.
            particle_index: Index of particle.
            distribution_id: Identifier of the source distribution.
            leapfrog_convergence_rounds: Number of leapfrog convergence rounds in the previous step.
        """
        self._particles.sort_values('r', kind=self.sort_kind, inplace=True)
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
                'distribution_id': self._particles['distribution_id'],
                'leapfrog_convergence_rounds': self._particles['leapfrog_convergence_rounds'],
            }
        )
        return data

    @property
    def particles_by_type(self) -> dict[str, table.QTable]:
        """Return the `particles` QTable split by particle type (as a dictionary)."""
        groups = self.particles.group_by('particle_type').groups
        return dict(zip(np.array(dict(groups.keys)['particle_type']), groups))

    @property
    def initial_particles_by_type(self) -> dict[str, table.QTable]:
        """Return the `initial_particles` QTable split by particle type (as a dictionary)."""
        groups = self.initial_particles.group_by('particle_type').groups
        return dict(zip(np.array(dict(groups.keys)['particle_type']), groups))

    @property
    def phase_space(self) -> PhaseSpace:
        """Return the phase space object for the halo's particles (all of them, currently)."""
        return PhaseSpace.from_particles(self.distributions[0], self.particles, save_path=self.save_path)

    @property
    def phase_space_snapshots(self) -> PhaseSpace:
        """Return the phase space object for the halo's particles (all of them), including all the snapshots in the historical mass grids."""
        return PhaseSpace.from_particles(
            self.distributions[0], snapshots=self.get_particle_states(), save_path=self.save_path
        )

    @property
    def phase_space_by_type(self) -> dict[str, PhaseSpace]:
        """Return the phase space object for the halo's particles split by particle type (as a dictionary)."""
        return {
            group['particle_type'][0]: PhaseSpace.from_particles(
                self.get_distribution(group['distribution_id'][0]), group, save_path=self.save_path
            )
            for group in self.particles.group_by('particle_type').groups
        }

    @property
    def phase_space_snapshots_by_type(self) -> dict[str, PhaseSpace]:
        """Return the phase space object for the halo's particles split by particle type (as a dictionary), including all the snapshots in the historical mass grids."""
        return {
            (particle_type := group['particle_type'][0]): PhaseSpace.from_particles(
                self.get_distribution(group['distribution_id'][0]),
                snapshots=self.get_particle_states(filter_particle_type=particle_type),
                save_path=self.save_path,
            )
            for group in self.particles.group_by('particle_type').groups
        }

    def get_distribution(self, id: int) -> Distribution:
        """Return the distribution with the given id."""
        for distribution in self.distributions:
            if distribution.id == id:
                return distribution
        raise ValueError(f'Distribution with id {id} not found')

    def get_particle_states(
        self,
        now: bool = True,
        snapshots: bool = True,
        initial: bool = False,
        filter_particle_type: ParticleType | None = None,
    ) -> table.QTable:
        """Return a table of particle snapshots, potentially including the initial and current states."""
        assert now or snapshots or initial, 'At least one of now, snapshots, or initial must be True'
        data_tables = []
        if now:
            data_tables += [self.particles]
        if snapshots:
            data_tables += [self.snapshots]
        if initial:
            data_tables += [self.initial_particles]
        data = table.QTable(table.vstack(data_tables))
        if filter_particle_type is not None:
            data = utils.slice_closest(data=data, value=filter_particle_type, key='particle_type')
        return data

    def preprocess_particle_states(
        self,
        data: table.QTable,
        filter_particle_type: ParticleType | None = None,
        filter_interacting: bool | None = None,
        mask: NDArray[np.bool_] | None = None,
        filter_indices: NDArray[np.int64] | list[int] | None = None,
        time: Quantity['time'] | Literal['start', 'end'] | None = None,
    ) -> table.QTable:
        """Preprocess particle data by applying filters and masks.

        Parameters:
            data: The data to plot.
            filter_particle_type: Whether to filter to only plot the specified particle type.
            filter_interacting: Whether to filter to only plot interacting/non-interacting particles based on `self.scatter_track_index`. If `None` ignores.
            filter_indices: Keep only the specified indices in `data` (based on the `particle_index` column).
            mask: Any additional mask to apply to the data. Must match the shape of the `data` (pre any other filtration).

        Returns:
            Preprocessed data table.
        """
        if mask is not None:
            data = cast(table.QTable, data[mask]).copy()
        if filter_particle_type is not None:
            data = utils.slice_closest(data, value=filter_particle_type, key='particle_type')
        if filter_interacting is not None:
            indices = np.unique(np.hstack(self.scatter_track_index))
            data = utils.filter_indices(data, indices)
        if filter_indices is not None:
            data = utils.filter_indices(data, filter_indices)
        if time is not None:
            if time == 'start':
                data = utils.slice_closest(data, value=data['time'].min())
            elif time == 'end':
                data = utils.slice_closest(data, value=data['time'].max())
            else:
                data = utils.slice_closest(data, value=time)
        return data

    @overload
    def inner_core_density(
        self,
        inner_core_radius: Quantity['length'] | None = None,
        filter_particle_type: ParticleType | None = 'dm',
        stat: Literal['density'] = 'density',
    ) -> tuple[Quantity['length'], Quantity['mass density']]: ...

    @overload
    def inner_core_density(
        self,
        inner_core_radius: Quantity['length'] | None = None,
        filter_particle_type: ParticleType | None = 'dm',
        stat: Literal['count', 'density ratio', 'fraction'] = 'density ratio',
    ) -> tuple[Quantity['length'], NDArray[np.float64]]: ...

    def inner_core_density(
        self,
        inner_core_radius: Quantity['length'] | None = None,
        filter_particle_type: ParticleType | None = 'dm',
        stat: Literal['count', 'density', 'density ratio', 'fraction'] = 'density ratio',
    ) -> tuple[Quantity['length'], Quantity['mass density'] | NDArray[np.float64]]:
        """Calculate statistics on the inner core density.

        Parameters:
            inner_core_radius: The radius of the inner core. If `None` use the internal value.
            filter_particle_type: Whether to filter to only plot the specified particle type.
            stat: The type of statistic to calculate. `ratio` returns the density divided by the initial density (post bootstrap).

        Returns:
            (time, core density ratio)
        """
        data = self.get_particle_states(filter_particle_type=filter_particle_type, now=False)
        inner_core_radius = inner_core_radius if inner_core_radius is not None else self.inner_core_radius
        agg = 'sum' if stat == 'density' else 'count'
        time, counts = (
            data[data['r'] < inner_core_radius].to_pandas().groupby('time')['m'].agg(agg).reset_index().to_numpy().T
        )
        if stat == 'density':
            counts = cast(Quantity, Quantity(counts, data['m'].unit) / (4 / 3 * np.pi * inner_core_radius**3))
        else:
            counts = np.array(counts)
        if stat == 'density ratio':
            counts /= counts[0]
        elif stat == 'fraction':
            counts /= data.to_pandas().groupby('time')['m'].agg('count').to_numpy()
        return Quantity(time, data['time'].unit), counts

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
        """Clean up the particles by dropping nullish values and particles outside the radius.

        Significantly faster if the DataFrame is presorted.
        """
        if self.cleanup_nullish_particles or self.cleanup_particles_by_radius:
            drop_indices = pd.Series(data=np.zeros(len(self._particles), dtype=np.bool_), index=self._particles.index)
            if self.cleanup_nullish_particles:
                drop_indices += self._particles['r'].isna()
            if self.cleanup_particles_by_radius:
                drop_indices += self._particles['r'] > self.r_max.value
            if drop_indices.any():
                if presorted:
                    end = drop_indices.argmax()
                    self._particles = self._particles.iloc[:end].copy()
                else:
                    self._particles.drop(index=drop_indices[drop_indices].index, inplace=True)

    #####################
    ##Physical properties
    #####################

    @property
    def plot(self) -> HaloPlotter:
        """Plotting object for the halo."""
        return HaloPlotter(self)

    @property
    def animate(self) -> HaloAnimator:
        """Animator object for the halo."""
        return HaloAnimator(self)

    @property
    def units(self) -> HaloUnits:
        """Units derived from the halo."""
        return HaloUnits(self)

    @property
    def r(self) -> Quantity['length']:
        """Particle radius."""
        return Quantity(self._particles['r'], units.length)

    @property
    def vx(self) -> Quantity['velocity']:
        """The first perpendicular component (to the radial direction) of the particle velocity."""
        return Quantity(self._particles['vx'], units.velocity)

    @property
    def vy(self) -> Quantity['velocity']:
        """The second perpendicular component (to the radial direction) of the particle velocity."""
        return Quantity(self._particles['vy'], units.velocity)

    @property
    def vr(self) -> Quantity['velocity']:
        """The radial component of the particle velocity."""
        return Quantity(self._particles['vr'], units.velocity)

    @property
    def v(self) -> Quantity['velocity']:
        """The velocity of the particle, as a 3-vector `(vx, vy, vr)`."""
        return Quantity(self._particles[['vx', 'vy', 'vr']], units.velocity)

    @property
    def time_step(self) -> Quantity['time']:
        """The time-step size"""
        return Quantity(1, self.units.time_step).decompose(units.system)

    @property
    def dynamical_time(self) -> Quantity['time']:
        """The dynamical time of the system"""
        return Quantity(1, self.units.dynamical_time).decompose(units.system)

    @property
    def t_c(self) -> Quantity['time']:
        """The collapse time estimate of the system"""
        return Quantity(1, self.units.t_c).decompose(units.system)

    @property
    def collapse_time(self) -> Quantity['time']:
        """Real core collapse time of the halo. Only calculated after the halo reached core collapse during its run."""
        return Quantity(1, self.units.core_collapse).decompose(units.system)

    @property
    def M(self) -> Quantity['mass']:
        """The enclosed mass below the particle."""
        halo_mass = physics.utils.enclosed_mass(r=self.r, m=self.m)
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
        return Quantity(self._particles['m'], units.mass)

    @property
    def internal_energy(self) -> Quantity['energy']:
        """The internal energy of the particle."""
        return 0.5 * self.m * self.v_norm**2

    @property
    def poisson_potential(self) -> Quantity['energy']:
        """The gravitational potential energy of the particle."""
        return cast(Quantity, physics.utils.poisson_potential(self.r, self.M, self.m))

    @property
    def potential(self) -> Quantity['specific energy']:
        """The relative gravitational potential energy of the particle."""
        return cast(Quantity, physics.utils.potential(self.r, self.M, self.m)).to(units.energy)
        # return (self.potential_reference - self.poisson_potential).to(run_units.energy)

    @property
    def E(self) -> Quantity['specific energy']:
        """The energy of the particle."""
        return (self.potential - self.internal_energy).to(units.energy)

    @property
    def local_density(self) -> Quantity['mass density']:
        """The local mass density around the particle."""
        return cast(
            Quantity['mass density'],
            physics.utils.local_density(
                self.r,
                self.m,
                self.scatter_params['max_radius_j'],
            ),
        )

    @property
    def local_density_by_type(self) -> dict[str, Quantity['mass density']]:
        """The local mass density around the particle. Split by particle type."""
        return {
            particle_type: cast(
                Quantity['mass density'],
                physics.utils.local_density(
                    cast(Quantity, data['r']),
                    cast(Quantity, data['m']),
                    self.scatter_params['max_radius_j'],
                ),
            )
            for particle_type, data in self.particles_by_type.items()
        }

    @property
    def n_scatters(self) -> NDArray[np.int64]:
        """The number of scatters every scattering round."""
        return np.array([len(x) / 2 for x in self.scatter_track_index], dtype=np.int64)

    @property
    def n_particles(self) -> dict[str, int]:
        """The total number of particles of every type in the halo."""
        return self._particles['particle_type'].value_counts().to_dict()

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
                self.runtime_track_simulation_time,
                fillvalue=np.nan,
            ),
            columns=['sort', 'cleanup', 'sidm', 'leapfrog', 'full step', 'real timestep'],
        )

    def unit_mass(self, distribution: Distribution) -> Quantity['mass']:
        """Return the unit mass of the given distribution."""
        return distribution.total_mass / self.n_particles[distribution.particle_type]

    @property
    def generator_state(self) -> Mapping[str, Any]:
        """Get the current state of the random number generator."""
        return self.rng.bit_generator.state

    @property
    def scatter_times(self) -> Quantity['time']:
        """Wrap `self.scatter_track_time` as a Quantity."""
        return Quantity(np.hstack(self.scatter_track_time), units.time)

    @property
    def scatter_track_time_raveled(self) -> Quantity['time']:
        """Get a raveled array with the scatter time matching each particle in the hstack-ed `self.scatter_track_index`."""
        return Quantity(
            np.hstack([[t.value] * len(i) for i, t in zip(self.scatter_track_index, self.scatter_times)]),
            units.time,
        )

    @cached_property
    def core_collapse_time_unit(self) -> Unit:
        """The core collapse time of the halo, as a unit"""
        return def_unit(
            'Tc',
            self.core_collapse_estimate(),
            doc='Core collapse time',
        )

    @cached_property
    def core_collapse_time(self) -> Quantity['time']:
        """The core collapse time of the halo, as a Quantity"""
        return Quantity(1, self.core_collapse_time_unit)

    def scatter_track_time_raveled_binned(self, time_bin_size: Quantity | None | Literal['save cadence']):
        """Get a raveled array with the scatter time matching each particle in the hstack-ed `self.scatter_track_index`, with the time binned to a fixed bin size."""
        time_array = self.scatter_track_time_raveled
        if time_bin_size is None:
            return time_array
        elif time_bin_size == 'save cadence':
            time_bin_size = self.save_every_time
        n_bins = int(time_array.max() / time_bin_size)
        time_array = (time_array // time_bin_size) / n_bins * time_array.max()
        return time_array

    def max_core_time(
        self,
        time_binning: Quantity['time'] = Quantity(100, 'Myr'),
        smoothing_sigma: int | None = 1,
        kind: str = 'cubic',
    ) -> Quantity['time']:
        """Calculate the time at which the halo reaches maximum core.

        The number of scattering events is aggregated over a fixed bin size and smoothed using a Gaussian filter, and then the argmin is taken.

        Parameters:
            time_binning: The binning resolution to aggregate the number of scattering events.
            smoothing_sigma: The smoothing factor over the number of scattering events.
            kind: The kind of interpolation to use.

        Returns:
            The maximal core time
        """
        n = int(time_binning / self.dt)
        time = self.scatter_times
        scatters = np.add.reduceat(self.n_scatters, np.arange(0, len(self.n_scatters), n))
        if smoothing_sigma is not None:
            scatters = scipy.ndimage.gaussian_filter1d(scatters, sigma=smoothing_sigma)
        return cast(
            Quantity,
            time[
                scipy.interpolate.interp1d(
                    time[::n].value,
                    scatters,
                    kind=kind,
                    bounds_error=False,
                    fill_value=np.inf,
                )(time).argmin()
            ],
        )

    def core_collapse_scatter_estimate(self, **kwargs: Any) -> Quantity['time']:
        """Calculate the time at which the halo starts major core collapse.

        Defined as the time at which the halo first reaches `cutoff` scatters per `time_binning` time.
        """
        return run_optimization.core_collapse_scatter_estimate(
            t=self.time, scatter_times=self.scatter_times, n_scatters=self.n_scatters, **kwargs
        )

    def core_collapse_core_density_estimate(
        self,
        inner_core_radius: Quantity['length'] | None = None,
        critical_ratio: float | None = None,
    ) -> Quantity['time']:
        """Calculate the time at which the halo starts major core collapse.

        Defined as the time at which the inner core density first exceeds `critical_ratio` times the initial density.

        Parameters:
            inner_core_radius: The radius of the inner core. If `None` use the internal value.
            critical_ratio: The critical ratio defining the core collapse.

        Returns:
            The core collapse start time
        """
        return run_optimization.core_collapse_core_density_estimate(
            snapshots=self.get_particle_states(),
            initial_r=utils.get_columns(self.initial_particles_by_type['dm'], ['r'])[0],
            inner_core_radius=inner_core_radius if inner_core_radius is not None else self.inner_core_radius,
            critical_ratio=critical_ratio if critical_ratio is not None else self.critical_ratio,
        )

    def core_collapse_estimate(
        self, method: Literal['core density', 'scatters'] = 'core density', **kwargs: Any
    ) -> Quantity['time']:
        """Calculate the time at which the halo starts major core collapse."""
        if method == 'core density':
            return self.core_collapse_core_density_estimate(**kwargs)
        else:
            return self.core_collapse_scatter_estimate(**kwargs)

    def core_density_ratio(self, inner_core_radius: Quantity['length'] | None = None) -> float:
        """Calculate the ratio of the core density to the initial density."""
        return run_optimization.core_density_ratio(
            r=utils.get_columns(self.particles_by_type['dm'], ['r'])[0],
            initial_r=utils.get_columns(utils.slice_closest(self.initial_particles, 'dm', 'particle_type'), ['r'])[0],
            inner_core_radius=inner_core_radius if inner_core_radius is not None else self.inner_core_radius,
        )

    #####################
    ##Dynamic evolution
    #####################

    def to_step(self, time: Quantity['time']) -> int:
        """Calculate the number of steps required to reach the given time."""
        return int(time / self.dt)

    def to_time(self, steps: int) -> Quantity['time']:
        """Calculate the duration of the given number of steps."""
        return steps * self.dt

    @property
    def current_step(self) -> int:
        """The current simulation step count (calculated based on the simulation time)."""
        return self.to_step(self.time)

    def save_snapshot(self, **kwargs: Unpack[types.SaveParams]) -> None:
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

    def check_early_quit(
        self, inner_core_radius: Quantity['length'] | None = None, critical_ratio: float | None = None
    ) -> bool:
        """Check if the simulation should be terminated early.

        Parameters:
            inner_core_radius: The inner core radius. If None, use the current inner core radius.
            critical_ratio: The critical ratio defining the core collapse.

        Returns:
            `True` if the simulation should be terminated early, `False` otherwise.
        """
        return run_optimization.check_early_quit(
            core_collapse_kwargs={
                'r': self.r,
                'initial_r': utils.get_columns(self.initial_particles, ['r'])[0],
                'inner_core_radius': inner_core_radius if inner_core_radius is not None else self.inner_core_radius,
                'critical_ratio': critical_ratio if critical_ratio is not None else self.critical_ratio,
            }
        )

    def early_quit(
        self, early_quit_kwargs: types.EarlyQuitParams | None = None, save_kwargs: types.SaveParams = {}
    ) -> bool:
        """Checks early quitting and handling the saving if needed to allow quit escape."""
        if not self.reached_core_collapse:
            self.reached_core_collapse = self.check_early_quit(**(early_quit_kwargs or {}))
        if early_quit_kwargs is not None and self.reached_core_collapse:
            if self.hard_save:
                self.save(**save_kwargs)
            print('Quiting early!')
            return True
        return False

    def step(self, in_bootstrap: bool = False, save_kwargs: types.SaveParams = {}) -> None:
        """Perform a single time-step of the simulation.

        Every step:
            - Sort particles by radius.
            - Clean up erroneous particles.
            - Save a snapshot if it's time.
            - Perform scattering. This is done before the leapfrog integration since it doesn't modify the particle positions and thus doesn't require resorting.
            - Perform leapfrog integration.
            - Update simulation time.

        Parameters:
            in_bootstrap: Whether the simulation is in bootstrap mode.
            save_kwargs: Keyword arguments for saving the snapshot.
        """

        self.runtime_realtime_track += [datetime.now().timestamp()]
        t_start = time.perf_counter()
        t0 = time.perf_counter()
        self._particles.sort_values('r', kind=self.sort_kind, inplace=True)
        self.runtime_track_sort += [time.perf_counter() - t0]
        t0 = time.perf_counter()
        self.cleanup_particles()
        self.runtime_track_cleanup += [time.perf_counter() - t0]
        if self.is_save_round():
            self.save_snapshot(**save_kwargs)
        r, vx, vy, vr, m, leapfrog_convergence_rounds = self._particles[
            ['r', 'vx', 'vy', 'vr', 'm', 'leapfrog_convergence_rounds']
        ].values.T
        if not in_bootstrap and self.scatter_params['sigma'] > sidm.no_sigma:
            t0 = time.perf_counter()
            mask = cast(NDArray[np.bool_], self._particles['interacting'].values)
            (
                vx[mask],
                vy[mask],
                vr[mask],
                indices,
                scatter_rounds,
                scatter_rounds_underestimated,
            ) = sidm.scatter(
                r=r[mask],
                vx=vx[mask],
                vy=vy[mask],
                vr=vr[mask],
                dt=self.dt,
                m=m[mask],
                generator=self.rng,
                **self.scatter_params,
            )
            self.scatter_track_index += [np.array(self._particles[mask].iloc[indices].index, dtype=np.int64)]
            self.scatter_track_time += [self.time.value]
            self.scatter_track_radius += [self.r[mask][indices]]
            self.scatter_rounds += [scatter_rounds]
            self.scatter_rounds_underestimated += [scatter_rounds_underestimated]
            self.runtime_track_sidm += [time.perf_counter() - t0]
        t0 = time.perf_counter()
        r, vx, vy, vr, leapfrog_convergence_rounds = leapfrog.step(
            r=r,
            vx=vx,
            vy=vy,
            vr=vr,
            m=m,
            first_mini_round=(leapfrog_convergence_rounds - 1).clip(min=0),
            M=self.M,
            dt=self.dt,
            **self.dynamics_params,
        )
        self._particles['r'] = r
        self._particles['vx'] = vx
        self._particles['vy'] = vy
        self._particles['vr'] = vr
        self._particles['leapfrog_convergence_rounds'] = leapfrog_convergence_rounds

        self.runtime_track_leapfrog += [time.perf_counter() - t0]
        if not in_bootstrap:
            self.time += self.dt
            self.ministep_size += [self.dt.value]
            self.steps += 1
        self.runtime_track_simulation_time += [self.time.to(units.time).value]
        self.runtime_track_full_step += [time.perf_counter() - t_start]

    def bootstrap(
        self,
        tqdm_kwargs: dict[str, Any] = {},
        save_kwargs: types.SaveParams = {},
        optimize_dt_kwargs: types.OptimizeDtParams | None = None,
    ):
        """Run bootstrapping phase if applicable. Only runs gravitational dynamics for the specified number of steps (i.e. no SIDM) to ensure relaxation into a steady state for the initial conditions."""
        if self.steps == 0 and self.bootstrap_steps > 0:
            if optimize_dt_kwargs is not None:
                self.dt = run_optimization.optimize_dt(self, **optimize_dt_kwargs)
            for _ in tqdm(range(self.bootstrap_steps), desc='Bootstrap phase', **tqdm_kwargs):
                self.step(in_bootstrap=True)
            self.save_snapshot(**save_kwargs)

    def evolve(
        self,
        n_steps: int | None = None,
        t: Quantity['time'] | None = None,
        until_t: Quantity['time'] | None = None,
        tqdm_kwargs: dict[str, Any] = {},
        save_kwargs: types.SaveParams = {},
        optimize_dt_kwargs: types.OptimizeDtParams | None = None,
        early_quit_kwargs: types.EarlyQuitParams | None = None,
    ) -> None:
        """Evolve the simulation for a given number of steps or time.

        Parameters:
            n_steps: Number of steps to evolve the simulation for. Takes precedence over `t`.
            t: Time to evolve the simulation for. Ignored if `n_steps` is specified, otherwise transformed into steps using `to_steps()`.
            until_t: Evolve the simulation until this time. Ignored if `n_steps` or `t` are specified, otherwise transformed into steps using `to_steps()`.
            tqdm_kwargs: Additional keyword arguments to pass to `tqdm` (NOTE this is the custom submodule defined in this project at `tqdm.py`).
            save_kwargs: Additional keyword arguments to pass to `save()`.
            optimize_dt_kwargs: Additional keyword arguments to pass to `optimize_dt()`. If `None`, avoid `dt` optimization.
            early_quit_kwargs: Additional keyword arguments to pass to `early_quit()`. If `None`, avoid early quit consideration.

        Returns:
            None
        """
        self.bootstrap(tqdm_kwargs=tqdm_kwargs, save_kwargs=save_kwargs, optimize_dt_kwargs=optimize_dt_kwargs)

        if early_quit_kwargs is not None:
            self.reached_core_collapse = None
        if n_steps is None:
            if t is not None:
                n_steps = self.to_step(t)
            elif until_t is not None:
                if self.time > until_t:
                    raise ValueError('current time is greater than the specified end time')
                n_steps = self.to_step(cast(Quantity, until_t - self.time))
            else:
                raise ValueError('Either `n_steps`, `t`, or `until_t` must be specified')

        start_points, reoptimize_rate = run_optimization.split_to_chunks(
            required_time=self.to_time(n_steps), optimize_dt_kwargs=optimize_dt_kwargs
        )
        for _ in tqdm(start_points, disable=len(start_points) == 1):
            if optimize_dt_kwargs is not None:
                self.dt = run_optimization.optimize_dt(self, **optimize_dt_kwargs)
            for _ in tqdm(range(self.to_step(reoptimize_rate)), start_time=self.time, dt=self.dt, **tqdm_kwargs):
                self.step(save_kwargs=save_kwargs)
                if self.early_quit(early_quit_kwargs=early_quit_kwargs, save_kwargs=save_kwargs):
                    return
            if self.early_quit(early_quit_kwargs=early_quit_kwargs, save_kwargs=save_kwargs):
                return
        if self.hard_save:
            self.save(**save_kwargs)

    #####################
    ##Save/Load
    #####################

    @property
    def results_path(self) -> Path:
        """Return the path to the results directory."""
        if self.save_path is None:
            raise ValueError('`save_path` is not set')
        return self.save_path.parents[1] / 'results' / self.save_path.stem

    @property
    def metadata(self) -> io.Metadata:
        """Return the metadata for the simulation. Used for saving."""
        return io.Metadata(**{key: getattr(self, key) for key in io.metadata_keys()})

    @property
    def heavy_payload(self) -> io.HeavyPayload:
        """Return the heavy payload for the simulation. Used for saving."""
        return io.HeavyPayload(**{key: getattr(self, key) for key in io.heavy_payload_keys()})

    def save(
        self,
        path: str | Path | None = None,
        two_steps: bool = False,
        keep_last_backup: bool = False,
        split_snapshots: bool = True,
        verbose: bool = False,
    ) -> None:
        """Save the simulation state to a directory.

        Parameters:
            path: Save path. If `path` is `None` attempts to use the internal save path.
            two_steps: If `True` saves the simulation state in two steps, to avoid rewriting the existing file with data that can be stopped midway (leaving just the 1 corrupted file). This means that for the duration of the saving the disk size used is doubled.
            keep_last_backup: If `True` keeps a full backup of the previous save, otherwise overwrite it based on `two_steps` rules. This option _always_ uses twice the disk space.
            split_snapshots: If `True` saves the snapshots QTable as separate files.
            verbose: If `True` prints progress information.

        Returns:
            None
        """
        io.save(
            path=self.save_path if path is None else path,
            static_tables={'particles': self.particles, 'initial_particles': self.initial_particles},
            splitable_table={'snapshots': self.snapshots},
            metadata_payload=self.metadata,
            heavy_payload=self.heavy_payload,
            distributions=self.distributions,
            background=self.background,
            two_steps=two_steps,
            keep_last_backup=keep_last_backup,
            split_tables=split_snapshots,
            verbose=verbose,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        update_save_path: bool = True,
        static: bool = False,
        undersample_snapshots: int | None = None,
        verbose: bool = True,
    ) -> Self:
        """Load the simulation state from a directory.

        Parameters:
            path: Save path to load from.
            update_save_path: Whether to update the internal save path to `path` (for example, if the directory was moved after the run).
            static: Whether to load the simulation with `hard_save=False` as a safeguard, to avoid accidentally evolving the simulation on a completed run (that was loaded for analysis).
            undersample_snapshots: If provided, undersample loading the snapshot tables by the given factor (i.e. load every 10th table, etc.).
            verbose: Whether to print progress information during loading.

        Returns:
            The loaded Halo object
        """
        tables = io.load_tables(path, undersample={'snapshots': undersample_snapshots}, verbose=verbose)
        assert tables['particles'] is not None, 'Particles table is missing'
        particles = tables['particles']
        r, vx, vy, vr, m = utils.get_columns(particles, columns=['r', 'vx', 'vy', 'vr', 'm'])
        output = cls(
            r=r,
            v=cast(Quantity, np.vstack([vx, vy, vr]).T),
            particle_type=cast(list[ParticleType], particles['particle_type']),
            m=m,
            distribution_id=np.array(particles['distribution_id']),
            distributions=io.load_distributions(path, verbose=verbose),
            background=io.load_background(path, verbose=verbose),
            **io.load_pickle(path=path, stem='metadata', verbose=verbose),
            **io.load_pickle(path, stem='heavy_payload', verbose=verbose),
            snapshots=tables['snapshots'],
        )
        if 'initial_particles' in tables and tables['initial_particles'] is not None:
            output._initial_particles = output.to_dataframe(qtable=tables['initial_particles'])
            output.initial_particles = tables['initial_particles']
        if update_save_path:
            output.save_path = Path(path).resolve()
        if static:
            output.hard_save = False
        return output

    @staticmethod
    def load_metadata(path: Path | str, verbose: bool = True) -> io.Metadata:
        """Load metadata from a pickle file."""
        return io.load_pickle(path=path, stem='metadata', verbose=verbose)

    @staticmethod
    def load_heavy_payload(path: Path | str, verbose: bool = True) -> io.HeavyPayload:
        """Load metadata from a pickle file."""
        return io.load_pickle(path=path, stem='heavy_payload', verbose=verbose)

    def rename(self, full_path: str | Path | None = None, stem: str | None = None) -> None:
        """Renames the halo save path (and existing output folder if it exists)."""
        assert full_path is None or stem is None, 'Only one of full_path or stem can be specified'
        if full_path is not None:
            save_path = Path(full_path)
        elif stem is not None:
            assert self.save_path is not None, 'save_path must be specified to use this option'
            save_path = Path(self.save_path).with_stem(stem)
        else:
            raise ValueError('Either full_path or stem must be specified')
        if self.save_path is not None and Path(self.save_path).exists():
            Path(self.save_path).rename(save_path)
        self.save_path = save_path

    #####################
    ##Plots
    #####################

    def save_plot(self, fig: Figure, save_kwargs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        # DEPRECATE
        """Saves the plot."""
        if save_kwargs is None:
            return
        if 'name' in save_kwargs:
            save_kwargs['save_path'] = self.results_path / save_kwargs.pop('name')
        plot.save(fig=fig, **save_kwargs)

    def fill_time_unit(self, unit: TimeUnitLike) -> UnitLike:
        # DEPRECATE
        """If `unit` is a halo-related time parameter return its unit, otherwise return `unit`."""
        if unit == 'dynamical time':
            return self.units.dynamical_time
        elif unit == 'core collapse time':
            return self.units.core_collapse
        elif unit == 'time step':
            return self.units.time_step
        return unit

    def energy_change_summary(self, filter_particle_type: ParticleType | None = None, **kwargs: Any) -> report.Report:
        """Generate a summary of the energy change during the simulation."""
        initial = self.initial_particles.copy()
        final = self.particles.copy()
        if filter_particle_type is not None:
            initial = utils.slice_closest(initial, value=filter_particle_type, key='particle_type')
            final = utils.slice_closest(final, value=filter_particle_type, key='particle_type')

        return report.Report(
            header=f'After {self.current_step} steps with dt={self.dt:.4f} | {self.time:.1f}',
            body_lines=[
                report.Line(title='Total energy at the start', value=initial['E'].sum(), format='.1f'),
                report.Line(title='Total energy at the end', value=final['E'].sum(), format='.1f'),
                report.Line(
                    title='Energy change',
                    value=np.abs(final['E'].sum() - initial['E'].sum()),
                    format='.1f',
                ),
                report.Line(
                    title='Energy change per step',
                    value=np.abs(final['E'].sum() - initial['E'].sum()) / self.current_step,
                    format='.1e',
                ),
                report.Line(
                    title='Energy change per dt',
                    value=np.abs(final['E'].sum() - initial['E'].sum()) / self.dt,
                    format='.1e',
                ),
                report.Line(
                    title='Relative energy change',
                    value=np.abs(final['E'].sum() - initial['E'].sum()) / initial['E'].sum(),
                    format='.3%',
                ),
                report.Line(
                    title='Relative energy change per step',
                    value=np.abs(final['E'].sum() - initial['E'].sum()) / initial['E'].sum() / self.current_step,
                    format='.1e',
                ),
                report.Line(
                    title='Relative energy change per dt',
                    value=np.abs(final['E'].sum() - initial['E'].sum()) / initial['E'].sum() / self.dt,
                    format='.3%',
                ),
                report.Line(
                    title='Mean velocity change',
                    value=np.abs(final['v_norm'].mean() - initial['v_norm'].mean()).to('km/second'),
                    format='.1f',
                ),
                report.Line(
                    title='Mean velocity change per step',
                    value=np.abs(final['v_norm'].mean() - initial['v_norm'].mean()).to('km/second') / self.current_step,
                    format='.1e',
                ),
                report.Line(
                    title='Mean velocity change per dt',
                    value=np.abs(final['v_norm'].mean() - initial['v_norm'].mean()).to('km/second') / self.dt,
                    format='.1e',
                ),
                report.Line(
                    title='Relative Mean velocity change',
                    value=np.abs(final['v_norm'].mean() - initial['v_norm'].mean()) / initial['v_norm'].mean(),
                    format='.3%',
                ),
            ],
            **kwargs,
        )

    def scatter_report(self, **kwargs: Any) -> report.Report:
        """Generate a summary of the scattering during the simulation."""
        core_collapse_start_time = self.core_collapse_scatter_estimate()
        max_core_time = self.max_core_time()
        n_scatter_cumsum = self.n_scatters.cumsum()
        scatters_to_collapse_start = n_scatter_cumsum[(self.scatter_times <= core_collapse_start_time).argmin()]
        n_scattering_particles = len(np.unique(np.hstack(self.scatter_track_index)))
        return report.Report(
            body_lines=[
                report.Line(title='Maximal core time', value=max_core_time.to('Gyr'), format='.1f'),
                report.Line(
                    title='Core collapse start time',
                    value=core_collapse_start_time.to('Gyr'),
                    format='.2f',
                ),
                report.Line(
                    title='Number of scatter events until core collapse started',
                    value=scatters_to_collapse_start,
                    format='',
                ),
                report.Line(
                    title='Number of scatter events after core collapse started',
                    value=n_scatter_cumsum[-1] - scatters_to_collapse_start,
                    format='',
                ),
                report.Line(title='Overall number of scatter events', value=n_scatter_cumsum[-1], format=''),
                report.Line(
                    title='Participating particles',
                    value=f'{n_scattering_particles:,}/{self.n_particles["dm"]:,}',
                    format='',
                ),
                report.Line(
                    title='Participating particles fraction',
                    value=n_scattering_particles / self.n_particles['dm'],
                    format='.1%',
                ),
                report.Line(
                    title='Average number of scatter events per particle until core collapse started',
                    value=scatters_to_collapse_start / self.n_particles['dm'],
                    format='.1f',
                ),
                report.Line(
                    title='Average number of scatter events per scattering particle until core collapse started',
                    value=scatters_to_collapse_start / n_scattering_particles,
                    format='.1f',
                ),
            ],
            **kwargs,
        )

    def add_automatic_guidelines(
        self,
        manual_times: Quantity['time'] = Quantity([], 'Myr'),
        manual_labels: list[str] = [],
        time_unit: TimeUnitLike = 'Gyr',
    ) -> tuple[Quantity['time'], list[str]]:
        """Automatically pull max core and collapse times for use in plotting."""
        time_unit = self.fill_time_unit(time_unit)
        times = []
        labels = []
        for t, label in zip(
            [Quantity(0, time_unit), self.max_core_time(), self.core_collapse_start_time(), self.time],
            ['start', 'max core', 'core collapse (start)', 'core collapse (deep)'],
        ):
            if t == np.inf:
                break
            times += [t.to(time_unit)]
            labels += [label]
        output = pd.DataFrame(
            {'time': np.hstack([manual_times.to(time_unit), *times]), 'label': manual_labels + labels}
        )
        times, labels = output.drop_duplicates().sort_values('time').to_numpy().T
        return Quantity(times, time_unit), labels.tolist()

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
        x_unit: UnitLike | None = None,
        ylabel: str | None = None,
        label: str | None = None,
        fig: Figure | None = None,
        ax: Axes | None = None,
        plt_kwargs: dict[str, Any] = {},
        save_kwargs: dict[str, Any] | None = None,
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
            x_unit: The x-axis units to use in the plot.
            ylabel: The label for the y-axis.
            label: The label for the histogram (legend).
            fig: The figure to plot on.
            ax: The axes to plot on.
            plt_kwargs: Additional keyword arguments to pass to the sns plotting function (`sns.histplot()` or `sns.kdeplot()`).
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.
            kwargs: Additional keyword arguments to pass to `plot.setup()`.

        Returns:
            fig, ax.
        """
        x_unit = plot.default_unit_type(key, x_unit)
        if filter_particle_type is not None:
            data = utils.slice_closest(data, value=filter_particle_type, key='particle_type')
        x = data[key].to(x_unit)
        if x_range is not None:
            x = x[(x > x_range[0]) * (x < x_range[1])]
        if absolute:
            x = np.abs(x)
        params = {
            **plot.default_plot_text(key, x_unit=x_unit),
            **utils.drop_None(title=title, xlabel=xlabel, ylabel=ylabel),
        }
        fig, ax = plot.setup(fig, ax, **params, **kwargs)
        if plot_type == 'kde':
            sns.kdeplot(x, cumulative=cumulative, ax=ax, label=label, **plt_kwargs)
        else:
            sns.histplot(x, cumulative=cumulative, ax=ax, stat=stat, label=label, **plt_kwargs)
        if x_plot_range is not None:
            ax.set_xlim(*x_plot_range.to(x_unit).value)
        self.save_plot(fig=fig, save_kwargs=save_kwargs)
        return fig, ax

    def plot_r_distribution(
        self,
        data: table.QTable,
        cumulative: bool = False,
        add_density: int | None = 0,
        x_unit: UnitLike = 'kpc',
        x_range: Quantity | None = None,
        hist_label: str | None = None,
        density_label: str | None = None,
        save_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the radial distribution of the halo. Wraps `plot_distribution()` with additional options.

        Parameters:
            data: The data to plot.
            cumulative: Whether to plot the cumulative distribution.
            add_density: Density distribution to plot on top of the plot (index from the distributions list). If `None` ignores.
            x_unit: The units to plot the x-axis in.
            x_range: The range of the x-axis.
            hist_label: The label for the histogram (legend).
            density_label: The label for the density distribution (legend).
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.
            kwargs: Additional keyword arguments to pass to `plot.setup()`.

        Returns:
            fig, ax.
        """
        fig, ax = self.plot_distribution(
            key='r',
            data=data,
            cumulative=cumulative,
            x_unit=x_unit,
            x_range=x_range,
            label=hist_label,
            **kwargs,
        )
        if add_density is not None:
            params: dict[str, Any] = (
                {'r_start': cast(Quantity, x_range[0]), 'r_end': cast(Quantity, x_range[1])}
                if x_range is not None
                else {}
            )
            fig, ax = self.distributions[add_density].plot_radius_distribution(
                cumulative=cumulative,
                length_unit=x_unit,
                fig=fig,
                ax=ax,
                label=density_label,
                **params,
            )
        self.save_plot(fig=fig, save_kwargs=save_kwargs)
        return fig, ax

    def plot_particle_evolution(
        self,
        include_start: bool = True,
        include_now: bool = False,
        filter_particle_type: ParticleType | None = None,
        radius_bins: Quantity = Quantity(np.linspace(1e-3, 5, 100), 'kpc'),
        time_range: Quantity | None = None,
        length_unit: UnitLike = 'kpc',
        time_unit: TimeUnitLike = 'dynamical time',
        xlabel: str | None = 'Radius',
        ylabel: str | None = 'Time',
        cbar_label: str | None = 'Particles',
        cmap: str = 'seismic',
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
            length_unit: Units to use for the radius axis.
            time_unit: Units to use for the time axis.
            xlabel: Label for the radius axis.
            ylabel: Label for the time axis.
            cbar_label: Label for the colorbar.
            cmap: The colormap to use for the plot.
            row_normalization: The normalization to apply to each row. If `None` no normalization is applied. If `float` it must be a percentile value (between 0 and 1), and the normalization will be based on this quantile of each row.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.heatmap()`).

        Returns:
            fig, ax.
        """

        time_unit = self.fill_time_unit(time_unit)
        data = self.get_particle_states(now=include_now, initial=include_start, snapshots=True)
        if filter_particle_type is not None:
            data = utils.slice_closest(data, value=filter_particle_type, key='particle_type')
        grid, extent, (x_range, y_range) = plot.aggregate_evolution_data(
            data=data,
            radius_bins=radius_bins,
            time_range=time_range,
            output_type='counts',
            row_normalization=row_normalization,
        )

        fig, ax = plot.heatmap(
            grid=grid,
            x_range=x_range,
            y_range=y_range,
            extent=extent,
            x_unit=length_unit,
            y_unit=time_unit,
            xlabel=xlabel,
            ylabel=ylabel,
            cbar_label=cbar_label,
            grid_row_normalization=row_normalization,
            cmap=cmap,
            **kwargs,
        )
        self.save_plot(fig=fig, **kwargs)
        return fig, ax

    def plot_temperature_evolution(
        self,
        include_start: bool = True,
        include_now: bool = False,
        filter_particle_type: ParticleType | None = None,
        radius_bins: Quantity = Quantity(np.linspace(1e-1, 5, 100), 'kpc'),
        time_range: Quantity | None = None,
        specific_energy_unit: UnitLike = 'km^2/second^2',
        length_unit: UnitLike = 'kpc',
        time_unit: TimeUnitLike = 'dynamical time',
        xlabel: str | None = 'Radius',
        ylabel: str | None = 'Time',
        cbar_label: str | None = r'$\propto$Temperature (velocity variance)',
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
            specific_energy_unit: Units to use for the specific energy.
            length_unit: Units to use for the radius axis.
            time_unit: Units to use for the time axis.
            xlabel: Label for the radius axis.
            ylabel: Label for the time axis.
            cbar_label: Label for the colorbar.
            row_normalization: The normalization to apply to each row. If `None` no normalization is applied. If `float` it must be a percentile value (between 0 and 1), and the normalization will be based on this quantile of each row.
            cmap: The colormap to use for the plot.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.heatmap()`).

        Returns:
            fig, ax.
        """
        time_unit = self.fill_time_unit(time_unit)
        data = self.get_particle_states(now=include_now, initial=include_start, snapshots=True)
        if filter_particle_type is not None:
            data = utils.slice_closest(data, value=filter_particle_type, key='particle_type')
        grid, extent, (x_range, y_range) = plot.aggregate_evolution_data(
            data=data,
            radius_bins=radius_bins,
            time_range=time_range,
            output_type='temperature',
            row_normalization=row_normalization,
            output_grid_unit=specific_energy_unit,
        )

        fig, ax = plot.heatmap(
            grid=grid,
            extent=extent,
            x_range=x_range,
            y_range=y_range,
            x_unit=length_unit,
            y_unit=time_unit,
            xlabel=xlabel,
            ylabel=ylabel,
            cbar_label=cbar_label,
            grid_row_normalization=row_normalization,
            cbar_label_autosuffix=True if row_normalization is not None else False,
            cmap=cmap,
            **kwargs,
        )
        self.save_plot(fig=fig, **kwargs)
        return fig, ax

    def plot_heat_flux_evolution(
        self,
        include_start: bool = True,
        include_now: bool = False,
        filter_particle_type: ParticleType | None = None,
        radius_bins: Quantity = Quantity(np.linspace(1e-3, 5, 100), 'kpc'),
        time_range: Quantity | None = None,
        v_axis: Literal['vx', 'vy', 'vr'] = 'vr',
        heat_unit: UnitLike = '1/Myr^3',
        length_unit: UnitLike = 'kpc',
        time_unit: TimeUnitLike = 'dynamical time',
        xlabel: str | None = 'Radius',
        ylabel: str | None = 'Time',
        cbar_label: str | None = 'Specific Heat flux',
        row_normalization: Literal['max', 'sum', 'integral'] | float | None = None,
        cmap: str = 'seismic',
        setup_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the heat flux evolution of the halo. Wraps `prep_2d_data()`.

        Parameters:
            include_start: Whether to include the initial particle distribution in the plot.
            include_now: Whether to include the current particle distribution in the plot.
            filter_particle_type: Whether to filter to only plot the specified particle type.
            radius_bins: The bins for the radius axis. Also used to define the radius range to consider.
            time_range: Range of times to consider (filters the data).
            v_axis: The velocity to calculate the heat flux in.
            velocity_unit: Units to use for the velocity.
            length_unit: Units to use for the radius axis.
            time_unit: Units to use for the time axis.
            xlabel: Label for the radius axis.
            ylabel: Label for the time axis.
            cbar_label: Label for the colorbar.
            row_normalization: The normalization to apply to each row. If `None` no normalization is applied. If `float` it must be a percentile value (between 0 and 1), and the normalization will be based on this quantile of each row.
            cmap: The colormap to use for the plot.
            setup_kwargs: Additional keyword arguments to pass to `plot.setup()`.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.heatmap()`).

        Returns:
            fig, ax.
        """
        time_unit = self.fill_time_unit(time_unit)
        data = self.get_particle_states(now=include_now, initial=include_start, snapshots=True)
        if filter_particle_type is not None:
            data = utils.slice_closest(data, value=filter_particle_type, key='particle_type')
        grid, extent, (x_range, y_range) = plot.aggregate_evolution_data(
            data=data,
            radius_bins=radius_bins,
            time_range=time_range,
            v_axis=v_axis,
            output_type='specific heat flux',
            row_normalization=row_normalization,
            output_grid_unit=heat_unit,
        )

        fig, ax = plot.heatmap(
            grid=grid,
            extent=extent,
            x_range=x_range,
            y_range=y_range,
            x_unit=length_unit,
            y_unit=time_unit,
            xlabel=xlabel,
            ylabel=ylabel,
            cbar_label=cbar_label,
            grid_row_normalization=row_normalization,
            cbar_label_autosuffix=True if row_normalization is not None else False,
            cmap=cmap,
            setup_kwargs=setup_kwargs,
            **kwargs,
        )
        self.save_plot(fig=fig, **kwargs)
        return fig, ax

    def plot_scattering_location_evolution(
        self,
        radius_bins: Quantity = Quantity(np.linspace(1e-3, 1.2, 100), 'kpc'),
        time_range: Quantity | None = None,
        time_bin_size: Quantity | None | Literal['save cadence'] = 'save cadence',
        normalize_by_n_particles: bool = False,
        length_unit: UnitLike = 'kpc',
        time_unit: TimeUnitLike = 'dynamical time',
        xlabel: str | None = 'Radius',
        ylabel: str | None = 'Time',
        cbar_label: str | None = 'Number of scattering events per {time}',
        cbar_label_time_unit: UnitLike = 'Myr',
        cbar_label_time_format: str = '.1f',
        cbar_log_scale: bool = True,
        row_normalization: Literal['max', 'sum', 'integral'] | float | None = None,
        cmap: str = 'jet',
        x_tick_format: str = '.1f',
        transparent_range: tuple[float, float] | None = (0, 100),
        setup_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the scattering location evolution of the halo. Wraps `prep_2d_data()`.

        Parameters:
            radius_bins: The bins for the radius axis. Also used to define the radius range to consider.
            time_range: Range of times to consider (filters the data).
            time_bin_size: The size of the time bins. If `save cadence`, the time bins will be set to the save cadence of the simulation. If `None`, avoid binning completely.
            normalize_by_n_particles: Whether to normalize the histogram by the number of particles in each bin.
            length_unit: Units to use for the radius axis.
            time_unit: Units to use for the time axis.
            xlabel: Label for the radius axis.
            ylabel: Label for the time axis.
            cbar_label: Label for the colorbar.
            cbar_label_time_unit: Units to use for time.
            cbar_label_time_format: Format string for time.
            cbar_log_scale: Whether to use a logarithmic scale for the colorbar.
            row_normalization: The normalization to apply to each row. If `None` no normalization is applied. If `float` it must be a percentile value (between 0 and 1), and the normalization will be based on this quantile of each row.
            cmap: The colormap to use for the plot.
            x_tick_format: Format string for the x-axis ticks.
            transparent_range: Range of values to turn transparent (i.e. plot as `NaN`). If `None` ignores.
            setup_kwargs: Additional keyword arguments to pass to `plot.setup()`.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.heatmap()`).

        Returns:
            fig, ax.
        """
        time_unit = self.fill_time_unit(time_unit)

        time_array = self.scatter_track_time_raveled_binned(time_bin_size).to(time_unit)

        if cbar_label is not None:
            cbar_label = cbar_label.format(
                time=np.unique(time_array)
                .diff()[0]
                .to(cbar_label_time_unit)
                .to_string(format='latex', formatter=cbar_label_time_format),
            )

        data = table.QTable(
            {
                'time': time_array,
                'r': Quantity(np.hstack(self.scatter_track_radius), units.length).to(length_unit),
            }
        )
        grid, extent, (x_range, y_range) = plot.aggregate_evolution_data(
            data=data,
            radius_bins=radius_bins,
            time_range=time_range,
            row_normalization=row_normalization,
        )

        if normalize_by_n_particles:
            time_bins = np.unique(time_array)
            location_data = self.get_particle_states(now=False).copy()
            location_data = cast(table.QTable, location_data[location_data['time'] <= time_bins.max()])
            location_data['time'] = time_bins[
                np.searchsorted(time_bins, cast(NDArray[np.float64], location_data['time']))
            ]

            location_grid, _, _ = plot.aggregate_evolution_data(
                data=location_data,
                radius_bins=radius_bins,
                time_range=time_range,
            )

            grid[location_grid == 0] = 0
            grid[location_grid != 0] = grid[location_grid != 0] / location_grid[location_grid != 0]

        fig, ax = plot.heatmap(
            grid=grid,
            extent=extent,
            x_range=x_range,
            y_range=y_range,
            x_unit=length_unit,
            y_unit=time_unit,
            xlabel=xlabel,
            ylabel=ylabel,
            cbar_label=cbar_label,
            log_scale=cbar_log_scale,
            grid_row_normalization=row_normalization,
            cbar_label_autosuffix=True if row_normalization is not None else False,
            cmap=cmap,
            transparent_range=transparent_range,
            x_tick_format=x_tick_format,
            setup_kwargs=setup_kwargs,
            **kwargs,
        )
        self.save_plot(fig=fig, **kwargs)
        return fig, ax

    def plot_scattering_location(
        self,
        title: str | None = 'Scattering location distribution within the first {time}, total of {n_scatters} events',
        xlabel: str | None = 'Radius',
        length_unit: UnitLike = 'kpc',
        time_unit: TimeUnitLike = 'Gyr',
        time_format: str = '.1f',
        figsize: tuple[int, int] = (12, 6),
        fig: Figure | None = None,
        ax: Axes | None = None,
        save_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot the histogram of scattering event locations.

        Flattens all the events (ignores time), and plots the location of the particles at each scattering event.

        Parameters:
            title: Title for the plot.
            xlabel: Label for the x-axis.
            length_unit: Units to use for the radius axis.
            time_unit: Units to use for time.
            time_format: Format string for time.
            figsize: Size of the figure.
            fig: Figure to use for the plot.
            ax: Axes to use for the plot.
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.

        Returns:
            fig, ax.

        """
        time_unit = self.fill_time_unit(time_unit)
        if title is not None:
            title = title.format(
                time=self.time.to(time_unit).to_string(format='latex', formatter=time_format),
                n_scatters=self.n_scatters.sum(),
            )
        fig, ax = plot.setup(
            fig, ax, figsize=figsize, minorticks=True, **utils.drop_None(title=title, xlabel=xlabel), x_unit=length_unit
        )
        sns.histplot(
            Quantity(np.hstack(self.scatter_track_radius), units.length).to(length_unit),
            ax=ax,
            log=True,
        )
        self.save_plot(fig=fig, save_kwargs=save_kwargs)
        return fig, ax

    def plot_scattering_distance(
        self,
        title: str | None = 'Interaction distance distribution',
        xlabel: str | None = 'Interaction distance',
        length_unit: UnitLike = 'pc',
        log_scale: bool = True,
        stat: str = 'density',
        fig: Figure | None = None,
        ax: Axes | None = None,
        setup_kwargs: dict[str, Any] = {},
        save_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the histogram of the distance between scattering particles during interaction.

        Flattens all the events (ignores time), and tracks the location of the particles at each scattering event.

        Parameters:
            title: Title for the plot.
            xlabel: Label for the x-axis.
            length_unit: Units to use for the radius axis.
            log_scale: Whether to use a logarithmic scale for the histogram.
            stat: Statistical function to use for the histogram, must be a valid input for `sns.histplot`.
            fig: Figure to use for the plot.
            ax: Axes to use for the plot.
            setup_kwargs: Additional keyword arguments to pass to `plot.setup()`.
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.
            kwargs: Additional keyword arguments passed to `sns.histplot`.

        Returns:
            fig, ax.

        """
        fig, ax = plot.setup(fig, ax, **utils.drop_None(title=title, xlabel=xlabel, x_unit=length_unit), **setup_kwargs)
        radius = np.diff(np.hstack(self.scatter_track_radius).reshape(-1, 2)).ravel().to(length_unit)
        sns.histplot(radius, log_scale=log_scale, stat=stat, ax=ax, **kwargs)
        self.save_plot(fig=fig, save_kwargs=save_kwargs)
        return fig, ax

    def plot_scattering_density(
        self,
        num: int = 500,
        xlabel: str | None = 'Radius',
        ylabel: str | None = 'Density',
        title: str | None = 'Scattering density within the first {time}, total of {n_scatters} events',
        length_unit: UnitLike = 'kpc',
        time_unit: TimeUnitLike = 'Gyr',
        time_format: str = '.1f',
        smooth_sigma: float = 5,
        smooth_interpolate_kind: str = 'linear',
        save_kwargs: dict[str, Any] | None = None,
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
            length_unit: Units to use for the radius axis.
            time_unit: Units to use for time.
            time_format: Format string for time.
            smooth_sigma: Smoothing factor for the density plot (sigma for a 1d Gaussian kernel).
            smooth_interpolate_kind: Interpolation kind for the density plot. Applied after the Gaussian smoothing to further smooth the plot data.
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.setup()`).

        Returns:
            fig, ax.
        """
        r = np.hstack(self.scatter_track_radius).to(length_unit)
        r_bins = np.linspace(0, r.max(), num=num)
        dr = r_bins[1] - r_bins[0]
        density = Quantity(
            [
                ((r >= low) * (r < high)).sum() / (4 * np.pi * dr * ((low + high) / 2) ** 2)
                for low, high in zip(r_bins[:-1], r_bins[1:])
            ]
        )
        r_bins = r_bins[:-1]
        density_unit = str(density.unit)
        interpolated_density = scipy.interpolate.interp1d(
            r_bins[density != 0], density[density != 0], kind=smooth_interpolate_kind
        )(r_bins)
        smoothed_density = scipy.ndimage.gaussian_filter1d(interpolated_density, sigma=smooth_sigma)

        time_unit = self.fill_time_unit(time_unit)
        if title is not None:
            title = title.format(
                time=self.time.to(time_unit).to_string(format='latex', formatter=time_format),
                n_scatters=self.n_scatters.sum(),
            )
        fig, ax = plot.setup(
            **kwargs,
            yscale='log',
            **utils.drop_None(title=title, xlabel=xlabel, ylabel=ylabel),
            x_unit=length_unit,
            y_unit=density_unit,
        )
        sns.lineplot(x=r_bins, y=smoothed_density, ax=ax)
        self.save_plot(fig=fig, save_kwargs=save_kwargs)
        return fig, ax

    def plot_scattering_amount_distribution(
        self,
        bins: list[int] | NDArray[np.int64] = [10, 50, 100, 200, 400, 1000, 2000],
        xlabel: str | None = 'Number of scattering per particle',
        ylabel: str | None = 'Fraction of scattering DM particles',
        title: str | None = 'Per particle scattering amount distribution',
        minorticks: bool = True,
        xscale: plot.Scale = 'log',
        plot_labels: bool = True,
        bar_kwargs: dict[str, Any] = {'align': 'center', 'edgecolor': 'black', 'alpha': 0.7},
        save_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the distribution of the scattering particles by the number of scatterings (fraction per bin).

        Parameters:
            bins: The bin edges to use for the number of scatterings (the dividers between bins - the start and end will be added automatically).
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            title: Title for the plot.
            minorticks: Whether to add the grid for the minor ticks.
            xscale: The scale of the x-axis.
            plot_labels: Whether to add text bubbles with the y-value above the bins,
            bar_kwargs: Keyword arguments to pass to `Axes.bar()`.
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.setup()`).

        Returns:
            fig, ax.
        """

        fig, ax = plot.setup(
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            minorticks=minorticks,
            xscale=xscale,
            y_axis_percent_formatter={'xmax': 1},
            **kwargs,
        )

        indices, counts = np.unique(np.hstack(self.scatter_track_index), return_counts=True)
        bins = np.hstack([1, np.array(bins), indices.max() + 1])
        heights = []
        bin_centers = []
        for low, high in zip(bins[:-1], bins[1:]):
            heights += [((counts >= low) * (counts < high)).mean()]
            bin_centers += [(low + high) / 2]
        widths = np.diff(bins, 1)
        ax.bar(bin_centers, heights, width=widths, **bar_kwargs)

        if plot_labels:
            for bin_center, height in zip(bin_centers, heights):
                ax.text(
                    s=f'{height:.0%}',
                    **plot.pretty_ax_text(x=bin_center, y=height + 0.01, verticalalignment='bottom'),
                )

        self.save_plot(fig=fig, save_kwargs=save_kwargs)
        return fig, ax

    def plot_scatter_rounds_over_time(
        self,
        rounds: bool = True,
        total_required: bool = True,
        underestimations: bool = False,
        time_unit: TimeUnitLike = 'Gyr',
        xlabel: str | None = 'Time',
        ylabel: str | None = 'Number of scattering subdivisions per time-step',
        title: str | None = 'Scattering subdivisions and underestimation over time',
        label_rounds: str | None = 'Rounds performed',
        label_total_required: str | None = 'Total amount required',
        label_underestimations: str | None = 'Underestimations',
        clip_max_rounds: float | None = None,
        clip_max_total_required: float | None = None,
        clip_max_underestimations: float | None = None,
        save_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the number of scattering rounds per `dt` time-step, and the number of underestimations.

        Parameters:
            rounds: Plot the number of scattering rounds performed per `dt` time-step.
            total_required: Plot the number of required scattering rounds per `dt` time-step (regardless of what actually happened).
            underestimations: Plot the scattering rounds underestimated per `dt` time-step ([required] - [actually happened]).
            time_unit: Units for the x-axis.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            title: The title of the plot.
            label_rounds: Label for the `rounds` plot (legend).
            label_total_required: Label for the `total_required` plot (legend).
            label_underestimations: Label for the `underestimation` plot (legend).
            clip_max_rounds: Maximum value to clip the `rounds` plot.
            clip_max_total_required: Maximum value to clip the `total_required` plot.
            clip_max_underestimations: Maximum value to clip the `underestimations` plot.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.setup()`).

        Returns:
            fig, ax.
        """
        time_unit = self.fill_time_unit(time_unit)
        fig, ax = plot.setup(xlabel=xlabel, ylabel=ylabel, x_unit=time_unit, title=title, **kwargs)
        x = (np.arange(len(self.scatter_rounds)) * self.dt).to(time_unit)
        if total_required:
            y = np.array(self.scatter_rounds) + np.array(self.scatter_rounds_underestimated)
            sns.lineplot(
                x=x,
                y=y if clip_max_total_required is None else y.clip(max=clip_max_total_required),
                ax=ax,
                label=label_total_required,
            )
        if rounds:
            y = np.array(self.scatter_rounds)
            sns.lineplot(
                x=x,
                y=y if clip_max_rounds is None else y.clip(max=clip_max_rounds),
                ax=ax,
                label=label_rounds,
            )
        if underestimations:
            y = np.array(self.scatter_rounds_underestimated)
            sns.lineplot(
                x=x,
                y=y if clip_max_underestimations is None else y.clip(max=clip_max_underestimations),
                ax=ax,
                label=label_underestimations,
            )
        if (
            (rounds and label_rounds is not None)
            or (total_required and label_total_required is not None)
            or (underestimations and label_underestimations is not None)
        ):
            ax.legend()
        self.save_plot(fig=fig, save_kwargs=save_kwargs)
        return fig, ax

    def plot_distributions_density(
        self,
        markers_on_first_only: bool = False,
        save_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the density profile (`rho`) of each of the provided distributions in the halo.

        Parameters:
            markers_on_first_only: If `True` only plot markers (`r_s` and `r_vir`) for the first density.
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.
            kwargs: Additional keyword arguments are passed to every call to the plotting function.

        Returns:
            fig, ax.
        """
        fig, ax = None, None
        for i, distribution in enumerate(self.distributions):
            fig, ax = distribution.plot_density(
                label=f'{distribution.label} ({distribution.title})',
                fig=fig,
                ax=ax,
                add_markers=(i == 0 or not markers_on_first_only),
                **kwargs,
            )
        assert fig is not None and ax is not None
        self.save_plot(fig=fig, save_kwargs=save_kwargs)
        return fig, ax

    def plot_scatter_distribution_at_time(
        self,
        time: Quantity,
        data: table.QTable | None = None,
        include_start: bool = True,
        include_now: bool = False,
        no_scatter_value: float = 0,
        only_past_scatters: bool = True,
        x_bins: Quantity = Quantity(np.geomspace(1e-3, 1e3, 100), 'kpc'),
        scatter_bins: Quantity = Quantity(np.geomspace(1, 6000, 100), ''),
        x_key: str = 'r',
        x_unit: UnitLike = 'kpc',
        cmap: str = 'jet',
        cbar_log_scale: bool = True,
        transparent_value: float | None = 0,
        xlabel: str | None = 'Radius',
        ylabel: str | None = 'Number of scattering events',
        title: str | None = 'Distribution by number of scattering events at {time}',
        title_suffix: str | None = None,
        cbar_label: str | None = 'Number of particles',
        time_unit: TimeUnitLike = 'Gyr',
        time_format: str = '.1f',
        xscale: plot.Scale = 'log',
        yscale: plot.Scale = 'log',
        plot_method: Literal['imshow', 'pcolormesh'] = 'pcolormesh',
        fig: Figure | None = None,
        ax: Axes | None = None,
        aggregate_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """
        Plot the number of scattering events as a function of a tracked property at the closest snapshot to the specified time.

        Parameters:
            time: The time to slice the snapshots (nearest).
            data: The data to plot. If `None` the data will be loaded from the halo snapshots.
            include_start: Whether to include the initial particle distribution in the data. Ignored if `data` is provided.
            include_now: Whether to include the current particle distribution in the data. Ignored if `data` is provided.
            no_scatter_value: Value to use for particles with no scattering events.
            only_past_scatters: Whether to only include past scattering events, or any event this particle will be a part of.
            x_bins: Bins for the x-axis.
            scatter_bins: Bins for the scatter axis.
            x_key: The key to use for the x-axis.
            x_unit: The units for the x-column in the data.
            cmap: The colormap to use for the plot.
            cbar_log_scale: Whether to plot the cbar in a log scale.
            transparent_value: Grid value to turn transparent (i.e. plot as `NaN`). If `None` ignores.
            xlabel: The label for the x-axis.
            ylabel: The label for the y-axis.
            title: The title of the plot.
            cbar_label: Label for the colorbar.
            time_unit: The time units to use in the plot's title.
            time_format: Format string for time to use in the plot's title.
            xscale: The scale for the x-axis.
            yscale: The scale for the y-axis.
            plot_method: Method to use for plotting.
            fig: Figure to plot on.
            ax: Axes to plot on.
            aggregate_kwargs: Additional keyword arguments to pass to the aggregation function (`plot.aggregate_2d_data()`).
            kwargs: Additional keyword arguments to pass to the plot function (`plot.heatmap()`).

        Returns:
            fig, ax.
        """
        time_unit = self.fill_time_unit(time_unit)
        if data is None:
            data = self.get_particle_states(now=include_now, initial=include_start, snapshots=True)
            data = utils.slice_closest(utils.slice_closest(data, value=time), value='dm', key='particle_type')

        if title is not None:
            title = title.format(time=time.to(time_unit).to_string(format='latex', formatter=time_format))

        if title_suffix is not None and title is not None:
            title += f' ({title_suffix})'

        index_track = (
            list(self.scatter_track_index)[: np.argmin(self.scatter_times < time)]
            if only_past_scatters
            else self.scatter_track_index
        )
        if len(index_track) == 0:
            data['n_scatters'] = Quantity(np.full(len(data), no_scatter_value))
        else:
            sub = pd.merge(
                data.to_pandas(),
                pd.DataFrame(
                    np.vstack(np.unique(np.hstack(index_track), return_counts=True)).T,
                    columns=['particle_index', 'n_scatters'],
                ),
                on='particle_index',
                how='left',
            )
            sub['n_scatters'] = sub['n_scatters'].fillna(no_scatter_value)
            data['n_scatters'] = Quantity(sub['n_scatters'])

        fig, ax = plot.heatmap(
            *plot.aggregate_2d_data(
                data, x_key=x_key, y_key='n_scatters', x_bins=x_bins, y_bins=scatter_bins, **aggregate_kwargs
            ),
            plot_method=plot_method,
            x_range=x_bins,
            y_range=scatter_bins,
            cmap=cmap,
            x_unit=x_unit,
            y_unit='',
            log_scale=cbar_log_scale,
            transparent_value=transparent_value,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            cbar_label=cbar_label,
            xscale=xscale,
            yscale=yscale,
            **kwargs,
        )
        self.save_plot(fig=fig, **kwargs)
        return fig, ax

    def plot_scatter_distribution_at_time_animation(
        self,
        include_start: bool = False,
        include_now: bool = False,
        save_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> None:
        """
        Plot the number of scattering events as a function of a tracked property at the closest snapshot to the specified time.

        Parameters:
            include_start: Whether to include the initial particle distribution in the data.
            include_now: Whether to include the current particle distribution in the data.
            save_kwargs: Additional keyword arguments to pass to `plot.save_plot()`.
            kwargs: Additional keyword arguments to pass to the plot function for each frame (`self.plot_scatter_distribution_at_time()`).

        Returns:
            fig, ax.
        """
        plot.save_images(
            plot.to_images(
                iterator=[
                    t
                    for t in np.unique(
                        cast(Quantity, self.get_particle_states(initial=include_start, now=include_now)['time'])
                    )
                    if t <= self.scatter_track_time[-1]
                ],
                plot_fn=lambda x: self.plot_scatter_distribution_at_time(time=x, **kwargs),
            ),
            **save_kwargs,
        )

    def plot_mean_scattering_distance_over_time(
        self,
        bin_edges: Quantity['time'] = Quantity(np.linspace(0, 13.5, 20), 'Gyr'),
        length_unit: UnitLike = 'pc',
        time_unit: TimeUnitLike = 'Gyr',
        xlabel: str | None = 'Time',
        ylabel: str | None = 'Interaction distance',
        title: str | None = 'Mean interaction distance over time',
        accuracy_factor: int = 3,
        plot_guidelines: dict[str, Any] | None = {
            'times': Quantity([[0, 1], [12.5, 13]], 'Gyr'),
            'labels': ['core\nexpanding', 'core\ncollapse'],
        },
        texts: list[dict[str, Any]] | None = None,
        vlines: list[dict[str, Any]] | None = None,
        save_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the mean and median interaction distance over time.

        Parameters:
            bin_edges: The edges of the time bins.
            length_unit: Units to use for distance.
            time_unit: Units to use for time.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            title: The title of the plot.
            accuracy_factor: The width of the error range (in units of standard deviations),
            plot_guidelines: A dictionary of guidelines for plotting, with keys `times` and `labels`. Where `times` is an array of time Quantities shaped (n_guidelines, 2), and `labels` is a list of strings of length n_guidelines that will be plotted at the center of each guideline tuple (row).
            texts: Overwrites the autogenerated text bubbles from `plot_guidelines`. If provided must be a list of dictionaries valid for `ax.text()`.
            vlines: Overwrites the autogenerated vertical lines from `plot_guidelines`. If provided must be a list of dictionaries valid for `ax.axvline()`.
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.
            kwargs: Additional keyword arguments passed to `plot.setup()`.

        Returns:
            fig, ax.
        """
        time_unit = self.fill_time_unit(time_unit)
        time_array = self.scatter_times
        values = []
        time_bins = []
        for time_range in tqdm(list(zip(bin_edges[:-1], bin_edges[1:]))):
            mask = (time_array >= time_range[0]) * (time_array <= time_range[1])
            if not mask.any():
                continue
            start, end = np.arange(len(mask))[mask][[0, -1]]
            values += [
                np.diff(np.hstack(list(self.scatter_track_radius)[start:end]).reshape(-1, 2)).ravel().to(length_unit)
            ]
            time_bins += [Quantity(time_range).mean()]

        time_bins = Quantity(time_bins)
        distance_mean = Quantity([v.mean() for v in values])
        distance_median = Quantity([np.median(v) for v in values])
        distance_std = Quantity([v.std() for v in values])
        bin_count = np.array([len(v) for v in values])
        distance_accuracy = accuracy_factor * distance_std / np.sqrt(bin_count)

        if plot_guidelines is None:
            vlines = [{}]
            texts = [{}]
        else:
            if texts is None:
                texts = [
                    plot.pretty_ax_text(**cast(dict[str, Any], t))
                    for t in pd.DataFrame(
                        {
                            's': plot_guidelines['labels'],
                            'x': plot_guidelines['times'].to(time_unit).mean(1).value,
                            'y': [0.07] * 2,
                            'horizontalalignment': ['center'] * 2,
                            'verticalalignment': ['bottom'] * 2,
                        }
                    ).to_dict('records')
                ]
            if vlines is None:
                vlines = [
                    {'x': t, 'color': 'red', 'linestyle': '--', 'linewidth': 0.5}
                    for t in plot_guidelines['times'].to(time_unit).ravel().value
                ]

        fig, ax = plot.setup(
            **utils.drop_None(
                xlabel=xlabel,
                ylabel=ylabel,
                title=title,
            ),
            x_unit=time_unit,
            y_unit=length_unit,
            vlines=vlines,
            texts=texts,
            **kwargs,
        )
        sns.lineplot(x=time_bins.value, y=distance_mean.value, ax=ax, label='Mean')
        ax.fill_between(
            time_bins.value,
            (distance_mean - distance_accuracy).value,
            (distance_mean + distance_accuracy).value,
            alpha=0.2,
        )
        sns.lineplot(x=time_bins.value, y=distance_median.value, ax=ax, label='Median')
        ax.fill_between(
            time_bins.value,
            (distance_median - distance_accuracy).value,
            (distance_median + distance_accuracy).value,
            alpha=0.2,
        )
        self.save_plot(fig=fig, save_kwargs=save_kwargs)
        return fig, ax
