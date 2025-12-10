import time
import pickle
import shutil
import itertools
from copy import deepcopy
from typing import Any, Self, Literal, cast
from pathlib import Path
from datetime import datetime
from collections import deque
from collections.abc import Mapping

import numpy as np
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from astropy import table
from numpy.typing import NDArray
from astropy.units import Unit, Quantity, def_unit
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from astropy.units.typing import UnitLike

from . import plot, utils, report, physics, run_units
from .tqdm import tqdm
from .types import ParticleType
from .physics import sidm, leapfrog
from .background import Mass_Distribution
from .distribution.distribution import Distribution

no_sigma = Quantity(0, run_units.cross_section)


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
        distributions: list[Distribution] | None = None,
        scatter_rounds: deque[int] | None = None,
        scatter_rounds_underestimated: deque[int] | None = None,
        ministep_size: deque[float] | None = None,
        scatter_track_time: deque[float] | None = None,
        scatter_track_index: deque[NDArray[np.int64]] | None = None,
        scatter_track_radius: deque[NDArray[np.float64]] | None = None,
        time: Quantity['time'] = 0 * run_units.time,
        steps: int | float = 0,
        background: Mass_Distribution | None = None,
        last_saved_time: Quantity['time'] = 0 * run_units.time,
        save_every_time: Quantity['time'] | None = None,
        save_every_n_steps: int | None = None,
        dynamics_params: leapfrog.Params | None = None,
        scatter_params: sidm.Params | None = None,
        max_allowed_subdivisions: int = 1,
        subdivide_on_scatter_chance: bool = False,
        subdivide_on_gravitational_step: bool = True,
        subdivide_on_startup: bool = False,
        snapshots: table.QTable | None = None,
        hard_save: bool = False,
        save_path: Path | str | None = None,
        Rmax: Quantity['length'] = Quantity(300, 'kpc'),
        bootstrap_steps: int = 100,
        scatters_to_collapse: int = 340,
        cleanup_nullish_particles: bool = False,
        cleanup_particles_by_radius: bool = False,
        runtime_realtime_track: deque[float] | None = None,
        runtime_track_sort: deque[float] | None = None,
        runtime_track_cleanup: deque[float] | None = None,
        runtime_track_sidm: deque[float] | None = None,
        runtime_track_leapfrog: deque[float] | None = None,
        runtime_track_full_step: deque[float] | None = None,
        generator: np.random.Generator | None = None,
        seed: int | None = None,
        generator_state: Mapping[str, Any] | None = None,
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
            scatter_rounds_underestimated: Number of underestimated scatter rounds the halo had every time step (due to `max_allowed_rounds` in `physics.sidm.scatter()`).
            ministep_size: The size of the ministep used for each ministep (to track changes in them).
            scatter_track_time: The time for each scatter track round, must match `scatter_track_index` and `scatter_track_radius` in shape.
            scatter_track_index: The interacting particles (particle index) at every time step.
            scatter_track_radius: The location of the interacting particles at every time step.
            time: Time of the halo.
            steps: number of steps made in the simulation (should match `self.time/self.dt` but left as a sanity check).
            background: Background mass distribution of the halo.
            last_saved_time: Last time a snapshot was saved.
            save_every_time: How often should a snapshot be saved, in time units.
            save_every_n_steps: How often should a snapshot be saved, in time-step units (integer).
            dynamics_params: Dynamics parameters of the halo, sent to the leapfrog integrator.
            scatter_params: Scatter parameters of the halo, used in the SIDM calculation.
            max_allowed_subdivisions: Maximum number of subdivisions allowed in each step.
            subdivide_on_scatter_chance: Whether to subdivide based on the scatter chance. Only relevant if `max_allowed_subdivision` is not `1`. If multiple subdivisions logic are provided, take the greater constraint.
            subdivide_on_gravitational_step: Whether to subdivide based on the ratio of vr*dt to the spacing to the nearest neighbor. Only relevant if `max_allowed_subdivision` is not `1`. If multiple subdivisions logic are provided, take the greater constraint.
            subdivide_on_startup: Whether to subdivide to the maximum allowed subdivisions until the 1 Gyr mark. Only relevant if `max_allowed_subdivision` is not `1`. If multiple subdivisions logic are provided, take the greater constraint.
            snapshots: Snapshots of the halo.
            hard_save: Whether to save the halo to memory at every snapshot save, or just keep in RAM.
            save_path: Path to save the halo to memory.
            Rmax: Maximum radius of the halo, particles outside of this radius get killed off. If `None` ignores.
            bootstrap_steps: Number of bootstrap rounds to perform before scattering begins. Time only begins counting after the bootstrap steps.
            scatters_to_collapse: Number of scatters required on average for every dark matter particle to reach core collapse. Only used for estimating core collapse time for the early stopping mechanism, has no effect on the physical calculation (which will reach core-collapse on its own independently).
            cleanup_nullish_particles: Whether to remove particles from the halo after each interaction if they are nullish.
            cleanup_particles_by_radius: Whether to remove particles from the halo based on their radius (r >= `Rmax`).
            generator: Random number generator. If provided ignore `seed` and `generator_state`.
            seed: Seed for the random number generator.
            generator_state: State of the random number generator. If not provided, will be set by the `seed`

        Returns:
            Halo object.
        """
        self._particles = self.to_dataframe(r, v, m, particle_type)
        self._particles.sort_values('r', inplace=True)
        self.time: Quantity['time'] = time.to(run_units.time)
        self.steps: int = int(steps)
        self.dt: Quantity['time'] = dt.to(run_units.time)
        self.distributions: list[Distribution] = utils.handle_default(distributions, [])
        self.Tdyn: Quantity['time']
        if Tdyn is not None:
            self.Tdyn = Tdyn
        elif len(self.distributions) > 0:
            self.Tdyn = self.distributions[0].Tdyn
        elif len(self.distributions) == 0:
            self.Tdyn = Quantity(1, run_units.time)
        self.background: Mass_Distribution | None = background
        self.Phi0: Quantity['energy'] = Phi0 if Phi0 is not None else physics.utils.Phi(self.r, self.M, self.m)[-1]
        self.snapshots: table.QTable = utils.handle_default(snapshots, table.QTable())
        self.save_every_n_steps = save_every_n_steps
        self.save_every_time: Quantity['time'] | None = (
            save_every_time if save_every_time is None else save_every_time.to(run_units.time)
        )
        self._dynamics_params: leapfrog.Params = leapfrog.normalize_params(dynamics_params, add_defaults=True)
        self._scatter_params: sidm.Params = sidm.normalize_params(scatter_params, add_defaults=True)
        self.max_allowed_subdivisions: int = max_allowed_subdivisions
        self.subdivide_on_scatter_chance: bool = subdivide_on_scatter_chance
        self.subdivide_on_gravitational_step: bool = subdivide_on_gravitational_step
        self.subdivide_on_startup: bool = subdivide_on_startup
        self.ministep_size: deque[float] = utils.handle_default(ministep_size, deque())
        self.scatter_track_time: deque[float] = utils.handle_default(scatter_track_time, deque())
        self.scatter_track_index: deque[NDArray[np.int64]] = utils.handle_default(scatter_track_index, deque())
        self.scatter_track_radius: deque[NDArray[np.float64]] = utils.handle_default(scatter_track_radius, deque())
        self._initial_particles = self._particles.copy()
        self.initial_particles = self.particles.copy()
        self.last_saved_time = last_saved_time
        self.scatter_rounds: deque[int] = utils.handle_default(scatter_rounds, deque())
        self.scatter_rounds_underestimated: deque[int] = utils.handle_default(scatter_rounds_underestimated, deque())
        self.hard_save: bool = hard_save
        self.save_path: Path | str | None = Path(save_path) if isinstance(save_path, str) else save_path
        self.Rmax: Quantity['length'] = Rmax.to(run_units.length)
        self.bootstrap_steps = bootstrap_steps
        self.scatters_to_collapse: int = scatters_to_collapse
        self.cleanup_nullish_particles = cleanup_nullish_particles
        self.cleanup_particles_by_radius = cleanup_particles_by_radius
        self.runtime_realtime_track: deque[float] = utils.handle_default(runtime_realtime_track, deque())
        self.runtime_track_sort: deque[float] = utils.handle_default(runtime_track_sort, deque())
        self.runtime_track_cleanup: deque[float] = utils.handle_default(runtime_track_cleanup, deque())
        self.runtime_track_sidm: deque[float] = utils.handle_default(runtime_track_sidm, deque())
        self.runtime_track_leapfrog: deque[float] = utils.handle_default(runtime_track_leapfrog, deque())
        self.runtime_track_full_step: deque[float] = utils.handle_default(runtime_track_full_step, deque())
        if generator is not None:
            self.rng = generator
            self.seed = generator.bit_generator.seed_seq.entropy
        else:
            self.seed = seed
            self.rng = np.random.default_rng(self.seed)
            if generator_state is not None:
                self.rng.bit_generator.state = generator_state

    def __repr__(self):
        scatter_params = dict(deepcopy(self.scatter_params))
        scatter_params['sigma'] = f'{scatter_params["sigma"].to("cm^2/g"):.1f}'
        description = {
            'Current time': f'{self.time:.1f}',
            'Time step size': f'{self.dt:.4f} = {self.dt.to(self.Tdyn):.1e}',
            '#particles': self.n_particles,
            'Save parameters': utils.drop_None(
                **{
                    'path': self.save_path,
                    'hard save': self.hard_save,
                    'Save every': f'{self.save_every_time:.1f}',
                    'Save every [n] steps': self.save_every_n_steps,
                }
            ),
            'Cleanup': {
                'NaN': self.cleanup_nullish_particles,
                'high radius': self.cleanup_particles_by_radius,
            },
            'Leapfrog parameters': dict(self.dynamics_params),
            'Scatter parameters': scatter_params,
            'Distributions': {f'#{i}': d for i, d in enumerate(self.distributions)},
        }

        description_strings = []
        for key, value in description.items():
            if isinstance(value, dict):
                description_strings += [f'{key}:']
                description_strings += [
                    '\n'.join([f'    {sub_key}: {sub_value}' for sub_key, sub_value in value.items()])
                ]
            else:
                description_strings += [f'{key}: {value}']
        return '\n'.join(description_strings)

    @classmethod
    def setup(
        cls,
        distributions: list[Distribution],
        n_particles: list[int | float],
        seed: int | None = None,
        **kwargs: Any,
    ) -> Self:
        """Initialize a Halo object from a given set of distributions.

        Parameters:
            distributions: List of distributions for each particle type.
            n_particles: List of number of particles for each particle type.
            seed: Seed for the random number generator.
            generator: If `None` use the default generator from `rng.generator`.
            kwargs: Additional keyword arguments, passed to the constructor.

        Returns:
            Halo object.
        """
        r, v, particle_type, m = [], [], [], []
        generator = np.random.default_rng(seed)
        Distribution.merge_distribution_grids(distributions)
        for distribution, n in zip(distributions, n_particles):
            # r_sub = distribution.sample_r(int(n)).to(run_units.length)
            # v_sub = distribution.sample_v(r_sub).to(run_units.velocity)
            r_sub, v_sub = distribution.sample(n, generator=generator)
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
            generator=generator,
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
        self.steps = 0
        self.last_saved_time = 0 * run_units.time
        self._particles = self._initial_particles.copy()
        self.scatter_rounds = deque()
        self.scatter_rounds_underestimated = deque()
        self.scatter_track_index = deque()
        self.scatter_track_radius = deque()
        self.snapshots = table.QTable()
        self.runtime_track_sort = deque()
        self.runtime_track_cleanup = deque()
        self.runtime_track_sidm = deque()
        self.runtime_track_leapfrog = deque()
        self.runtime_track_full_step = deque()
        self.rng = np.random.default_rng(self.seed)

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
                    self._particles = self._particles.iloc[:end].copy()
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
        """The first perpendicular component (to the radial direction) of the particle velocity."""
        return Quantity(self._particles['vx'], run_units.velocity)

    @property
    def vy(self) -> Quantity['velocity']:
        """The second perpendicular component (to the radial direction) of the particle velocity."""
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
        return np.array([len(x) / 2 for x in self.scatter_track_index], dtype=np.int64)

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

    def unit_mass(self, distribution: Distribution) -> Quantity['mass']:
        """Return the unit mass of the given distribution."""
        return distribution.Mtot / self.n_particles[distribution.particle_type]

    @property
    def generator_state(self) -> Mapping[str, Any]:
        """Get the current state of the random number generator."""
        return self.rng.bit_generator.state

    @property
    def scatter_times(self) -> Quantity['time']:
        """Wrap `self.scatter_track_time` as a Quantity."""
        return Quantity(np.hstack(self.scatter_track_time), run_units.time)

    @property
    def scatter_track_time_raveled(self) -> Quantity['time']:
        """Get a raveled array with the scatter time matching each particle in the hstack-ed `self.scatter_track_index`."""
        return Quantity(
            np.hstack([[t.value] * len(i) for i, t in zip(self.scatter_track_index, self.scatter_times)]),
            run_units.time,
        )

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

    def core_collapse_start_time(
        self,
        time_binning: Quantity['time'] = Quantity(100, 'Myr'),
        cutoff: int | float = 1e5,
        kind: str = 'linear',
    ) -> Quantity['time']:
        """Calculate the time at which the halo starts major core collapse.

        Defined as the time at which the halo first reaches `cutoff` scatters per `time_binning` time.

        Parameters:
            time_binning: The binning resolution to aggregate the number of scattering events.
            cutoff: The number of scatters per `time_binning` time at which the core collapse is considered to have started.
            kind: The kind of interpolation to use.

        Returns:
            The core collapse start time
        """
        n = int(time_binning / self.dt)
        return Quantity(
            scipy.interpolate.interp1d(
                *utils.joint_clean(
                    arrays=[
                        np.add.reduceat(self.n_scatters, np.arange(0, len(self.n_scatters), n)),
                        self.scatter_times.value[::n],
                    ]
                ),
                kind=kind,
                bounds_error=False,
                fill_value=np.inf,
            )(cutoff),
            self.dt.unit,
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

    def step(
        self,
        in_bootstrap: bool = False,
        subdivisions: int | None = None,
        save_kwargs: dict[str, Any] = {},
    ) -> None:
        """Perform a single time step of the simulation.

        Every step:
            - Sort particles by radius.
            - Cleanup erroneous particles.
            - Save a snapshot if it's time.
            - Perform scattering. This is done before the leapfrog integration since it doesn't modify the particle positions and thus doesn't require resorting.
            - Perform leapfrog integration.
            - Update simulation time.

        Parameters:
            in_bootstrap: Whether the simulation is in bootstrap mode.
            subdivisions: Number of time subdivisions within the step.
            save_kwargs: Keyword arguments for saving the snapshot.
        """

        if in_bootstrap or self.scatter_params.get('sigma', no_sigma) == no_sigma:
            subdivisions = 1
        elif subdivisions is None:
            subdivision_values: list[int] = []
            assert self.subdivide_on_scatter_chance or self.subdivide_on_gravitational_step, (
                'If subdivisioning is used, at least one of `subdivide_on_scatter_chance` or `subdivide_on_gravitational_step` must be True.'
            )
            if self.subdivide_on_scatter_chance:
                subdivision_values += [
                    sidm.fast_scatter_rounds(
                        scatter_chance=sidm.scatter_chance_shortcut(
                            r=self._particles.r,
                            vx=self._particles.vx,
                            vy=self._particles.vy,
                            vr=self._particles.vr,
                            dt=self.dt,
                            m=self._particles.m,
                            sigma=self.scatter_params.get('sigma', no_sigma),
                            max_radius_j=self.scatter_params.get('max_radius_j', 10),
                        ),
                        kappa=self.scatter_params.get('kappa', 1),
                        max_allowed_rounds=self.max_allowed_subdivisions,
                    ).max()
                ]
            if self.subdivide_on_gravitational_step:
                neighborhood_size = self.scatter_params.get('neighborhood_size', 10) // 3
                subdivision_values += [
                    np.ceil(
                        ((self._particles.vr * self.dt * np.sqrt(2)) / self._particles.r.diff(-neighborhood_size))
                        .abs()
                        .quantile(0.99)
                    ).astype(int)
                ]
            if self.subdivide_on_startup and self.time <= Quantity(1, 'Gyr'):
                subdivision_values += [self.max_allowed_subdivisions]
            if len(subdivision_values) == 0:
                subdivisions = 1
            else:
                subdivisions = int(min(self.max_allowed_subdivisions, max(1, *subdivision_values)))
            assert subdivisions is not None and subdivisions > 0, 'Error in subdivision calculation'

        for _ in range(subdivisions):
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
            dt = self.dt / subdivisions
            if not in_bootstrap and self.scatter_params.get('sigma', no_sigma) > no_sigma:
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
                    dt=dt,
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
            r, vx, vy, vr = leapfrog.step(r=r, vx=vx, vy=vy, vr=vr, m=m, M=self.M, dt=dt, **self.dynamics_params)
            self._particles['r'] = r
            self._particles['vx'] = vx
            self._particles['vy'] = vy
            self._particles['vr'] = vr

            self.runtime_track_leapfrog += [time.perf_counter() - t0]
            if not in_bootstrap:
                self.time += dt
                self.ministep_size += [dt.value]
                self.steps += 1
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
        if self.bootstrap_steps > 0 and self.steps == 0:
            start_time = self.time - self.bootstrap_steps * self.dt
            n_steps += self.bootstrap_steps
        else:
            start_time = self.time
        for step in tqdm(range(n_steps), start_time=cast(Quantity, start_time), dt=self.dt, **tqdm_kwargs):
            self.step(
                in_bootstrap=(step < self.bootstrap_steps and self.steps == 0),
                save_kwargs=save_kwargs,
                subdivisions=None if self.max_allowed_subdivisions != 1 else 1,
            )
            if (
                np.sign(t_after_core_collapse) >= 0
                and self.n_scatters.sum() > self.scatters_to_collapse * self.n_particles['dm']
            ):
                if self.time > self.core_collapse_time + t_after_core_collapse:
                    print(f'Core collapse detected at time {self.time}')
                    break
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

    @staticmethod
    def payload_keys() -> list[str]:
        """Return the keys of the payload dictionary, used for saving and loading halos. A `@staticmethod` and not a `@property` to allow getting it from an uninitialized cls during `@classmethod`."""
        return [
            'time',
            'steps',
            'dt',
            'distributions',
            'save_every_n_steps',
            'save_every_time',
            'dynamics_params',
            'scatter_params',
            'max_allowed_subdivisions',
            'subdivide_on_scatter_chance',
            'subdivide_on_gravitational_step',
            'subdivide_on_startup',
            'ministep_size',
            'scatter_track_time',
            'scatter_track_index',
            'scatter_track_radius',
            'background',
            'last_saved_time',
            'scatter_rounds',
            'scatter_rounds_underestimated',
            'hard_save',
            'save_path',
            'Rmax',
            'scatters_to_collapse',
            'cleanup_nullish_particles',
            'cleanup_particles_by_radius',
            'runtime_realtime_track',
            'runtime_track_sort',
            'runtime_track_cleanup',
            'runtime_track_sidm',
            'runtime_track_leapfrog',
            'runtime_track_full_step',
            'seed',
            'generator_state',
        ]

    @staticmethod
    def save_table(data: table.QTable, path: str | Path, **kwargs: Any) -> None:
        """Save a QTable to a file, splitting the strings from the Quantity data, and saving into `{}_strings.csv` and `{}.fits`."""
        data[[column for column in data.colnames if data[column].dtype != np.dtype('O')]].write(
            path.with_name(f'{path.stem}.fits'), **kwargs
        )
        data[[column for column in data.colnames if data[column].dtype == np.dtype('O')]].write(
            path.with_name(f'strings_{path.stem}.csv'), **kwargs
        )

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

    def save(
        self,
        path: str | Path | None = None,
        two_steps: bool = False,
        keep_last_backup: bool = False,
        split_snapshots: bool = True,
    ) -> None:
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
                if '_backup.' in file.name:
                    continue
                if file.is_dir():
                    shutil.copytree(file, file.with_stem(f'{file.stem}_backup'), dirs_exist_ok=True)
                else:
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
            (path / 'split_snapshots').mkdir(exist_ok=True)
            if len(self.snapshots) > 0:
                for i, group in enumerate(self.snapshots.group_by('time').groups):
                    self.save_table(group, path / f'split_snapshots/snapshot_{i}.fits', overwrite=True)

    @classmethod
    def load(cls, path: str | Path, update_save_path: bool = True, static: bool = False) -> Self:
        """Load the simulation state from a directory.

        Parameters:
            path: Save path to load from.
            update_save_path: Whether to update the internal save path to `path` (for example, if the directory was moved after the run).
            static: Whether to load the simulation with `hard_save=False` as a safeguard, to avoid accidently evolving the simulation on a completed run (that was loaded for analysis).

        Returns:
            The loaded Halo object
        """
        path = Path(path)
        with open(path / 'halo_payload.pkl', 'rb') as f:
            payload = {key: value for key, value in pickle.load(f).items() if key in cls.payload_keys()}

        tables = {}

        if (path / 'split_snapshots').exists():
            table_list = [cls.load_table(file) for file in (path / 'split_snapshots').glob('*.fits')]
            if len(table_list) > 0:
                tables['snapshots'] = cast(
                    table.QTable,
                    table.vstack([cls.load_table(file) for file in (path / 'split_snapshots').glob('*.fits')]),
                )
        elif (path / 'snapshots.fits').exists():
            tables['snapshots'] = cls.load_table(path / 'snapshots.fits')
        else:
            tables['snapshots'] = None

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
        if static:
            output.hard_save = False
        return output

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
        """Saves the plot."""
        if save_kwargs is None:
            return
        if 'name' in save_kwargs:
            save_kwargs['save_path'] = self.results_path / save_kwargs.pop('name')
        plot.save_plot(fig=fig, **save_kwargs)

    def fill_time_unit(self, unit: UnitLike) -> UnitLike:
        """If the `unit` is `Tdyn` return `self.Tdyn`. If it's `time step` return `self.time_step`, otherwise return `unit`."""
        if unit == 'Tdyn':
            return self.Tdyn
        elif unit == 'time step':
            return self.time_step
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
        core_collapse_start_time = self.core_collapse_start_time()
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
        save_kwargs: dict[str, Any] | None = None,
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
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.
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
            data = utils.slice_closest(data, value=filter_particle_type, key='particle_type')

        indices = indices if indices is not None else list(range(len(np.unique(np.array(data['time'])))))

        colors = sns.color_palette(color_palette, len(indices)) if color_palette is not None else None

        time_units = self.fill_time_unit(time_units)
        fig, ax = plot.setup_plot(
            **utils.drop_None(title=title, xlabel=utils.add_label_unit(xlabel, x_units)), **kwargs
        )
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

        self.save_plot(fig=fig, save_kwargs=save_kwargs)
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
            x_units: The x-axis units to use in the plot.
            ylabel: The label for the y-axis.
            label: The label for the histogram (legend).
            fig: The figure to plot on.
            ax: The axes to plot on.
            plt_kwargs: Additional keyword arguments to pass to the sns plotting function (`sns.histplot()` or `sns.kdeplot()`).
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.
            kwargs: Additional keyword arguments to pass to `plot.setup_plot()`.

        Returns:
            fig, ax.
        """
        x_units = plot.default_plot_unit_type(key, x_units)
        if filter_particle_type is not None:
            data = utils.slice_closest(data, value=filter_particle_type, key='particle_type')
        x = data[key].to(x_units)
        if x_range is not None:
            x = x[(x > x_range[0]) * (x < x_range[1])]
        if absolute:
            x = np.abs(x)
        params = {
            **plot.default_plot_text(key, x_units=x_units),
            **utils.drop_None(title=title, xlabel=xlabel, ylabel=ylabel),
        }
        fig, ax = plot.setup_plot(fig, ax, **params, **kwargs)
        if plot_type == 'kde':
            sns.kdeplot(x, cumulative=cumulative, ax=ax, label=label, **plt_kwargs)
        else:
            sns.histplot(x, cumulative=cumulative, ax=ax, stat=stat, label=label, **plt_kwargs)
        if x_plot_range is not None:
            ax.set_xlim(*x_plot_range.to(x_units).value)
        self.save_plot(fig=fig, save_kwargs=save_kwargs)
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
        save_kwargs: dict[str, Any] | None = None,
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
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.
            kwargs: Additional keyword arguments to pass to `plot.setup_plot()`.

        Returns:
            fig, ax.
        """
        fig, ax = self.plot_distribution(
            key='r',
            data=data,
            cumulative=cumulative,
            x_units=x_units,
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
                length_units=x_units,
                fig=fig,
                ax=ax,
                label=density_label,
                **params,
            )
        self.save_plot(fig=fig, save_kwargs=save_kwargs)
        return fig, ax

    def plot_phase_space_evolution(
        self,
        include_start: bool = True,
        include_now: bool = False,
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
            data = utils.slice_closest(data, value=filter_particle_type, key='particle_type')

        images = plot.evolution_to_images(
            data=data,
            plot_fn=lambda x: self.plot_phase_space(
                data=x,
                texts=[
                    {
                        's': f'{x["time"][0].to("Gyr"):.2f}',
                        **plot.pretty_ax_text(x=0.05, y=0.95, transform='transAxes'),
                    }
                ],
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
        filter_interacting: bool | None = None,
        mask: NDArray[np.bool_] | None = None,
        filter_indices: NDArray[np.int64] | list[int] | None = None,
        x_bins: Quantity['length'] = Quantity(np.linspace(1e-2, 35, 200), 'kpc'),
        y_bins: Quantity['velocity'] = Quantity(np.linspace(0, 60, 200), 'km/second'),
        x_key: str = 'r',
        y_key: str = 'v_norm',
        x_adjust_bins_edges_to_data: bool = False,
        y_adjust_bins_edges_to_data: bool = False,
        cmap: str = 'jet',
        transparent_value: float | None = 0,
        xlabel: str | None = 'Radius',
        ylabel: str | None = 'Velocity',
        x_units: UnitLike = 'kpc',
        y_units: UnitLike = 'km/second',
        return_grid: bool = False,
        adjust_data_to_EL: bool = False,
        **kwargs: Any,
    ) -> tuple[Figure, Axes] | tuple[Quantity, tuple[Quantity, Quantity, Quantity, Quantity]]:
        """Plot the phase space distribution of the data.

        Parameters:
            data: The data to plot.
            filter_particle_type: Whether to filter to only plot the specified particle type.
            filter_interacting: Whether to filter to only plot interacting/non-interacting particles based on `self.scatter_track_index`. If `None` ignores.
            filter_indices: Keep only the specified indices in `data` (based on the `particle_index` column).
            mask: Any additional mask to apply to the data. Must match the shape of the `data` (pre any other filtration).
            x_bins: The bins for the x-axis (mainly - radius). Also used to define the range to consider.
            y_bins: The bins for the y-axis (mainly - velocity). Also used to define the range to consider.
            x_adjust_bins_edges_to_data: Overwrite `x_bins` edges to match the data range.
            y_adjust_bins_edges_to_data: Overwrite `y_bins` edges to match the data range.
            x_key: The key for the x-axis in `data` (mainly - radius).
            y_key: The key for the y-axis in `data` (mainly - velocity).
            cmap: The colormap to use for the plot.
            transparent_value: Grid value to turn transparent (i.e. plot as `NaN`). If `None` ignores.
            xlabel: The label of the x-axis.
            ylabel: The label of the y-axis.
            x_units: Units to use for the x-axis.
            y_units: Units to use for the y-axis.
            return_grid: Return the grid+extent variables instead of plotting.
            adjust_data_to_EL: Adds a specific angular momentum column to the data (`data['L'] = data['r'] * data['vp']`) and transforms the energy to specific energy.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.plot_2d()`).

        Returns:
            fig, ax.
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

        if adjust_data_to_EL:
            data['L'] = data['r'] * cast(Quantity, data['vp'])
            data['E'] = data['E'] / cast(Quantity, data['m'])

        grid, extent = plot.aggregate_2d_data(
            data=data,
            x_key=x_key,
            y_key=y_key,
            x_bins=x_bins,
            y_bins=y_bins,
            x_adjust_bins_edges_to_data=x_adjust_bins_edges_to_data,
            y_adjust_bins_edges_to_data=y_adjust_bins_edges_to_data,
            data_x_units=x_units,
            data_y_units=y_units,
        )
        if return_grid:
            return grid, extent
        fig, ax = plot.plot_2d(
            grid=grid,
            extent=extent,
            xlabel=xlabel,
            ylabel=ylabel,
            x_units=x_units,
            y_units=y_units,
            cmap=cmap,
            transparent_value=transparent_value,
            **kwargs,
        )
        self.save_plot(fig=fig, **kwargs)
        return fig, ax

    def plot_phase_space_by_scattering_amount(
        self,
        data: table.QTable | None = None,
        bins: list[int] | NDArray[np.int64] = [10, 50, 100, 200, 400, 1000, 2000],
        save_basename: str = 'Phase space DM',
        **kwargs: Any,
    ) -> None:
        """Plot the phase space distribution of the data.

        Parameters:
            data: The data to plot. If `None` create a gif animation of all snapshots of the simulation.
            bins: The bin edges to use for the number of scatterings (the dividers between bins, the start and end will be added automatically).
            save_basename: Base name for the saved files (`f'{save_basename} [{low},{high}].png'`)
            kwargs: Additional keyword arguments to pass to the plot function (`self.plot_phase_space()`).

        Returns:
            fig, ax.
        """

        indices, counts = np.unique(np.hstack(self.scatter_track_index), return_counts=True)
        bins = np.hstack([0, 1, np.array(bins), indices.max() + 1])

        particle_index = self.initial_particles[self.initial_particles['particle_type'] == 'dm']['particle_index']

        for low, high in tqdm(list(zip(bins[:-1], bins[1:]))):
            mask = pd.Series(False, index=np.array(particle_index))
            if low == 0:
                mask.loc[mask.index.isin(indices)] = True
                mask = ~mask
                title = "Initial phase space distribution for particles that didn't scatter"
            else:
                mask.loc[mask.index.isin(indices[(counts >= low) * (counts < high)])] = True
                title = f'Initial phase space distribution for particles\nthat scattred between {low} and {high} nubmer of times'
            if data is None:
                self.plot_phase_space_evolution(
                    save_path=self.results_path / f'{save_basename} [{low},{high}].gif',
                    filter_particle_type='dm',
                    frame_plot_kwargs={**kwargs, 'filter_indices': np.array(mask[mask].index), 'title': title},
                )
            else:
                self.plot_phase_space(
                    data=cast(table.QTable, data[data['particle_type'] == 'dm']),
                    filter_particle_type='dm',
                    mask=np.array(mask),
                    title=title,
                    save_kwargs={'save_path': self.results_path / f'{save_basename} [{low},{high}].png'},
                    **kwargs,
                )
                plt.close()

    def plot_inner_core_density(
        self,
        include_start: bool = True,
        include_now: bool = False,
        radius: Quantity['length'] = Quantity(0.2, 'kpc'),
        filter_particle_type: ParticleType | None = None,
        time_units: UnitLike = 'Tdyn',
        xlabel: str | None = 'Time',
        ylabel: str | None = 'Particles',
        title: str | None = 'Particles in inner core ({radius})',
        aggregation_type: Literal['amount', 'percent'] = 'amount',
        title_radius_format: str = '.1f',
        label: str | None = None,
        fig: Figure | None = None,
        ax: Axes | None = None,
        line_kwargs: dict[str, Any] = {},
        save_kwargs: dict[str, Any] | None = None,
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
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.setup_plot()`).

        Returns:
            fig, ax.
        """
        base_data = self.get_particle_states(now=include_now, initial=include_start, snapshots=True)
        data_time_units = base_data['time'].unit
        data_length_units = base_data['r'].unit
        data = base_data[['r', 'time', 'particle_type']].to_pandas()

        if filter_particle_type is not None:
            data = utils.slice_closest(data, value=filter_particle_type, key='particle_type')
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

        fig, ax = plot.setup_plot(
            fig,
            ax,
            **utils.drop_None(title=title, xlabel=utils.add_label_unit(xlabel, time_units), ylabel=ylabel),
            **kwargs,
        )
        sns.lineplot(
            x=np.array(Quantity(agg_data['time'], data_time_units).to(time_units)),
            y=agg_data['in_radius'],
            ax=ax,
            label=label,
            **line_kwargs,
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
        length_units: UnitLike = 'kpc',
        time_units: UnitLike = 'Tdyn',
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
            length_units: Units to use for the radius axis.
            time_units: Units to use for the time axis.
            xlabel: Label for the radius axis.
            ylabel: Label for the time axis.
            cbar_label: Label for the colorbar.
            cmap: The colormap to use for the plot.
            row_normalization: The normalization to apply to each row. If `None` no normalization is applied. If `float` it must be a percentile value (between 0 and 1), and the normalization will be based on this quantile of each row.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.plot_2d()`).

        Returns:
            fig, ax.
        """

        time_units = self.fill_time_unit(time_units)
        data = self.get_particle_states(now=include_now, initial=include_start, snapshots=True)
        if filter_particle_type is not None:
            data = utils.slice_closest(data, value=filter_particle_type, key='particle_type')
        grid, extent = plot.aggregate_evolution_data(
            data=data,
            radius_bins=radius_bins,
            time_range=time_range,
            output_type='counts',
            row_normalization=row_normalization,
        )

        fig, ax = plot.plot_2d(
            grid=grid,
            extent=extent,
            x_units=length_units,
            y_units=time_units,
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
        specific_energy_units: UnitLike = 'km^2/second^2',
        length_units: UnitLike = 'kpc',
        time_units: UnitLike = 'Tdyn',
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
            data = utils.slice_closest(data, value=filter_particle_type, key='particle_type')
        grid, extent = plot.aggregate_evolution_data(
            data=data,
            radius_bins=radius_bins,
            time_range=time_range,
            output_type='temperature',
            row_normalization=row_normalization,
            output_grid_units=specific_energy_units,
        )

        fig, ax = plot.plot_2d(
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
        self.save_plot(fig=fig, **kwargs)
        return fig, ax

    def plot_heat_flux_evolution(
        self,
        include_start: bool = True,
        include_now: bool = True,
        filter_particle_type: ParticleType | None = None,
        radius_bins: Quantity = Quantity(np.linspace(1e-3, 5, 100), 'kpc'),
        time_range: Quantity | None = None,
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
        """Plot the heat flux evolution of the halo. Wraps `prep_2d_data()`.

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
            setup_kwargs: Additional keyword arguments to pass to `plot.setup_plot()`.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.plot_2d()`).

        Returns:
            fig, ax.
        """
        time_units = self.fill_time_unit(time_units)
        data = self.get_particle_states(now=include_now, initial=include_start, snapshots=True)
        if filter_particle_type is not None:
            data = utils.slice_closest(data, value=filter_particle_type, key='particle_type')
        grid, extent = plot.aggregate_evolution_data(
            data=data,
            radius_bins=radius_bins,
            time_range=time_range,
            v_axis=v_axis,
            output_type='specific heat flux',
            row_normalization=row_normalization,
            output_grid_units=heat_units,
        )

        fig, ax = plot.plot_2d(
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
        self.save_plot(fig=fig, **kwargs)
        return fig, ax

    def plot_scattering_location_evolution(
        self,
        radius_bins: Quantity = Quantity(np.linspace(1e-3, 1.2, 100), 'kpc'),
        time_range: Quantity | None = None,
        time_bin_size: Quantity | None | Literal['save cadence'] = 'save cadence',
        normalize_by_n_particles: bool = False,
        length_units: UnitLike = 'kpc',
        time_units: UnitLike = 'Tdyn',
        xlabel: str | None = 'Radius',
        ylabel: str | None = 'Time',
        cbar_label: str | None = 'Number of scattering events per {time}',
        cbar_label_time_units: UnitLike = 'Myr',
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
            length_units: Units to use for the radius axis.
            time_units: Units to use for the time axis.
            xlabel: Label for the radius axis.
            ylabel: Label for the time axis.
            cbar_label: Label for the colorbar.
            cbar_label_time_units: Units to use for time.
            cbar_label_time_format: Format string for time.
            cbar_log_scale: Whether to use a logarithmic scale for the colorbar.
            row_normalization: The normalization to apply to each row. If `None` no normalization is applied. If `float` it must be a percentile value (between 0 and 1), and the normalization will be based on this quantile of each row.
            cmap: The colormap to use for the plot.
            x_tick_format: Format string for the x-axis ticks.
            transparent_range: Range of values to turn transparent (i.e. plot as `NaN`). If `None` ignores.
            setup_kwargs: Additional keyword arguments to pass to `plot.setup_plot()`.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.plot_2d()`).

        Returns:
            fig, ax.
        """
        time_units = self.fill_time_unit(time_units)

        time_array = self.scatter_track_time_raveled_binned(time_bin_size).to(time_units)

        if cbar_label is not None:
            cbar_label = cbar_label.format(
                time=np.unique(time_array)
                .diff()[0]
                .to(cbar_label_time_units)
                .to_string(format='latex', formatter=cbar_label_time_format),
            )

        data = table.QTable(
            {
                'time': time_array,
                'r': Quantity(np.hstack(self.scatter_track_radius), run_units.length).to(length_units),
            }
        )
        grid, extent = plot.aggregate_evolution_data(
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

            location_grid, _ = plot.aggregate_evolution_data(
                data=location_data,
                radius_bins=radius_bins,
                time_range=time_range,
            )

            grid[location_grid == 0] = 0
            grid[location_grid != 0] = grid[location_grid != 0] / location_grid[location_grid != 0]

        fig, ax = plot.plot_2d(
            grid=grid,
            extent=extent,
            x_units=length_units,
            y_units=time_units,
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
        save_kwargs: dict[str, Any] | None = None,
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
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.
            **kwargs: Additional keyword arguments to pass to the distribution plotting function (`self.plot_distribution()`), for *both* distributions. Overwritten by start_kwargs/end_kwargs as needed.

        Returns:
            fig, ax.
        """
        time_units = self.fill_time_unit(time_units)
        fig, ax = self.plot_distribution(
            key=key,
            data=self.initial_particles,
            fig=fig,
            ax=ax,
            label=label_start,
            **{**kwargs, **start_kwargs},
        )
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
        self.save_plot(fig=fig, save_kwargs=save_kwargs)
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
        save_kwargs: dict[str, Any] | None = None,
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
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.

        Returns:
            fig, ax.

        """
        xlabel = utils.add_label_unit(xlabel, length_units)
        time_units = self.fill_time_unit(time_units)
        if title is not None:
            title = title.format(
                time=self.time.to(time_units).to_string(format='latex', formatter=time_format),
                n_scatters=self.n_scatters.sum(),
            )
        fig, ax = plot.setup_plot(
            fig, ax, figsize=figsize, minorticks=True, **utils.drop_None(title=title, xlabel=xlabel)
        )
        sns.histplot(
            Quantity(np.hstack(self.scatter_track_radius), run_units.length).to(length_units),
            ax=ax,
            log=True,
        )
        self.save_plot(fig=fig, save_kwargs=save_kwargs)
        return fig, ax

    def plot_scattering_distance(
        self,
        title: str | None = 'Interaction distance distribution',
        xlabel: str | None = 'Interaction distance',
        length_units: UnitLike = 'pc',
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
            length_units: Units to use for the radius axis.
            log_scale: Whether to use a logarithmic scale for the histogram.
            stat: Statistical function to use for the histogram, must be a valid input for `sns.histplot`.
            fig: Figure to use for the plot.
            ax: Axes to use for the plot.
            setup_kwargs: Additional keyword arguments to pass to `plot.setup_plot()`.
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.
            kwargs: Additional keyword arguments passed to `sns.histplot`.

        Returns:
            fig, ax.

        """
        fig, ax = plot.setup_plot(
            fig, ax, **utils.drop_None(title=title, xlabel=utils.add_label_unit(xlabel, length_units)), **setup_kwargs
        )
        radius = np.diff(np.hstack(self.scatter_track_radius).reshape(-1, 2)).ravel().to(length_units)
        sns.histplot(radius, log_scale=log_scale, stat=stat, ax=ax, **kwargs)
        self.save_plot(fig=fig, save_kwargs=save_kwargs)
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
            length_units: Units to use for the radius axis.
            time_units: Units to use for time.
            time_format: Format string for time.
            smooth_sigma: Smoothing factor for the density plot (sigma for a 1d Gaussian kernel).
            smooth_interpolate_kind: Interpolation kind for the density plot. Applied after the gaussian smoothing to further smooth the plot data.
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.setup_plot()`).

        Returns:
            fig, ax.
        """
        r = np.hstack(self.scatter_track_radius).to(length_units)
        r_bins = np.linspace(0, r.max(), num=num)
        dr = r_bins[1] - r_bins[0]
        density = Quantity(
            [
                ((r >= low) * (r < high)).sum() / (4 * np.pi * dr * ((low + high) / 2) ** 2)
                for low, high in zip(r_bins[:-1], r_bins[1:])
            ]
        )
        r_bins = r_bins[:-1]
        density_units = str(density.unit)
        interpolated_density = scipy.interpolate.interp1d(
            r_bins[density != 0], density[density != 0], kind=smooth_interpolate_kind
        )(r_bins)
        smoothed_density = scipy.ndimage.gaussian_filter1d(interpolated_density, sigma=smooth_sigma)

        xlabel = utils.add_label_unit(xlabel, length_units)
        ylabel = utils.add_label_unit(ylabel, density_units)
        time_units = self.fill_time_unit(time_units)
        if title is not None:
            title = title.format(
                time=self.time.to(time_units).to_string(format='latex', formatter=time_format),
                n_scatters=self.n_scatters.sum(),
            )
        fig, ax = plot.setup_plot(**kwargs, ax_set={'yscale': 'log'}, **utils.drop_None(title=title, xlabel=xlabel))
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
        ax_set: dict[str, Any] = {'xscale': 'log'},
        plot_labels: bool = True,
        bar_kwargs: dict[str, Any] = {'align': 'center', 'edgecolor': 'black', 'alpha': 0.7},
        save_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the distribution of the scattering particles by the number of scatterings (fraction per bin).

        Parameters:
            bins: The bin edges to use for the number of scatterings (the dividers between bins, the start and end will be added automatically).
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            title: Title for the plot.
            minorticks: Whether to add the grid for the minor ticks.
            ax_set: Additional keyword arguments to pass to `Axes.set()`. e.g `{'xscale': 'log'}`.
            plot_labels: Whether to add text bubbles with the y-value above the bins,
            bar_kwargs: Keyword arguments to pass to `Axes.bar()`.
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.setup_plot()`).

        Returns:
            fig, ax.
        """

        fig, ax = plot.setup_plot(
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            minorticks=minorticks,
            ax_set=ax_set,
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
        save_kwargs: dict[str, Any] | None = None,
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
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.setup_plot()`).

        Returns:
            fig, ax.
        """
        x = self.r
        time_units = self.fill_time_unit(time_units)
        local_density = self.local_density.to(density_units)
        smoothed_local_density = (
            scipy.ndimage.gaussian_filter1d(local_density, sigma=smooth_sigma) if smooth_sigma > 0 else local_density
        )
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
        self.save_plot(fig=fig, save_kwargs=save_kwargs)
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
        save_kwargs: dict[str, Any] | None = None,
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
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.
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
        sns.histplot(
            self.local_density.to(density_units),
            log_scale=log_scale,
            stat=stat,
            cumulative=cumulative,
            **hist_kwargs,
        )
        self.save_plot(fig=fig, save_kwargs=save_kwargs)
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
        save_kwargs: dict[str, Any] | None = None,
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
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.setup_plot()`).

        Returns:
            fig, ax.
        """
        return plot.plot_trace(
            key=key,
            data=data
            if data is not None
            else table.QTable(table.vstack([self.initial_particles, self.snapshots, self.particles])).copy(),
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
        undersample: int | None = None,
        xlabel: str | None = 'Time',
        ylabel: str | None = 'Cumulative number of scattering events',
        title: str | None = 'Cumulative number of scattering events',
        label: str | None = None,
        ax_set: dict[str, Any] = {'yscale': 'log'},
        lineplot_kwargs: dict[str, Any] = {},
        save_kwargs: dict[str, Any] | None = None,
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
            lineplot_kwargs: Additional keyword arguments to pass to `sns.lineplot()`.
            save_kwargs: Additional keyword arguments to pass to `plot.save_plot()`.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.setup_plot()`).

        Returns:
            fig, ax.
        """
        fig, ax = plot.setup_plot(
            xlabel=utils.add_label_unit(xlabel, time_unit),
            ylabel=ylabel,
            title=title,
            ax_set=ax_set,
            **kwargs,
        )
        x = self.scatter_times.to(time_unit)
        y = self.n_scatters.cumsum()
        if undersample is not None:
            x = x[::undersample]
            y = y[::undersample]
        sns.lineplot(
            x=x,
            y=y,
            ax=ax,
            label=label,
            **lineplot_kwargs,
        )
        if label is not None:
            ax.legend()
        self.save_plot(fig=fig, save_kwargs=save_kwargs)
        return fig, ax

    def plot_cumulative_scattering_amount_per_particle_over_time(
        self,
        time_unit: UnitLike = 'Gyr',
        xlabel: str | None = 'Time',
        ylabel: str | None = 'Cumulative number of scattering events',
        title: str | None = 'Mean cumulative number of scattering events per particle',
        label: str | None = None,
        per_dm_particle: bool = False,
        save_kwargs: dict[str, Any] | None = None,
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
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.setup_plot()`).

        Returns:
            fig, ax.
        """
        fig, ax = plot.setup_plot(xlabel=utils.add_label_unit(xlabel, time_unit), ylabel=ylabel, title=title, **kwargs)
        sns.lineplot(
            x=(np.arange(len(self.n_scatters)) * self.dt).to(time_unit),
            y=self.n_scatters.cumsum() / self.n_particles['dm'],
            ax=ax,
            label=label,
        )
        if label is not None:
            ax.legend()
        self.save_plot(fig=fig, save_kwargs=save_kwargs)
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
        ax_set: dict[str, str] | None = {'yscale': 'log'},
        save_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the number of scattering events over time, binned.

        Parameters:
            time_binning: Binning for the x-axis.
            time_unit: Units for the x-axis.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            title: Title for the plot.
            label: Label for the plot (legend).
            time_format: Format for the time in the title.
            title_time_unit: Units for the time displayed in the title.
            ax_set: Additional keyword arguments to pass to `Axes.set()`. e.g `{'yscale': 'log'}`.
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.setup_plot()`).

        Returns:
            fig, ax.
        """
        n = int(time_binning / self.dt)
        x = (np.arange(len(self.n_scatters)) * self.dt).to(time_unit)[::n]
        scatters = np.add.reduceat(self.n_scatters, np.arange(0, len(self.n_scatters), n))

        if title is not None:
            title = title.format(time=time_binning.to(title_time_unit).to_string(format='latex', formatter=time_format))

        fig, ax = plot.setup_plot(
            xlabel=utils.add_label_unit(xlabel, time_unit),
            ylabel=ylabel,
            title=title,
            ax_set=ax_set,
            **kwargs,
        )
        sns.lineplot(x=x[:-1], y=scatters[:-1], ax=ax, label=label)
        if label is not None:
            ax.legend()
        self.save_plot(fig=fig, save_kwargs=save_kwargs)
        return fig, ax

    def plot_scatter_rounds_over_time(
        self,
        rounds: bool = True,
        total_required: bool = True,
        underestimations: bool = False,
        time_unit: UnitLike = 'Gyr',
        xlabel: str | None = 'Time',
        ylabel: str | None = 'Number of scattering subdivisions per time step',
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
        """Plot the number of scattering rounds per `dt` time step, and the number of underestimations.

        Parameters:
            rounds: Plot the number of scattering rounds performed per `dt` time step.
            total_required: Plot the number of required scattering rounds per `dt` time step (regardless of what actually happened).
            underestimations: Plot the scattering rounds underestimated per `dt` time step ([required] - [actually happened]).
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
            ax_set: Additional keyword arguments to pass to `Axes.set()`.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.setup_plot()`).

        Returns:
            fig, ax.
        """
        fig, ax = plot.setup_plot(xlabel=utils.add_label_unit(xlabel, time_unit), ylabel=ylabel, title=title, **kwargs)
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

    def plot_distributions_rho(
        self,
        markers_on_first_only: bool = False,
        save_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the density profile (`rho`) of each of the provided distributions in the halo.

        Parameters:
            markers_on_first_only: If `True` only plot markers (`Rs` and `Rvir`) for the first density.
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.
            kwargs: Additional keyword arguments are passed to every call to the plotting function.

        Returns:
            fig, ax.
        """
        fig, ax = None, None
        for i, distribution in enumerate(self.distributions):
            fig, ax = distribution.plot_rho(
                label=f'{distribution.label} ({distribution.title})',
                fig=fig,
                ax=ax,
                add_markers=(i == 0 or not markers_on_first_only),
                **kwargs,
            )
        assert fig is not None and ax is not None
        self.save_plot(fig=fig, save_kwargs=save_kwargs)
        return fig, ax

    def plot_distributions_over_time(
        self,
        times: Quantity['time'] = Quantity([0, 1, 11, 14.5], 'Gyr'),
        data: table.QTable | None = None,
        include_start: bool = True,
        include_now: bool = False,
        labels: list[str] = ['start', 'max core', 'core collapse (start)', 'core collapse (deep)'],
        radius_bins: Quantity['length'] = Quantity(np.geomspace(3e-2, 5e2, 100), 'kpc'),
        limit_radius_by_Rvir: bool = True,
        distributions: list[int] | None = None,
        ax_set: dict[str, Any] = {'xscale': 'log', 'yscale': 'log'},
        fig: Figure | None = None,
        ax: Axes | None = None,
        setup_kwargs: dict[str, Any] = {},
        save_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the density profiles of the halo over time.

        Parameters:
            times: The times at which to plot the density profiles.
            data: The data to plot. If `None` the data will be loaded from the halo snapshots.
            include_start: Whether to include the initial particle distribution in the data. Ignored if `data` is provided.
            include_now: Whether to include the current particle distribution in the data. Ignored if `data` is provided.
            labels: The labels for the density profiles.
            radius_bins: The radius bins for the density profile calculations.
            limit_radius_by_Rvir: Whether to limit the radius bins by the virial radius.
            distributions: The distributions to plot (indices from `self.distributions`). If `None` plot all distributions.
            ax_set: Additional keyword arguments to pass to `Axes.set()`. e.g `{'xscale': 'log'}`.
            fig: The figure to plot on.
            ax: The axes to plot on.
            setup_kwargs: Additional keyword arguments to pass to `plot.setup_plot()`.
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.
            kwargs: Additional keyword arguments are passed to every call to the plotting function.

        Returns:
            fig, ax.
        """
        if data is None:
            data = self.get_particle_states(now=include_now, initial=include_start, snapshots=True)
        fig, ax = plot.setup_plot(fig=fig, ax=ax, ax_set=ax_set, **setup_kwargs)
        add_distribution_label = distributions is None or len(distributions) > 1

        for i, distribution in enumerate(self.distributions):
            if distributions is not None and i not in distributions:
                continue
            for label, t in zip(labels, times):
                sub = utils.slice_closest(utils.slice_closest(data, value=t), value='dm', key='particle_type')
                fig, ax = plot.plot_density(
                    cast(Quantity, sub['r']),
                    unit_mass=self.unit_mass(distribution),
                    bins=radius_bins
                    if not limit_radius_by_Rvir
                    else cast(Quantity, radius_bins[radius_bins <= distribution.Rvir]),
                    label=f'{distribution.label} {label}' if add_distribution_label else label,
                    fig=fig,
                    ax=ax,
                    **kwargs,
                )
        assert fig is not None and ax is not None
        self.save_plot(fig=fig, save_kwargs=save_kwargs)
        return fig, ax

    def plot_distributions_over_time_animation(
        self,
        radius_bins: Quantity['length'] = Quantity(np.geomspace(3e-2, 5e2, 100), 'kpc'),
        limit_radius_by_Rvir: bool = True,
        distributions: list[int] | None = None,
        xlim: list[Quantity['length'] | None | Literal['bins']] = ['bins', 'bins'],
        ylim: list[Quantity['mass density'] | None] = [
            Quantity([1e3, 1e11], 'Msun/kpc^3'),
            Quantity([1e-1, 1e7], 'Msun/kpc^3'),
        ],
        label_units: UnitLike = 'Gyr',
        label_format: str = '.1f',
        density_guidelines_kwargs: dict[str, Any] | None = {
            'times': Quantity([0, 1, 12], 'Gyr'),
            'line_kwargs': {'linestyle': '--'},
        },
        multiplicity_guidelines: int | None = 10,
        save_kwargs: dict[str, Any] = {},
    ) -> None:
        """Plot the density profiles of the halo as animations over time.

        Parameters:
            radius_bins: The radius bins for the density profile calculations.
            limit_radius_by_Rvir: Whether to limit the radius bins by the virial radius.
            distributions: The distributions to plot (indices from `self.distributions`). If `None` plot all distributions.
            xlim: List matching `distributions`. Consistent limits of the x-axis throughout the animation. If `None` ignores. If 'bins', uses the radius bins as the x-axis limits.
            ylim: List matching `distributions`. Consistent limits of the y-axis throughout the animation. If `None` ignores.
            label_units: Units for the time label.
            label_format: String format for the time label.
            density_guidelines_kwargs: Keyword arguments to pass to `plot.plot_distributions_over_time()` for plotting the density at fixed timestamps throughout the animation, serving as guidelines (i.e. initial distribution, max core, final distribution, etc.). If `None` doesn't plot the guidelines.
            multiplicity_guidelines: Number of frames to print for when the animation reaches the guidelines. If `None` ignores.
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. If it doesn't include `save_path`, it will use the default path (`self.results_path`).

        Returns:
            fig, ax.
        """

        data = self.get_particle_states()
        if 'save_path' in save_kwargs:
            save_path = Path(save_kwargs.pop('save_path'))
        else:
            save_path = self.results_path
        if 'name' in save_kwargs:
            save_path = save_path / save_kwargs.pop('name')

        for i, (distribution, xlim_, ylim_) in enumerate(zip(self.distributions, xlim, ylim)):
            bins = cast(
                Quantity,
                radius_bins if not limit_radius_by_Rvir else radius_bins[radius_bins <= distribution.Rvir],
            )
            if isinstance(xlim_, str) and xlim_ == 'bins':
                xlim_ = bins
            if xlim_ is not None:
                xlim_ = np.array(utils.to_extent(xlim_, force_array=True))
            if ylim_ is not None:
                ylim_ = np.array(utils.to_extent(ylim_, force_array=True))
            if distributions is not None and i not in distributions:
                continue

            if density_guidelines_kwargs is not None:
                unique_times = np.unique(
                    cast(
                        NDArray[np.float64],
                        data[data['particle_type'] == distribution.particle_type]['time'],
                    )
                )
                multiplicity_table = table.QTable(
                    {'time': unique_times, 'factor': np.ones(len(unique_times), dtype=np.int64)}
                )
                for t in density_guidelines_kwargs['times']:
                    multiplicity_table['factor'][np.abs(multiplicity_table['time'] - t).argmin()] = (
                        multiplicity_guidelines
                    )
                multiplicity = np.array(multiplicity_table['factor'], dtype=np.int64)
            else:
                multiplicity = None

            plot.save_images(
                images=plot.to_images(
                    iterator=data[data['particle_type'] == distribution.particle_type].group_by('time').groups,
                    plot_fn=lambda x: plot.plot_density(
                        x['r'],
                        unit_mass=self.unit_mass(distribution),
                        bins=bins,
                        label=f'{distribution.label} at {x["time"][0].to(label_units):{label_format}}',
                        xlim=xlim_,
                        ylim=ylim_,
                        fig=(
                            guidelines := self.plot_distributions_over_time(
                                radius_bins=radius_bins,
                                limit_radius_by_Rvir=limit_radius_by_Rvir,
                                distributions=[i],
                                **density_guidelines_kwargs,
                                xlim=xlim_,
                                ylim=ylim_,
                            )
                            if density_guidelines_kwargs is not None
                            else [None, None]
                        )[0],
                        ax=guidelines[1],
                    ),
                    multiplicity=multiplicity,
                    tqdm_kwargs={'desc': distribution.label},
                ),
                save_path=save_path.with_stem(f'{save_path.stem} {distribution.label}'),
                **save_kwargs,
            )

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
        x_units: UnitLike = 'kpc',
        cmap: str = 'jet',
        cbar_log_scale: bool = True,
        transparent_value: float | None = 0,
        xlabel: str | None = 'Radius',
        ylabel: str | None = 'Number of scattering events',
        title: str | None = 'Distribution by number of scattering events at {time}',
        title_suffix: str | None = None,
        cbar_label: str | None = 'Number of particles',
        time_unit: UnitLike = 'Gyr',
        time_format: str = '.1f',
        x_log: bool = True,
        y_log: bool = True,
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
            x_units: The units for the x-column in the data.
            cmap: The colormap to use for the plot.
            cbar_log_scale: Wheather to plot the cbar in a log scale.
            transparent_value: Grid value to turn transparent (i.e. plot as `NaN`). If `None` ignores.
            xlabel: The label for the x-axis.
            ylabel: The label for the y-axis.
            title: The title of the plot.
            cbar_label: Label for the colorbar.
            time_units: The time units to use in the plot's title.
            time_format: Format string for time to use in the plot's title.
            x_log: Sets the x-axis to a log scale.
            y_log: Sets the y-axis to a log scale.
            plot_method: Method to use for plotting.
            fig: Figure to plot on.
            ax: Axes to plot on.
            aggregate_kwargs: Additional keyword arguments to pass to the aggregation function (`plot.aggregate_2d_data()`).
            kwargs: Additional keyword arguments to pass to the plot function (`plot.plot_2d()`).

        Returns:
            fig, ax.
        """
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

        fig, ax = plot.plot_2d(
            *plot.aggregate_2d_data(
                data, x_key=x_key, y_key='n_scatters', x_bins=x_bins, y_bins=scatter_bins, **aggregate_kwargs
            ),
            plot_method=plot_method,
            x_range=x_bins,
            y_range=scatter_bins,
            cmap=cmap,
            x_units=x_units,
            y_units='',
            log_scale=cbar_log_scale,
            transparent_value=transparent_value,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            cbar_label=cbar_label,
            x_log=x_log,
            y_log=y_log,
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
        length_units: UnitLike = 'pc',
        time_units: UnitLike = 'Gyr',
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
            length_units: Units to use for distance.
            time_units: Units to use for time.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            title: The title of the plot.
            accuracy_factor: The width of the plotted error range (in units of standard deviations),
            plot_guidelines: A dictionary of guidelines for plotting. A dict with keys `times` and `labels`, where `times` is an array of time Quantities shaped (n_guidelines, 2), and `labels` is a list of strings of length n_guidelines that will be plotted at the center of each guideline tuple (row).
            texts: Overwrites the autogenerated text bubbles from `plot_guidelines`. If provided must be a list of dictionaries valid for `ax.text()`.
            vlines: Overwrites the autogenerated vertical lines from `plot_guidelines`. If provided must be a list of dictionaries valid for `ax.axvline()`.
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.
            kwargs: Additional keyword arguments passed to `plot.setup_kwargs()`

        Returns:
            fig, ax.
        """

        time_array = self.scatter_times
        values = []
        time_bins = []
        for time_range in tqdm(list(zip(bin_edges[:-1], bin_edges[1:]))):
            mask = (time_array >= time_range[0]) * (time_array <= time_range[1])
            if not mask.any():
                continue
            start, end = np.arange(len(mask))[mask][[0, -1]]
            values += [
                np.diff(np.hstack(list(self.scatter_track_radius)[start:end]).reshape(-1, 2)).ravel().to(length_units)
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
                            'x': plot_guidelines['times'].to(time_units).mean(1).value,
                            'y': [0.07] * 2,
                            'horizontalalignment': ['center'] * 2,
                            'verticalalignment': ['bottom'] * 2,
                        }
                    ).to_dict('records')
                ]
            if vlines is None:
                vlines = [
                    {'x': t, 'color': 'red', 'linestyle': '--', 'linewidth': 0.5}
                    for t in plot_guidelines['times'].to(time_units).ravel().value
                ]

        fig, ax = plot.setup_plot(
            **utils.drop_None(
                xlabel=utils.add_label_unit(xlabel, time_units),
                ylabel=utils.add_label_unit(ylabel, length_units),
                title=title,
            ),
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
