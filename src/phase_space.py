import pickle
from typing import Any, Literal, TypedDict, cast
from pathlib import Path

import numpy as np
import scipy
import pandas as pd
import seaborn as sns
from astropy import table, constants
from matplotlib import colors
from numpy.typing import NDArray
from astropy.units import Unit, Quantity, def_unit
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from astropy.units.typing import UnitLike

from . import rng, plot, utils, physics, run_units
from .tqdm import tqdm
from .physics.leapfrog import FactorGuessKwargs
from .distribution.distribution import Distribution


class PhaseSpace:
    """Phase space of a distribution function."""

    class SampleKwargs(TypedDict, total=False):
        """Keyword arguments for `PhaseSpace.sample_particles()`."""

        mass_cutoff: Quantity['mass'] | None
        N: int | None

    def __init__(
        self,
        distribution: Distribution | None = None,
        mass_grid: Quantity['mass'] | None = None,
        f_grid: Quantity[run_units.f_unit] | None = None,
        r_range: Quantity['length'] = Quantity(np.geomspace(1e-5, 1e3, 500), 'kpc'),
        v_range: Quantity['velocity'] = Quantity(np.linspace(0, 1e3, 200), 'km/second'),
        t: Quantity['time'] = Quantity(0, run_units.time),
        dt: Quantity['time'] | float = 1,
        gravitation_subdivision: int = 5,
        gravitation_mass_cutoff: Quantity['mass'] = Quantity(1e-1, 'Msun'),
        scatter_factor: float = 1,
        save_every_t: Quantity['time'] | None = Quantity(1, 'Myr'),
        generator: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> None:
        assert distribution is not None or mass_grid is not None or f_grid is not None, (
            'Either `distribution` or a `grid` must be provided.'
        )

        self.distribution = distribution
        self.r_array: Quantity = r_range.to(run_units.length)
        self.v_array: Quantity = v_range.to(run_units.velocity)
        self.gravitation_subdivision = gravitation_subdivision
        self.gravitation_mass_cutoff = gravitation_mass_cutoff
        self.scatter_factor = scatter_factor

        self.r_grid, self.v_grid = cast(tuple[Quantity, Quantity], np.meshgrid(self.r_array, self.v_array))

        self.dr, self.dv = map(utils.diff, [self.r_array, self.v_array])
        self.dr_grid, self.dv_grid = cast(tuple[Quantity, Quantity], np.meshgrid(self.dr, self.dv))
        self.volume_element = cast(Quantity, self.dv_grid * self.dr_grid)

        if self.distribution is not None:
            self.f_grid = self.distribution.f(v=self.v_grid, r=self.r_grid)
        elif f_grid is not None:
            self.f_grid = f_grid
        elif mass_grid is not None:
            self.mass_grid = mass_grid
        else:
            raise NotImplementedError('Either `distribution`, `f_grid`, or `mass_grid` must be provided.')

        self.mass_grids: Quantity = self.mass_grid[np.newaxis, ...].copy()
        self.grids_time: Quantity = t[np.newaxis, ...].copy()

        self.initial_f_grid: Quantity = self.f_grid.copy()
        self.initial_grid: Quantity = self.grid.copy()
        self.initial_mass_grid: Quantity = self.mass_grid.copy()

        if generator is None:
            self.seed = seed
            self.generator = np.random.default_rng(self.seed)
            if self.seed is None:
                self.seed = rng.get_seed(self.generator)
        else:
            self.generator = generator
            self.seed = rng.get_seed(self.generator)

        if self.distribution is not None:
            self.Tdyn = self.distribution.Tdyn
        else:
            self.Tdyn = self.calculate_Tdyn()
        self.time: Quantity = t.copy()
        self.dt: Quantity = (dt if isinstance(dt, Quantity) else self.Tdyn * dt).to(run_units.time)
        self.save_every_t = save_every_t

    @classmethod
    def from_particles(
        cls,
        distribution: Distribution,
        data: table.QTable | pd.DataFrame | None = None,
        snapshots: table.QTable | None = None,
        r: Quantity['length'] | None = None,
        vx: Quantity['velocity'] | None = None,
        vy: Quantity['velocity'] | None = None,
        vr: Quantity['velocity'] | None = None,
        m: Quantity['mass'] | None = None,
        t: Quantity['time'] | None = None,
        r_range: Quantity['length'] | int | None = 50,
        v_range: Quantity['velocity'] | int | None = 50,
        verbose: bool = True,
        **kwargs: Any,
    ) -> 'PhaseSpace':
        """Creates a phase space object from a set of particles' positions and velocities.

        Parameters:
            distribution: The base distribution for the particles. Defines metadata and doesn't need to be exact.
            data: The particles' data. If provided prefered over `r`, `vx`, `vy`, `vr`, and `m`. If provided as a `DataFrame`, assumes the default units, otherwise (`QTable`) draws directly from the table.
            snapshots: The particles' data snapshots. If provided prefered over `data`. The latest time will be treated as "current", and other time snapshots will be saved to `mass_grids`.
            r: The particles' position. Mandatory if `data=None`, otherwise ignored.
            vx: The particles' first perpendicular component (to the radial direction) of the velocity. Mandatory if `data=None`, otherwise ignored.
            vy: The particles' second perpendicular component (to the radial direction) of the velocity. Mandatory if `data=None`, otherwise ignored.
            vr: The particles' radial velocity. Mandatory if `data=None`, otherwise ignored.
            m: The particles' mass. Mandatory if `data=None`, otherwise ignored.
            t: The current time.
            r_range: Passed on to the `PhaseSpace` constructor to define the integration grid. If an `int`, create a logarithmic grid from the minimum to maximum radiuses with the provided number of posts. If `None` use the default constructor value.
            v_range: Passed on to the `PhaseSpace` constructor to define the integration grid. If an `int` create a logarithmic grid from the minimum to maximum velocity norms with with the provided number of posts. If `None` use the default constructor value.
            verbose: Whether to print progress information when loading snapshots.
            **kwargs: Additional keyword arguments passed to the `PhaseSpace` constructor.
        """
        if snapshots is not None:
            groups = snapshots.group_by('time').groups
            data = groups[-2]
            if isinstance(r_range, int):
                r = utils.get_columns(snapshots, ['r'])[0]
                r_range = cast(Quantity, np.geomspace(r.min(), r.max(), r_range))
            if isinstance(v_range, int):
                vx, vy, vr = utils.get_columns(snapshots, ['vx', 'vy', 'vr'])
                v = np.sqrt(vx**2 + vy**2 + vr**2)
                v_range = cast(Quantity, np.geomspace(v.min(), v.max(), v_range))
        else:
            groups = None
        if data is not None:
            if isinstance(data, table.QTable):
                r, vx, vy, vr, m = utils.get_columns(data, columns=['r', 'vx', 'vy', 'vr', 'm'])
                if 'time' in data.columns:
                    t = utils.get_columns(data, ['time'])[0]
            else:
                r, vx, vy, vr, m = (
                    Quantity(data['r'], run_units.length),
                    Quantity(data['vx'], run_units.velocity),
                    Quantity(data['vy'], run_units.velocity),
                    Quantity(data['vr'], run_units.velocity),
                    Quantity(data['m'], run_units.mass),
                )
                if 'time' in data.columns:
                    t = Quantity(data['time'], run_units.time)
        assert r is not None, 'Failed to parse `r`'
        assert vx is not None, 'Failed to parse `vx`'
        assert vy is not None, 'Failed to parse `vy`'
        assert vr is not None, 'Failed to parse `vr`'
        assert m is not None, 'Failed to parse `m`'
        r, vx, vy, vr, m = utils.unmask_quantity(r, vx, vy, vr, m)

        if isinstance(r_range, int):
            r_range = cast(Quantity, np.geomspace(r.min(), r.max(), r_range))
        if isinstance(v_range, int):
            v = np.sqrt(vx**2 + vy**2 + vr**2)
            v_range = cast(Quantity, np.geomspace(v.min(), v.max(), v_range))
        ps = cls(distribution, **utils.drop_None(r_range=r_range, v_range=v_range, t=t), **kwargs)
        ps.mass_grid = ps.particles_to_mass_grid(r=r, vx=vx, vy=vy, vr=vr, m=m)
        if groups is not None:
            mass_grids = []
            grids_time = []
            for group in tqdm(groups, desc='Creating snapshot grids', disable=not verbose):
                r, vx, vy, vr, m, t = utils.get_columns(group, ['r', 'vx', 'vy', 'vr', 'm', 'time'])
                mass_grids += [ps.particles_to_mass_grid(r=r, vx=vx, vy=vy, vr=vr, m=m)]
                grids_time += [t[0]]
            ps.mass_grids = np.stack(mass_grids)
            ps.grids_time = Quantity(grids_time)
        return ps

    @property
    def jacobian_r(self) -> Quantity:
        """Jacobian of the distribution function with respect to radius"""
        return 4 * np.pi * self.r_array**2

    @property
    def jacobian_v(self) -> Quantity:
        """Jacobian of the distribution function with respect to velocity"""
        return 4 * np.pi * self.v_array**2

    @property
    def jacobian_rv(self) -> Quantity:
        """2d Jacobian of the distribution function"""
        return self.jacobian_v[..., np.newaxis] * self.jacobian_r[np.newaxis, ...]

    @property
    def mass_grid(self) -> Quantity['mass']:
        """Mass grid of the distribution function"""
        return (self.grid * self.volume_element).to(run_units.mass)

    @mass_grid.setter
    def mass_grid(self, grid: Quantity['mass']) -> None:
        self.grid = grid / self.volume_element

    @property
    def probability_grid(self) -> NDArray[np.float64]:
        """Probability grid of the distribution function (normalized mass grid)"""
        return self.mass_grid / self.Mtot

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the phase space grid"""
        return self.grid.shape

    @property
    def size(self) -> int:
        """Size of the phase space grid"""
        return self.grid.size

    @property
    def grid(self) -> Quantity:
        """Grid of the distribution function with the Jacobian included"""
        return (self.jacobian_rv * self.f_grid).to(
            run_units.f_unit * run_units.length**2 * run_units.velocity**2 * run_units.mass
        )

    @grid.setter
    def grid(self, grid: Quantity) -> None:
        self.f_grid = (grid / self.jacobian_rv).to(run_units.f_unit * run_units.mass)

    @property
    def Mtot(self) -> Quantity['mass']:
        """Total mass of the distribution function"""
        return self.mass_grid.sum()

    def calculate_Tdyn(self, mass_grid: Quantity | None = None) -> Unit:
        """Calculate the dynamical time of the distribution function. If `mass_grid` is not provided, the internal (current) grid will be used."""
        r_m = Quantity(scipy.stats.gmean(self.r_array.value), self.r_array.unit)
        M_m = Quantity(
            scipy.interpolate.interp1d(
                x=self.r_array.value,
                y=(M := (mass_grid if mass_grid is not None else self.mass_grid)).sum(axis=0).cumsum().value,
            )(r_m.value),
            M.unit,
        )

        # https://galaxiesbook.org/chapters/I-01.-Gravitation_3-Circular-velocity-and-dynamical-time.html
        # $ T = sqrt((3 pi)/(G dot expval(rho))) = sqrt((3 pi)/(G dot (M(<r_m))/(4/3 pi r_m^3))) = 2 pi sqrt(r^3/(G dot M)) $
        # Should it have the factor of 2*pi at the start? the paper drops it.
        return def_unit(
            'Tdyn',
            2 * np.pi * np.sqrt(r_m**3 / (constants.G * M_m)).decompose(run_units.system),
            doc='phase space dynamic time',
        )

    def to_f_grid(self, mass_grid: Quantity['mass']) -> Quantity[run_units.f_unit]:
        """Calcualte the f grid from the provided mass grid"""
        return (mass_grid / (self.jacobian_rv * self.volume_element)).decompose(run_units.system)

    def integrate(
        self,
        integrand: Quantity | None = None,
        full_grid: Quantity | None = None,
        axis: Literal['r', 'v', 'rv'] = 'rv',
    ) -> Quantity:
        """Integrate the grid over the provided axis.

        Parameters:
            integrand: The integrand to integrate over the grid. Either `integrand` or `full_grid` must be provided. Will perform: `integral (integrand)*f*J dV`
            full_grid: Same as `integrand` but assumes the distribution function (df) is already contained in the input, i.e. calculating `integral (full_grid)*J dV`. Either `integrand` or `full_grid` must be provided, and `full_grid` takes priority.
            axis: The axis to integrate over. `r` gives `G(v) = integral g(r,v)*4*pi*r^2 dr`, `v` gives `G(r) = integral g(r,v)*4*pi*r^2 v^2 dv`, and `rv` gives `G = integral g(r,v)*8*pi^2*r^2*v^2 drdv`
        """
        assert integrand is not None or full_grid is not None, 'Either `integrand` or `full_grid` must be provided'
        if full_grid is not None:
            grid = full_grid
        else:
            grid = integrand * self.f_grid
        if axis == 'r':
            return (grid * self.jacobian_r[np.newaxis, ...] * self.dr_grid).sum(axis=1)
        elif axis == 'v':
            return (grid * self.jacobian_v[..., np.newaxis] * self.dv_grid).sum(axis=0)
        else:
            return (grid * self.jacobian_rv * self.volume_element).sum()

    def calculate_rho(self, mass_grid: Quantity['mass']) -> Quantity['mass density']:
        """Calcualte the density as a function of radius (without the Jacobian) for the provided mass grid"""
        return self.integrate(full_grid=self.to_f_grid(mass_grid), axis='v').to(run_units.density)

    def calculate_temperature(self, mass_grid: Quantity['mass']) -> Quantity['specific energy']:
        """Calcualte the temperature as a function of radius for the provided mass grid"""
        num = self.integrate(full_grid=self.v_grid**2 * self.to_f_grid(mass_grid), axis='v')
        den = self.calculate_rho(mass_grid)
        temperature = Quantity(np.full(num.shape, np.nan), cast(Unit, num.unit) / cast(Unit, den.unit))
        temperature[den != 0] = num[den != 0] / den[den != 0]
        return temperature.to(run_units.specific_energy)

    @property
    def rho(self) -> Quantity['mass density']:
        """Density as a function of radius (without the Jacobian)"""
        return self.calculate_rho(self.mass_grid)

    @property
    def temperature(self) -> Quantity['specific energy']:
        """Temperature as a function of radius"""
        return self.calculate_temperature(self.mass_grid)

    def fill_time_unit(self, unit: UnitLike) -> UnitLike:
        """If the `unit` is `Tdyn` return `self.Tdyn`. Otherwise return `unit`."""
        if unit == 'Tdyn':
            return self.Tdyn
        return unit

    def is_save_round(self) -> bool:
        """Check if it's time to save a snapshot"""
        return bool(self.grids_time[-1] + self.save_every_t < self.time)

    def save_snapshot(self, grid: Quantity['mass'] | None = None, t: Quantity['time'] | None = None):
        """Save a snapshot of the mass grid of the distribution function.

        If an argument isn't provided, the object's phase space value will be used."""
        if t is None:
            t = self.time
        if grid is None:
            grid = self.mass_grid
        self.mass_grids = np.concatenate([self.mass_grids, grid[np.newaxis, ...].copy()], axis=0)
        self.grids_time = np.hstack([self.grids_time, t])

    @property
    def snapshots(self) -> list[tuple[Quantity['mass'], Quantity['time']]]:
        """List of snapshots of the distribution function with the appropriate time"""
        return list(zip(self.mass_grids, self.grids_time))

    def closest_snapshots(self, times: Quantity) -> list[tuple[Quantity['mass'], Quantity['time']]]:
        """List of the closest snapshots for each specified time"""
        indices = np.unique([np.argmin(np.abs(self.grids_time - t)) for t in times])
        return list(zip(self.mass_grids[indices], self.grids_time[indices]))

    def match_1d_index(self, value: Quantity, axis: Literal['r', 'v']) -> NDArray[np.int64]:
        """Match the value's index in the coordinate array (find it's position on the grid in that axis)"""
        array = self.r_array if axis == 'r' else self.v_array
        return (np.searchsorted(array, value, side='right') - 1).clip(min=0, max=len(array) - 2)

    def particles_to_mass_grid(
        self,
        r: Quantity['length'],
        vx: Quantity['velocity'],
        vy: Quantity['velocity'],
        vr: Quantity['velocity'],
        m: Quantity['mass'],
    ) -> Quantity['mass']:
        """Burn a set of particles into a mass grid"""
        v = cast(Quantity, np.sqrt(vx**2 + vy**2 + vr**2))

        return Quantity(
            np.bincount(
                np.ravel_multi_index(
                    (self.match_1d_index(v, 'v'), self.match_1d_index(r, 'r')),
                    self.grid.shape,
                ),
                weights=m.ravel(),
                minlength=self.grid.size,
            ).reshape(self.grid.shape),
            self.mass_grid.unit,
        )

    def sample_particles(
        self,
        mass_cutoff: Quantity['mass'] | None = None,
        N: int | None = None,
    ) -> tuple[
        Quantity['length'],
        Quantity['velocity'],
        Quantity['velocity'],
        Quantity['velocity'],
        Quantity['mass'],
        Quantity['mass'],
        Quantity['mass'],
    ]:
        """Sample particles from the phase space distribution.

        Parameters:
            mass_cutoff: The minimum grid cell mass to consider for sampling particles. If `None`, use `self.gravitation_mass_cutoff`.
            N: The number of particles to sample per grid cell. If `None`, use `self.gravitation_subdivision`.

        Returns:
            r: The positions of the sampled particles.
            vx: The first perpendicular velocities of the sampled particles.
            vy: The second perpendicular velocities of the sampled particles.
            vr: The radial velocities of the sampled particles.
            m: The masses of the sampled particles.
            M: The cumulative masses enclosed by the radiuses of the sampled particles.
            unused_mass: The mass_grid with the cells that were not sampled (and zero elsewhere).
        """
        if mass_cutoff is None:
            mass_cutoff = self.gravitation_mass_cutoff
        if N is None:
            N = self.gravitation_subdivision
        mask = self.mass_grid.ravel() > mass_cutoff
        unused_mass = cast(Quantity, np.where(~mask.reshape(self.mass_grid.shape), self.mass_grid, 0).copy())

        r = Quantity(
            np.random.uniform(
                self.r_grid.ravel()[mask], (self.r_grid + self.dr_grid).ravel()[mask], (N, mask.sum())
            ).ravel(),
            self.r_grid.unit,
        )
        v_norm = Quantity(
            np.random.uniform(
                self.v_grid.ravel()[mask], (self.v_grid + self.dv_grid).ravel()[mask], (N, mask.sum())
            ).ravel(),
            self.v_array.unit,
        )
        vx, vy, vr = cast(tuple[Quantity, Quantity, Quantity], utils.split_3d(v_norm))

        m = cast(Quantity, np.tile(self.mass_grid.ravel()[mask] / N, N))
        M = cast(Quantity, np.tile(np.repeat(self.mass_grid.sum(axis=0).cumsum(), len(self.v_array))[mask], N))

        return r, vx, vy, vr, m, M, unused_mass

    def sample_weighted_particles(
        self, n_particles: int | float, generator: np.random.Generator | None = None, velocity_d3: bool = True
    ) -> tuple[Quantity['length'], Quantity['velocity']]:
        """Samples a fixed number of particles from the phase space weighted by their probability (normalized phase space mass distribution):
            - The mass distribution grid is normalized to sum to 1, each pixel is sampled weighted by it's value.
            - The sampled indices are jittered to provide a uniform random value from within the sampled pixel.
            - Depending on `velocity_d3`, the sampled velocities are split into 3d coordinates.

        Parameters:
            n_particles: Number of particles to sample.
            generator: If not provided, use the object's generator.
            velocity_d3: If `True` return the velocity split into 3d coordinates, otherwise return the velocity norm.

        Returns:
            A tuple of two vectors:
                Sampled radius values for each particle, shaped `(num_particles,)`
                Either (based on `velocity_d3`):
                    - Corresponding 3d velocities for each particle, shaped `(num_particles,3)`.
                    OR
                    - Corresponding velocity norm for each particle, shaped `(num_particles,)`.

        """
        if generator is None:
            generator = self.generator

        indices = np.unravel_index(
            generator.choice(a=self.size, size=int(n_particles), p=self.probability_grid.ravel()),
            self.shape,
        )

        v_indices, r_indices = indices + generator.uniform(
            [[0], [0]],
            [[1], [1]],
            (2, int(n_particles)),
        )

        r_indices, v_indices = r_indices[index_sort := np.argsort(r_indices)], v_indices[index_sort]

        v_interp = scipy.interpolate.interp1d(np.arange(self.shape[0]), self.v_array.value)
        r_interp = scipy.interpolate.interp1d(np.arange(self.shape[1]), self.r_array.value)

        radius, velocity = (
            Quantity(r_interp(r_indices), self.r_array.unit),
            Quantity(v_interp(v_indices), self.v_array.unit),
        )
        if velocity_d3:
            velocity = utils.split_3d_quantity(velocity, generator=self.generator)
        return radius, velocity

    def particles_gravitational_step(
        self,
        r: Quantity['length'],
        vx: Quantity['velocity'],
        vy: Quantity['velocity'],
        vr: Quantity['velocity'],
        m: Quantity['mass'],
        M: Quantity['mass'],
        max_minirounds: int = 5,
        raise_warning: bool = False,
        factors: NDArray[np.int64] | None = None,
        **kwargs: Any,
    ) -> tuple[Quantity['length'], Quantity['velocity'], Quantity['velocity'], Quantity['velocity']]:
        """Advance a set of particles gravitationally in time.

        Parameters:
            r: The positions of the sampled particles.
            vx: The first perpendicular velocities of the sampled particles.
            vy: The second perpendicular velocities of the sampled particles.
            vr: The radial velocities of the sampled particles.
            m: The masses of the sampled particles.
            M: The cumulative masses enclosed by the radiuses of the sampled particles.
            max_minirounds: Maximum number of mini-rounds to perform.
            raise_warning: Raise a warning if a particle fails to converge.
            kwargs: Additional keyword arguments to pass to the leapfrog step function.

        Returns:
            The post-step positions and velocities of the particles.
        """
        r_, vx_, vy_, vr_, _ = physics.leapfrog.step(
            r=r,
            vx=vx,
            vy=vy,
            vr=vr,
            m=m,
            M=M,
            dt=self.dt,
            factor=factors,
            max_minirounds=max_minirounds,
            grid_window_radius=0,
            raise_warning=raise_warning,
            **kwargs,
        )
        return Quantity(r_, r.unit), Quantity(vx_, vx.unit), Quantity(vy_, vy.unit), Quantity(vr_, vr.unit)

    def gravitational_step(
        self,
        sample_kwargs: SampleKwargs = {},
        guess_dt_factor: bool = True,
        guess_dt_factor_kwargs: FactorGuessKwargs = FactorGuessKwargs(base=10),
        **kwargs: Any,
    ):
        """Perform a gravitational step on the phase space:
        1. Sample particles from the distribution function.
        2. Perform gravitational step on particles.
        3. Reintegrate the particles to the mass grid.

        Parameters:
            sample_kwargs: Keyword arguments for sampling particles.
            guess_dt_factor: Guess the value of `factor` for each particle based on `r/v`.
            **kwargs: Additional keyword arguments for the gravitational step.
        """
        r, vx, vy, vr, m, M, unused_mass = self.sample_particles(**sample_kwargs)
        r, vx, vy, vr = self.particles_gravitational_step(
            r=r,
            vx=vx,
            vy=vy,
            vr=vr,
            m=m,
            M=M,
            guess_dt_factor=guess_dt_factor,
            guess_dt_factor_kwargs=guess_dt_factor_kwargs,
            **kwargs,
        )
        self.mass_grid = unused_mass + self.particles_to_mass_grid(r=r, vx=vx, vy=vy, vr=vr, m=m)

    def evolve(
        self,
        n_steps: int | None = None,
        t: Quantity['time'] | None = None,
        until_t: Quantity['time'] | None = None,
        gravitational_step_kwargs: dict[str, Any] = {},
        tqdm_kwargs: dict[str, Any] = {},
    ):
        """Evolve the phase space.

        Parameters:
            n_steps: Number of steps to evolve for. Takes precedence over `t`.
            t: Time to evolve for. Ignored if `n_steps` is specified.
            until_t: Evolve until this time. Ignored if `n_steps` or `t` are specified.
            tqdm_kwargs: Additional keyword arguments to pass to `tqdm` (NOTE this is the custom submodule defined in this project at `tqdm.py`).
        """

        if n_steps is None:
            if t is not None:
                n_steps = int(t / self.dt)
            elif until_t is not None:
                if self.time > until_t:
                    raise ValueError('current time is greater than the specified end time')
                n_steps = int((until_t - self.time) / self.dt)
            else:
                raise ValueError('Either `n_steps`, `t`, or `until_t` must be specified')

        for _ in tqdm(range(n_steps), start_time=self.time.copy(), dt=self.dt):
            # mini_scattering(r_grid=r_grid,v_grid=v_grid,mass_grid=mass_grid,dr=dr,dv=dv,dt=dt,k=k)
            self.gravitational_step(**gravitational_step_kwargs)
            self.time += self.dt
            if self.is_save_round():
                self.save_snapshot()

    def save(self, save_path: str | Path) -> None:
        """Save the phase space object to a pickle file."""
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> 'PhaseSpace':
        """Load the phase space object from a pickle file."""
        with open(path, 'rb') as f:
            return pickle.load(f)

    def plot_grid(
        self,
        grid: Quantity | None = None,
        t: Quantity['time'] | None = None,
        reduce_resolution: int = 1,
        x_unit: UnitLike = 'kpc',
        y_unit: UnitLike = 'km/second',
        xlabel: str = 'Radius',
        ylabel: str = 'Velocity',
        plot_method: Literal['pcolormesh', 'imshow'] = 'pcolormesh',
        cmap: str = 'jet',
        norm: colors.Normalize = colors.LogNorm(1, 1e7),
        title: str | None = 'Phase space distribution at time={time}',
        title_time_unit: UnitLike | str = 'Tdyn',
        title_time_format: str = '.4f',
        cbar_label: str | None = None,
        cbar_label_autosuffix: bool = False,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the given grid as a heatmap If `grid` is `None`, use the current mass grid. All parameters are passed on to `plot.heatmap()`."""
        if t is None:
            t = self.time
        if title is not None:
            title = title.format(time=f'{t.to(self.fill_time_unit(title_time_unit)):{title_time_format}}')
        if grid is None and cbar_label is None:
            cbar_label = 'Mass'
        grid = grid if grid is not None else self.mass_grid
        if reduce_resolution > 1:
            r_array = cast(Quantity, self.r_array[::reduce_resolution])
            v_array = cast(Quantity, self.v_array[::reduce_resolution])
            grid = cast(
                Quantity,
                grid.reshape(len(v_array), reduce_resolution, len(r_array), reduce_resolution).sum(axis=(1, 3)),
            )
        else:
            r_array = self.r_array
            v_array = self.v_array

        return plot.heatmap(
            grid=grid,
            x_range=r_array,
            y_range=v_array,
            plot_method=plot_method,
            x_unit=x_unit,
            y_unit=y_unit,
            xlabel=xlabel,
            ylabel=ylabel,
            cmap=cmap,
            norm=norm,
            title=title,
            cbar_label=cbar_label,
            cbar_label_autosuffix=cbar_label_autosuffix,
            **kwargs,
        )

    def animate_mass_grid(self, save_path: str | Path, undersample: int | None = None, **kwargs: Any) -> None:
        """Animate the phase space (mass grid) of the distribution function, and save as a gif.

        Parameters:
            save_path: Path to save the animation.
            undersample: Undersample the phase space snapshots by this factor. Ignored if `None`.
            **kwargs: Additional keyword arguments passed to `plot_grid()`.
        """
        plot.save_images(
            plot.to_images(
                iterator=self.snapshots if undersample is None else self.snapshots[::undersample],
                plot_fn=lambda x: self.plot_grid(*x, **kwargs),
            ),
            save_path=save_path,
        )

    def plot_rho(
        self,
        mass_grid: Quantity | None = None,
        smoothing_sigma: float | None = None,
        smooth_holes: bool = True,
        plot_distribution: bool = False,
        title: str | None = r'Mass density ($\rho$)',
        xlabel: str | None = 'Radius',
        ylabel: str | None = r'$\rho$',
        length_unit: UnitLike = 'kpc',
        density_unit: UnitLike = 'Msun/kpc^3',
        xscale: plot.Scale = 'log',
        yscale: plot.Scale = 'log',
        lineplot_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the radial density profile of the distribution function.

        Parameters:
            mass_grid: Mass grid to calculate the density at. If `None`, the current density is used.
            plot_distribution: Whether to plot the theoretical density from the source distribution.
            smoothing_sigma: Smoothing parameter for the density profile. If `None`, no smoothing is applied.
            title: The title of the plot.
            xlabel: The label of the x-axis.
            ylabel: The label of the y-axis.
            length_unit: Unit of length.
            density_unit: Unit of density.
            lineplot_kwargs: Additional keyword arguments passed to `sns.lineplot()`.
            xscale: The scale of the x-axis.
            yscale: The scale of the y-axis.
            **kwargs: Additional keyword arguments passed to `plot.setup()` if `plot_distribution` is `False` or to `distribution.plot_rho()` if `plot_distribution` is `True`.
        """
        if plot_distribution and self.distribution is not None:
            fig, ax = self.distribution.plot_rho(
                xscale=xscale, yscale=yscale, title=title, xlabel=xlabel, ylabel=ylabel, **kwargs
            )
        else:
            fig, ax = plot.setup(
                xscale=xscale,
                yscale=yscale,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                x_unit=length_unit,
                y_unit=density_unit,
                **kwargs,
            )

        x = self.r_array
        y = self.rho if mass_grid is None else self.calculate_rho(mass_grid)
        y = utils.smooth_holes_1d(x=x, y=y, include_zero=True)
        if smoothing_sigma is not None:
            y = Quantity(scipy.ndimage.gaussian_filter(y.value, smoothing_sigma), y.unit)

        sns.lineplot(
            x=x.to(length_unit).value,
            y=y.to(density_unit).value,
            ax=ax,
            **lineplot_kwargs,
        )
        return fig, ax

    def plot_temperature(
        self,
        mass_grid: Quantity | None = None,
        smoothing_sigma: float | None = None,
        title: str | None = r'Temperature',
        xlabel: str | None = 'Radius',
        ylabel: str | None = r'Temperature',
        length_unit: UnitLike = 'kpc',
        temperature_unit: UnitLike = 'km^2/second^2',
        xscale: plot.Scale = 'log',
        yscale: plot.Scale = 'log',
        lineplot_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the radial temperature profile of the distribution function.

        Parameters:
            mass_grid: Mass grid to calculate the density at. If `None`, the current density is used.
            smoothing_sigma: Smoothing parameter for the temperature profile. If `None`, no smoothing is applied.
            title: The title of the plot.
            xlabel: The label of the x-axis.
            ylabel: The label of the y-axis.
            length_unit: Unit of length.
            temperature_unit: Unit of density.
            xscale: The scale of the x-axis.
            yscale: The scale of the y-axis.
            lineplot_kwargs: Additional keyword arguments passed to `sns.lineplot()`.
            **kwargs: Additional keyword arguments passed to `plot.setup()`.
        """
        fig, ax = plot.setup(
            xscale=xscale,
            yscale=yscale,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            x_unit=length_unit,
            y_unit=temperature_unit,
            **kwargs,
        )

        x = self.r_array
        y = self.temperature if mass_grid is None else self.calculate_temperature(mass_grid)
        y = utils.smooth_holes_1d(x=x, y=y, include_zero=True)
        if smoothing_sigma is not None:
            y = Quantity(scipy.ndimage.gaussian_filter(y.value, smoothing_sigma), y.unit)

        sns.lineplot(
            x=x.to(length_unit).value,
            y=y.to(temperature_unit).value,
            ax=ax,
            **lineplot_kwargs,
        )
        return fig, ax

    def animate_plot(
        self,
        save_path: str | Path,
        plot_type: Literal['rho', 'temperature'] = 'rho',
        times: Quantity['time'] | None = None,
        undersample: int | None = 1,
        color_palette: str | None = None,
        text_label_text: Literal['auto'] | str = 'auto',
        text_label_unit: UnitLike | str = 'Tdyn',
        text_label_format: str = '.4f',
        **kwargs: Any,
    ) -> None:
        """Animate the radial profile of the distribution function, and save as a gif.

        Parameters:
            save_path: Path to save the animation.
            plot_type: Type of plot to generate. Either 'rho' or 'temperature'.
            times: Plot the closest times from `self.grids_time`. Takes priority over `undersample`.
            undersample: Plot every `undersample`-th time from `self.grids_time`.
            text_label_unit: Unit used for the time label in the plot.
            text_label_format: Format string for the time label.
            **kwargs: Additional keyword arguments passed to `plot_rho()`.
        """
        if text_label_text == 'auto':
            text_label_text = r'$rho$' if plot_type == 'rho' else 'Temperature'
        assert times is not None or undersample is not None, "Either 'times' or 'undersample' must be provided."

        snapshots = self.closest_snapshots(times) if times is not None else self.snapshots[::undersample]
        palette = sns.color_palette(color_palette, len(snapshots))
        plot_fn = self.plot_rho if plot_type == 'rho' else self.plot_temperature

        plot.save_images(
            plot.to_images(
                iterator=list(enumerate(snapshots)),
                plot_fn=lambda x: plot_fn(
                    mass_grid=x[1][0],
                    lineplot_kwargs={
                        'label': f'{text_label_text} (t={x[1][1].to(self.fill_time_unit(text_label_unit)):{text_label_format}})',
                        'color': palette[x[0]] if color_palette is not None else None,
                        **kwargs.get('lineplot_kwargs', {}),
                    },
                    **kwargs,
                ),
            ),
            save_path=save_path,
        )

    def plot_multi_line(
        self,
        plot_type: Literal['rho', 'temperature'] = 'rho',
        times: Quantity['time'] | None = None,
        undersample: int | None = None,
        color_palette: str | None = None,
        text_label_unit: UnitLike | str = 'Tdyn',
        text_label_format: str = '.4f',
        lineplot_kwargs: dict[str, Any] = {},
        setup_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the radial profile of the distribution function at given times.

        Parameters:
            plot_type: Type of plot to generate. Either 'rho' or 'temperature'.
            times: Plot the closest times from `self.grids_time`. Takes priority over `undersample`.
            undersample: Plot every `undersample`-th time from `self.grids_time`.
            color_palette: Color palette used for the plot.
            text_label_unit: Unit used for the time label in the plot.
            text_label_format: Format string for the time label.
            lineplot_kwargs: Additional keyword arguments passed to `sns.lineplot()`.
            setup_kwargs: Additional keyword arguments passed to `plot.setup()`
            **kwargs: Additional keyword arguments passed to `plot_temperature()`.
        """

        assert times is not None or undersample is not None, "Either 'times' or 'undersample' must be provided."
        snapshots = self.closest_snapshots(times) if times is not None else self.snapshots[::undersample]
        fig, ax = plot.setup(**setup_kwargs)
        palette = sns.color_palette(color_palette, len(snapshots))
        plot_fn = self.plot_rho if plot_type == 'rho' else self.plot_temperature
        for i, (mass_grid, t) in enumerate(snapshots):
            fig, ax = plot_fn(
                mass_grid=mass_grid,
                lineplot_kwargs={
                    'label': f'{t.to(text_label_unit):{text_label_format}}',
                    'color': palette[i],
                    **lineplot_kwargs,
                },
                early_quit=False,
                fig=fig,
                ax=ax,
                **kwargs,
            )
        return fig, ax
