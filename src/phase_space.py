import pickle
from typing import Any, Literal, TypedDict, cast
from pathlib import Path

import numpy as np
import scipy
import seaborn as sns
from matplotlib import colors
from numpy.typing import NDArray
from astropy.units import Quantity
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
        distribution: Distribution,
        r_range: Quantity['length'] = Quantity(np.geomspace(1e-5, 1e3, 500), 'kpc'),
        v_range: Quantity['velocity'] = Quantity(np.geomspace(1e-5, 1e3, 400), 'km/second'),
        t: Quantity['time'] = Quantity(0, run_units.time),
        dt: Quantity['time'] | float = 1,
        gravitation_subdivision: int = 5,
        gravitation_mass_cutoff: Quantity['mass'] = Quantity(1e-1, 'Msun'),
        scatter_factor: float = 1,
        save_every_t: Quantity['time'] | None = Quantity(1, 'Myr'),
        generator: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> None:
        self.distribution = distribution
        self.Tdyn = self.distribution.Tdyn
        self.r_array: Quantity = r_range.to(run_units.length)
        self.v_array: Quantity = v_range.to(run_units.velocity)
        self.time: Quantity = t.copy()
        self.dt: Quantity = (dt if isinstance(dt, Quantity) else self.distribution.Tdyn * dt).to(run_units.time)
        self.gravitation_subdivision = gravitation_subdivision
        self.gravitation_mass_cutoff = gravitation_mass_cutoff
        self.scatter_factor = scatter_factor
        self.save_every_t = save_every_t

        self.r_grid, self.v_grid = cast(tuple[Quantity, Quantity], np.meshgrid(self.r_array, self.v_array))

        self.dr, self.dv = [
            cast(Quantity, np.pad(np.diff(array), (0, 1), mode='edge')) for array in [self.r_array, self.v_array]
        ]
        self.dr_grid, self.dv_grid = cast(tuple[Quantity, Quantity], np.meshgrid(self.dr, self.dv))
        self.volume_element = cast(Quantity, self.dv_grid * self.dr_grid)

        self.f_grid: Quantity = self.distribution.f_from_rv(v=self.v_grid, r=self.r_grid)

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

    def calculate_rho(self, mass_grid: Quantity['mass']) -> Quantity['mass density']:
        """Calcualte the density as a function of radius (without the Jacobian) for the provided mass grid"""
        return (mass_grid.sum(axis=0) / (self.dr * self.jacobian_r)).to(run_units.density)

    @property
    def rho(self) -> Quantity['mass density']:
        """Density as a function of radius (without the Jacobian)"""
        return self.calculate_rho(self.mass_grid)

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
        cbar_label_autosuffix: bool = False,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the given grid as a heatmap If `grid` is `None`, use the current mass grid. All parameters are passed on to `plot.heatmap()`."""
        if t is None:
            t = self.time
        if title is not None:
            title = title.format(time=f'{t.to(self.fill_time_unit(title_time_unit)):{title_time_format}}')
        if grid is None and 'cbar_label' not in kwargs:
            kwargs['cbar_label'] = 'Mass'
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
            cbar_label_autosuffix=cbar_label_autosuffix,
            **kwargs,
        )

    def animate_mass_grid(self, save_path: str | Path, undersample: int | None = None, **kwargs: Any) -> None:
        """Animate the phase space (mass grid) of the distribution function, and save as a gif.

        Parameters:
            save_path: Path to save the animation.
            undersample: Undersample the phase space snapshots by this factor. Ignored if `None`.
            **kwargs: Additional keyword arguments passed to `plot_grid`.
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
        plot_distribution: bool = False,
        length_unit: UnitLike = 'kpc',
        density_unit: UnitLike = 'Msun/kpc^3',
        lineplot_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the radial density profile of the distribution function.

        Parameters:
            mass_grid: Mass grid to calculate the density at. If `None`, the current density is used.
            plot_distribution: Whether to plot the theoretical density from the source distribution.
            smoothing_sigma: Smoothing parameter for the density profile. If `None`, no smoothing is applied.
            length_unit: Unit of length.
            density_unit: Unit of density.
            lineplot_kwargs: Additional keyword arguments passed to `sns.lineplot`.
            **kwargs: Additional keyword arguments passed to `plot.setup` if `plot_distribution` is `False` or to `distribution.plot_rho` if `plot_distribution` is `True`.
        """
        if plot_distribution:
            fig, ax = self.distribution.plot_rho(**kwargs)
        else:
            fig, ax = plot.setup(**kwargs)

        rho = self.rho if mass_grid is None else self.calculate_rho(mass_grid)
        if smoothing_sigma is not None:
            rho = Quantity(scipy.ndimage.gaussian_filter(rho.value, smoothing_sigma), rho.unit)

        sns.lineplot(
            x=self.r_array.to(length_unit).value,
            y=rho.to(density_unit).value,
            ax=ax,
            **lineplot_kwargs,
        )
        return fig, ax

    def animate_rho(
        self,
        save_path: str | Path,
        undersample: int | None = None,
        text_label_unit: UnitLike | str = 'Tdyn',
        text_label_format: str = '.4f',
        **kwargs: Any,
    ) -> None:
        """Animate the radial density profile of the distribution function, and save as a gif.

        Parameters:
            save_path: Path to save the animation.
            undersample: Undersample the phase space snapshots by this factor. Ignored if `None`.
            text_label_unit: Unit used for the time label in the plot.
            text_label_format: Format string for the time label.
            **kwargs: Additional keyword arguments passed to `plot_rho`.
        """
        plot.save_images(
            plot.to_images(
                iterator=self.snapshots if undersample is None else self.snapshots[::undersample],
                plot_fn=lambda x: self.plot_rho(
                    mass_grid=x[0],
                    lineplot_kwargs={
                        'label': rf'$\rho$(t={x[1].to(self.fill_time_unit(text_label_unit)):{text_label_format}})',
                        **kwargs.get('lineplot_kwargs', {}),
                    },
                    **kwargs,
                ),
            ),
            save_path=save_path,
        )
