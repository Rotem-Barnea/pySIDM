from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import scipy
import seaborn as sns
from numba import njit, prange
from astropy import constants, cosmology
from numpy.typing import NDArray
from astropy.units import Unit, Quantity, def_unit
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from astropy.units.typing import UnitLike

from .. import rng, plot, utils, report, physics, run_units
from ..types import FloatOrArray, ParticleType
from ..physics.eddington import QuantitySpline

if TYPE_CHECKING:
    from ..phase_space import PhaseSpace


class Distribution:
    """General mass distribution profile."""

    def __init__(
        self,
        Rmin: Quantity['length'] = Quantity(1e-4, 'kpc'),
        Rmax: Quantity['length'] | None = None,
        Rs: Quantity['length'] | None = None,
        c: int | float | None | Literal['Dutton14'] = None,
        Rvir: Quantity['length'] | None = None,
        Mtot: Quantity['mass'] | None = None,
        rho_s: Quantity['mass density'] | None = None,
        space_steps: float | int = 1e3,
        spline_s: float | None = None,
        truncate: bool = False,
        truncate_power: int = 4,
        initialize_grids: list[str] = [],
        label: str = '',
        particle_type: ParticleType = 'dm',
        name: str = '',
        id: int | None = None,
    ) -> None:
        """General mass distribution profile.

        Parameters:
            Rmin: Minimum radius of the density profile, used for calculating the `internal logarithmic grid` and set internal cutoffs.
            Rmax: Maximum radius of the density profile, used for calculating the `internal logarithmic grid` and set internal cutoffs.
            Rs: Scale radius of the distribution profile.
            c: Concentration parameter of the distribution profile (such that Rvir = c * Rs). If 'Dutton14', calculate it based on the total mass (must be provided via `Mtot`).
            Rvir: Virial radius of the distribution profile.
            rho_s: Scale density of the distribution profile. Either `Mtot` or `rho_s` must be provided, and the other will be calculated from the rest of the parameters. If both are provided, they are hard set with no attempts to reconcile the parameters.
            Mtot: Total mass of the distribution profile. Either `Mtot` or `rho_s` must be provided, and the other will be calculated from the rest of the parameters. If both are provided, they are hard set with no attempts to reconcile the parameters.
            space_steps: Number of space steps for the `internal logarithmic grid`.
            spline_s: Spline smoothing parameter for calculating the drho/dPsi derivative.
            truncate: Whether to truncate the density at the virial radius.
            truncate_power: The power law used for truncation.
            initialize_grid: Grids to initialize at startup, otherwise they will only be calculated at runtime as needed.
            label: Label for the density profile.
            name: Additional name for the distribution profile.
            id: Unique identifier for the distribution profile.

        Returns:
            General mass distribution object.
        """

        if c == 'Dutton14':
            assert Mtot is not None, 'Mtot must be provided when using Dutton14'
            c = self.c_from_M_Dutton14(Mtot)

        assert sum([Rs is not None, Rvir is not None, c is not None]) == 2, (
            'Exactly two of Rs, Rvir, and c must be specified'
        )
        if Rs is not None and Rvir is not None:
            c = (Rvir / Rs).decompose(run_units.system).value
        elif Rs is not None and c is not None:
            Rvir = Quantity(c * Rs.to(run_units.length))
        elif Rvir is not None and c is not None:
            Rs = Quantity(Rvir.to(run_units.length) / c)

        assert Rs is not None, 'Failed to evaluate Rs'
        assert Rvir is not None, 'Failed to evaluate Rvir'
        assert c is not None, 'Failed to evaluate c'
        assert Mtot is not None or rho_s is not None, 'Either Mtot or rho_s must be specified'

        self.Rs: Quantity['length'] = Rs.to(run_units.length)
        self.Rvir: Quantity['length'] = Rvir.to(run_units.length)
        self.c: float = c

        self.space_steps: int = int(space_steps)
        self.title = 'General'
        self.particle_type: ParticleType = particle_type
        self._label: str = label
        self.name: str = name
        self.Rmin: Quantity['length'] = Rmin.to(run_units.length)
        self.Rmax: Quantity['length'] = (Rmax if Rmax is not None else 85 * self.Rs).to(run_units.length)
        if Mtot is not None:
            self.Mtot: Quantity['mass'] = Mtot.to(run_units.mass)
        if rho_s is not None:
            self.rho_s: Quantity['mass density'] = rho_s.to(run_units.density)
        else:
            self.rho_s = self.calculate_rho_scale()
        if Mtot is None:
            self.Mtot = self.calculate_M_tot()

        self.spline_s = spline_s
        self.truncate = truncate
        self.truncate_power = truncate_power

        self.memoization = {}
        for grid in initialize_grids:
            getattr(self, grid)

        self.id = utils.make_id(id)

    def __repr__(self):
        warn = '' if self.physical else 'WARNING: This distribution is not physical.\n\n'
        return str(
            report.Report(
                body_lines=[
                    report.Line(title='name', value=self.name),
                    report.Line(title='particle type', value=self.particle_type),
                    report.Line(title='Rs', value=self.Rs, format='.4f'),
                    report.Line(title='c', value=self.c, format='.1f'),
                    report.Line(title='Rvir', value=self.Rvir, format='.4f'),
                    report.Line(title='Mtot', value=self.Mtot, format='.3e'),
                    report.Line(title='rho_s', value=self.rho_s, format='.4f'),
                    report.Line(title='Tdyn', value=(1 * self.Tdyn).to(run_units.time), format='.4f'),
                    report.Line(title='Rmin', value=self.Rmin, format='.4f'),
                    report.Line(title='Rmax', value=self.Rmax, format='.4f'),
                    report.Line(title='space steps', value=self.space_steps, format='.0e'),
                ],
                header=f'{warn}{self.title} density function (ID={self.id})',
                body_prefix='  - ',
            )
        )

    def __call__(self, r: Quantity['length']) -> Quantity['mass density']:
        """Calculate `rho(r)`"""
        return self.rho(r)

    def to_scale(self, x: Quantity['length']) -> Quantity['dimensionless']:
        """Scale the distance, i.e. `x/Rs`"""
        return x.to(self.Rs.unit) / self.Rs

    @staticmethod
    def c_from_M_Dutton14(M: Quantity['mass']) -> float:
        """Calculate the concentration parameter `c` from the total mass `M` based on Dutton & Maccio (2014)."""
        return 10 ** (
            1.025 - 0.097 * np.log10((M.to('Msun') / 1e12 * cosmology.Planck18.H0).decompose(run_units.system).value)
        )

    @property
    def label(self) -> str:
        """Return the label of the profile."""
        if self._label == '':
            if self.particle_type == 'dm':
                return 'DM'
            elif self.particle_type == 'baryon':
                return 'Baryonic matter'
        return self._label

    @label.setter
    def label(self, label: str) -> None:
        self._label = label

    @property
    def physical(self) -> bool:
        """Return whether the profile is physical."""
        return True

    @property
    def Tdyn(self) -> Unit:
        """Calculate the dynamic time of the profile, returning it as a `Unit` object (memoized)."""
        if 'Tdyn' not in self.memoization:
            self.memoization['Tdyn'] = def_unit(
                'Tdyn',
                np.sqrt(self.Rs**3 / (constants.G * self.Mtot)).to(run_units.time),
                doc=f'{self.title} dynamic time',
            )
        return self.memoization['Tdyn']

    @Tdyn.setter
    def Tdyn(self, Tdyn: Unit) -> None:
        self.memoization['Tdyn'] = Tdyn

    @property
    def geomspace_grid(self) -> Quantity['length']:
        """Calculate the `internal logarithmic grid` (memoized)."""
        if 'geomspace_grid' not in self.memoization:
            self.memoization['geomspace_grid'] = cast(Quantity, np.geomspace(self.Rmin, self.Rmax, self.space_steps))
        return self.memoization['geomspace_grid']

    @property
    def linspace_grid(self) -> Quantity['length']:
        """Calculate the  `internal linear grid` (memoized)."""
        if 'linspace_grid' not in self.memoization:
            self.memoization['linspace_grid'] = cast(
                Quantity, np.linspace(start=self.Rmin, stop=self.Rmax, num=self.space_steps)
            )
        return self.memoization['linspace_grid']

    @staticmethod
    @njit
    def calculate_rho(
        r: FloatOrArray,
        rho_s: float = 1,
        Rs: float = 1,
        Rvir: float = 1,
        truncate: bool = False,
        truncate_power: int = 4,
    ) -> FloatOrArray:
        """Calculate the density (`rho`) at a given radius.

        This method is meant to be overwritten by subclasses. The function gets called by njit parallelized functions and must be njit compatible.

        Parameters:
            r: The radius at which to calculate the density.
            rho_s: The scale density.
            Rs: The scale radius.
            Rvir: The virial radius.
            truncate: Whether to truncate the density at the virial radius.
            truncate_power: The power law used for truncation.

        Returns:
            The density at the given radius.
        """
        return r

    def rho(self, r: Quantity['length']) -> Quantity['mass density']:
        """Calculate the density (`rho`) at a given radius."""
        return Quantity(
            self.calculate_rho(
                r=r.to(run_units.length).value,
                rho_s=self.rho_s.decompose(run_units.system).value,
                Rs=self.Rs.decompose(run_units.system).value,
                Rvir=self.Rvir.decompose(run_units.system).value,
                truncate=self.truncate,
                truncate_power=self.truncate_power,
            ),
            run_units.density,
        )

    @property
    def rho_grid(self) -> Quantity['mass density']:
        """Calculate the density (`rho`) at the  `internal logarithmic grid` (memoized)."""
        if 'rho_grid' not in self.memoization:
            self.memoization['rho_grid'] = self.rho(self.geomspace_grid)
        return self.memoization['rho_grid']

    def rho_r2(self, r: Quantity['length']) -> Quantity['linear density']:
        """Calculate the density (`rho`) times the jacobian (`r^2`) at a given radius."""
        return self.rho(r) * r.to(run_units.length) ** 2

    @property
    def rho_r2_grid(self) -> Quantity['linear density']:
        """Calculate the density (`rho`) times the jacobian (`r^2`) at the  `internal logarithmic grid` (memoized)."""
        if 'rho_r2_grid' not in self.memoization:
            self.memoization['rho_r2_grid'] = self.rho_r2(self.geomspace_grid)
        return self.memoization['rho_r2_grid']

    def spherical_rho_integrate(self, r: Quantity['length'], use_rho_s: bool = True) -> Quantity['mass']:
        """Calculate the density (`rho`) integral in `[0,r]` assuming spherical symmetry. `use_rho_s` is used internally to calculate the density scale and shouldn't be used."""
        rho_s = self.rho_s.decompose(run_units.system).value if use_rho_s else 1
        integral = utils.fast_spherical_rho_integrate(
            np.atleast_1d(r.to(run_units.length).value),
            self.calculate_rho,
            rho_s,
            self.Rs.decompose(run_units.system).value,
            self.Rvir.decompose(run_units.system).value,
        )
        return Quantity(integral, run_units.mass)

    def M(self, r: Quantity['length']) -> Quantity['mass']:
        """Calculate the enclosed mass (`M(<=r)`) at a given radius. Integrates the density function (`rho`)."""
        scalar_input = np.isscalar(r)
        if len(r.shape) == 2:
            M = Quantity([self.spherical_rho_integrate(r_) for r_ in r])
        else:
            M = self.spherical_rho_integrate(r)
        if scalar_input:
            return Quantity(np.array(M)[0], run_units.mass)
        return M

    @property
    def M_grid(self) -> Quantity['mass']:
        """Calculate the enclosed mass (`M(<=r)`) at the  `internal logarithmic grid` (memoized)."""
        if 'M_grid' not in self.memoization:
            self.memoization['M_grid'] = self.M(self.geomspace_grid)
        return self.memoization['M_grid']

    def calculate_M_tot(self) -> Quantity['mass']:
        """Calculate the total mass, i.e. the integral over `[0, Rmax]`."""
        return cast(Quantity, self.spherical_rho_integrate(self.Rmax, True)[0])

    def calculate_rho_scale(self) -> Quantity['mass density']:
        """Calculate the density scale to set the integral over `[0, Rmax]` to equal `Mtot`."""
        return self.Mtot / self.spherical_rho_integrate(self.Rmax, False)[0] * run_units.density

    def mass_pdf(self, r: Quantity['length']) -> FloatOrArray:
        """Mass probability density function (pdf) at radius `r`. Normalized `rho*r^2`."""
        mass_pdf = self.rho_r2(r).value
        mass_pdf /= np.trapezoid(mass_pdf, r.decompose(run_units.system).value)
        return mass_pdf

    def mass_cdf(self, r: Quantity['length']) -> FloatOrArray:
        """Mass cumulative probability density function (cdf) at radius `r`. Normalized enclosed mass."""
        return (self.M(r) / self.Mtot).decompose(run_units.system).value

    def pdf(self, r: Quantity['length']) -> FloatOrArray:
        """Mass probability density function (pdf) interpolated at a given radius (memoized)."""
        if 'pdf' not in self.memoization:
            self.memoization['pdf'] = scipy.interpolate.interp1d(
                self.geomspace_grid.value,
                self.mass_pdf(self.geomspace_grid),
                kind='cubic',
                bounds_error=False,
                fill_value=(0, 1),
            )
        return self.memoization['pdf'](r.to(run_units.length).value)

    def cdf(self, r: Quantity['length']) -> FloatOrArray:
        """Mass cumulative probability density function (cdf) interpolated at a given radius `r` (memoized)."""
        if 'cdf' not in self.memoization:
            self.memoization['cdf'] = scipy.interpolate.interp1d(
                self.geomspace_grid.value,
                self.mass_cdf(self.geomspace_grid),
                kind='cubic',
                bounds_error=False,
                fill_value=(0, 1),
            )
        return self.memoization['cdf'](r.to(run_units.length).value)

    def quantile_function(self, p: FloatOrArray) -> Quantity['length']:
        """Mass quantile function (inversed cdf) interpolated at a given radius `r` (memoized)."""
        if 'quantile_function' not in self.memoization:
            cdf, rs = utils.joint_clean(arrays=[self.mass_cdf(self.geomspace_grid), self.geomspace_grid.value])
            self.memoization['quantile_function'] = scipy.interpolate.interp1d(
                cdf, rs, kind='cubic', bounds_error=False, fill_value=(self.Rmin.value, self.Rmax.value)
            )
        return Quantity(self.memoization['quantile_function'](p), run_units.length)

    def Phi(self, r: Quantity['length']) -> Quantity['specific energy']:
        """Calculate the gravitational potential energy (`Phi`), at a given radius `r`."""
        xs = Quantity(np.geomspace(self.Rmin, r, 1000), 'kpc').T
        return np.trapezoid(y=constants.G * self.M(xs) / xs**2, x=xs).to(run_units.specific_energy)

    @property
    def Phi0(self) -> Quantity['specific energy']:
        """Calculate the relative value for the gravitational potential energy (`Phi0`), i.e. at `infinity` (memoized)."""
        if 'Phi0' not in self.memoization:
            self.memoization['Phi0'] = self.Phi(self.Rmax * 100)
        return self.memoization['Phi0']

    def calculate_Psi(self, r: Quantity['length']) -> Quantity['specific energy']:
        """Calculate the relative gravitational potential energy (`Psi`) at a given radius `r`."""
        return cast(Quantity, self.Phi0 - self.Phi(r))

    @property
    def Psi_grid(self) -> Quantity['specific energy']:
        """Calculate the relative gravitational potential (`Psi`) at the  `internal logarithmic grid` (memoized)."""
        if 'Psi_grid' not in self.memoization:
            self.memoization['Psi_grid'] = self.calculate_Psi(self.geomspace_grid)
        return self.memoization['Psi_grid']

    @property
    def F(self) -> QuantitySpline:
        """Calculate the spline function for the antiderivative (`F`) of the distribution function (`f`)"""
        if 'F' not in self.memoization:
            self.memoization['F'] = physics.eddington.make_F_spline(
                Psi_grid=self.Psi_grid,
                rho_Psi_spline=physics.eddington.make_rho_Psi_spline(
                    Psi_grid=self.Psi_grid,
                    rho_grid=self.rho_grid,
                    s=self.spline_s,
                ),
            )
        return self.memoization['F']

    def f(self, E: Quantity, reject_negative: bool = True) -> Quantity:
        """Calculate the distribution function (`f`) for the given energy."""
        return physics.eddington.f(E=E, F_spline=self.F, reject_negative=reject_negative)

    @property
    def Psi(self) -> QuantitySpline:
        """Interpolate the relative gravitational potential (`Psi`) based on the  `internal logarithmic grid` (memoized)."""
        if 'Psi' not in self.memoization:
            # self.memoization['Psi'] = scipy.interpolate.interp1d(
            #     self.geomspace_grid, self.Psi_grid, bounds_error=False, fill_value=0
            # )
            self.memoization['Psi'] = QuantitySpline(
                x=self.geomspace_grid.value,
                y=self.Psi_grid.value,
                in_unit=str(self.geomspace_grid.unit),
                out_unit=str(self.Psi_grid.unit),
                s=0,
            )
        return self.memoization['Psi']

    def E(
        self,
        v: Quantity['velocity'],
        r: Quantity['length'] | None = None,
        Psi: Quantity['specific energy'] | None = None,
    ) -> Quantity['specific energy']:
        """Interpolate the internal energy (`E`) At the given radius `r` using `Psi()`."""
        if r is not None:
            return cast(Quantity, self.Psi(r) - 1 / 2 * v**2)
        elif Psi is not None:
            return cast(Quantity, Psi - 1 / 2 * v**2)
        raise ValueError('Either `r` or `Psi` must be provided')

    def recalculate(self, key: str, inplace: bool = False) -> Any:
        """Recalculate the memoized value of the given key."""
        if key in self.memoization:
            self.memoization.pop(key)
        new_value = getattr(self, key)
        if not inplace:
            return new_value

    @staticmethod
    def merge_distribution_grids(distributions: list['Distribution'], inplace: bool = False) -> None:
        """Merges the `Psi` grid values of the given distributions. Used to combine the potentials of multiple distributions to sample via Eddington's inversion."""
        physical_distributions = [density for density in distributions if density.physical]
        for distribution in physical_distributions:
            distribution.memoization = {}
        Psi_grid = sum([getattr(density, 'Psi_grid') for density in physical_distributions])
        for distribution in physical_distributions:
            distribution.memoization['Psi_grid'] = Psi_grid

    ## Roll initial setup

    def sample_r(
        self,
        n_particles: int | float,
        sampling_method: Literal['random', 'uniform'] = 'random',
        generator: np.random.Generator | None = None,
    ) -> Quantity['length']:
        """Sample particle positions from the distribution quantile function.

        Parameters:
            n_particles: Number of particles to sample.
            sampling_method: Sampling method to space the quantiles before applying the inverse CDF. If 'random' sample from a uniform random distribution. If 'uniform' use a uniform grid in the range (0,1). If 'uniform with boundaries' use a uniform grid in the range [0,1].
            generator: If not provided, use the default generator defined in `rng.generator`.

        Returns:
            Sampled particle positions.
        """
        if sampling_method == 'random':
            if generator is None:
                generator = rng.generator
            points = generator.random(int(n_particles))
        elif sampling_method == 'uniform':
            points = cast(NDArray[np.float64], np.linspace(0, 1, int(n_particles)))
        return cast(Quantity, np.sort(self.quantile_function(points)))

    @staticmethod
    @njit(parallel=True)
    def sample_v_norm_fast(
        Psi: NDArray[np.float64],
        E_grid: NDArray[np.float64],
        f_grid: NDArray[np.float64],
        rolls: NDArray[np.float64],
        num: int = 100000,
    ) -> NDArray[np.float64]:
        """Sample particle velocity from the distribution function. Internal njit accelerated function. Prioritize using `roll_v()`."""
        output = np.empty_like(Psi)
        for particle in prange(len(Psi)):
            vs_grid = np.linspace(0, np.sqrt(2 * Psi[particle]), num=num)
            vs = np.empty_like(vs_grid, dtype=np.float64)
            vs[:] = vs_grid
            pdf = np.zeros_like(vs)
            for i, v in enumerate(vs):
                pdf[i] = v**2 * utils.linear_interpolation(E_grid, f_grid, Psi[particle] - v**2 / 2)
            pdf /= pdf.sum()
            cdf = np.cumsum(pdf)
            i = np.searchsorted(cdf, rolls[particle]) - 1
            if i < 0:
                i = 0
            elif i >= len(cdf) - 1:
                i = len(cdf) - 2
            output[particle] = vs[i]
        return output

    def sample_v_norm(
        self,
        r: Quantity['length'],
        num: int = 1000,
        generator: np.random.Generator | None = None,
    ) -> Quantity['velocity']:
        """Sample particle velocity (norm) from the distribution function.

        For full 3d velocity vectors, use `roll_v_3d()`.

        Parameters:
            r: Radius of the particles (required to sample the velocity, should have been sampled using `roll_v()` prior).
            num: Resolution parameters, defines the number of steps to use in the df integral.

        Returns:
            Sampled velocity norm of the particle, shaped `(num_particles,)`.
        """
        if generator is None:
            generator = rng.generator
        return Quantity(
            self.sample_v_norm_fast(
                Psi=self.Psi(r).to(run_units.specific_energy),
                E_grid=self.E_grid,
                f_grid=self.f_grid,
                rolls=generator.random(len(r)),
                num=num,
            ),
            run_units.velocity,
        )

    def sample_v(
        self,
        r: Quantity['length'],
        num: int = 1000,
        generator: np.random.Generator | None = None,
    ) -> Quantity['velocity']:
        """Sample particle velocity (3d vectors) from the distribution function. Shape `(num_particles, 3)` with `(vx,vy,vr)`.

        The velocity is split into radial and perpendicular components by a uniform cosine distributed angle, and then the perpendicular component is split into x and y components by a uniform angle.
        See `roll_v()` for parameter details.
        """
        return cast(
            Quantity, np.vstack(utils.split_3d(self.sample_v_norm(r, generator=generator), generator=generator)).T
        )

    def sample_legacy(
        self,
        n_particles: int | float,
        sampling_method: Literal['random', 'uniform'] = 'random',
        num: int = 1000,
        generator: np.random.Generator | None = None,
    ) -> tuple[Quantity['length'], Quantity['velocity']]:
        """Sample particles In two steps:
            - Sample radius from the inversed CDF.
            - Calculate the velocity CDF for every particle's sampled radius, and sample from it's inverse.

        Parameters:
            n_particles: Number of particles to sample.
            sampling_method: Sampling method to space the quantiles before applying the inverse CDF. If 'random' sample from a uniform random distribution. If 'uniform' use a uniform grid in the range (0,1). If 'uniform with boundaries' use a uniform grid in the range [0,1].
            num: Resolution parameters, defines the number of steps to use in the df integral.
            generator: If not provided, use the default generator defined in `rng.generator`.

        Returns:
            A tuple of two vectors:
                Sampled radius values for each particle, shaped `(num_particles,)`
                Corresponding 3d velocities for each particle, shaped `(num_particles,3)`.
        """
        r = self.sample_r(n_particles, sampling_method=sampling_method, generator=generator).to(run_units.length)
        v = self.sample_v(r, num=num, generator=generator).to(run_units.velocity)
        return r, v

    def sample(
        self,
        n_particles: int | float,
        radius_min_value: Quantity['length'] | None = None,
        radius_max_value: Quantity['length'] | None = None,
        velocity_min_value: Quantity['velocity'] = Quantity(0, 'km/second'),
        velocity_max_value: Quantity['velocity'] | None = None,
        radius_resolution: int | float = 10000,
        velocity_resolution: int | float = 10000,
        radius_range: Quantity['length'] | None = None,
        velocity_range: Quantity['velocity'] | None = None,
        radius_noise: float = 1,
        velocity_noise: float = 1,
        fail_on_negative: bool = False,
        generator: np.random.Generator | None = None,
    ) -> tuple[Quantity['length'], Quantity['velocity']]:
        """Sample particles from the joint phase space distribution:
            - The distribution function `f` is transformed into a joint pdf proportional to `r^2*v^2*f(r,v)*drdv`.
            - The joint pdf is discretized on a grid of radius and velocity bins. If not provided directly (`radius_range`, `velocity_range`), a linear grid is constructed based on the rest of the parameters. The final row and column is dropped to facilitate the calculation of the `drdv` term.
            - The discretized distribution is flattened and sampled from using `generator.choice` with probability weights set by the bin value.
            - The sampled radius and velocity are perturbed by a uniform noise term to provide sub-pixel results.
            - Angles for the velocity split are sampled by `utils.split_3d()`.

        Parameters:
            n_particles: Number of particles to sample.
            radius_min_value: Minimum radius value to consider for the phase space distribution. If `None` use the quantile value for `0`. Regardless, this value is capped by `Rmin` even if provided.
            radius_max_value: Maximum radius value to consider for the phase space distribution. If `None` use the quantile value for `1`. Regardless, this value is capped by `Rmax` even if provided.
            velocity_min_value: Minimum velocity norm value to consider for the phase space distribution.
            velocity_max_value: Maximum velocity norm value to consider for the phase space distribution. If `None` use 1.5x the square maximum energy grid value.
            radius_resolution: Resolution of the radius grid.
            velocity_resolution: Resolution of the velocity grid.
            radius_range: Radius bins to use. If provided override the `radius_min_value`, `radius_max_value`, and `radius_resolution` parameters. If `None` ignores.
            velocity_range: Velocity bins to use. If provided override the `velocity_min_value`, `velocity_max_value`, and `velocity_resolution` parameters. If `None` ignores.
            radius_noise: Noise factor for the sampled radius values. The samples are perturbed by a uniform distribution in a symmetric interval with half-width `radius_noise * pixel width / 2`.
            velocity_noise: Noise factor for the sampled velocity values. Same logic as `radius_noise`.
            fail_on_negative: Whether to raise an error if the df grid is negative at any values, otherwise fill holes and continue.
            generator: If not provided, use the default generator defined in `rng.generator`.

        Returns:
            A tuple of two vectors:
                Sampled radius values for each particle, shaped `(num_particles,)`
                Corresponding 3d velocities for each particle, shaped `(num_particles,3)`.
        """
        if generator is None:
            generator = rng.generator
        if radius_range is None:
            if radius_min_value is None:
                radius_min_value = self.quantile_function(0)
            radius_min_value = max(radius_min_value, self.Rmin)
            if radius_max_value is None:
                radius_max_value = self.quantile_function(1)
            radius_max_value = min(radius_max_value, self.Rmax)
            radius_range = Quantity(np.linspace(radius_min_value, radius_max_value, int(radius_resolution)))
        if velocity_range is None:
            if velocity_max_value is None:
                # velocity_max_value = 1.5 * Quantity(np.sqrt(self.E_grid.max()))
                velocity_max_value = 1.5 * Quantity(np.sqrt(self.Psi_grid.max())).to(velocity_min_value.unit)
            velocity_range = Quantity(np.linspace(velocity_min_value, velocity_max_value, int(velocity_resolution)))

        r_grid, v_grid = cast(
            tuple[Quantity, Quantity], np.meshgrid(radius_range[:-1], velocity_range[:-1], indexing='ij')
        )

        drdv = np.prod(np.meshgrid(radius_range.diff(), velocity_range.diff(), indexing='ij'), axis=0)
        probability_grid = np.array(16 * np.pi * r_grid**2 * v_grid**2 * self.f(self.E(v=v_grid, r=r_grid)) * drdv)
        probability_grid /= probability_grid.sum()
        flat_probability_grid = probability_grid.ravel()
        if (probability_grid < 0).any():
            if fail_on_negative:
                raise ValueError(f'Negative probability density encountered, {self}')
            else:
                probability_grid = cast(NDArray[np.float64], utils.smooth_holes_2d(probability_grid))
                probability_grid[probability_grid < 0] = 0

        indices = np.unravel_index(
            generator.choice(a=flat_probability_grid.size, size=int(n_particles), p=flat_probability_grid),
            probability_grid.shape,
        )

        r_indices, v_indices = indices + generator.uniform(
            [[min(1 - radius_noise, 0)], [min(1 - velocity_noise, 0)]],
            [[radius_noise], [velocity_noise]],
            (2, int(n_particles)),
        )

        r_indices, v_indices = np.abs(r_indices), np.abs(v_indices)
        r_indices[r_indices > radius_resolution - 2] = (
            2 * (radius_resolution - 2) - r_indices[r_indices > radius_resolution - 2]
        )
        v_indices[v_indices > velocity_resolution - 2] = (
            2 * (velocity_resolution - 2) - v_indices[v_indices > velocity_resolution - 2]
        )

        r_indices, v_indices = r_indices[np.argsort(r_indices)], v_indices[np.argsort(r_indices)]

        r_interp = scipy.interpolate.interp1d(np.arange(radius_resolution - 1), radius_range[:-1])
        v_interp = scipy.interpolate.interp1d(np.arange(velocity_resolution - 1), velocity_range[:-1])

        velocity = Quantity(v_interp(v_indices), velocity_range.unit)

        return (
            Quantity(r_interp(r_indices), radius_range.unit),
            cast(Quantity, np.vstack(utils.split_3d(velocity, generator=generator)).T),
        )

    def phase_space(self, **kwargs: Any) -> 'PhaseSpace':
        """Returns the phase space object matching the distribution."""
        from ..phase_space import PhaseSpace

        return PhaseSpace(self, **kwargs)

    def full_sample(
        self,
        sample_method: Literal['distribution', 'phase space', 'legacy'] = 'distribution',
        phase_space_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> tuple[
        Quantity['length'],
        Quantity['velocity'],
        Quantity['mass'],
        NDArray[np.str_],
        NDArray[np.int64],
    ]:
        """Samples particles from the distribution. Wraps the sample method and returns additional particle properties.

        Parameters
            sample_method: The method to use for sampling particles.
              - 'distribution': Samples particles from the distribution's distribution function (`df`) grid. Only works with linear grids.
              - 'phase space': Samples particles from the phase space object. Works with any grid.
              - 'legacy': Samples the particles' radius by sampling from the quantile m(r) distribution, and then for each particle calculate the velocity pdf from the distribution function (`df`), and sample from it's CDF. THERE IS SOME BUG IN THIS METHOD.

            **phase_space_kwargs: Additional keyword arguments to pass to the phase space object during initialization. Only relevant if `sample_method='phase space'`.
            **kwargs: Additional keyword arguments to pass to the sample method.

        Returns:
            The particles' position, velocity, mass, particle type, and distribution ID.
        """
        if sample_method == 'distribution':
            r, v = self.sample(**kwargs)
        elif sample_method == 'phase space':
            r, v = self.phase_space(**phase_space_kwargs).sample_weighted_particles(**kwargs)
        elif sample_method == 'legacy':
            r, v = self.sample_legacy(**kwargs)
        return (
            r,
            v,
            cast(Quantity, np.ones(len(r)) * self.Mtot / len(r)),
            np.full(len(r), self.particle_type),
            np.full(len(r), self.id),
        )

    ##Plots

    def plot_phase_space(
        self,
        r_range: Quantity['length'] = Quantity(np.linspace(1e-2, 35, 200), 'kpc'),
        v_range: Quantity['velocity'] = Quantity(np.linspace(0, 60, 200), 'km/second'),
        velocity_unit: UnitLike = 'km/second',
        cmap: str = 'jet',
        transparent_value: float | None = 0,
        fix_negative_values: bool = True,
        plot_energy: bool = False,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the phase space distribution of the density profile.

        Parameters:
            r_range: Range of radial distances to plot.
            v_range: Range of velocities to plot.
            velocity_unit: Units to use for the velocity axis.
            cmap: The colormap to use for the plot.
            transparent_value: Grid value to turn transparent (i.e. plot as `NaN`). If `None` ignores.
            fix_negative_values: Whether to fix negative values by interpolating them.
            plot_energy: Whether to plot the energy distribution instead of the density distribution.
            kwargs: Additional keyword arguments to pass to `plot.plot_phase_space()`.

        Returns:
            fig, ax.
        """
        r, v = cast(tuple[Quantity, Quantity], np.meshgrid(r_range, v_range))
        if plot_energy:
            grid = self.E(v=v, r=r)
        else:
            grid = 16 * np.pi * r**2 * v**2 * self.f(self.E(v=v, r=r))

        if fix_negative_values and not plot_energy and (grid < 0).any():
            y, x = np.indices(grid.shape)
            mask = grid < 0

            grid[mask] = Quantity(
                scipy.interpolate.griddata(
                    points=(x[~mask], y[~mask]), values=grid[~mask], xi=(x[mask], y[mask]), method='linear'
                ),
                grid.unit,
            )

        return plot.phase_space(
            grid,
            r_range,
            v_range,
            velocity_unit=velocity_unit,
            cmap=cmap,
            transparent_value=transparent_value,
            **kwargs,
        )

    def add_plot_R_markers(self, ax: Axes, ymax: float, x_unit: UnitLike = 'kpc') -> Axes:
        """Add markers for the scale radius and virial radius to the plot."""
        ax.vlines(
            x=[self.Rs.to(x_unit).value, self.Rvir.to(x_unit).value],
            ymin=0,
            ymax=ymax,
            linestyles='dashed',
            colors='black',
        )
        ax.text(x=self.Rs.to(x_unit).value, y=ymax, s='Rs')
        ax.text(x=self.Rvir.to(x_unit).value, y=ymax, s='Rvir')
        return ax

    def plot_rho(
        self,
        r_start: Quantity['length'] | None = None,
        r_end: Quantity['length'] | None = Quantity(1e4, 'kpc'),
        density_unit: UnitLike = 'Msun/kpc^3',
        length_unit: UnitLike = 'kpc',
        label: str | None = None,
        add_markers: bool = True,
        xscale: str = 'log',
        yscale: str = 'log',
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the density distribution (`rho`) of the density profile.

        Parameters:
            r_start: The starting radius for the plot. If `None`, uses `Rmin`.
            r_end: The ending radius for the plot. If `None` uses `Rmax`.
            density_unit: The unit for the density axis.
            length_unit: The unit for the radius axis.
            xscale: The scale of the x-axis.
            yscale: The scale of the y-axis.
            kwargs: Additional keyword arguments to pass to `plot.setup()`.

        Returns:
            fig, ax.
        """
        fig, ax = plot.setup(
            title='Density distribution (rho)',
            xlabel=utils.add_label_unit('Radius', length_unit),
            ylabel=utils.add_label_unit('Density', density_unit),
            xscale=xscale,
            yscale=yscale,
            **kwargs,
        )

        if r_start is None:
            r_start = self.Rmin
        if r_end is None:
            r_end = self.Rmax

        r = cast(Quantity, np.geomspace(r_start, r_end, self.space_steps))
        rho = self.rho(r)
        sns.lineplot(x=r.to(length_unit).value, y=rho.to(density_unit).value, ax=ax, label=label)

        if add_markers:
            ax = self.add_plot_R_markers(ax, ymax=rho.max().to(density_unit).value, x_unit=length_unit)

        if label is not None:
            ax.legend()
        return fig, ax

    def plot_radius_distribution(
        self,
        r_start: Quantity['length'] | None = None,
        r_end: Quantity['length'] | None = None,
        cumulative: bool = False,
        length_unit: UnitLike = 'kpc',
        label: str | None = None,
        add_markers: bool = True,
        color: str = 'red',
        fig: Figure | None = None,
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the radius distribution (`pdf`/`cdf`) of the density profile.

        Parameters:
            r_start: The starting radius for the plot. If `None` uses `Rmin`.
            r_end: The ending radius for the plot. If `None` uses `Rmax`.
            cumulative: Plot the `cdf`, if `False` plot the `pdf` instead.
            length_unit: The unit for the radius axis.
            label: The label for the plot (legend).
            add_markers: Whether to add markers to the plot.
            color: The color for the plot.
            fig: The figure to plot on.
            ax: The axes to plot on.
            kwargs: Additional keyword arguments for the plotting function (`sns.lineplot()`).

        Returns:
            fig, ax.
        """
        title = 'Particle cumulative range distribution (cdf)' if cumulative else 'Particle range distribution (pdf)'
        fig, ax = plot.setup(fig, ax, title=title, xlabel=utils.add_label_unit('Radius', length_unit), ylabel='Density')

        if r_start is None:
            r_start = self.Rmin
        if r_end is None:
            r_end = self.Rmax

        r = cast(Quantity, np.geomspace(r_start, r_end, self.space_steps))
        y = self.mass_cdf(r) if cumulative else self.mass_pdf(r)
        sns.lineplot(x=r.to(length_unit).value, y=y, color=color, ax=ax, label=label, **kwargs)
        if add_markers:
            ax = self.add_plot_R_markers(ax, ymax=y.max(), x_unit=length_unit)
        return fig, ax

    def plot_drho_dPsi(
        self,
        xlabel: str | None = r'$\Psi$',
        ylabel: str | None = r'$\frac{\mathrm{d}\rho}{\mathrm{d}\Psi}$',
        title: str | None = 'Density derivative (log-log)',
        energy_unit: UnitLike = 'km^2/second^2',
        y_unit: UnitLike = 'Msun*second^2/(kpc^3*km^2)',
        fig: Figure | None = None,
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot the density derivative with respect to the relative potential energy `drho/dPsi`.

        Parameters:
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            energy_unit: Units for the energy axis.
            y_unit: Units for the y-axis.
            fig: Figure to plot on.
            ax: Axes to plot on.

        Returns:
            fig, ax
        """

        xlabel = utils.add_label_unit(xlabel, energy_unit)
        ylabel = utils.add_label_unit(ylabel, y_unit)
        fig, ax = plot.setup(
            fig,
            ax,
            minorticks=True,
            xscale='log',
            yscale='log',
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )
        sns.lineplot(x=np.array(self.Psi_grid.to(energy_unit)), y=np.array(self.drho_dPsi_grid.to(y_unit)), ax=ax)
        return fig, ax

    def plot_f(
        self,
        E: Quantity['specific energy'] = Quantity(np.geomspace(50, 1800, 1000), 'km^2/second^2'),
        xlabel: str | None = r'$\mathcal{E}$',
        ylabel: str | None = 'f',
        title: str | None = 'Particle distribution function',
        energy_unit: UnitLike = 'km^2/second^2',
        f_unit: UnitLike = 'second^3/(kpc^3*km^3)',
        fig: Figure | None = None,
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot the distribution function `f(E)`.

        Parameters:
            E: Energy values to plot.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            energy_unit: Units for the energy axis.
            f_unit: Units for the density function axis.
            fig: Figure to plot on.
            ax: Axes to plot on.

        Returns:
            fig, ax
        """

        xlabel = utils.add_label_unit(xlabel, energy_unit)
        ylabel = utils.add_label_unit(ylabel, f_unit)
        fig, ax = plot.setup(
            fig,
            ax,
            minorticks=True,
            xscale='log',
            yscale='log',
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )
        sns.lineplot(x=np.array(E.to(energy_unit)), y=np.array(self.f(E).to(f_unit)), ax=ax)
        return fig, ax
