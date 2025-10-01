import numpy as np
import scipy
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numba import njit, prange
from numpy.typing import NDArray
from typing import cast, Any
from astropy import constants
from astropy.units import Quantity, Unit, def_unit
from astropy.units.typing import UnitLike
from .. import utils, run_units
from ..types import FloatOrArray


class Density:
    """General mass distribution profile."""

    def __init__(
        self,
        Rmin: Quantity['length'] = Quantity(1e-4, 'kpc'),
        Rmax: Quantity['length'] | None = None,
        Rs: Quantity['length'] = Quantity(1, 'kpc'),
        Rvir: Quantity['length'] = Quantity(1, 'kpc'),
        Mtot: Quantity['mass'] = Quantity(1, 'Msun'),
        unit_mass: Quantity['mass'] = Quantity(1, 'Msun'),
        space_steps: float | int = 1e4,
    ) -> None:
        """General mass distribution profile.

        Parameters:
            Rmin: Minimum radius of the density profile, used for calculating the internal logarithmic grid and set internal cutoffs.
            Rmax: Maximum radius of the density profile, used for calculating the internal logarithmic grid and set internal cutoffs.
            Rs: Scale radius of the distribution profile.
            Rvir: Virial radius of the distribution profile.
            Mtot: Total mass of the distribution profile.
            unit_mass: Unit mass of the distribution profile particles.
            space_steps: Number of space steps for the internal logarithmic grid.

        Returns:
            General mass distribution object.
        """
        self.space_steps: int = int(space_steps)
        self.Mtot: Quantity['mass'] = Mtot.to(run_units.mass)
        self.title = 'Density'
        self.unit_mass: Quantity['mass'] = unit_mass.to(run_units.mass)
        self.Rs: Quantity['length'] = Rs.to(run_units.length)
        self.Rvir: Quantity['length'] = Rvir.to(run_units.length)
        self.Rmin: Quantity['length'] = Rmin.to(run_units.length)
        self.Rmax: Quantity['length'] = (Rmax or 85 * self.Rs).to(run_units.length)
        self.rho_s: Quantity['mass density'] = self.calculate_rho_scale()
        self.memoization = {}

    def __repr__(self):
        return f"""General mass density function
  - Rmin = {self.Rmin:.4f}
  - Rmax = {self.Rmax:.4f}
  - space_steps = {self.space_steps:.0e}
  - Mtot = {self.Mtot:.3e}
  - particle mass = {self.unit_mass:.3e}"""

    def __call__(self, r: Quantity['length']) -> Quantity['mass density']:
        """Calculate rho(r)"""
        return self.rho(r)

    def to_scale(self, x: Quantity['length']) -> Quantity['dimensionless']:
        """Scale the distance, i.e. x/Rs"""
        return x.to(self.Rs.unit) / self.Rs

    @property
    def Tdyn(self) -> Unit:
        """Calculate the dynamic time of the profile, returning it as a Unit object (memoized)."""
        if 'Tdyn' not in self.memoization:
            self.memoization['Tdyn'] = def_unit(
                'Tdyn', np.sqrt(self.Rs**3 / (constants.G * self.Mtot)).to(run_units.time), doc=f'{self.title} dynamic time'
            )
        return self.memoization['Tdyn']

    @Tdyn.setter
    def Tdyn(self, Tdyn: Unit) -> None:
        self.memoization['Tdyn'] = Tdyn

    @property
    def geomspace_grid(self) -> Quantity['length']:
        """Calculate the internal logarithmic grid (memoized)."""
        if 'geomspace_grid' not in self.memoization:
            self.memoization['geomspace_grid'] = cast(Quantity['length'], np.geomspace(self.Rmin, self.Rmax, self.space_steps))
        return self.memoization['geomspace_grid']

    @property
    def linspace_grid(self) -> Quantity['length']:
        """Calculate the internal linear grid (memoized)."""
        if 'linspace_grid' not in self.memoization:
            self.memoization['linspace_grid'] = cast(Quantity['length'], np.linspace(start=self.Rmin, stop=self.Rmax, num=self.space_steps))
        return self.memoization['linspace_grid']

    @staticmethod
    @njit
    def calculate_rho(r: FloatOrArray, rho_s: float = 1, Rs: float = 1, Rvir: float = 1) -> FloatOrArray:
        """Calculate the density (rho) at a given radius.

        This method is meant to be overwritten by subclasses. The function gets called by njit parallelized functions and must be njit compatible.

        Parameters:
            r: The radius at which to calculate the density.
            rho_s: The scale density.
            Rs: The scale radius.
            Rvir: The virial radius.

        Returns:
            The density at the given radius.
        """
        return r

    def rho(self, r: Quantity['length']) -> Quantity['mass density']:
        """Calculate the density (rho) at a given radius."""
        return Quantity(self.calculate_rho(r.to(run_units.length).value, self.rho_s.value, self.Rs.value, self.Rvir.value), run_units.density)

    @property
    def rho_grid(self) -> Quantity['mass density']:
        """Calculate the density (rho) at the density internal logarithmic grid (memoized)."""
        if 'rho_grid' not in self.memoization:
            self.memoization['rho_grid'] = self.rho(self.geomspace_grid)
        return self.memoization['rho_grid']

    def rho_r2(self, r: Quantity['length']) -> Quantity['linear density']:
        """Calculate the density (rho) times the jacobian (r^2) at a given radius."""
        return self.rho(r) * r.to(run_units.length) ** 2

    @property
    def rho_r2_grid(self) -> Quantity['linear density']:
        """Calculate the density (rho) times the jacobian (r^2) at the density internal logarithmic grid (memoized)."""
        if 'rho_r2_grid' not in self.memoization:
            self.memoization['rho_r2_grid'] = self.rho_r2(self.geomspace_grid)
        return self.memoization['rho_r2_grid']

    def spherical_rho_integrate(self, r: Quantity['length'], use_rho_s: bool = True) -> Quantity['mass']:
        """Calculate the density (rho) integral in [0,r] assuming spherical symmetry. use_rho_s is used internally to calculate the density scale and shouldn't be used."""
        rho_s = self.rho_s.value if use_rho_s else 1
        integral = utils.fast_spherical_rho_integrate(
            np.atleast_1d(r.to(run_units.length).value), self.calculate_rho, rho_s, self.Rs.value, self.Rvir.value
        )
        return Quantity(integral, run_units.mass)

    def M(self, r: Quantity['length']) -> Quantity['mass']:
        """Calculate the enclosed mass (M(<=r)) at a given radius. Integrates the density function (rho)."""
        scalar_input = np.isscalar(r)
        M = self.spherical_rho_integrate(r)
        if scalar_input:
            return Quantity(np.array(M)[0], run_units.mass)
        return M

    @property
    def M_grid(self) -> Quantity['mass']:
        """Calculate the enclosed mass (M(<=r)) at the density internal logarithmic grid (memoized)."""
        if 'M_grid' not in self.memoization:
            self.memoization['M_grid'] = self.M(self.geomspace_grid)
        return self.memoization['M_grid']

    def calculate_rho_scale(self) -> Quantity['mass density']:
        """Calculate the density scale to set the integral over [0, Rmax] to equal Mtot."""
        return self.Mtot / self.spherical_rho_integrate(self.Rmax, False)[0] * run_units.density

    def Phi(self, r: Quantity['length']) -> Quantity['specific energy']:
        """Calculate the gravitational potential (Phi) at a given radius (memoized). The calculation utilizes a cubic interpolant."""
        if 'Phi' not in self.memoization:
            r_grid = self.geomspace_grid
            M_grid = self.M(r_grid)
            Phi_grid = -constants.G.to(run_units.G_units).value * scipy.integrate.cumulative_trapezoid(
                y=M_grid.value / r_grid.value**2, x=r_grid.value, initial=0
            )
            Phi_grid -= Phi_grid[-1]
            Phi_grid *= -1
            self.memoization['Phi'] = scipy.interpolate.interp1d(r_grid.value, Phi_grid, kind='cubic', bounds_error=False, fill_value=(0, 0))
        return Quantity(self.memoization['Phi'](r.to(run_units.length).value), run_units.specific_energy)

    @property
    def Phi_grid(self) -> Quantity['specific energy']:
        """Calculate the gravitational potential (Phi) at the density internal logarithmic grid (memoized)."""
        if 'Phi_grid' not in self.memoization:
            self.memoization['Phi_grid'] = self.Phi(self.geomspace_grid)
        return self.memoization['Phi_grid']

    @property
    def Phi0(self) -> Quantity['specific energy']:
        """Reference potential at the maximum radius ~infinity (memoized)."""
        if 'Phi0' not in self.memoization:
            self.memoization['Phi0'] = self.Phi(self.Rmax)
        return self.memoization['Phi0']

    def Psi(self, r: Quantity['length']) -> Quantity['specific energy']:
        """Relative gravitational potential (Psi) at radius r."""
        return cast(Quantity['specific energy'], self.Phi0 - self.Phi(r))

    @property
    def Psi_grid(self) -> Quantity['specific energy']:
        """Calculate the relative gravitational potential (Psi) at the density internal logarithmic grid (memoized)."""
        if 'Psi_grid' not in self.memoization:
            self.memoization['Psi_grid'] = self.Psi(self.geomspace_grid)
        return self.memoization['Psi_grid']

    def mass_pdf(self, r: Quantity['length']) -> FloatOrArray:
        """Mass probability density function (pdf) at radius r. Normalized rho*r^2."""
        mass_pdf = self.rho_r2(r).value
        mass_pdf /= np.trapezoid(mass_pdf, r.value)
        return mass_pdf

    def mass_cdf(self, r: Quantity['length']) -> FloatOrArray:
        """Mass cumulative probability density function (cdf) at radius r. Normalized enclosed mass."""
        return (self.M(r) / self.Mtot).value

    def pdf(self, r: Quantity['length']) -> FloatOrArray:
        """Mass probability density function (pdf) interpolated at a given radius (memoized)."""
        if 'pdf' not in self.memoization:
            self.memoization['pdf'] = scipy.interpolate.interp1d(
                self.geomspace_grid.value, self.mass_pdf(self.geomspace_grid), kind='cubic', bounds_error=False, fill_value=(0, 1)
            )
        return self.memoization['pdf'](r.to(run_units.length).value)

    def cdf(self, r: Quantity['length']) -> FloatOrArray:
        """Mass cumulative probability density function (cdf) interpolated at a given radius (memoized)."""
        if 'cdf' not in self.memoization:
            self.memoization['cdf'] = scipy.interpolate.interp1d(
                self.geomspace_grid.value, self.mass_cdf(self.geomspace_grid), kind='cubic', bounds_error=False, fill_value=(0, 1)
            )
        return self.memoization['cdf'](r.to(run_units.length).value)

    def quantile_function(self, p: FloatOrArray) -> Quantity['length']:
        """Mass quantile function (inversed cdf) interpolated at a given radius (memoized)."""
        if 'quantile_function' not in self.memoization:
            rs, cdf = utils.joint_clean([self.geomspace_grid.value, self.mass_cdf(self.geomspace_grid)], ['rs', 'cdf'], 'cdf')
            self.memoization['quantile_function'] = scipy.interpolate.interp1d(
                cdf, rs, kind='cubic', bounds_error=False, fill_value=(self.Rmin.value, self.Rmax.value)
            )
        return Quantity(self.memoization['quantile_function'](p), run_units.length)

    def Psi_to_r(self, Psi: Quantity['specific energy']) -> Quantity['length']:
        """Inverse of the relative gravitational potential (Psi), i.e. calculate the radius giving a set Psi value, using a cubic interpolation (memoized)."""
        if 'Psi_to_r' not in self.memoization:
            r_grid, Psi_grid = utils.joint_clean([self.geomspace_grid.value, self.Psi_grid.value], ['r', 'Psi'], 'Psi')
            self.memoization['Psi_to_r'] = scipy.interpolate.interp1d(
                Psi_grid, r_grid, kind='cubic', bounds_error=False, fill_value=(self.Rmin.value, self.Rmax.value)
            )
        return Quantity(self.memoization['Psi_to_r'](Psi.to(run_units.specific_energy).value), run_units.length)

    def Psi_to_rho(self, Psi: Quantity['specific energy']) -> Quantity['mass density']:
        """Calculate the density at the radius that would give the input Psi value, using a cubic interpolation."""
        return self.rho(self.Psi_to_r(Psi))

    def drhodPsi(self, Psi: Quantity['specific energy']) -> Quantity:
        """Calculate the derivative of the density with respect to Psi."""
        return utils.quantity_derivate(Psi, self.Psi_to_rho)

    def drho2dPsi2(self, Psi: Quantity['specific energy']) -> Quantity:
        """Calculate the second order derivative of the density with respect to Psi."""
        return utils.quantity_derivate2(Psi, self.Psi_to_rho)

    def calculate_f(self, E: Quantity['specific energy']) -> Quantity[run_units.f_units]:
        """Calculate the distribution function (df) at a given specific energy value, using the Eddington inversion method. Internal function, prioritize using f()."""
        scalar_input = np.isscalar(E)
        Psi = cast(Quantity['specific energy'], np.linspace(self.Psi_grid.min(), E.to(run_units.specific_energy).max(), 1000))
        drho2dPsi2 = self.drho2dPsi2(Psi)
        integral = Quantity(np.zeros_like(np.atleast_1d(E.value)), drho2dPsi2.unit * np.sqrt(1 * run_units.specific_energy))
        for i, e in enumerate(np.atleast_1d(E)):
            mask = Psi < e
            integral[i] = scipy.integrate.trapezoid(x=Psi[mask].value, y=(drho2dPsi2[mask] / np.sqrt(e - Psi[mask])).value) * integral.unit
        if scalar_input:
            integral = integral[0]
        return 1 / (self.unit_mass * np.sqrt(8) * np.pi**2) * (self.drhodPsi(0 * run_units.specific_energy) / np.sqrt(E) + integral)

    def E(self, r: Quantity['length'], v: Quantity['velocity']) -> Quantity['specific energy']:
        """Calculate the specific energy (E) for a particle at a given radius and velocity (norm)."""
        return cast(Quantity['specific energy'], self.Psi(r) - v**2 / 2)

    def f(self, E: Quantity['specific energy']) -> Quantity[run_units.f_units]:
        """Calculate the distribution function (df) at a given specific energy value (memoized)."""
        if 'f' not in self.memoization:
            E_grid = cast(Quantity['specific energy'], np.linspace(0, self.Psi_grid.max(), int(1e3))[1:])
            f_grid = self.calculate_f(E_grid)
            self.memoization['f'] = scipy.interpolate.interp1d(E_grid, f_grid, kind='cubic', bounds_error=False, fill_value=(0, 0))
        return Quantity(self.memoization['f'](E.to(run_units.specific_energy).value), run_units.f_units)

    ## Roll initial setup

    def roll_r(self, n_particles: int | float) -> Quantity['length']:
        """Sample particle positions from the distribution quantile function."""
        rolls = np.random.rand(int(n_particles))
        return self.quantile_function(rolls)

    @staticmethod
    @njit(parallel=True)
    def roll_v_fast(Psi: NDArray[np.float64], E_grid: NDArray[np.float64], f_grid: NDArray[np.float64], num: int = 100000) -> NDArray[np.float64]:
        """Sample particle velocity from the distribution function. Internal njit accelerated function. Prioritize using roll_v()."""
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
            p = np.random.rand()
            i = np.searchsorted(cdf, p) - 1
            if i < 0:
                i = 0
            elif i >= len(cdf) - 1:
                i = len(cdf) - 2
            output[particle] = vs[i]
        return output

    def roll_v(self, r: Quantity['length'], num: int = 1000) -> Quantity['velocity']:
        """Sample particle velocity (norm) from the distribution function.

        For full 3d velocity vectors, use roll_v_3d().

        Parameters:
            r: Radius of the particles (required to sample the velocity, should have been sampled using roll_v() prior).
            num: Resolution parameters, defines the number of steps to use in the df integral.

        Returns:
            Sampled velocity norm of the particle, shaped (num_particles,).
        """
        Psi = self.Psi(r).to(run_units.specific_energy)
        E_grid = cast(Quantity['specific energy'], np.linspace(0, self.Psi_grid.max(), int(num))[1:])
        return Quantity(self.roll_v_fast(Psi.value, E_grid.value, f_grid=self.calculate_f(E_grid).value, num=num), run_units.velocity)

    def roll_v_3d(self, r: Quantity['length'], num: int = 1000) -> Quantity['velocity']:
        """Sample particle velocity (3d vectors) from the distribution function. Shape (num_particles, 3) with (vx,vy,vr).

        The velocity is split into radial and perpendicular components by a uniform cosine distributed angle, and then the perpendicular component is split into x and y components by a uniform angle.
        See roll_v() for parameter details.
        """
        return cast(Quantity['velocity'], np.vstack(utils.split_3d(self.roll_v(r))).T)

    def roll_initial_angle(self, n_particles: int) -> NDArray[np.float64]:
        """Sample initial angle off the radial direction from a uniform cosine distribution."""
        theta = np.acos(np.random.rand(n_particles) * 2 - 1)
        return theta

    ##Plots

    def plot_phase_space(
        self,
        r_range: Quantity['length'] = Quantity(np.linspace(1e-2, 50, 200), 'kpc'),
        v_range: Quantity['velocity'] = Quantity(np.linspace(0, 100, 200), 'km/second'),
        velocity_units: UnitLike = 'km/second',
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plots the phase space distribution of the density profile.

        Parameters:
            r_range: Range of radial distances to plot.
            v_range: Range of velocities to plot.
            velocity_units: Units to use for the velocity axis. Set to default to 'km/second' and not 'kpc/Myr' and thus explicitly mentioned (the length unit use the default of the plotting function, and can be passed on as optional keyword arguments)
            kwargs: Additional keyword arguments to pass to utils.plot_phase_space().

        Returns:
            fig, ax.
        """
        r, v = cast(tuple[Quantity['length'], Quantity['velocity']], np.meshgrid(r_range, v_range))
        f = self.f(self.E(r, v))
        grid = np.asarray((16 * np.pi * r**2 * v**2 * f).value)
        fig, ax = utils.plot_phase_space(grid, r_range, v_range, velocity_units=velocity_units, **kwargs)
        return fig, ax

    def add_plot_R_markers(self, ax: Axes, ymax: float, x_units: UnitLike = 'kpc') -> Axes:
        """Adds markers for the scale radius and virial radius to the plot."""
        ax.vlines(x=[self.Rs.to(x_units).value, self.Rvir.to(x_units).value], ymin=0, ymax=ymax, linestyles='dashed', colors='black')
        ax.text(x=self.Rs.to(x_units).value, y=ymax, s='Rs')
        ax.text(x=self.Rvir.to(x_units).value, y=ymax, s='Rvir')
        return ax

    def plot_rho(
        self,
        r_start: Quantity['length'] | None = None,
        r_end: Quantity['length'] | None = Quantity(1e4, 'kpc'),
        density_units: UnitLike = 'Msun/kpc^3',
        length_units: UnitLike = 'kpc',
        fig: Figure | None = None,
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """Plots the density distribution (rho) of the density profile.

        Parameters:
            r_start: The starting radius for the plot. If None, uses Rmin.
            r_end: The ending radius for the plot. If None, uses Rmax.
            density_units: The units for the density axis.
            length_units: The units for the radius axis.
            fig: The figure to plot on. Creates a new one if fig/ax are not given.
            ax: The axes to plot on.  Creates a new one if fig/ax are not given.

        Returns:
            fig, ax.
        """
        fig, ax = utils.setup_plot(
            fig,
            ax,
            title='Density distribution (rho)',
            xlabel=utils.add_label_unit('Radius', length_units),
            ylabel=utils.add_label_unit('Density', density_units),
        )

        if r_start is None:
            r_start = self.Rmin
        if r_end is None:
            r_end = self.Rmax

        r = cast(Quantity['length'], np.geomspace(r_start, r_end, self.space_steps))
        rho = self.rho(r)
        sns.lineplot(x=r.to(length_units).value, y=rho.to(density_units).value, ax=ax)
        ax.set(xscale='log', yscale='log')

        ax = self.add_plot_R_markers(ax, ymax=rho.max().to(density_units).value, x_units=length_units)
        return fig, ax

    def plot_radius_distribution(
        self,
        r_start: Quantity['length'] | None = None,
        r_end: Quantity['length'] | None = None,
        cumulative: bool = False,
        length_units: UnitLike = 'kpc',
        fig: Figure | None = None,
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """Plots the radius distribution (pdf/cdf) of the density profile.

        Parameters:
            r_start: The starting radius for the plot. If None, uses Rmin.
            r_end: The ending radius for the plot. If None, uses Rmax.
            cumulative: Plot the cdf, if False plot the pdf instead.
            length_units: The units for the radius axis.
            fig: The figure to plot on. Creates a new one if fig/ax are not given.
            ax: The axes to plot on.  Creates a new one if fig/ax are not given.

        Returns:
            fig, ax.
        """
        title = 'Particle cumulative range distribution (cdf)' if cumulative else 'Particle range distribution (pdf)'
        fig, ax = utils.setup_plot(fig, ax, title=title, xlabel=utils.add_label_unit('Radius', length_units), ylabel='Density')

        if r_start is None:
            r_start = self.Rmin
        if r_end is None:
            r_end = self.Rmax

        r = cast(Quantity['length'], np.geomspace(r_start, r_end, self.space_steps))
        y = self.mass_cdf(r) if cumulative else self.mass_pdf(r)
        sns.lineplot(x=r.to(length_units).value, y=y, color='r', ax=ax)
        ax = self.add_plot_R_markers(ax, ymax=y.max(), x_units=length_units)
        return fig, ax
