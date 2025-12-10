from typing import Any, cast

import numpy as np
import scipy
import seaborn as sns
from numba import njit, prange
from astropy import constants
from numpy.typing import NDArray
from astropy.units import Unit, Quantity, def_unit
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from astropy.units.typing import UnitLike

from .. import rng, plot, utils, run_units
from ..types import FloatOrArray, ParticleType


class Distribution:
    """General mass distribution profile."""

    def __init__(
        self,
        Rmin: Quantity['length'] = Quantity(1e-4, 'kpc'),
        Rmax: Quantity['length'] | None = None,
        Rs: Quantity['length'] | None = None,
        c: int | float | None = None,
        Rvir: Quantity['length'] | None = None,
        Mtot: Quantity['mass'] | None = None,
        rho_s: Quantity['mass density'] | None = None,
        space_steps: float | int = 1e3,
        h: Quantity['length'] = Quantity(1e-5, 'kpc'),
        initialize_grids: list[str] = [],
        label: str = '',
        particle_type: ParticleType = 'dm',
    ) -> None:
        """General mass distribution profile.

        Parameters:
            Rmin: Minimum radius of the density profile, used for calculating the `internal logarithmic grid` and set internal cutoffs.
            Rmax: Maximum radius of the density profile, used for calculating the `internal logarithmic grid` and set internal cutoffs.
            Rs: Scale radius of the distribution profile.
            c: Concentration parameter of the distribution profile (such that Rvir = c * Rs).
            Rvir: Virial radius of the distribution profile.
            rho_s: Scale density of the distribution profile. Either `Mtot` or `rho_s` must be provided, and the other will be calculated from the rest of the parameters. If both are provided, they are hard set with no attempts to reconcile the parameters.
            Mtot: Total mass of the distribution profile. Either `Mtot` or `rho_s` must be provided, and the other will be calculated from the rest of the parameters. If both are provided, they are hard set with no attempts to reconcile the parameters.
            space_steps: Number of space steps for the `internal logarithmic grid`.
            h: Radius step size for numerical differentiation.
            initialize_grid: Grids to initialize at startup, otherwise they will only be calculated at runtime as needed.
            label: Label for the density profile.

        Returns:
            General mass distribution object.
        """

        assert sum([Rs is not None, Rvir is not None, c is not None]) == 2, (
            'Exactly two of Rs, Rvir, and c must be specified'
        )
        if Rs is not None and Rvir is not None:
            c = float(Rvir / Rs)
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
        self.h: Quantity['length'] = h.to(run_units.length)
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

        self.memoization = {}
        for grid in initialize_grids:
            getattr(self, grid)

    def __repr__(self):
        return f"""{self.title} density function
  - particle_type = {self.particle_type}
  - Rs = {self.Rs:.4f}
  - c = {self.c:.1f}
  - Rvir = {self.Rvir:.4f}
  - Mtot = {self.Mtot:.3e}
  - rho_s = {self.rho_s:.4f}
  - Tdyn = {(1 * self.Tdyn).to(run_units.time):.4f}

  - Rmin = {self.Rmin:.4f}
  - Rmax = {self.Rmax:.4f}
  - space_steps = {self.space_steps:.0e}"""

    def __call__(self, r: Quantity['length']) -> Quantity['mass density']:
        """Calculate `rho(r)`"""
        return self.rho(r)

    def to_scale(self, x: Quantity['length']) -> Quantity['dimensionless']:
        """Scale the distance, i.e. `x/Rs`"""
        return x.to(self.Rs.unit) / self.Rs

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
    def calculate_rho(r: FloatOrArray, rho_s: float = 1, Rs: float = 1, Rvir: float = 1) -> FloatOrArray:
        """Calculate the density (`rho`) at a given radius.

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
        """Calculate the density (`rho`) at a given radius."""
        return Quantity(
            self.calculate_rho(r.to(run_units.length).value, self.rho_s.value, self.Rs.value, self.Rvir.value),
            run_units.density,
        )

    @property
    def rho_grid(self) -> Quantity['mass density']:
        """Calculate the density (`rho`) at the  `internal logarithmic grid` (memoized)."""
        if 'rho_grid' not in self.memoization:
            self.memoization['rho_grid'] = self.rho(self.geomspace_grid)
        return self.memoization['rho_grid']

    @property
    def rho_grid_h(self) -> Quantity['mass density']:
        """Calculate the density (`rho`) at the  `internal logarithmic grid` with a shift step `h` (memoized). Used for derivative calculations."""
        if 'rho_grid_h' not in self.memoization:
            self.memoization['rho_grid_h'] = self.rho(cast(Quantity, self.geomspace_grid + self.h))
        return self.memoization['rho_grid_h']

    @property
    def rho_grid_2h(self) -> Quantity['mass density']:
        """Calculate the density (`rho`) at the  `internal logarithmic grid` with two shift steps `2h` (memoized). Used for derivative calculations."""
        if 'rho_grid_2h' not in self.memoization:
            self.memoization['rho_grid_2h'] = self.rho(cast(Quantity, self.geomspace_grid + 2 * self.h))
        return self.memoization['rho_grid_2h']

    @property
    def drho_dr_grid(self) -> Quantity:
        """Calculate the derivative of the density (`rho`) at the  `internal logarithmic grid` (memoized)."""
        if 'drho_dr_grid' not in self.memoization:
            self.memoization['drho_dr_grid'] = Quantity((self.rho_grid_h - self.rho_grid) / self.h)
        return self.memoization['drho_dr_grid']

    @property
    def d2rho_dr2_grid(self) -> Quantity:
        """Calculate the second order derivative of the density (`rho`) at the  `internal logarithmic grid` (memoized)."""
        if 'd2rho_dr2_grid' not in self.memoization:
            self.memoization['d2rho_dr2_grid'] = Quantity(
                (self.rho_grid_2h - 2 * self.rho_grid_h + self.rho_grid) / self.h**2
            )
        return self.memoization['d2rho_dr2_grid']

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
        rho_s = self.rho_s.value if use_rho_s else 1
        integral = utils.fast_spherical_rho_integrate(
            np.atleast_1d(r.to(run_units.length).value), self.calculate_rho, rho_s, self.Rs.value, self.Rvir.value
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
        mass_pdf /= np.trapezoid(mass_pdf, r.value)
        return mass_pdf

    def mass_cdf(self, r: Quantity['length']) -> FloatOrArray:
        """Mass cumulative probability density function (cdf) at radius `r`. Normalized enclosed mass."""
        return (self.M(r) / self.Mtot).value

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

    def Psi(self, r: Quantity['length']) -> Quantity['specific energy']:
        """Calculate the relative gravitational potential energy (`Psi`) at a given radius `r`."""
        return cast(Quantity, self.Phi0 - self.Phi(r))

    @property
    def Psi_grid(self) -> Quantity['specific energy']:
        """Calculate the relative gravitational potential (`Psi`) at the  `internal logarithmic grid` (memoized)."""
        if 'Psi_grid' not in self.memoization:
            self.memoization['Psi_grid'] = self.Psi(self.geomspace_grid)
        return self.memoization['Psi_grid']

    @property
    def Psi_grid_h(self) -> Quantity['specific energy']:
        """Calculate the relative gravitational potential (`Psi`) at the  `internal logarithmic grid` with a shift step `h` (memoized). Used for derivative calculations."""
        if 'Psi_grid_h' not in self.memoization:
            self.memoization['Psi_grid_h'] = self.Psi(cast(Quantity, self.geomspace_grid + self.h))
        return self.memoization['Psi_grid_h']

    @property
    def Psi_grid_2h(self) -> Quantity['specific energy']:
        """Calculate the relative gravitational potential (`Psi`) at the  `internal logarithmic grid` with two shift steps `2h` (memoized). Used for derivative calculations."""
        if 'Psi_grid_2h' not in self.memoization:
            self.memoization['Psi_grid_2h'] = self.Psi(cast(Quantity, self.geomspace_grid + 2 * self.h))
        return self.memoization['Psi_grid_2h']

    @property
    def dPsi_dr_grid(self) -> Quantity:
        """Calculate the derivative of the relative gravitational potential (`Psi`) at the  `internal logarithmic grid` (memoized)."""
        if 'dPsi_dr_grid' not in self.memoization:
            self.memoization['dPsi_dr_grid'] = Quantity((self.Psi_grid_h - self.Psi_grid) / self.h)
        return self.memoization['dPsi_dr_grid']

    @property
    def d2Psi_dr2_grid(self) -> Quantity:
        """Calculate the second order derivative of the relative gravitational potential (`Psi`) at the  `internal logarithmic grid` (memoized)."""
        if 'd2Psi_dr2_grid' not in self.memoization:
            self.memoization['d2Psi_dr2_grid'] = Quantity(
                (self.Psi_grid_2h - 2 * self.Psi_grid_h + self.Psi_grid) / self.h**2
            )
        return self.memoization['d2Psi_dr2_grid']

    @property
    def drho_dPsi_grid(self) -> Quantity:
        r"""Calculate the derivative of the density with respect to the relative gravitational potential (`Psi`) at the  `internal logarithmic grid` (memoized).

        Simplified using the chain rule to allow calculating based on the spatial derivatives at a consistent grid:
            $\frac{\mathrm{d}\rho}{\mathrm{d}\Psi} =
            \frac{\mathrm{d}\rho}{\mathrm{d}r}\frac{\mathrm{d}r}{\mathrm{d}\Psi} =
            \rho^\prime\left(\frac{\mathrm{d}\Psi}{\mathrm{d}r}\right)^{-1} =
            \frac{\rho^\prime}{\Psi^\prime}$
        """
        if 'drho_dPsi_grid' not in self.memoization:
            self.memoization['drho_dPsi_grid'] = self.drho_dr_grid / self.dPsi_dr_grid
        return self.memoization['drho_dPsi_grid']

    @property
    def d2rho_dPsi2_grid(self) -> Quantity:
        r"""Calculate the second order derivative of the density with respect to the relative gravitational potential (`Psi`) at the  `internal logarithmic grid` (memoized).

        Simplified using the chain rule (Faà di Bruno's formula) to allow calculating based on the spatial derivatives at a consistent grid:
            $\frac{\mathrm{d}^2\rho}{\mathrm{d}\Psi^2} =
            \frac{\mathrm{d}^2\rho}{\mathrm{d}r^2}\left(\frac{\mathrm{d}r}{\mathrm{d}\Psi}\right)^2 +
            \frac{\mathrm{d}\rho}{\mathrm{d}r}\frac{\mathrm{d}^2r}{\mathrm{d}\Psi^2}$

            $\frac{\mathrm{d}^2r}{\mathrm{d}\Psi^2} &=
            \frac{\mathrm{d}}{\mathrm{d}\Psi}\left(\frac{\mathrm{d}r}{\mathrm{d}\Psi}\right) =
            \frac{\mathrm{d}}{\mathrm{d}\Psi}\left(\left(\frac{\mathrm{d}\Psi}{\mathrm{d}r}\right)^{-1}\right) =
            -\left(\frac{\mathrm{d}\Psi}{\mathrm{d}r}\right)^{-2}\frac{\mathrm{d}}{\mathrm{d}\Psi}\left(\frac{\mathrm{d}\Psi}{\mathrm{d}r}\right) =$
            $= -\left(\frac{\mathrm{d}\Psi}{\mathrm{d}r}\right)^{-2}\frac{\frac{\mathrm{d}}{\mathrm{d}r}\left(\frac{\mathrm{d}\Psi}{\mathrm{d}r}\right)}{\frac{\mathrm{d}\Psi}{\mathrm{d}r}} =
            -\left(\frac{\mathrm{d}\Psi}{\mathrm{d}r}\right)^{-3}\frac{\mathrm{d}^2\Psi}{\mathrm{d}r^2} =
            -\left(\Psi^\prime\right)^{-3}\Psi^{\prime\prime}$

            $\frac{\mathrm{d}^2\rho}{\mathrm{d}\Psi^2} =
            \rho^{\prime\prime}\left(\frac{\mathrm{d}r}{\mathrm{d}\Psi}\right)^2 -
            \frac{\rho^{\prime}\Psi^{\prime\prime}}{\left(\Psi^\prime\right)^3} =
            \rho^{\prime\prime}\left(\frac{\mathrm{d}\Psi}{\mathrm{d}r}\right)^{-2} - \frac{\rho^{\prime}\Psi^{\prime\prime}}{\left(\Psi^\prime\right)^3} =
            \frac{\rho^{\prime\prime}}{\left(\Psi^\prime\right)^2} - \frac{\rho^{\prime}\Psi^{\prime\prime}}{\left(\Psi^\prime\right)^3}$
        """
        if 'd2rho_dPsi2_grid' not in self.memoization:
            self.memoization['d2rho_dPsi2_grid'] = Quantity(
                self.d2rho_dr2_grid / self.dPsi_dr_grid**2
                - self.drho_dr_grid / self.dPsi_dr_grid**3 * self.d2Psi_dr2_grid
            )
        return self.memoization['d2rho_dPsi2_grid']

    def d2rho_dPsi2(self, Psi: Quantity['specific energy']) -> Quantity:
        """Interpolate the second order derivative of the density with respect to the relative gravitational potential (`Psi`) based on the  `internal logarithmic grid` (memoized)."""
        if 'd2rho_dPsi2' not in self.memoization:
            self.memoization['d2rho_dPsi2'] = scipy.interpolate.interp1d(
                self.Psi_grid, self.d2rho_dPsi2_grid, bounds_error=False, fill_value=0
            )
        return Quantity(self.memoization['d2rho_dPsi2'](Psi.to(self.Psi_grid.unit)), self.d2rho_dPsi2_grid.unit)

    def calculate_f(self, E: Quantity, num: int = 10000) -> Quantity:
        """Calculate the distribution function (`f`) for the given energy."""
        t = Quantity(np.linspace(0, np.sqrt(E), num))[:-1]
        integral = 2 * np.trapezoid(self.d2rho_dPsi2(cast(Quantity, E - t**2)), t, axis=0)
        return 1 / (self.Mtot.unit * np.sqrt(8) * np.pi**2) * (1 / np.sqrt(E) * self.drho_dPsi_grid[-1] + integral)

    @property
    def E_grid(self) -> Quantity['specific energy']:
        """Set the `internal energy grid` to a logarithmic scale with `1,000` steps from the minimum `Psi_grid` value to the maximum one (memoized)."""
        if 'E_grid' not in self.memoization:
            self.memoization['E_grid'] = np.geomspace(self.Psi_grid.min(), self.Psi_grid.max(), 1000)
        return self.memoization['E_grid']

    @property
    def f_grid(self) -> Quantity['specific energy']:
        """Calculate the distribution function (`f`) on the `internal energy grid` (memoized)."""
        if 'f_grid' not in self.memoization:
            self.memoization['f_grid'] = self.calculate_f(self.E_grid, 10000).to(run_units.f_units)
        return self.memoization['f_grid']

    def f(self, E: Quantity['specific energy']) -> Quantity:
        """Interpolate the distribution function (`f`) based on the `internal energy grid` (memoized)."""
        if 'f' not in self.memoization:
            self.memoization['f'] = scipy.interpolate.interp1d(
                self.E_grid, self.f_grid, bounds_error=False, fill_value=0
            )
        return Quantity(self.memoization['f'](E.to(self.E_grid.unit)), self.f_grid.unit)

    def Psi_interpolate(self, r: Quantity['length']) -> Quantity['specific energy']:
        """Interpolate the relative gravitational potential (`Psi`) based on the  `internal logarithmic grid` (memoized)."""
        if 'Psi_interpolate' not in self.memoization:
            self.memoization['Psi_interpolate'] = scipy.interpolate.interp1d(
                self.geomspace_grid, self.Psi_grid, bounds_error=False, fill_value=0
            )
        return Quantity(self.memoization['Psi_interpolate'](r.to(self.geomspace_grid.unit)), self.Psi_grid.unit)

    def E(self, r: Quantity['length'], v: Quantity['velocity']) -> Quantity['specific energy']:
        """Interpolate the internal energy (`E`) At the given radius `r` using `Psi_interpolate()`."""
        return cast(Quantity, self.Psi_interpolate(r) - 1 / 2 * v**2)

    def recalculate(self, key: str, inplace: bool = False) -> Any:
        """Recalculate the memoized value of the given key."""
        if key in self.memoization:
            self.memoization.pop(key)
        new_value = getattr(self, key)
        if not inplace:
            return new_value

    @staticmethod
    def merge_distribution_grids(distributions: list['Distribution'], grid_base_name: list[str] = ['Psi']):
        """Merges the grid values of the given distributions of the given types. Used to combine the potentials of multiple distributions to sample via Eddington's inversion."""
        for grid_name in grid_base_name:
            for suffix in ['', '_h', '_2h']:
                grid = sum([getattr(density, f'{grid_name}_grid{suffix}') for density in distributions])
                for distribution in distributions:
                    distribution.memoization[f'{grid_name}_grid{suffix}'] = grid

    ## Roll initial setup

    def sample_r(self, n_particles: int | float, generator: np.random.Generator | None = None) -> Quantity['length']:
        """Sample particle positions from the distribution quantile function."""
        if generator is None:
            generator = rng.generator
        return self.quantile_function(generator.random(int(n_particles)))

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
                Psi=self.Psi_interpolate(r).to(run_units.specific_energy),
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

    def sample_old(
        self,
        n_particles: int | float,
        num: int = 1000,
        generator: np.random.Generator | None = None,
    ) -> tuple[Quantity['length'], Quantity['velocity']]:
        """Sample particles In two steps:
            - Sample radius from the inversed CDF.
            - Calculate the velocity CDF for every particle's sampled radius, and sample from it's inverse.

        Parameters:
            n_particles: Number of particles to sample.
            num: Resolution parameters, defines the number of steps to use in the df integral.
            generator: If not provided, use the default generator defined in `rng.generator`.

        Returns:
            A tuple of two vectors:
                Sampled radius values for each particle, shaped `(num_particles,)`
                Corresponding 3d velocities for each particle, shaped `(num_particles,3)`.
        """
        r = self.sample_r(n_particles, generator=generator).to(run_units.length)
        v = self.sample_v(r, num=num, generator=generator).to(run_units.velocity)
        return r, v

    def sample(
        self,
        n_particles: int | float,
        radius_min_value: Quantity['length'] = Quantity(1e-4, 'kpc'),
        radius_max_value: Quantity['length'] | None = None,
        velocity_min_value: Quantity['velocity'] = Quantity(0, 'km/second'),
        velocity_max_value: Quantity['velocity'] = Quantity(100, 'km/second'),
        radius_resolution: int | float = 10000,
        velocity_resolution: int | float = 10000,
        radius_range: Quantity['length'] | None = None,
        velocity_range: Quantity['velocity'] | None = None,
        radius_noise: float = 0.1,
        velocity_noise: float = 0.1,
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
            radius_min_value: Minimum radius value to consider for the phase space distribution.
            radius_max_value: Maximum radius value to consider for the phase space distribution. If `None` use the `Rmax` value of the distribution. Regardless, this value is capped by `Rmax` even if provided.
            velocity_min_value: Minimum velocity norm value to consider for the phase space distribution.
            velocity_max_value: Maximum velocity norm value to consider for the phase space distribution.
            radius_resolution: Resolution of the radius grid.
            velocity_resolution: Resolution of the velocity grid.
            radius_range: Radius bins to use. If provided override the `radius_min_value`, `radius_max_value`, and `radius_resolution` parameters. If `None` ignores.
            velocity_range: Velocity bins to use. If provided override the `velocity_min_value`, `velocity_max_value`, and `velocity_resolution` parameters. If `None` ignores.
            radius_noise: Noise factor for the sampled radius values. The samples are perturbed by a uniform distribution in a symmetric interval with half-width `radius_noise` * `r` (i.e. every particle is pertubed by a relative noise of `radius_noise`), and at most the actual radius resolution in `radius_range`.
            velocity_noise: Noise factor for the sampled velocity values. Same logic as `radius_noise`.
            generator: If not provided, use the default generator defined in `rng.generator`.

        Returns:
            A tuple of two vectors:
                Sampled radius values for each particle, shaped `(num_particles,)`
                Corresponding 3d velocities for each particle, shaped `(num_particles,3)`.
        """
        if generator is None:
            generator = rng.generator
        if radius_range is None:
            if radius_max_value is None:
                radius_max_value = self.Rmax
            else:
                radius_max_value = cast(Quantity, np.min(radius_max_value, self.Rmax))
            radius_range = Quantity(np.linspace(radius_min_value, radius_max_value, int(radius_resolution)))
        if velocity_range is None:
            velocity_range = Quantity(np.linspace(velocity_min_value, velocity_max_value, int(velocity_resolution)))

        r_grid, v_grid = cast(tuple[Quantity, Quantity], np.meshgrid(radius_range[:-1], velocity_range[:-1]))

        drdv = np.prod(np.meshgrid(radius_range.diff(), velocity_range.diff()), axis=0)
        grid = np.array(16 * np.pi * r_grid**2 * v_grid**2 * self.f(self.E(r_grid, v_grid)) * drdv)
        grid /= grid.sum()
        flat_grid = grid.ravel()
        if (grid < 0).any():
            raise ValueError(f'Negative probability density encountered, {self}')
        indices = np.unravel_index(
            generator.choice(a=flat_grid.size, size=int(n_particles), p=flat_grid),
            grid.shape,
        )

        jittered_indices = indices + np.array(
            [generator.uniform(0, f, size=int(n_particles)) for f in [radius_noise, velocity_noise]]
        )

        radius, velocity = tuple(
            Quantity(scipy.ndimage.map_coordinates(grid, jittered_indices, order=1, mode='nearest'), grid.unit)
            for grid in [r_grid, v_grid]
        )

        return (
            cast(Quantity, np.abs(radius)),
            cast(Quantity, np.vstack(utils.split_3d(np.abs(velocity), generator=generator)).T),
        )

    ##Plots

    def plot_phase_space(
        self,
        r_range: Quantity['length'] = Quantity(np.linspace(1e-2, 35, 200), 'kpc'),
        v_range: Quantity['velocity'] = Quantity(np.linspace(0, 60, 200), 'km/second'),
        velocity_units: UnitLike = 'km/second',
        cmap: str = 'jet',
        transparent_value: float | None = 0,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the phase space distribution of the density profile.

        Parameters:
            r_range: Range of radial distances to plot.
            v_range: Range of velocities to plot.
            velocity_units: Units to use for the velocity axis.
            cmap: The colormap to use for the plot.
            transparent_value: Grid value to turn transparent (i.e. plot as `NaN`). If `None` ignores.
            kwargs: Additional keyword arguments to pass to `plot.plot_phase_space()`.

        Returns:
            fig, ax.
        """
        r, v = cast(tuple[Quantity, Quantity], np.meshgrid(r_range, v_range))
        f = self.f(self.E(r, v))
        grid = 16 * np.pi * r**2 * v**2 * f
        fig, ax = plot.plot_phase_space(
            grid,
            r_range,
            v_range,
            velocity_units=velocity_units,
            cmap=cmap,
            transparent_value=transparent_value,
            **kwargs,
        )
        return fig, ax

    def add_plot_R_markers(self, ax: Axes, ymax: float, x_units: UnitLike = 'kpc') -> Axes:
        """Add markers for the scale radius and virial radius to the plot."""
        ax.vlines(
            x=[self.Rs.to(x_units).value, self.Rvir.to(x_units).value],
            ymin=0,
            ymax=ymax,
            linestyles='dashed',
            colors='black',
        )
        ax.text(x=self.Rs.to(x_units).value, y=ymax, s='Rs')
        ax.text(x=self.Rvir.to(x_units).value, y=ymax, s='Rvir')
        return ax

    def plot_rho(
        self,
        r_start: Quantity['length'] | None = None,
        r_end: Quantity['length'] | None = Quantity(1e4, 'kpc'),
        density_units: UnitLike = 'Msun/kpc^3',
        length_units: UnitLike = 'kpc',
        label: str | None = None,
        add_markers: bool = True,
        ax_set: dict[str, Any] = {'xscale': 'log', 'yscale': 'log'},
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the density distribution (`rho`) of the density profile.

        Parameters:
            r_start: The starting radius for the plot. If `None`, uses `Rmin`.
            r_end: The ending radius for the plot. If `None` uses `Rmax`.
            density_units: The units for the density axis.
            length_units: The units for the radius axis.
            ax_set: Additional keyword arguments to pass to `Axes.set()`. e.g `{'xscale': 'log'}`.
            kwargs: Additional keyword arguments to pass to `plot.setup_plot()`.

        Returns:
            fig, ax.
        """
        fig, ax = plot.setup_plot(
            title='Density distribution (rho)',
            xlabel=utils.add_label_unit('Radius', length_units),
            ylabel=utils.add_label_unit('Density', density_units),
            ax_set=ax_set,
            **kwargs,
        )

        if r_start is None:
            r_start = self.Rmin
        if r_end is None:
            r_end = self.Rmax

        r = cast(Quantity, np.geomspace(r_start, r_end, self.space_steps))
        rho = self.rho(r)
        sns.lineplot(x=r.to(length_units).value, y=rho.to(density_units).value, ax=ax, label=label)

        if add_markers:
            ax = self.add_plot_R_markers(ax, ymax=rho.max().to(density_units).value, x_units=length_units)

        if label is not None:
            ax.legend()
        return fig, ax

    def plot_radius_distribution(
        self,
        r_start: Quantity['length'] | None = None,
        r_end: Quantity['length'] | None = None,
        cumulative: bool = False,
        length_units: UnitLike = 'kpc',
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
            length_units: The units for the radius axis.
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
        fig, ax = plot.setup_plot(
            fig, ax, title=title, xlabel=utils.add_label_unit('Radius', length_units), ylabel='Density'
        )

        if r_start is None:
            r_start = self.Rmin
        if r_end is None:
            r_end = self.Rmax

        r = cast(Quantity, np.geomspace(r_start, r_end, self.space_steps))
        y = self.mass_cdf(r) if cumulative else self.mass_pdf(r)
        sns.lineplot(x=r.to(length_units).value, y=y, color=color, ax=ax, label=label, **kwargs)
        if add_markers:
            ax = self.add_plot_R_markers(ax, ymax=y.max(), x_units=length_units)
        return fig, ax

    def plot_drho_dPsi(
        self,
        xlabel: str | None = r'$\Psi$',
        ylabel: str | None = r'$\frac{\mathrm{d}\rho}{\mathrm{d}\Psi}$',
        title: str | None = 'Density derivative (log-log)',
        energy_units: UnitLike = 'km^2/second^2',
        y_units: UnitLike = 'Msun*second^2/(kpc^3*km^2)',
        fig: Figure | None = None,
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot the density derivative with respect to the relative potential energy `drho/dPsi`.

        Parameters:
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            energy_units: Units for the energy axis.
            y_units: Units for the y-axis.
            fig: Figure to plot on.
            ax: Axes to plot on.

        Returns:
            fig, ax
        """

        xlabel = utils.add_label_unit(xlabel, energy_units)
        ylabel = utils.add_label_unit(ylabel, y_units)
        fig, ax = plot.setup_plot(
            fig,
            ax,
            minorticks=True,
            ax_set={'xscale': 'log', 'yscale': 'log'},
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )
        sns.lineplot(x=np.array(self.Psi_grid.to(energy_units)), y=np.array(self.drho_dPsi_grid.to(y_units)), ax=ax)
        return fig, ax

    def plot_f(
        self,
        E: Quantity['specific energy'] = Quantity(np.geomspace(50, 1800, 1000), 'km^2/second^2'),
        xlabel: str | None = r'$\mathcal{E}$',
        ylabel: str | None = 'f',
        title: str | None = 'Particle distribution function',
        energy_units: UnitLike = 'km^2/second^2',
        f_units: UnitLike = 'second^3/(kpc^3*km^3)',
        fig: Figure | None = None,
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot the distribution function `f(E)`.

        Parameters:
            E: Energy values to plot.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            energy_units: Units for the energy axis.
            f_units: Units for the density function axis.
            fig: Figure to plot on.
            ax: Axes to plot on.

        Returns:
            fig, ax
        """

        xlabel = utils.add_label_unit(xlabel, energy_units)
        ylabel = utils.add_label_unit(ylabel, f_units)
        fig, ax = plot.setup_plot(
            fig,
            ax,
            minorticks=True,
            ax_set={'xscale': 'log', 'yscale': 'log'},
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )
        sns.lineplot(x=np.array(E.to(energy_units)), y=np.array(self.f(E).to(f_units)), ax=ax)
        return fig, ax
