from typing import Any, cast

import numpy as np
import scipy
from numpy.typing import NDArray
from astropy.units import Unit, Quantity
from scipy.interpolate import UnivariateSpline
from astropy.units.typing import UnitLike

from src import run_units

from ..tqdm import tqdm


class QuantitySpline(UnivariateSpline):
    """Wrapper around `scipy.interpolate.UnivariateSpline` that accepts and returns `astropy.units.Quantity` objects and handles unit conversions."""

    def __init__(self, in_unit: UnitLike = '', out_unit: UnitLike = '', *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.in_unit = Unit(str(in_unit))
        self.out_unit = Unit(str(out_unit))
        self.input_args = args
        self.input_kwargs = kwargs

    def __call__(self, x: Quantity, nu: int = 0, ext: int | None = None) -> Quantity:
        """Evaluate the spline"""
        return Quantity(super().__call__(x.to(self.in_unit).value, nu=nu, ext=ext), self.out_unit)

    def derivative_at(self, x: Quantity) -> Quantity:
        """Evaluate the derivative of the spline at a given point"""
        return Quantity(super().derivative()(x.to(self.in_unit).value), self.out_unit / self.in_unit)

    def to_scipy(self) -> UnivariateSpline:
        """Returns a regular `scipy.interpolate.UnivariateSpline` object"""
        return UnivariateSpline(*self.input_args, **self.input_kwargs)


def F(
    E: float,
    spline: UnivariateSpline,
    limit: int = 100,
    epsabs: float = 1e-8,
    epsrel: float = 1e-6,
    difficulties_bound_factor: tuple[float, float] | list[float] | NDArray[np.float64] = (0.95, 0.9999),
    **kwargs: Any,
) -> float:
    """Calculate the antiderivative of the distribution function `df`. Internal function that intentionally doesn't support units.

    Parameters:
        E: The energy value to calculate the antiderivative at.
        spline: A `scipy` spline object for `rho` as a function of `Psi`.
        limit: Passed on to `scipy.integrate.quad()`.
        epsabs: Passed on to `scipy.integrate.quad()`.
        epsrel: Passed on to `scipy.integrate.quad()`.
        difficulties_bound_factor: Defines a list of breakpoints around difficulties to help `scipy.integrate.quad()` converge. The values are multiplied by `E`. The maximum value is taken as the end-point of the integral (there's always a pole there).
        kwargs: Additional keyword arguments to pass to `scipy.integrate.quad()`.


    Returns:
        The antiderivative value at `E`.
    """
    return scipy.integrate.quad(
        func=lambda x: (1 / np.sqrt(E - x)) * spline.derivative()(x),
        a=0,
        b=E * np.array(difficulties_bound_factor).max(),
        limit=limit,
        epsabs=epsabs,
        epsrel=epsrel,
        points=np.array(difficulties_bound_factor) * E,
        **kwargs,
    )[0] / (np.sqrt(8) * np.pi**2)


def make_rho_Psi_spline(
    Psi_grid: Quantity['specific energy'],
    rho_grid: Quantity['mass density'],
    s: float | None = 1e-2,
    **kwargs: Any,
) -> QuantitySpline:
    """Create a spline for the mass density as a function of Psi.

    Parameters:
        Psi_grid: A grid of Psi values to calculate the spline on.
        rho_grid: A grid of mass density values corresponding to the Psi grid.
        s: The smoothing factor for the spline.
        **kwargs: Additional keyword arguments to pass to the spline constructor.

    Returns:
        The spline of mass density as a function of specific energy.
    """
    return QuantitySpline(
        x=Psi_grid[indices := np.argsort(Psi_grid)].value,
        y=rho_grid[indices].value,
        s=s,
        in_unit=str(rho_grid.unit),
        out_unit=str(rho_grid.unit),
        **kwargs,
    )


def make_F_spline(
    Psi_grid: Quantity['specific energy'],
    rho_Psi_spline: QuantitySpline,
    ext: int = 1,
    F_kwargs: dict[str, Any] = {},
    tqdm_kwargs: dict[str, Any] = {'desc': 'Calculating `F`'},
    **kwargs: Any,
) -> QuantitySpline:
    """Calculate a spline for the antiderivative `F` of the distribution function `df`.

    Parameters:
        Psi_grid: A grid of Psi values to calculate the spline on.
        rho_Psi_spline: A `scipy` spline object for `rho` as a function of `Psi`.
        ext: Extrapolation mode for the spline.
        F_kwargs: Additional keyword arguments to pass to the integrator `F()`.
        tqdm_kwargs: Additional keyword arguments to pass to the tqdm progress bar.
        kwargs: Additional keyword arguments to pass to the spline object.

    Returns:
        The spline of `F`.
    """
    spline = rho_Psi_spline.to_scipy()
    F_grid = np.array([F(E=e, spline=spline, **F_kwargs) for e in tqdm(Psi_grid.value, **tqdm_kwargs)])
    return QuantitySpline(
        x=Psi_grid[indices := np.argsort(Psi_grid)].value,
        y=F_grid[indices],
        ext=ext,
        in_unit=str(Psi_grid.unit),
        out_unit=run_units.F_unit,
        **kwargs,
    )


def make_f_spline(
    Psi_grid: Quantity['specific energy'],
    F_spline: QuantitySpline,
    out_unit: UnitLike = run_units.f_unit,
    **kwargs: Any,
) -> QuantitySpline:
    """Calculate a spline for the distribution function `f`."""
    f_grid = F_spline.derivative_at(Psi_grid)
    return QuantitySpline(
        x=Psi_grid[indices := np.argsort(Psi_grid)].value,
        y=f_grid[indices],
        in_unit=str(Psi_grid.unit),
        out_unit=str(f_grid.unit),
        **kwargs,
    )


def f(
    E: Quantity['specific energy'],
    F_spline: QuantitySpline,
    reject_negative: bool = True,
) -> Quantity:
    """Calculate the distribution function `f` from the antiderivative `F`."""
    value = F_spline.derivative_at(E)
    if reject_negative:
        return cast(Quantity, value.clip(min=0))
    return value
