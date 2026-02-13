"""Eddington's inversion calculations"""

from typing import Any, cast

import numpy as np
import scipy
from astropy.units import Quantity
from scipy.interpolate import UnivariateSpline
from astropy.units.typing import UnitLike

from src import units
from src.types import QuantitySpline

from ..tqdm import tqdm


def integral_f(
    E: float,
    spline: UnivariateSpline,
    limit: int = 200,
    **kwargs: Any,
) -> float:
    """Calculate the antiderivative of the distribution function `df`. Internal function that intentionally doesn't support units.

    Parameters:
        E: The energy value to calculate the antiderivative at.
        spline: A `scipy` spline object for `density` as a function of `potential`.
        limit: Passed on to `scipy.integrate.quad()`.
        epsrel: Passed on to `scipy.integrate.quad()`.
        kwargs: Additional keyword arguments to pass to `scipy.integrate.quad()`.


    Returns:
        The antiderivative value at `E`.
    """
    return scipy.integrate.quad(
        func=lambda x: spline.derivative()(x),
        a=0,
        b=E,
        limit=limit,
        weight='alg',
        wvar=(0, -0.5),  # (t, s) where weight = (x-a)^t * (b-x)^s. t=-0.5 gives (E-x)^(-0.5) = 1/√(E-x)
        **kwargs,
    )[0] / (np.sqrt(8) * np.pi**2)


def make_density_potential_spline(
    potential_grid: Quantity['specific energy'],
    density_grid: Quantity['mass density'],
    s: float | None = 1e-2,
    **kwargs: Any,
) -> QuantitySpline:
    """Create a spline for the mass density as a function of potential.

    Parameters:
        potential_grid: A grid of potential values to calculate the spline on.
        density_grid: A grid of mass density values corresponding to the potential grid.
        s: The smoothing factor for the spline.
        **kwargs: Additional keyword arguments to pass to the spline constructor.

    Returns:
        The spline of mass density as a function of specific energy.
    """
    return QuantitySpline(
        x=potential_grid[indices := np.argsort(potential_grid)].value,
        y=density_grid[indices].value,
        s=s,
        in_unit=str(density_grid.unit),
        out_unit=str(density_grid.unit),
        **kwargs,
    )


def make_integral_f_spline(
    potential_grid: Quantity['specific energy'],
    density_potential_spline: QuantitySpline,
    ext: int = 1,
    integral_f_kwargs: dict[str, Any] = {},
    tqdm_kwargs: dict[str, Any] = {'desc': 'Calculating `F`'},
    **kwargs: Any,
) -> QuantitySpline:
    """Calculate a spline for the antiderivative `F` of the distribution function `df`.

    Parameters:
        potential_grid: A grid of potential values to calculate the spline on.
        density_potential_spline: A `scipy` spline object for `rho` as a function of `potential`.
        ext: Extrapolation mode for the spline.
        integral_f_kwargs: Additional keyword arguments to pass to the integrator `integral_f()`.
        tqdm_kwargs: Additional keyword arguments to pass to the tqdm progress bar.
        kwargs: Additional keyword arguments to pass to the spline object.

    Returns:
        The spline of `F`.
    """
    spline = density_potential_spline.to_scipy()
    integral_f_grid = np.array(
        [integral_f(E=e, spline=spline, **integral_f_kwargs) for e in tqdm(potential_grid.value, **tqdm_kwargs)]
    )
    return QuantitySpline(
        x=potential_grid[indices := np.argsort(potential_grid)].value,
        y=integral_f_grid[indices],
        ext=ext,
        in_unit=str(potential_grid.unit),
        out_unit=units.integral_f_unit,
        **kwargs,
    )


def make_f_spline(
    potential_grid: Quantity['specific energy'],
    integral_f_spline: QuantitySpline,
    out_unit: UnitLike = units.f_unit,
    **kwargs: Any,
) -> QuantitySpline:
    """Calculate a spline for the distribution function `df`."""
    f_grid = integral_f_spline.derivative_at(potential_grid)
    return QuantitySpline(
        x=potential_grid[indices := np.argsort(potential_grid)].value,
        y=f_grid[indices],
        in_unit=str(potential_grid.unit),
        out_unit=str(f_grid.unit),
        **kwargs,
    )


def f(
    E: Quantity['specific energy'],
    integral_f_spline: QuantitySpline,
    reject_negative: bool = True,
) -> Quantity:
    """Calculate the distribution function `df` from the antiderivative `F`."""
    value = integral_f_spline.derivative_at(E) * units.mass
    if reject_negative:
        return cast(Quantity, value.clip(min=0))
    return value
