"""Defines types used in the simulation"""

from typing import Any, Literal, TypeVar, cast

import numpy as np
import scipy
import pandas as pd
from numpy.typing import NDArray
from astropy.units import Unit, Quantity
from astropy.units.typing import UnitLike

FloatOrArray = TypeVar('FloatOrArray', float, NDArray[np.float64])
QuantityOrArray = Quantity | NDArray[np.float64] | pd.Series
QuantityLike = TypeVar('QuantityLike', Quantity, NDArray[np.float64])
ParticleType = Literal['dm', 'baryon', 'cdm']
TimeUnitLike = UnitLike | Literal['time step', 'dynamical time', 'core collapse', 't_c']
ErrorHandle = Literal['error', 'warning', 'suppress']
PhysicalProperty = Literal[
    'density', 'velocity dispersion', 'temperature', 'pressure', 'internal energy', 'enclosed_mass'
]

T = TypeVar('T')


def regulate_arguments(scheme: type[T], **kwargs: Any) -> T:
    """Regulates the kwargs argument to only insert the relevant ones to `scheme` (a TypedDict)."""
    valid_keys = set(scheme.__annotations__.keys())
    filtered_kwargs: dict[valid_keys, Any] = {k: v for k, v in kwargs.items() if k in valid_keys}
    return cast(scheme, filtered_kwargs)


class QuantitySpline(scipy.interpolate.UnivariateSpline):
    """Wrapper around `scipy.interpolate.UnivariateSpline` that accepts and returns `astropy.units.Quantity` objects and handles unit conversions."""

    def __init__(
        self, in_unit: UnitLike | None = None, out_unit: UnitLike | None = None, *args: Any, **kwargs: Any
    ) -> None:
        from src.utils import utils

        kwargs = kwargs.copy()
        in_unit = utils.guess_unit(in_unit, kwargs.get('x', None))
        out_unit = utils.guess_unit(out_unit, kwargs.get('y', None))
        super().__init__(*utils.strip_args_units(*args), **utils.strip_kwargs_units(**kwargs))
        self.in_unit = Unit(str(in_unit))
        self.out_unit = Unit(str(out_unit))
        self.input_args = args
        self.input_kwargs = kwargs

    def __call__(self, x: Quantity | NDArray[np.float64] | float, nu: int = 0, ext: int | None = None) -> Quantity:
        """Evaluate the function"""
        if isinstance(x, Quantity):
            x = x.to(self.in_unit).value
        return Quantity(super().__call__(x, nu=nu, ext=ext), self.out_unit)

    def derivative_at(self, x: Quantity) -> Quantity:
        """Evaluate the derivative of the spline at a given point"""
        return Quantity(super().derivative()(x.to(self.in_unit).value), self.out_unit / self.in_unit)

    def roots(self) -> Quantity:
        "Return the zeros of the spline"
        return Quantity(super().roots(), self.in_unit)

    def to_scipy(self) -> scipy.interpolate.UnivariateSpline:
        """Returns the regular `scipy` object"""
        return scipy.interpolate.UnivariateSpline(*self.input_args, **self.input_kwargs)


class QuantityInterpolate(scipy.interpolate.interp1d):
    """Wrapper around `scipy.interpolate.interp1d` that accepts and returns `astropy.units.Quantity` objects and handles unit conversions."""

    def __init__(
        self, in_unit: UnitLike | None = None, out_unit: UnitLike | None = None, *args: Any, **kwargs: Any
    ) -> None:
        from src.utils import utils

        kwargs = kwargs.copy()
        in_unit = utils.guess_unit(in_unit, kwargs.get('x', None))
        out_unit = utils.guess_unit(out_unit, kwargs.get('y', None))
        super().__init__(*utils.strip_args_units(*args), **utils.strip_kwargs_units(**kwargs))
        self.in_unit = Unit(str(in_unit))
        self.out_unit = Unit(str(out_unit))
        self.input_args = args
        self.input_kwargs = kwargs

    def __call__(self, x: Quantity | NDArray[np.float64] | float) -> Quantity:
        """Evaluate the function"""
        if isinstance(x, Quantity):
            x = x.to(self.in_unit).value
        return Quantity(super().__call__(x), self.out_unit)

    def to_scipy(self) -> scipy.interpolate.UnivariateSpline:
        """Returns the regular `scipy` object"""
        return scipy.interpolate.UnivariateSpline(*self.input_args, **self.input_kwargs)
