"""Defines types used in the simulation"""

from typing import Any, Literal, TypeVar, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from astropy.units import Unit, Quantity
from scipy.interpolate import UnivariateSpline
from astropy.units.typing import UnitLike

FloatOrArray = TypeVar('FloatOrArray', float, NDArray[np.float64])
QuantityOrArray = Quantity | NDArray[np.float64] | pd.Series
ParticleType = Literal['dm', 'baryon']


class QuantitySpline(UnivariateSpline):
    """Wrapper around `scipy.interpolate.UnivariateSpline` that accepts and returns `astropy.units.Quantity` objects and handles unit conversions."""

    def __init__(self, in_unit: UnitLike | None = None, out_unit: UnitLike | None = None, *args: Any, **kwargs: Any):
        from . import utils

        kwargs = kwargs.copy()
        in_unit = self.guess_unit(in_unit, kwargs.get('x', None))
        out_unit = self.guess_unit(out_unit, kwargs.get('y', None))
        super().__init__(*utils.strip_args_units(*args), **utils.strip_kwargs_units(**kwargs))
        self.in_unit = Unit(str(in_unit))
        self.out_unit = Unit(str(out_unit))
        self.input_args = args
        self.input_kwargs = kwargs

    @staticmethod
    def guess_unit(unit: UnitLike | None = None, array: Quantity | None = None) -> UnitLike:
        """Pull the desired unit from the array if not provided."""
        if unit is None:
            if array is None:
                unit = ''
            else:
                unit = cast(Unit, array.unit)
        return unit

    def __call__(self, x: Quantity, nu: int = 0, ext: int | None = None) -> Quantity:
        """Evaluate the spline"""
        return Quantity(super().__call__(x.to(self.in_unit).value, nu=nu, ext=ext), self.out_unit)

    def derivative_at(self, x: Quantity) -> Quantity:
        """Evaluate the derivative of the spline at a given point"""
        return Quantity(super().derivative()(x.to(self.in_unit).value), self.out_unit / self.in_unit)

    def to_scipy(self) -> UnivariateSpline:
        """Returns a regular `scipy.interpolate.UnivariateSpline` object"""
        return UnivariateSpline(*self.input_args, **self.input_kwargs)
