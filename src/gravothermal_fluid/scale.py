"""Normalization scheme for the gravitational fluid"""

from typing import Any, Literal, cast, get_args, overload

import numpy as np
import pandas as pd
from astropy import constants
from numpy.typing import NDArray
from astropy.units import Unit, Quantity
from astropy.units.typing import UnitLike

from src import units
from src.distribution.distribution import Distribution

QuantityOrArray = Quantity | NDArray[np.float64] | pd.Series

ScaleType = Literal[
    'length',
    'volume',
    'density',
    'mass',
    'velocity',
    'time',
    'cross section',
    'luminosity',
    'luminosity gradient',
    'internal energy',
    'internal energy gradient',
    'pressure',
]


class Scale:
    """Scale management class for the physical properties in the gravitational fluid ODEs"""

    def __init__(
        self,
        r_s: Quantity['length'],
        rho_s: Quantity['mass density'],
        sigma: Quantity[units.cross_section],
        a: float = 4 / np.sqrt(np.pi),
    ):
        """Initialize a Scale object.

        Parameters:
            r_s: Scale length
            rho_s: Scale density
            sigma: Scale cross-section
            a: Scale factor
        """
        self.r_s = r_s.decompose(units.system)
        self.rho_s = rho_s.decompose(units.system)
        self.sigma = sigma.decompose(units.system)
        self.a = a

    @classmethod
    def from_distribution(cls, distribution: Distribution, **kwargs: Any):
        """Create a Scale instance from a Distribution object"""
        return cls(r_s=distribution.r_s, rho_s=distribution.rho_s, **kwargs)

    @property
    def scales(self) -> list[Quantity]:
        """A list of all supported scales"""
        return [self[key] for key in get_args(ScaleType)]

    def __getitem__(self, y: ScaleType) -> Quantity:
        """Get the scale for a given scale type"""
        return getattr(self, y.replace(' ', '_'))

    def get_unit(self, scale_type: ScaleType) -> Unit:
        """Get the scale unit for a given scale type"""
        return cast(Unit, self[scale_type].unit)

    @overload
    def __call__(self, x: Quantity, unit: UnitLike | ScaleType | None = None) -> NDArray[np.float64]: ...

    @overload
    def __call__(self, x: NDArray[np.float64], unit: UnitLike | ScaleType) -> Quantity: ...

    @overload
    def __call__(self, x: NDArray[np.float64], unit: None = None) -> NDArray[np.float64]: ...

    def __call__(self, x: QuantityOrArray, unit: UnitLike | ScaleType | None = None) -> QuantityOrArray:
        """Transform a quantity to a dimensionless array, or a dimensionless array to the dimensionfull quantity"""
        if isinstance(x, Quantity):
            for scale in self.scales:
                if cast(Unit, scale.unit).is_equivalent(x.unit):
                    return (x.to(scale.unit) / scale).value
            raise IOError(
                f"Input doesn't match a known any scale units: {x} of physical type {cast(Unit, x.unit).physical_type}"
            )
        elif str(unit) in get_args(ScaleType):
            return x * self[cast(ScaleType, unit)]
        elif unit is not None:
            for scale in self.scales:
                if cast(Unit, scale.unit).is_equivalent(unit):
                    return (x * scale).to(unit)
            raise IOError(
                f"Input unit doesn't match a known scale: {unit} of physical type {Unit(str(unit)).physical_type}"
            )
        raise IOError("`x` isn't a Quantity and `unit` is missing, unclear instructions")

    @property
    def length(self) -> Quantity['length']:
        """Length scale"""
        return self.r_s

    @property
    def volume(self) -> Quantity['length']:
        """Volume scale"""
        return cast(Quantity, self.length**3)

    @property
    def density(self) -> Quantity['mass density']:
        """Density scale"""
        return self.rho_s

    @property
    def mass(self) -> Quantity['mass']:
        """Mass scale"""
        return (4 * np.pi * self.volume * self.density).decompose(units.system)

    @property
    def velocity(self) -> Quantity['velocity']:
        """Velocity scale"""
        return np.sqrt(constants.G * self.mass / self.length).decompose(units.system)

    @property
    def time(self) -> Quantity['time']:
        """Time scale"""
        return 1 / (self.a * self.sigma * self.velocity * self.density).decompose(units.system)

    @property
    def cross_section(self) -> Quantity[units.cross_section]:
        """Cross section scale"""
        return 1 / (self.length * self.density).decompose(units.system)

    @property
    def luminosity(self) -> Quantity['radiant flux']:
        """Luminosity scale"""
        return (constants.G * self.mass**2 / (self.length * self.time)).decompose(units.system)

    @property
    def luminosity_gradient(self) -> Quantity:
        """Luminosity gradient (dL/dm) scale"""
        return (self.luminosity / self.mass).decompose(units.system)

    @property
    def internal_energy(self) -> Quantity['specific energy']:
        """Energy scale"""
        return (self.velocity**2).decompose(units.system)

    @property
    def internal_energy_gradient(self) -> Quantity:
        """Energy gradient (du/dM) scale"""
        return (self.internal_energy / self.mass).decompose(units.system)

    @property
    def pressure(self) -> Quantity['pressure']:
        """Pressure scale"""
        return (self.density * self.velocity**2).decompose(units.system)
