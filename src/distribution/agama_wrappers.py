from typing import Any, TypeVar, Callable, cast
from functools import wraps

import agama
import numpy as np
from numpy.typing import NDArray
from astropy.units import Unit, Quantity

from src.types import QuantityOrArray

from .. import utils

T = TypeVar('T')


def vectorize(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator that ravels the first argument before calling the function,
    then reshapes the result back to the original shape.

    Handles scalar inputs specially by preserving scalar output.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        if 'r' in kwargs:
            r = kwargs.pop('r')
            self = args[0]
            rest_args = args[1:]
        elif len(args) < 2:
            raise TypeError(f'{func.__name__} missing required positional argument')
        else:
            self, r = args[0], args[1]
            rest_args = args[2:]

        if np.isscalar(r) or (hasattr(r, 'shape') and r.shape == ()):
            # For scalar input, call function directly and return scalar result
            result = func(*args, **kwargs)
            # Ensure scalar output (in case function returns array with shape ())
            if hasattr(result, 'shape') and result.shape == ():
                return result.item() if hasattr(result, 'item') else result
            return result

        return func(self, r.ravel(), *rest_args, **kwargs).reshape(r.shape)

    return wrapper


def length() -> Unit:
    """Shorthand for the agama length unit"""
    return agama.getUnits()['length']


def time() -> Unit:
    """Shorthand for the agama time unit"""
    return agama.getUnits()['time']


def velocity() -> Unit:
    """Shorthand for the agama velocity unit"""
    return agama.getUnits()['velocity']


def mass() -> Unit:
    """Shorthand for the agama mass unit"""
    return agama.getUnits()['mass']


def to_3d(x: QuantityOrArray) -> QuantityOrArray:
    """Convert a spherical norm value to a 3D vector such that the values are on the x axis.

    Parameters:
        x: The input. Must be either a number or a 1d array/Quantity.

    Returns:
        Array of shape `(len(x),3)`
    """
    return np.squeeze(np.pad(np.atleast_2d(x), ((0, 2), (0, 0))).T)


class Potential:
    """Wrapper for the agama Potential class"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.potential = agama.Potential(*[(x.potential if isinstance(x, Potential) else x) for x in args], **kwargs)

    def to_action(self, *args: Any, **kwargs: Any) -> NDArray[np.float64]:
        """Transforms the given coordinates to actions."""
        return agama.ActionFinder(self.potential)(*args, **kwargs)

    @vectorize
    def density(self, r: Quantity['length']) -> Quantity['mass density']:
        """Calculate the density at a given radius."""
        return Quantity(self.potential.density(to_3d(r).to(length()).value), mass() / length() ** 3)

    @vectorize
    def Phi(self, r: Quantity['length']) -> Quantity['specific energy']:
        """Calculate the potential at a given radius."""
        return Quantity(self.potential.potential(to_3d(r).to(length()).value), velocity() ** 2)


class DistributionFunction:
    """Wrapper for the agama DistributionFunction class"""

    def __init__(self, potential: Any, density: Any, *args: Any, **kwargs: Any) -> None:
        self.potential = potential
        self.density = density
        if isinstance(potential, Potential):
            potential = potential.potential
        if isinstance(density, Potential):
            density = density.potential
        self.df = agama.DistributionFunction(potential=potential, density=density, *args, **kwargs)

    def __call__(
        self, r: Quantity['length'], v: Quantity['velocity'], potential: Any = None, zero_null: bool = True
    ) -> Quantity:
        """Evaluate the distribution function at the given coordinates."""
        if potential is None:
            potential = self.potential
        f = self.df(
            potential.to_action(np.hstack([to_3d(r.ravel()).to(length()).value, to_3d(v.ravel()).to(velocity()).value]))
        )
        if zero_null:
            if np.isscalar(f):
                f = 0 if np.isnan(f) else f
            else:
                f[np.isnan(f)] = 0
                f = f.reshape(r.shape)
        return Quantity(f, mass() / (length() ** 3 * velocity() ** 3))

    @property
    def Mtot(self) -> Quantity['mass']:
        """Calculate the total mass."""
        return Quantity(self.df.totalMass(), mass())

    def to_model(self) -> 'GalaxyModel':
        """Convert the distribution function to an agama model."""
        return GalaxyModel(potential=self.potential, df=self.df)

    def sample(self, n: int | float) -> tuple[Quantity['length'], Quantity['velocity'], Quantity['mass']]:
        """Sample particles from the model."""
        return self.to_model().sample(n=n)


class GalaxyModel:
    """Wrapper for the agama GalaxyModel class"""

    def __init__(self, potential: Any, df: Any, *args: Any, **kwargs: Any) -> None:
        if isinstance(potential, Potential):
            potential = potential.potential
        if isinstance(df, Potential):
            df = df.potential
        self.model = agama.GalaxyModel(potential=potential, df=df, *args, **kwargs)

    def sample(self, n: int | float) -> tuple[Quantity['length'], Quantity['velocity'], Quantity['mass']]:
        """Sample particles from the model."""
        posvel, m = self.model.sample(n=int(n))

        x, y, z, vx, vy, vz = posvel.T
        x, y, z = Quantity(np.vstack([x, y, z]), length())
        vx, vy, vz = Quantity(np.vstack([vx, vy, vz]), velocity())
        m = Quantity(m, mass())

        r = cast(Quantity, np.sqrt(x**2 + y**2 + z**2))
        vr = cast(Quantity, (x * vx + y * vy + z * vz) / r)
        v = cast(Quantity, np.vstack([*utils.split_2d(np.sqrt(vx**2 + vy**2 + vz**2 - vr**2), acos=False), vr]).T)
        return r, v, m
