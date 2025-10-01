import numpy as np
import scipy
from typing import Literal, cast
from astropy import constants
from astropy.units import Quantity, Unit
from ..types import QuantityOrArray
from .. import utils, run_units

Mass_calculation_methods = Literal['rank presorted', 'rank unsorted']


def M(
    r: Quantity['length'],
    m: Quantity['mass'] | None = None,
    count_self: bool = True,
    M_below: Quantity['mass'] = Quantity(0, run_units.mass),
    method: Mass_calculation_methods = 'rank unsorted',
) -> Quantity['mass']:
    masses = m if m is not None else Quantity([1] * len(r), run_units.mass)
    if method == 'rank unsorted':
        masses = masses[utils.rank_array(r)]
    M = masses.cumsum()
    if not count_self:
        M -= masses
    M += M_below
    return cast(Quantity['mass'], M)


def local_density(
    r: QuantityOrArray,
    m: QuantityOrArray,
    max_radius_j: int = 10,
    volume_kind: Literal['density', 'shell'] = 'shell',
    mass_kind: Literal['sum', 'single'] = 'single',
) -> QuantityOrArray:
    """Assumes the array is sorted"""
    x = np.array(r)
    x_end = np.zeros_like(x)
    x_end = np.pad(x, (0, max_radius_j), mode='edge')[max_radius_j:]

    if mass_kind == 'sum':
        y = np.pad(np.array(m), (0, max_radius_j), mode='edge').cumsum()
        mass = np.zeros_like(x)
        mass = y[max_radius_j:] - y[:-max_radius_j]
    else:
        mass = np.array(m)

    volume = np.zeros_like(x)
    if volume_kind == 'density':
        volume = 4 / 3 * np.pi * (x_end**3 - x**3)
    else:
        volume = 4 * np.pi * x**2 * (x_end - x)
    density = mass[:-1] / volume[:-1]
    density = np.hstack([density, density[-1]])
    if isinstance(m, Quantity) and isinstance(r, Quantity):
        return Quantity(density, m.unit / cast(Unit, r.unit) ** 3)
    return density


def Phi(r: QuantityOrArray, M: QuantityOrArray, m: QuantityOrArray) -> QuantityOrArray:
    integral = scipy.integrate.cumulative_trapezoid(y=constants.G.to(run_units.G_units).value * M * m / r**2, x=r, initial=0)
    if isinstance(r, Quantity):
        return Quantity(integral, run_units.energy)
    return integral


def Psi(r: QuantityOrArray, M: QuantityOrArray, m: QuantityOrArray) -> QuantityOrArray:
    integral = scipy.integrate.cumulative_trapezoid(y=constants.G.to(run_units.G_units).value * M * m / r**2, x=r, initial=0)
    integral = integral[-1] - integral
    if isinstance(r, Quantity):
        return Quantity(integral, run_units.energy)
    return integral
