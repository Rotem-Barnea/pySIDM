import numpy as np
from typing import Literal, cast
from astropy.units import Quantity, Unit
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


def local_density(r: Quantity['length'], m: Quantity['mass'], max_radius_j: int = 10) -> Quantity['mass density']:
    """Assumes the array is sorted"""
    x = r.value
    x_end = np.zeros_like(x)
    x_end[:-max_radius_j] = x[max_radius_j:]
    x_end[-max_radius_j:] = x[-1]
    n = np.full(len(x), max_radius_j, dtype=np.int64)
    n[-max_radius_j:] = np.arange(max_radius_j - 1, -1, -1)

    y = m.value.cumsum()
    mass = np.zeros_like(y)
    mass[:-max_radius_j] = y[max_radius_j:]
    mass[-max_radius_j:] = y[-1]

    volume = np.zeros_like(x)
    volume = 4 / 3 * np.pi * (x_end**3 - x**3)
    density = mass[:-1] / volume[:-1]
    density = np.hstack([density, density[-1]])
    return Quantity(density, m.unit / cast(Unit, r.unit) ** 3)
