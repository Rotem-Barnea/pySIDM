from typing import Literal, cast

import numpy as np
import scipy
from astropy import constants
from astropy.units import Unit, Quantity

from .. import run_units
from ..types import QuantityOrArray


def M(
    r: Quantity['length'],
    m: Quantity['mass'] | None = None,
    count_self: bool = False,
    M_below: Quantity['mass'] | None = None,
) -> Quantity['mass']:
    """Calculate the cumulative mass of a set of particles.

    Assumes the array is sorted.

    Parameters
        r: The positions of the particles.
        m: The masses of the particles. If `None` all particles are assumed to have a mass of 1 `Msun` (`run_units.mass`).
        count_self: Whether to include the mass of the particle at the current position in the cumulative mass.
        M_below: Additional mass below the current position from an external source.

    Returns
        The cumulative mass of the particles.
    """
    masses = m if m is not None else Quantity([1] * len(r), run_units.mass)
    M = masses.cumsum()
    if not count_self:
        M -= masses
    if M_below is not None:
        M += M_below
    return cast(Quantity, M)


def local_density(
    r: QuantityOrArray,
    m: QuantityOrArray,
    max_radius_j: int = 10,
    volume_include_final_unit_cell: bool = True,
    volume_kind: Literal['density', 'shell'] = 'shell',
    mass_kind: Literal['sum', 'single'] = 'single',
) -> QuantityOrArray:
    """Calculate the local density of a set of particles.

    Supports generally 2 modes:
        3d mass density (`sum(m)/(4/3*pi*(rmax^3-rmin^3))`) = `volume_kind`='density' and `mass_kind`='sum'
        scattering density term (`m/(4*pi*rmin^2*(rmax-rmin))`) = `volume_kind`='shell' and `mass_kind`='single'

    Assumes the array is sorted.

    Parameters
        r: The positions of the particles.
        m: The masses of the particles. If `None` all particles are assumed to have a mass of 1 `Msun` (`run_units.mass`).
        max_radius_j: Maximum index radius for partners for scattering.
        volume_include_final_unit_cell: If `True` counts the volume from each particle to it's `max_radius_j+1` neighbor, effectively including the unit cell of the final particle as well (without including the mass at the end of that unit cell).
        volume_kind: The kind of volume to calculate (thick shell or approximation using thin-shell).
        mass_kind: Either calculate the total mass enclosed, or count just a single mass (used for the scattering term).

    Returns
        The local density of the particles.
    """
    x = np.array(r)
    x_end = np.pad(x, (0, max_radius_j + volume_include_final_unit_cell), mode='edge')[
        max_radius_j + volume_include_final_unit_cell :
    ]

    if mass_kind == 'sum':
        y = np.pad(np.array(m), (0, max_radius_j), mode='edge').cumsum()
        mass = y[max_radius_j:] - y[:-max_radius_j]
    else:
        mass = np.array(m)

    if volume_kind == 'density':
        volume = 4 / 3 * np.pi * (x_end**3 - x**3)
    else:
        volume = 4 * np.pi * x**2 * (x_end - x)
    density = mass[:-1] / volume[:-1]
    # The final element has volume=0, this just circumvents this and uses the value from the second-to-last
    if (volume[:-1] == 0).any():
        raise ValueError('Volume cannot be zero')  # TODO: Handle this case more gracefully
    density = np.hstack([density, density[-1]])
    if isinstance(m, Quantity) and isinstance(r, Quantity):
        return Quantity(density, m.unit / cast(Unit, r.unit) ** 3)
    return density


def Phi(r: QuantityOrArray, M: QuantityOrArray, m: QuantityOrArray) -> QuantityOrArray:
    """Calculate the gravitational potential at a given radius.

    Performed using integration of the gravitational force `G*M(<=r)*m/r^2`.

    Parameters
        r: The position of the particles.
        M: The total enclosed mass (`M(<=r)`) at any particle position.
        m: The mass of each particle.

    Returns
        The gravitational potential at the given radius.
    """
    integral = scipy.integrate.cumulative_trapezoid(
        y=constants.G.to(run_units.G_unit).value * M * m / r**2, x=r, initial=0
    )
    if isinstance(r, Quantity):
        return Quantity(integral, run_units.energy)
    return integral


def Psi(r: QuantityOrArray, M: QuantityOrArray, m: QuantityOrArray) -> QuantityOrArray:
    """Calculate the relative gravitational potential at a given radius.

    Recalculates `Phi0` (the value at infinity) based on the maximal value for the given particle array.

    See `Phi()` for details.
    """
    integral = scipy.integrate.cumulative_trapezoid(
        y=constants.G.to(run_units.G_unit).value * M * m / r**2, x=r, initial=0
    )
    integral = integral[-1] - integral
    if isinstance(r, Quantity):
        return Quantity(integral, run_units.energy)
    return integral
