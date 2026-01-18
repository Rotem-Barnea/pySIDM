"""Managing setting up physical (real world) examples of galactic halos"""

from typing import Any, Literal, cast, get_args
from functools import partial

import numpy as np
import regex
import scipy
from astropy.units import Quantity
from astropy.units.typing import UnitLike

from src import report

from .nfw import NFW
from .hernquist import Hernquist
from .distribution import Distribution

physical_examples = Literal['Sague-1', 'Draco', 'Fornax dSph', 'default', 'Daneng2024:DM11+baryon']
distribution_options = Literal['dm_only', 'b_only', None]


def by_name(
    name: physical_examples = 'default',
    suffix: distribution_options = None,
    Rmin: Quantity['length'] = Quantity(1e-5, 'kpc'),
    Rmax: Quantity['length'] = Quantity(300, 'kpc'),
    dm_kwargs: dict[str, Any] = {},
    b_kwargs: dict[str, Any] = {},
    verbose: bool = False,
    **kwargs: Any,
) -> list[Distribution]:
    """Return a predefined set of distributions for a physical example (mimicking a real galaxy).

    Parameters:
        name: Name of the physical example.
        Rmin: Minimum radius of the distribution. Set for all distributions to match internal grids.
        Rmax: Maximum radius of the distribution. Set for all distributions to match internal grids.
        dm_kwargs: Keyword arguments for the dark matter distribution.
        b_kwargs: Keyword arguments for the baryonic distribution.
        verbose: Whether to print information about the distributions.
        **kwargs: Additional keyword arguments passed to both distributions.

    Returns:
        A list of distributions representing the physical example.
    """
    if verbose:
        print('Setup distributions')
        print('running example', name)
    distributions: list[Distribution] = []
    if suffix == 'dm_only' or suffix is None:
        distributions += [NFW.from_example(name, Rmin=Rmin, Rmax=Rmax, particle_type='dm', **dm_kwargs, **kwargs)]
    if suffix == 'b_only' or suffix is None:
        distributions += [
            Hernquist.from_example(name, Rmin=Rmin, Rmax=Rmax, particle_type='baryon', **b_kwargs, **kwargs)
        ]
    Distribution.merge_distributions(distributions)
    return distributions


def validate_input(name: str) -> tuple[physical_examples, distribution_options]:
    """Validate that the given name is a known physical example."""
    suffix = None
    for option in filter(lambda x: x, get_args(distribution_options)):
        if name.endswith(option):
            name = regex.sub(rf'_{option}$', '', name)
            suffix = option
    assert name in get_args(physical_examples), f'Unknown physical example: {name}'
    return cast(physical_examples, name), suffix


def calculate_J_integral(
    a: float = 1e-3,
    b: float = 1,
    rho0: float | None = None,
    r0: float | None = None,
    nfw_dist: NFW | None = None,
    hernquist_dist: Hernquist | None = None,
) -> tuple[float, float]:
    """Calculate the J integral for an NFW-Hernquist mixed system. `rho0=0` is equivilent to an NFW-only distribution.

    If not provided, `rho0` defaults to 0 and `r0` defaults to 1. If `nfw_dist` and `hernquist_dist` are provided, calculate `rho0` and `r0` from them instead.
    """

    def integrand(x: float, rho0: float, r0: float) -> float:
        part1 = (x / (rho0 * r0**3 * (x / r0) ** 2 / (2 * (1 + x / r0) ** 2) + np.log(1 + x) - x / (1 + x))) ** (5 / 2)
        part2 = rho0 * r0**2 * x / (2 * (1 + x / r0)) + np.log(1 + x)
        return part1 * part2

    if rho0 is None and r0 is None and nfw_dist is not None and hernquist_dist is not None:
        rho0 = (hernquist_dist.rho_s / nfw_dist.rho_s).to('').value
        r0 = (hernquist_dist.Rs / nfw_dist.Rs).to('').value

    return scipy.integrate.quad(
        func=partial(integrand, rho0=rho0 or 0, r0=r0 or 1),
        a=a,
        b=b,
    )


def describe_distributions(
    distributions: tuple['Distribution', 'Distribution'] | list['Distribution'],
    length_unit: UnitLike = 'kpc',
    mass_unit: UnitLike = 'Msun',
    density_unit: UnitLike = 'Msun/kpc^3',
) -> report.Report:
    """Print a description of a pair distribution."""
    assert len(distributions) == 2, 'Must be 2 distributions'
    if distributions[0].particle_type != 'dm' and distributions[1].particle_type == 'dm':
        distributions = (distributions[1], distributions[0])
    dm, baryon = distributions
    if dm.name != '' and baryon.name != '':
        name = f'DM={dm.name} / Stellar={baryon.name}' if dm.name != baryon.name else dm.name
    elif dm.name != '':
        name = dm.name
    elif baryon.name != '':
        name = baryon.name
    else:
        name = ''

    return report.Report(
        body_lines=[
            report.Line(title='DM Rs', value=dm.Rs, unit=length_unit, format='.3f'),
            report.Line(title='DM Mtot', value=dm.Mtot, unit=mass_unit, format='.2e'),
            report.Line(title='DM rho0', value=dm.rho_s, unit=density_unit, format='.2e'),
            report.Line(title='Baryon Rs', value=baryon.Rs, unit=length_unit, format='.3f'),
            report.Line(title='Baryon Mtot', value=baryon.Mtot, unit=mass_unit, format='.2e'),
            report.Line(title='Baryon rho0', value=baryon.rho_s, unit=density_unit, format='.2e'),
            report.Line(title='Tilde Rs', value=baryon.Rs / dm.Rs, format='.2f'),
            report.Line(title='Tilde rho0', value=baryon.rho_s / dm.rho_s, format='.2f'),
        ],
        header=f'Description for {name}',
        body_prefix='  - ',
    )
