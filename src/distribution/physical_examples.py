from typing import Any, Literal, cast, get_args

import regex
from astropy.units import Quantity

from .nfw import NFW
from .hernquist import Hernquist
from .distribution import Distribution

physical_examples = Literal['Sague-1', 'Draco', 'Fornax', 'default']
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
        distributions += [NFW.from_example(name, Rmin=Rmin, Rmax=Rmax, **dm_kwargs, **kwargs)]
    if suffix == 'b_only' or suffix is None:
        distributions += [Hernquist.from_example(name, Rmin=Rmin, Rmax=Rmax, **b_kwargs, **kwargs)]
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
