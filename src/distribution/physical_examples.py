from typing import Any, Literal

from astropy.units import Quantity

from .nfw import NFW
from .hernquist import Hernquist
from .distribution import Distribution


def by_name(
    name: Literal['Sague-1', 'Draco', 'Fornax', 'default'] = 'default',
    Rmin: Quantity['length'] = Quantity(1e-5, 'kpc'),
    Rmax: Quantity['length'] = Quantity(300, 'kpc'),
    dm_kwargs: dict[str, Any] = {},
    b_kwargs: dict[str, Any] = {},
    **kwargs: Any,
) -> list[Distribution]:
    """Return a predefined set of distributions for a physical example (mimicking a real galaxy).

    Parameters:
        name: Name of the physical example.
        Rmin: Minimum radius of the distribution. Set for all distributions to match internal grids.
        Rmax: Maximum radius of the distribution. Set for all distributions to match internal grids.
        dm_kwargs: Keyword arguments for the dark matter distribution.
        b_kwargs: Keyword arguments for the baryonic distribution.
        **kwargs: Additional keyword arguments passed to both distributions.

    Returns:
        A list of distributions representing the physical example.
    """
    dm_distribution = NFW.from_example(name, Rmin=Rmin, Rmax=Rmax, **dm_kwargs, **kwargs)
    b_distribution = Hernquist.from_example(name, Rmin=Rmin, Rmax=Rmax, **b_kwargs, **kwargs)
    distributions = [dm_distribution, b_distribution]
    Distribution.merge_distribution_grids(distributions)
    return distributions
