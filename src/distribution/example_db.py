from typing import TYPE_CHECKING, Any, TypedDict
from pathlib import Path

import pandas as pd
from astropy.units import Quantity

if TYPE_CHECKING:
    from .physical_examples import physical_examples


class ExampleParameters(TypedDict):
    """Parameters of the example galaxy, loaded from the galaxy db table."""

    mass_stellar: Quantity['mass']
    mass_half_light: Quantity['mass']
    r_half_light: Quantity['length']


def load_db(path: str | Path | None = None):
    """Loads the galaxy db table. Defaults to the local volume database csv file.

    See `https://local-volume-database.readthedocs.io/en/latest/index.html` for reference.
    """
    if path is None:
        path = Path(__file__).parent / 'local_volume_database' / 'comb_all.csv'
    return pd.read_csv(path, index_col='key')


def get_db_parameters(name: 'physical_examples', **kwargs: Any) -> ExampleParameters:
    """Return the parameters of the given physical example from the database. Additional keyword arguments are passed to the loading function `load_db()`."""
    db = load_db(**kwargs)
    if name == 'Sague-1':
        raise ValueError('Sague-1 is not available in the database')
    elif name == 'Draco':
        key = 'draco_1'
    elif name == 'Fornax dSph':
        key = 'fornax_1'
    else:
        raise ValueError(f'Unknown physical example in db: {name}')
    entry = db.loc[key]
    return ExampleParameters(
        mass_stellar=Quantity(10 ** entry['mass_stellar'], 'Msun'),
        mass_half_light=Quantity(10 ** entry['mass_dynamical_wolf'], 'Msun'),
        r_half_light=Quantity(entry['rhalf_physical'], 'pc'),
    )
