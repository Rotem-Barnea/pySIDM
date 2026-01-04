import pickle
from typing import Any
from pathlib import Path

import agama

from . import agama_wrappers
from .distribution import Distribution


def save_agama_potential(
    path: str | Path,
    stem: str,
    potential: agama_wrappers.Potential,
    type: str = 'Multipole',
    gridSizeR: int = 100,
    rmax: int = 100,
    **kwargs: Any,
) -> None:
    """Save an Agama potential to a file. The potential is repackaged as a potential with the provided arguments, to avoid resolution degradation in the base agama export method."""
    agama.Potential(type=type, density=potential.potential, gridSizeR=gridSizeR, rmax=rmax, **kwargs).export(
        str(Path(path) / f'{stem}.ini')
    )


def load_agama_potential(path: str | Path, stem: str) -> agama_wrappers.Potential | None:
    """Load an Agama potential from a file."""
    if (Path(path) / f'{stem}.ini').exists():
        return agama_wrappers.Potential(str(Path(path) / f'{stem}.ini'))
    elif (Path(path) / f'{stem}.pkl').exists():
        with open(Path(path) / f'{stem}.pkl', 'rb') as f:
            params = pickle.load(f)
        return agama_wrappers.Potential(*params['args'], **params['kwargs'])
    return None


def save_distribution(path: str | Path, stem: str, distribution: Distribution, **kwargs: Any):
    """Save a distribution to a file."""
    agama_potential = distribution.agama_potential
    if agama_potential is not None:
        save_agama_potential(path, f'{stem}_potential', agama_potential, **kwargs)
        distribution.agama_potential = None
    agama_total_potential = distribution.agama_total_potential
    if agama_total_potential is not None:
        save_agama_potential(path, f'{stem}_total_potential', agama_total_potential, **kwargs)
        distribution.agama_total_potential = None
    with open(Path(path) / f'{stem}.pkl', 'wb') as f:
        pickle.dump(distribution, f)
    distribution.agama_potential = agama_potential
    distribution.agama_total_potential = agama_total_potential


def load_distribution(path: str | Path, stem: str):
    """Save a distribution to a file."""
    with open(Path(path) / f'{stem}.pkl', 'rb') as f:
        distribution = pickle.load(f)
    distribution.agama_potential = load_agama_potential(path, f'{stem}_potential')
    distribution.agama_total_potential = load_agama_potential(path, f'{stem}_total_potential')
    return distribution
