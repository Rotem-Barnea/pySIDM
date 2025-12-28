import pickle
from typing import Any
from pathlib import Path

from . import agama_wrappers
from .distribution import Distribution


def save_agama_potential(path: str | Path, stem: str, potential: Any):
    """Save an Agama potential to a file."""
    potential.export(str(Path(path) / f'{stem}.ini'))


def load_agama_potential(path: str | Path, stem: str) -> agama_wrappers.Potential:
    """Load an Agama potential from a file."""
    return agama_wrappers.Potential(str(Path(path) / f'{stem}.ini'))


def save_distribution(path: str | Path, stem: str, distribution: Distribution):
    """Save a distribution to a file."""
    agama_potential = distribution.agama_potential
    if agama_potential is not None:
        save_agama_potential(path, f'{stem}_potential', agama_potential.potential)
        distribution.agama_potential = None
    agama_total_potential = distribution.agama_total_potential
    if agama_total_potential is not None:
        save_agama_potential(path, f'{stem}_total_potential', agama_total_potential.potential)
        distribution.agama_total_potential = None
    with open(Path(path) / f'{stem}.pkl', 'wb') as f:
        pickle.dump(distribution, f)
    distribution.agama_potential = agama_potential
    distribution.agama_total_potential = agama_total_potential


def load_distribution(path: str | Path, stem: str):
    """Save a distribution to a file."""
    with open(Path(path) / f'{stem}.pkl', 'rb') as f:
        distribution = pickle.load(f)
    if (Path(path) / f'{stem}_potential.ini').exists():
        distribution.agama_potential = load_agama_potential(path, f'{stem}_potential')
    if (Path(path) / f'{stem}_total_potential.ini').exists():
        distribution.agama_total_potential = load_agama_potential(path, f'{stem}_total_potential')
    return distribution
