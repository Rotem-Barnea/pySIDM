"""Utility functions related to unit handling"""

import re
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from astropy.units import Unit, Quantity
from astropy.units.typing import UnitLike

from src import units


def add_to_label(label: str | None, plot_unit: UnitLike | None = None) -> str | None:
    """Add the units to the `label` in a LaTeX formatted string and enclosed in brackets. Ignore if label is `None` or '' (unit-less)."""
    if label is None:
        return None
    if plot_unit is None or plot_unit == '':
        return label
    string_unit = f'{Unit(cast(str, plot_unit)):latex}'
    return rf'{label} $\left[{string_unit.strip("$")}\right]$'


def update_label(label: str | None, plot_unit: UnitLike | None = None) -> str | None:
    """Replace the label unit with another unit."""
    if label is None:
        return None
    return add_to_label(label=re.sub(r' \$\\left\[.*\\right\]\$$', '', label), plot_unit=plot_unit)


def strip_args(*args: Any) -> list[Any]:
    """Strip units from positional arguments if they are quantities. Also decompose them to the `run_unit` system."""
    out_args = []
    for arg in args:
        out_args += [arg.decompose(units.system).value if isinstance(arg, Quantity) else arg]
    return out_args


def strip_kwargs(**kwargs: Any) -> dict[str, Any]:
    """Strip units from keyword arguments if they are quantities. Also decompose them to the `run_unit` system."""
    out_kwargs = {}
    for key, value in kwargs.items():
        out_kwargs[key] = value.decompose(units.system).value if isinstance(value, Quantity) else value
    return kwargs


def guess(unit: UnitLike | None = None, array: NDArray[np.float64] | Quantity | None = None) -> UnitLike:
    """Pull the desired unit from the array if not provided."""
    if unit is None:
        if isinstance(array, Quantity):
            unit = cast(Unit, array.unit)
        else:
            unit = ''
    return unit
