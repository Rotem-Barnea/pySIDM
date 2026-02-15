"""Utility functions for data cleaning"""

from typing import Any, Literal, cast

import numpy as np
import scipy
import pandas as pd
from astropy import table
from numpy.typing import NDArray
from astropy.units import Quantity

from src import rng, types


def default(value: Any, default: Any) -> Any:
    """Handles setting the default value, avoiding pointer issues with Python by allowing the default function argument to be `None`"""
    if value is None:
        return default
    return value


def to_join(
    arrays: list[NDArray[Any]],
    keys: list[str] | None = None,
    clean_by: str | int = 0,
) -> NDArray[Any]:
    """Clean a list of arrays by removing duplicates and sorting them by a given key.

    Parameters:
        arrays: The arrays to clean.
        keys: Names for each array, to be used with `clean_by`. If `None` defaults to "column_{j}".
        clean_by: The column to sort and drop duplicates by. If a string, must match `keys`. If `int` must be smaller than the number of columns, and the value will be treated as the selected index. Defaults to 0 (the first column).

    Returns:
        The cleaned arrays.
    """
    if keys is None:
        keys = [f'column_{i}' for i in range(len(arrays))]
    data = pd.DataFrame(dict(zip(keys, arrays)))
    if isinstance(clean_by, int):
        clean_by = data.columns[clean_by]
    data = data.drop_duplicates(clean_by).sort_values(clean_by)
    return data.to_numpy().T


def pairs(
    pairs: NDArray[np.int64],
    blacklist: list[int] | NDArray[np.int64] = [],
    shuffle: bool = False,
    generator: np.random.Generator | None = None,
) -> NDArray[np.int64]:
    """Clean a list of pairs by removing duplicates.

    Ensures no particle is considered multiple times.

    Parameters:
        pairs: The raw array of pairs, of shape (n_pairs, 2).
        blacklist: List of blacklisted particles. If provided (and not empty), any pair containing a blacklisted particle is removed *after* any other filtering (which might lead to over-dropping).
        shuffle: Whether to shuffle the raw pairs before cleaning, to avoid biasing on smaller indices (and thus smaller radii). The shuffle is performed in-place without prior copy (will affect the input `pairs`).

    Returns:
        The cleaned array of pairs, of shape (n_cleaned_pairs, 2).
    """
    if generator is None:
        generator = rng.generator
    if shuffle:
        generator.shuffle(pairs)
    _, indices = np.unique(pairs.ravel(), return_index=True)
    first_occurrence = np.zeros(2 * len(pairs), dtype=np.bool_)
    first_occurrence[indices] = True
    first_occurrence = first_occurrence.reshape(pairs.shape)
    cleaned_pairs = pairs[first_occurrence.all(axis=1)]
    if len(blacklist) > 0:
        cleaned_pairs = np.array(
            [pair for pair in cleaned_pairs if pair[0] not in blacklist and pair[1] not in blacklist]
        )
    return cleaned_pairs


def smooth_holes_1d(
    x: types.QuantityLike,
    y: types.QuantityLike,
    mask: NDArray[np.bool_] | None = None,
    include_zero: bool = False,
    assume_sorted: bool = False,
    bounds_error: bool = False,
    fill_value: Literal['extrapolate']
    | float
    | tuple[float, float]
    | NDArray[np.float64]
    | tuple[NDArray[np.float64], NDArray[np.float64]] = 'extrapolate',
    **kwargs: Any,
) -> types.QuantityLike:
    """Smooths holes in a 1D array, defined by the provided mask.

    Smoothing is done by interpolating the values around the holes.

    Parameters:
        x: The x values used for the interpolation.
        y: The y values used for the interpolation.
        mask: The mask indicating the holes to be smoothed. If `None` treat all negative values as holes.
        include_zero: Only relevant if `mask` is not provided. Define "hole" as any `y<=0`, otherwise only fill `y<0`.
        assume_sorted: Whether the x values are sorted.
        bounds_error: Whether to raise an error if the x values are out of bounds.
        fill_value: The value to use for extrapolation. Must be accepted by `scipy.interpolate.interp1d()`.
        kwargs: Additional keyword arguments to pass to the interpolation function.

    Returns:
        The smoothed y values.
    """
    if mask is None:
        if include_zero:
            mask = np.array(y) <= 0
        else:
            mask = np.array(y) < 0
    smoothed = np.array(y).copy()
    smoothed[mask] = scipy.interpolate.interp1d(
        x=np.array(x[~mask]),
        y=np.array(y[~mask]),
        assume_sorted=assume_sorted,
        bounds_error=bounds_error,
        fill_value=fill_value,
        **kwargs,
    )(np.array(x[mask]))
    if isinstance(y, Quantity):
        return Quantity(smoothed, y.unit)
    return cast(types.QuantityLike, smoothed)


def smooth_holes_2d(
    data: types.QuantityLike, mask: NDArray[np.bool_] | None = None, include_zero: bool = False, **kwargs: Any
) -> types.QuantityLike:
    """Smooths holes in a 2D array, defined by the provided mask.

    Smoothing is done by interpolating the values around the holes.

    Parameters:
        data: The data to be smoothed.
        mask: The mask indicating the holes to be smoothed. If `None` treat all negative values as holes.
        include_zero: Only relevant if `mask` is not provided. Define "hole" as any `data<=0`, otherwise only fill `data<0`.
        kwargs: Additional keyword arguments to pass to the interpolation function.

    Returns:
        The smoothed data values.
    """
    if mask is None:
        if include_zero:
            mask = np.array(data) <= 0
        else:
            mask = np.array(data) < 0
    smoothed = np.array(data).copy()
    y, x = np.indices(data.shape)
    smoothed[mask] = scipy.interpolate.griddata(
        points=(x[~mask], y[~mask]),
        values=data[~mask],
        xi=(x[mask], y[mask]),
        **kwargs,
    )
    if isinstance(data, Quantity):
        return Quantity(smoothed, data.unit)
    return cast(types.QuantityLike, smoothed)


def filter_indices(
    data: table.QTable | pd.DataFrame,
    indices: list[int] | NDArray[np.int64],
    copy: bool = True,
) -> table.QTable:
    """Filter the data to only keep the specified indices.

    Parameters:
        data: The data to filter.
        indices: The indices to filter by.
        copy: Whether to return a copy of the sliced data.

    Returns:
        The filtered data.
    """
    mask = pd.Series(False, index=np.array(data['particle_index']))
    mask.loc[mask.index.isin(indices)] = True
    output = cast(table.QTable, data[np.array(mask)])
    if copy:
        return output.copy()
    return output
