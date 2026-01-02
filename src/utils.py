import datetime
from typing import Any, Literal, Callable, cast

import numpy as np
import scipy
import pandas as pd
from numba import njit, prange
from astropy import table
from numpy.typing import NDArray, ArrayLike
from astropy.units import Unit, Quantity
from astropy.units.typing import UnitLike

from . import rng
from .types import FloatOrArray, QuantityOrArray


def random_angle(
    like: NDArray[np.float64] | int,
    acos: bool,
    generator: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Generate an array of random angles.

    Parameters:
        like: Array who's shape to mimic. if `int` treat it as the length of the array.
        acos: If `False` generate a uniform random angle. If `True` generate a uniform random `cos(angle)`, and then applies arccos to retrieve the angle.
        generator: If `None` use the default generator from `rng.generator`.

    Returns:
        Array of random angles.
    """
    if generator is None:
        generator = rng.generator
    if isinstance(like, int):
        rolls = generator.random(like)
    elif len(like.shape) == 1:
        rolls = generator.random(len(like))
    else:
        rolls = generator.random(*like.shape)
    if acos:
        return np.acos(rolls * 2 - 1)
    return rolls * 2 * np.pi


def from_radial(
    r: NDArray[np.float64], theta: NDArray[np.float64], quick_sin: bool = True
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert radial coordinates to Cartesian coordinates.

    Parameters:
        r: Radius.
        theta: Angle.
        quick_sin: If `True` use a faster `sin` calculation. Otherwise calculate `sin(theta)` explicitly.

    Returns:
        x,y coordinates.
    """
    cos: NDArray[np.float64] = np.cos(theta)
    sin: NDArray[np.float64] = np.sqrt(1 - cos**2) * np.sign(np.pi - theta) if quick_sin else np.sin(theta)
    return r * cos, r * sin


def split_2d(
    r: NDArray[np.float64],
    acos: bool,
    generator: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split an array of radiuses to x,y coordinates using a random angle. See `random_angle()` for details on the angle calculation."""
    return from_radial(r, theta=random_angle(r, acos, generator=generator))


def split_3d(
    r: NDArray[np.float64],
    generator: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Split an array of radiuses to `x`,`y`,`z` coordinates using a `random acos angle` for the `z` coordinate (i.e. radial in the halo), and a `random uniform angle` for the `x-y` plane (i.e. tangential plane in the halo)."""
    radial, perp = from_radial(r, theta=random_angle(r, acos=True, generator=generator))
    x, y = from_radial(perp, theta=random_angle(perp, acos=False, generator=generator))
    return x, y, radial


def split_3d_quantity(x: Quantity, generator: np.random.Generator | None = None) -> Quantity:
    """Wrapper for `split_3d` that handles quantities."""
    return cast(Quantity, np.vstack(split_3d(x, generator=generator)).T)


def joint_clean(
    arrays: list[NDArray[Any]],
    keys: list[str] | None = None,
    clean_by: str | int = 0,
) -> NDArray[Any]:
    """Clean a list of arrays by removing duplicates and sorting them by a given key.

    Parameters:
        arrays: The arrays to clean.
        keys: Names for each array, to be used with `clean_by`. If `None` defaults to "column_{i}".
        clean_by: The column to sort and drop duplicates by. If 'str' must match `keys`. If `int` must be smaller than the number of columns, and the value will be treated as the selected index. Defaults to 0 (the first column).

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


def clean_pairs(
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


def drop_None(**kwargs: Any) -> dict[Any, Any]:
    """Remove key-value pairs where the value is `None`."""
    return {key: value for key, value in kwargs.items() if value is not None}


def rank_array(r: NDArray[Any]) -> NDArray[np.int64]:
    """Calculate the rank of every element in the array."""
    return r.argsort().argsort()


def derivate(x: FloatOrArray, y_fn: Callable[[FloatOrArray], FloatOrArray], h: float = 1e-4) -> FloatOrArray:
    """Calculate the derivative of a function at a point.

    Calculates a forward numerical derivative: `(y_fn(x + h) - y_fn(x)) / h`

    Parameters:
        x: The point/points at which to calculate the derivative.
        y_fn: The function to differentiate.
        h: The step size for numerical differentiation.

    Returns:
        The derivative of the function at the given point/points.
    """
    return (y_fn(x + h) - y_fn(x)) / h


def derivate2(x: FloatOrArray, y_fn: Callable[[FloatOrArray], FloatOrArray], h: float = 1e-4) -> FloatOrArray:
    """Calculate the second order derivative of a function at a point.

    Calculates a forward numerical derivative: `(y_fn(x + 2 * h) - 2 * y_fn(x + h) + y_fn(x)) / h**2`

    Parameters:
        x: The point/points at which to calculate the derivative.
        y_fn: The function to differentiate.
        h: The step size for numerical differentiation.

    Returns:
        The derivative of the function at the given point/points.
    """
    return (y_fn(x + 2 * h) - 2 * y_fn(x + h) + y_fn(x)) / h**2


def quantity_derivate(x: Quantity, y_fn: Callable[[Quantity], Quantity], h: float = 1e-4) -> Quantity:
    """Calculate the derivative of a function at a point. Wrapper for `derivate()` to handle and output `Quantity` objects."""
    t = Quantity(h, x.unit)
    return cast(Quantity, (y_fn(cast(Quantity, x + t)) - y_fn(x)) / t)


def quantity_derivate2(x: Quantity, y_fn: Callable[[Quantity], Quantity], h: float = 1e-4) -> Quantity:
    """Calculate the second order derivative of a function at a point. Wrapper for `derivate2()` to handle and output `Quantity` objects."""
    t = Quantity(h, x.unit)
    return cast(Quantity, (y_fn(cast(Quantity, x + 2 * t)) - 2 * y_fn(cast(Quantity, x + t)) + y_fn(x)) / t**2)


@njit
def linear_interpolation(xs: NDArray[np.float64], ys: NDArray[np.float64], x: float) -> NDArray[np.float64]:
    """Calculate the linear interpolation from a grid (`xs`, `ys`) at a point `x`. njit compliant.

    `xs` must be sorted in ascending order (relies on `np.searchsorted()`).
    """
    i = np.searchsorted(xs, x) - 1
    if i < 0:
        i = 0
    elif i >= len(xs) - 1:
        i = len(xs) - 2
    w = (x - xs[i]) / (xs[i + 1] - xs[i])
    return (1 - w) * ys[i] + w * ys[i + 1]


@njit(parallel=True)
def fast_assign(indices: NDArray[np.int64], array: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fast assignment of `array` elements using `indices`. njit accelerated."""
    output = np.empty_like(indices, dtype=np.float64)
    for i in prange(len(indices)):
        output[i] = array[indices[i]]
    return output


@njit(parallel=True)
def fast_spherical_rho_integrate(
    r: NDArray[np.float64],
    rho_fn: Callable[..., NDArray[np.float64]],
    rho_s: float = 1,
    Rs: float = 1,
    Rvir: float = 1,
    start: float = 0,
    num_steps: int = 10000,
) -> NDArray[np.float64]:
    """Integrate the density function (`rho`) assuming spherical symmetry. njit accelerated.

    Parameters:
        r: The radius points at which to calculate the density.
        rho_fn: The density function to integrate. must be njit compliant.
        rho_s: The scale density.
        Rs: The scale radius.
        Rvir: The virial radius.
        start: The starting radius for the integration.
        num_steps: The number of radius steps for the integration (from start to `r`).

    Returns:
        The enclosed mass at the given radius (integral of the density).
    """
    integral = np.empty_like(r, dtype=np.float64)
    for i in prange(len(r)):
        x_grid = np.linspace(start, r[i], num_steps)[1:]
        x = np.empty_like(x_grid, dtype=np.float64)
        x[:] = x_grid
        J = 4 * np.pi * x**2
        ys = rho_fn(x, rho_s=rho_s, Rs=Rs, Rvir=Rvir)
        integral[i] = np.trapezoid(y=ys * J, x=x)
    return integral


@njit(parallel=True)
def fast_unique_mask(x: NDArray[np.int64]) -> NDArray[np.int64]:
    """Calculate the number of occurrences of each element in the array. njit accelerated.

    Use with `np.where(fast_unique_mask(x) > 1)[0]` to get all unique elements."""
    output = np.zeros_like(x, dtype=np.int64)
    for i in prange(len(x)):
        output[x[i]] += 1
    return output


def aggregate_QTable(
    data: table.QTable,
    groupby: str | list[str],
    keys: str | list[str],
    agg_fn: str | Callable[[Any], Any],
    final_unit: dict[str, UnitLike] | None = None,
) -> table.QTable:
    """Shorthand for aggregating a QTable function by transforming to a pandas DataFrame and back.

    Done by:
        1. transform the data to a pandas DataFrame using `data.to_pandas()`.
        2. group by the specified columns `.groupby(groupby)`.
        3. slice by the desired keys `[keys]`.
        4. aggregate using the specified function `.agg(agg_fn)`.
        5. transform the aggregated DataFrame back to a table.Table and set the right units `table.Table.from_pandas(...,index=True,units=final_units)`.
        6. transform to a QTable `QTable(...)`.

    Parameters:
        data: The QTable to aggregate.
        groupby: The column(s) to group by.
        keys: The column(s) to aggregate.
        agg_fn: The aggregation function. Anything acceptable by `pandas.DataFrame.agg()`.
        final_unit: The units to set for the aggregated columns (otherwise will be left unitless).

    Returns:
        The aggregated QTable.
    """
    return table.QTable(
        table.Table.from_pandas(
            pd.DataFrame(data.to_pandas().groupby(groupby)[keys].agg(agg_fn)), index=True, units=final_unit
        )
    )


def add_label_unit(label: str | None, plot_unit: UnitLike | None = None) -> str | None:
    """Add the units to the `label` in a latex formatted string and enclosed in brackets. Ignore if label is `None`."""
    if label is None:
        return None
    if plot_unit is None or plot_unit == '':
        return label
    string_unit = f'{Unit(cast(str, plot_unit)):latex}'
    return rf'{label} $\left[{string_unit.strip("$")}\right]$'


@njit(parallel=True)
def fast_norm(x: NDArray[np.float64], square: bool = False) -> NDArray[np.float64]:
    """Compute the norm of each row in the array `x`. If `square` is `True`, return the square of the norm. njit accelerated."""
    output = np.empty(len(x), dtype=np.float64)
    for i in prange(len(x)):
        s = (x[i] ** 2).sum()
        if square:
            output[i] = s
        else:
            output[i] = np.sqrt(s)
    return output


def fast_quantity_norm(x: Quantity, square: bool = False) -> Quantity:
    """Compute the norm of each row in the array `x`. Wrapper around `fast_norm()`."""
    out_unit = cast(Unit, x.unit) ** 2 if square else x.unit
    return Quantity(fast_norm(x.value, square=square), unit=out_unit)


@njit(parallel=True)
def indices_to_mask(indices: NDArray[np.int64], length: int) -> NDArray[np.bool_]:
    """Create a mask of length `length` with `True` at the indices specified in `indices`. njit accelerated."""
    mask = np.zeros(length, dtype=np.bool_)
    for i in prange(len(indices)):
        mask[indices[i]] = True
    return mask


def backfill_kernel(n: int) -> NDArray[np.int64] | NDArray[np.float64]:
    """Create a kernel for backfilling a mask. The kernel has `n + 1` ones followed by `n` zeros.

    When convolving a mask with the kernel it will fill the previous `n` elements with `True`.
    """
    return np.hstack([np.ones(n + 1), np.zeros(n)])


_EXPAND_KERNEL_10 = backfill_kernel(10)


def expand_mask_back(mask: NDArray[np.bool_], n: int) -> NDArray[np.bool_]:
    """Expand a mask by `n` elements to the left.

    I.e. the `n` places to the right of every `True` element are also filled with `True`.
    """
    kernel = _EXPAND_KERNEL_10 if n == 10 else backfill_kernel(n)
    return np.convolve(mask.astype(int), kernel, mode='same') > 0


def to_extent(
    *args: NDArray[np.float64] | Quantity, force_array: bool = False
) -> tuple[float, ...] | tuple[Quantity, ...]:
    """Convert the input arrays to a tuple extent of the shape (min, max, min, max, ...).

    Args:
        *args: The input arrays to convert.
        force_array: Whether to force the output to be an array if Quantity.

    Returns:
        A tuple of the extent.
    """
    output = []
    for arg in args:
        output += [arg.min(), arg.max()]
    if force_array:
        output = [float(o.value) if isinstance(o, Quantity) else o for o in output]
    return tuple(output)


def slice_closest(
    data: table.QTable | pd.DataFrame,
    value: Quantity | float | str,
    key: str = 'time',
    copy: bool = True,
) -> table.QTable:
    """Slice the data to only keep the values closest to the input at the key.

    For example, given a table which concatenate values at different times, this method will return the subset of records where the time parameter is the closest to the requested.

    Parameters:
        data: The data to slice.
        value: The value to slice to. If a string is provided, it will be matched exactly.
        key: The key to slice by.
        copy: Whether to return a copy of the sliced data.

    Returns:
        The sliced data.
    """
    if isinstance(value, str):
        closest_value = value
    else:
        unique_values = np.unique(cast(Quantity, data[key]))
        closest_value = unique_values[np.argmin(np.abs(unique_values - value))]
    output = cast(table.QTable, data[data[key] == closest_value])
    if copy:
        return output.copy()
    return output


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


def handle_default(value: Any, default: Any) -> Any:
    """Handles setting the default value, avoiding pointer issues with Python by allowing the default function argument to be `None`"""
    if value is None:
        return default
    return value


def smooth_holes_1d(
    x: QuantityOrArray,
    y: QuantityOrArray,
    mask: NDArray[np.bool_] | None = None,
    include_zero: bool = False,
    assume_sorted: bool = False,
    bounds_error: bool = False,
    fill_value: str = 'extrapolate',
    **kwargs: Any,
) -> QuantityOrArray:
    """Smooths holes in a 1D array, defined by the provided mask.

    Smoothing is done by interpolating the values around the holes.

    Parameters:
        x: The x values used for the interpolation.
        y: The y values used for the interpolation.
        mask: The mask indicating the holes to be smoothed. If `None` treat all negative values as holes.
        include_zero: Only relevant if `mask` is not provided. Define "hole" as any `y<=0`, otherwise only fill `y<0`.
        assume_sorted: Whether the x values are sorted.
        bounds_error: Whether to raise an error if the x values are out of bounds.
        fill_value: The value to use for extrapolation. Must be accepted by zscipy.interpolate.interp1d()`.
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
    return smoothed


def smooth_holes_2d(
    data: QuantityOrArray, mask: NDArray[np.bool_] | None = None, include_zero: bool = False, **kwargs: Any
) -> QuantityOrArray:
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
    return smoothed


def make_id(id: Any | None = None, method: Literal['timestamp'] = 'timestamp') -> int:
    """Generates a unique identifier.

    Parameters:
        method: The method to use for generating the ID. Currently only 'timestamp' is supported.

    Returns:
        A unique identifier.
    """
    if id is not None:
        return id
    return int(datetime.datetime.now().timestamp() * 1000)


def guess_scale(
    array: Quantity | NDArray[Any] | list[float],
    allow_log_zero: bool = True,
    quantile_bounds: float = 0.1,
    log_scale_cutoff: float = 2,
    quantile_method: Literal['closest_observation', 'linear'] = 'closest_observation',
) -> Literal['linear', 'log']:
    """Guesses the required scale for plotting the data (linear or log).

    Parameters:
        array: The data to be analyzed.
        allow_log_zero: Whether to allow considering log scale even though the data contains zero values.
        quantile_bounds: The bounds for the quantiles used to calculate the scale.
        log_scale_cutoff: The cutoff for the log scale.
        quantile_method: The method to use for calculating the quantiles.

    Returns:
        The guessed scale.
    """
    x = np.array(array)
    if (x < 0).any() or (not allow_log_zero and (x == 0).any()):
        return 'linear'
    x = x[x > 0]
    if len(x) < 2:
        return 'linear'
    if (
        np.log10(
            np.quantile(x, 1 - quantile_bounds, method=quantile_method)
            / np.quantile(x, quantile_bounds, method=quantile_method)
        )
        > log_scale_cutoff
    ):
        return 'log'
    return 'linear'


def mask_edge_zeros(grid: NDArray[Any] | Quantity, axis: int | None = None) -> NDArray[np.bool_]:
    """Masks the edges of a grid that are fully zero.

    Parameters:
        grid: The grid to be masked.
        axis: The axis along which to mask the zeros.

    Returns:
        A boolean array indicating which elements are not edge zeros.
    """

    grid = np.array(grid)
    zeros = (grid == 0).all(axis=axis)
    non_zero_indices = np.where(~zeros)[0]
    indices = np.arange(len(zeros))
    return np.where((indices >= non_zero_indices[0]) * (indices <= non_zero_indices[-1]), True, False)


def diff(x: QuantityOrArray, pad_width: ArrayLike = (0, 1), mode: str = 'edge', **kwargs: Any) -> QuantityOrArray:
    """Returns the difference between consecutive elements of an array.

    By default, extend the difference array to match the original shape by duplicating the final value.

    Parameters:
        x: The array.
        pad_width: The width of the padding to be added to the array.
        mode: The mode of the padding.
        **kwargs: Additional keyword arguments to be passed to the padding function.

    Returns:
        The difference between consecutive elements of the quantity array.
    """
    kwargs = kwargs.copy()
    if 'mode' not in kwargs:
        kwargs['mode'] = mode
    if 'pad_width' not in kwargs:
        kwargs['pad_width'] = pad_width
    return cast(type(x), np.pad(np.diff(x), **kwargs))


def unmask_quantity(*args: Quantity) -> tuple[Quantity, ...]:
    """Safely unmasks masked quantity."""
    return tuple(cast(Quantity, arg.unmasked) if hasattr(arg, 'mask') else arg for arg in args)


def get_columns(data: table.QTable, columns: list[str], unmask: bool = True) -> tuple[Quantity, ...]:
    """Returns selected columns of a QTable as a tuple of quantities."""
    output = list(data[columns].values())
    if unmask:
        return unmask_quantity(*output)
    return tuple(output)
