"""General purpose utility functions"""

import datetime
from typing import Any, Literal, Callable, cast
from collections.abc import Sequence

import numpy as np
import scipy
import pandas as pd
from numba import njit, prange
from astropy import table
from numpy.typing import NDArray, ArrayLike
from astropy.units import Unit, Quantity
from astropy.units.typing import UnitLike

from src import rng, types


def random_angle(
    like: NDArray[np.float64] | int,
    arccos: bool,
    generator: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Generate an array of random angles.

    Parameters:
        like: Array who's shape to mimic. if `int` treat it as the length of the array.
        arccos: If `False` generate a uniform random angle. If `True` generate a uniform random `cos(angle)`, and then applies arccos to retrieve the angle.
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
    if arccos:
        return np.acos(rolls * 2 - 1)
    return rolls * 2 * np.pi


def from_radial(
    r: NDArray[np.float64], theta: NDArray[np.float64], quick_sin: bool = True
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert radial coordinates to Cartesian coordinates.

    Parameters:
        r: Radius.
        theta: Angle.
        quick_sin: If `True` use a faster `sin` calculation. Otherwise, calculate `sin(theta)` explicitly.

    Returns:
        x, y coordinates.
    """
    cos: NDArray[np.float64] = np.cos(theta)
    sin: NDArray[np.float64] = np.sqrt(1 - cos**2) * np.sign(np.pi - theta) if quick_sin else np.sin(theta)
    return r * cos, r * sin


def split_2d(
    r: NDArray[np.float64],
    arccos: bool,
    generator: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split an array of radii into x, y coordinates using a random angle. See `random_angle()` for details on the angle calculation."""
    return from_radial(r, theta=random_angle(r, arccos, generator=generator))


def split_3d(
    r: NDArray[np.float64],
    generator: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Split an array of radii into x, y, z coordinates using a `random arccos angle` for the `z` coordinate (i.e. radial in the halo), and a `random uniform angle` for the `x-y` plane (i.e. tangential plane in the halo)."""
    radial, perp = from_radial(r, theta=random_angle(r, arccos=True, generator=generator))
    x, y = from_radial(perp, theta=random_angle(perp, arccos=False, generator=generator))
    return x, y, radial


def split_3d_quantity(x: Quantity, generator: np.random.Generator | None = None) -> Quantity:
    """Wrapper for `split_3d` that handles quantities."""
    return cast(Quantity, np.vstack(split_3d(x, generator=generator)).T)


def drop_None(**kwargs: Any) -> dict[Any, Any]:
    """Remove key-value pairs where the value is `None`."""
    return {key: value for key, value in kwargs.items() if value is not None}


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
def fast_spherical_density_integrate(
    r: NDArray[np.float64],
    density_fn: Callable[..., NDArray[np.float64]],
    rho_s: float = 1,
    r_s: float = 1,
    r_vir: float = 1,
    start: float = 0,
    num_steps: int = 10000,
) -> NDArray[np.float64]:
    """Integrate the density function (`rho`) assuming spherical symmetry. njit accelerated.

    Parameters:
        r: The radius points at which to calculate the density.
        density_fn: The density function to integrate. must be njit compliant.
        rho_s: The scale density.
        r_s: The scale radius.
        r_vir: The virial radius.
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
        ys = density_fn(x, rho_s=rho_s, r_s=r_s, r_vir=r_vir)
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

    Parameters:
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


def diff(x: types.QuantityLike, pad_width: ArrayLike = (0, 1), mode: str = 'edge', **kwargs: Any) -> types.QuantityLike:
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
    return cast(types.QuantityLike, np.pad(np.diff(x), **kwargs))


def unmask_quantity(*args: Quantity) -> tuple[Quantity, ...]:
    """Safely unmasks masked quantity."""
    return tuple(cast(Quantity, arg.unmasked) if hasattr(arg, 'mask') else arg for arg in args)


def get_columns(data: table.QTable, columns: list[str], unmask: bool = True) -> tuple[Quantity, ...]:
    """Returns selected columns of a QTable as a tuple of quantities."""
    output = list(data[columns].values())
    if unmask:
        return unmask_quantity(*output)
    return tuple(output)


def to_center(
    edges: types.QuantityLike, method: Literal['algebric', 'geometric'] = 'geometric', guard_geometric: bool = True
) -> types.QuantityLike:
    """Calculates the center of each grid cell"""
    assert method in ['algebric', 'geometric'], ValueError(f"Invalid method '{method}'")
    if method == 'geometric':
        centers = np.sqrt(np.abs(edges[:-1] * edges[1:])) if guard_geometric else np.sqrt(edges[:-1] * edges[1:])
    elif method == 'algebric':
        centers = (edges[:-1] + edges[1:]) / 2
    return cast(type(edges), centers)


def to_edge(
    center_value: types.QuantityLike,
    low_bound: Literal[0, 'constant', 'center'] = 'constant',
    upper_bound: Literal[0, 'constant', 'center'] = 'constant',
) -> types.QuantityLike:
    """Interpolate the array to the shell edges from the center value"""
    if low_bound == 0:
        low = 0
    elif low_bound == 'constant':
        low = (center_value[0] + center_value[1]) / 2
    else:
        low = center_value[0]
    if upper_bound == 0:
        high = 0
    elif upper_bound == 'constant':
        high = (center_value[-2] + center_value[-1]) / 2
    else:
        high = center_value[-1]

    return cast(type(center_value), np.hstack([low, (center_value[:-1] + center_value[1:]) / 2, high]))


def fit_curve(
    fn: Callable[Any, NDArray[np.float64]],
    x: types.QuantityLike,
    y: types.QuantityLike,
    data_mask: NDArray[np.bool_] | None = None,
    xlog: bool = False,
    ylog: bool = False,
    output_scheme: Sequence[UnitLike | Literal['x', 'y'] | None] = ['x', 'y'],
    p0: tuple[float, ...] | None = None,
    bounds: tuple[float, float] = (0, np.inf),
    max_nfev: int = 10000,
    **kwargs: Any,
) -> list[Quantity | float]:
    """Fit data to a curve.

    Parameters:
        fn: Function to fit to.
        x: Array of input x values.
        y: Array of input y values.
        data_mask: A mask to filter out only parts of the data for the fitting.
        xlog: If True, apply a log to the x-values whenever used during fitting.
        ylog: If True, apply a log to the y-values whenever used during fitting.
        output_scheme: A list of units to wrap the results in. `'x'` and `'y`' are interpreted as the units of `x` and `y` respectively (will raise an error if they are not a Quantity object). `None` will leave the parameter as a float.
        p0: Initial guess, passed on to the solver (scipy.optimize.curve_fit()). Also defines the number of parameters to solve (positional from `fn`).
        bounds: Bounds for the parameters, passed on to the solver (scipy.optimize.curve_fit()).
        max_nfev: max number of iterations, passed on to the solver (scipy.optimize.curve_fit()).
        kwargs: Additional keyword arguments passed on to the solver (scipy.optimize.curve_fit()).

    Returns:
        A list of optimized parameters.
    """
    xdata, ydata = (np.log(np.array(value)) if log else np.array(value) for value, log in zip([x, y], [xlog, ylog]))

    if data_mask is not None:
        xdata, ydata = xdata[data_mask], ydata[data_mask]
    popt, _ = scipy.optimize.curve_fit(
        f=(lambda *x: np.log(fn(*x))) if ylog else fn,
        xdata=xdata,
        ydata=ydata,
        p0=p0,
        bounds=bounds,
        max_nfev=max_nfev,
        **kwargs,
    )
    return [
        Quantity(p, x.unit if unit == 'x' else (y.unit if unit == 'y' else unit)) if unit is not None else p
        for p, unit in zip(popt, output_scheme)
    ]


def gaussian_filter1d(input: types.QuantityLike, sigma: float | None, **kwargs: Any) -> types.QuantityLike:
    """Performs a 1d Gaussian filter on the data, preserving units."""
    if sigma is None or sigma == 0:
        return input
    filtered = scipy.ndimage.gaussian_filter1d(np.array(input), sigma=sigma, **kwargs)
    if isinstance(input, Quantity):
        return Quantity(filtered, input.unit)
    return filtered
