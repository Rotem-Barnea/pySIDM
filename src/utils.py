import numpy as np
import pandas as pd
from numba import njit, prange
from numpy.typing import NDArray
from typing import Callable, Any, cast
from astropy import table
from astropy.units import Quantity, Unit
from astropy.units.typing import UnitLike
from .types import FloatOrArray


def random_angle(like: NDArray[np.float64], acos: bool) -> NDArray[np.float64]:
    """Generate an array of random angles.

    Parameters:
        like: Array who's shape to mimic.
        acos: If False, generate a uniform random angle. If True, generate a uniform random cos(angle), and then applies arccos to retrieve the angle.

    Returns:
        Array of random angles.
    """
    rolls = np.random.rand(len(like)) if len(like.shape) == 1 else np.random.rand(*like.shape)
    if acos:
        return np.acos(rolls * 2 - 1)
    return rolls * 2 * np.pi


def from_radial(r: NDArray[np.float64], theta: NDArray[np.float64], quick_sin: bool = True) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert radial coordinates to Cartesian coordinates.

    Parameters:
        r: Radius.
        theta: Angle.
        quick_sin: If True, use a faster sin calculation. Otherwise calculate sin(theta) explicitly.

    Returns:
        x,y: coordinates.
    """
    cos: NDArray[np.float64] = np.cos(theta)
    sin: NDArray[np.float64] = np.sqrt(1 - cos**2) * np.sign(np.pi - theta) if quick_sin else np.sin(theta)
    return r * cos, r * sin


def split_2d(r: NDArray[np.float64], acos: bool) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split an array of radiuses to x,y coordinates using a random angle. See random_angle() for details on the angle calculation."""
    return from_radial(r, theta=random_angle(r, acos))


def split_3d(r: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Split an array of radiuses to x,y,z coordinates using a random acos angle for the z coordinate (i.e. radial in the halo), and a random uniform angle for the x-y plane (i.e. tangential plane in the halo)."""
    radial, perp = from_radial(r, theta=random_angle(r, acos=True))
    x, y = from_radial(perp, theta=random_angle(perp, acos=False))
    return x, y, radial


def joint_clean(arrays: list[NDArray[Any]], keys: list[str], clean_by: str) -> NDArray[Any]:
    """Clean a list of arrays by removing duplicates and sorting them by a given key."""
    data = pd.DataFrame(dict(zip(keys, arrays)))
    data = data.drop_duplicates(clean_by).sort_values(clean_by)
    return data.to_numpy().T


def clean_pairs(pairs: NDArray[np.int64], blacklist: list[int] | NDArray[np.int64] = []) -> NDArray[np.int64]:
    """Clean a list of pairs by removing duplicates.

    Ensures no particle is considered multiple times.
    If a blacklist is provided, also exclude pairs involving blacklisted particles.
    """
    cleaned_pairs = pd.DataFrame(pairs).drop_duplicates(0).drop_duplicates(1).to_numpy()  # Ensures there are no particles considered multiple times
    if len(blacklist) > 0:
        cleaned_pairs = np.array([pair for pair in cleaned_pairs if pair[0] not in blacklist and pair[1] not in blacklist])
    return cleaned_pairs


def drop_None(**kwargs: Any) -> dict[Any, Any]:
    """Remove key-value pairs where the value is None."""
    return {key: value for key, value in kwargs.items() if value is not None}


def rank_array(r: NDArray[Any]) -> NDArray[np.int64]:
    """Calculate the rank of every element in the array."""
    return r.argsort().argsort()


def derivate(x: FloatOrArray, y_fn: Callable[[FloatOrArray], FloatOrArray], h: float = 1e-4) -> FloatOrArray:
    """Calculate the derivative of a function at a point.

    Calculates a forward numerical derivative: (y_fn(x + h) - y_fn(x)) / h

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

    Calculates a forward numerical derivative: (y_fn(x + 2 * h) - 2 * y_fn(x + h) + y_fn(x)) / h**2

    Parameters:
        x: The point/points at which to calculate the derivative.
        y_fn: The function to differentiate.
        h: The step size for numerical differentiation.

    Returns:
        The derivative of the function at the given point/points.
    """
    return (y_fn(x + 2 * h) - 2 * y_fn(x + h) + y_fn(x)) / h**2


def quantity_derivate(x: Quantity, y_fn: Callable[[Quantity], Quantity], h: float = 1e-4) -> Quantity:
    """Calculate the derivative of a function at a point. Wrapper for derivate() to handle and output Quantity objects."""
    t = Quantity(h, x.unit)
    return cast(Quantity, (y_fn(cast(Quantity, x + t)) - y_fn(x)) / t)


def quantity_derivate2(x: Quantity, y_fn: Callable[[Quantity], Quantity], h: float = 1e-4) -> Quantity:
    """Calculate the second order derivative of a function at a point. Wrapper for derivate2() to handle and output Quantity objects."""
    t = Quantity(h, x.unit)
    return cast(Quantity, (y_fn(cast(Quantity, x + 2 * t)) - 2 * y_fn(cast(Quantity, x + t)) + y_fn(x)) / t**2)


@njit
def linear_interpolation(xs: NDArray[np.float64], ys: NDArray[np.float64], x: float) -> NDArray[np.float64]:
    """Calculate the linear interpolation from a grid (xs,ys) at a point x. njit compliant.

    xs must be sorted in ascending order (relies on np.searchsorted()).
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
    """Fast assignment of array elements using indices. njit accelerated."""
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
    """Integrate the density function (rho) assuming spherical symmetry. njit accelerated.

    Parameters:
        r: The radius points at which to calculate the density.
        rho_fn: The density function to integrate. must be njit compliant.
        rho_s: The scale density.
        Rs: The scale radius.
        Rvir: The virial radius.
        start: The starting radius for the integration.
        num_steps: The number of radius steps for the integration (from start to r).

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

    Use with np.where(fast_unique_mask(x) > 1)[0] to get all unique elements."""
    output = np.zeros_like(x, dtype=np.int64)
    for i in prange(len(x)):
        output[x[i]] += 1
    return output


def aggregate_QTable(
    data: table.QTable,
    groupby: str | list[str],
    keys: str | list[str],
    agg_fn: str | Callable[[Any], Any],
    final_units: dict[str, UnitLike] | None = None,
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
        agg_fn: The aggregation function. Anything acceptable by pandas.DataFrame.agg().
        final_units: The units to set for the aggregated columns (otherwise will be left unitless).

    Returns:
        The aggregated QTable.
    """
    return table.QTable(table.Table.from_pandas(pd.DataFrame(data.to_pandas().groupby(groupby)[keys].agg(agg_fn)), index=True, units=final_units))


def add_label_unit(label: str | None, plot_units: UnitLike | None = None) -> str | None:
    """Add the units to the label in a latex formatted string and enclosed in brackets. Ignore if label is None."""
    if label is None:
        return None
    if plot_units is None:
        return label
    return f'{label} [{Unit(cast(str, plot_units)):latex}]'


@njit(parallel=True)
def fast_norm(x: NDArray[np.float64], square: bool = False) -> NDArray[np.float64]:
    """Compute the norm of each row in the array x. If square is True, return the square of the norm. njit accelerated."""
    output = np.empty(len(x), dtype=np.float64)
    for i in prange(len(x)):
        s = (x[i] ** 2).sum()
        if square:
            output[i] = s
        else:
            output[i] = np.sqrt(s)
    return output


def fast_quantity_norm(x: Quantity, square: bool = False) -> Quantity:
    """Compute the norm of each row in the array x. Wrapper around fast_norm()."""
    out_units = cast(Unit, x.unit) ** 2 if square else x.unit
    return Quantity(fast_norm(x.value, square=square), unit=out_units)


@njit(parallel=True)
def indices_to_mask(indices: NDArray[np.int64], length: int) -> NDArray[np.bool_]:
    """Create a mask of length `length` with True at the indices specified in `indices`. njit accelerated."""
    mask = np.full(length, False, dtype=np.bool_)
    for i in prange(len(indices)):
        mask[indices[i]] = True
    return mask


def expand_mask_back(mask: NDArray[np.bool_], n: int) -> NDArray[np.bool_]:
    """Expand a mask by `n` elements to the left.

    I.e. the `n` places to the right of every True element are also filled with True.
    """
    return np.convolve(mask.astype(int), np.hstack([np.ones(n + 1), np.zeros(n)]), mode='same') > 0
