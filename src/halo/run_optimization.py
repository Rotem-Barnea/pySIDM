"""Manage the optimizations made  mid-run, such as time step (`dt`) optimization and early quiting on core collapse"""

import time
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import scipy
from astropy import table
from numpy.typing import NDArray
from astropy.units import Quantity

from src import utils
from src.tqdm import tqdm

from . import types

if TYPE_CHECKING:
    from .halo import Halo


def optimize_dt(
    halo: 'Halo',
    max_factor: int = 20,
    min_factor: int = 1,
    factor_steps: int = 30,
    factor_steps_down: int | None = 4,
    min_dt: Quantity['time'] | None = None,
    max_dt: Quantity['time'] | None = None,
    test_steps: int = 100,
    include_scatters: bool = False,
    verbose: bool = True,
    tqdm_leave: bool = False,
    **kwargs: Any,
) -> Quantity['time']:
    """Optimize dt to minimize the time taken for a given number of steps.

    The optimization is performed by running consecutive steps with decreasing `dt` value. A larger `dt` might require more adaptive rounds to converge leading to a slower runtime, while a smaller `dt` would require more steps all-together to reach a predefined `T`.

    Parameters:
        max_factor: Maximum factor to divide the initial `dt` by.
        min_factor: Minimum factor to divide the initial `dt` by.
        factor_steps: Number of factors tested between 1 and `max_factor`.
        factor_steps_down: Number of factors tested between 1 and `min_factor`. If `None` use the same value as `factor_steps`.
        min_dt: Minimum allowed `dt` value.
        max_dt: Minimum allowed `dt` value.
        test_steps: Number of steps to take when testing `dt`.
        include_scatters: Include scatters in the optimization, otherwise optimize only over the leapfrog integrator.
        verbose: Use a tqdm-style progress bar for the optimization process.
        tqdm_leave: Leave the progress bar after the optimization is complete.
        **kwargs: Additional unused keyword arguments.
    """
    progress_speed = []
    factor = np.hstack(
        [
            1 / np.linspace(1, min_factor, factor_steps_down or factor_steps),
            np.linspace(1, max_factor + 1, factor_steps),
        ]
    )
    dt_candidates = halo.unoptimized_dt.copy() / factor
    if min_dt is not None:
        dt_candidates = dt_candidates.clip(min=min_dt)
    if max_dt is not None:
        dt_candidates = dt_candidates.clip(max=max_dt)
    dt_candidates = np.unique(dt_candidates)
    for dt in tqdm(dt_candidates, desc='Optimizing `dt` value', disable=not verbose, leave=tqdm_leave):
        temp_halo = halo.copy()
        temp_halo.dt = dt
        start = time.perf_counter()
        for _ in range(test_steps):
            temp_halo.step(in_bootstrap=not include_scatters)
        end = time.perf_counter()
        progress_speed += [dt.value / (end - start)]
        del temp_halo
    optimized_dt: Quantity = dt_candidates[np.argmax(progress_speed)].copy()
    if verbose:
        print(
            f'Optimized factor: {halo.unoptimized_dt / optimized_dt:.2f} = 1/{optimized_dt / halo.unoptimized_dt:.2f}, `dt` value used: {optimized_dt}'
        )
    return optimized_dt


def split_to_chunks(
    required_time: Quantity['time'],
    reoptimize_rate: Quantity['time'] = Quantity(1, 'Gyr'),
    optimize_dt_kwargs: types.OptimizeDtParams | None = None,
) -> tuple[NDArray[np.int64], Quantity['time']]:
    """Split the runtime of an evolution look into chunks based on the reoptimization rate provided in `optimize_dt_kwargs` if provided, otherwise use the keyword argument directly (set to handle default)."""
    if optimize_dt_kwargs is not None:
        reoptimize_rate = optimize_dt_kwargs.get('reoptimize_rate', reoptimize_rate)
        start_points = np.arange(0, (required_time / reoptimize_rate).decompose().value, 1, dtype=np.int64)
    else:
        reoptimize_rate = required_time
        start_points = np.array([0], dtype=np.int64)
    return start_points, reoptimize_rate


def check_early_quit(core_collape_kwargs: types.CoreCollapseDensityEstimateParams | None = None) -> bool:
    """Check if the simulation should be terminated early.

    Parameters:
        core_collape_kwargs: Keyword arguments passed on to `core_is_collapsing()` to determine an early quit. If None ignores this check.

    Returns:
        `True` if the simulation should be terminated early, `False` otherwise.
    """
    if core_collape_kwargs is not None:
        return core_is_collapsing(**core_collape_kwargs)
    return False


def core_density_ratio(
    r: Quantity['length'],
    initial_r: Quantity['length'],
    inner_core_radius: Quantity['length'],
    **kwargs: Any,
) -> bool:
    """Calcualtes the density in the inner core, divided by the initial density

    Parameters:
        r: The radius of each particle.
        initial_r: The initial radius of each particle.
        inner_core_radius: The inner core radius.
        **kwargs: Additional unused keyword arguments.

    Returns:
        density ratio in the inner core
    """
    return (r < inner_core_radius).sum() / (initial_r < inner_core_radius).sum()


def core_is_collapsing(
    r: Quantity['length'],
    initial_r: Quantity['length'],
    inner_core_radius: Quantity['length'],
    critical_ratio: float = 2,
    **kwargs: Any,
) -> bool:
    """Check if the simulation should be terminated early.

    Parameters:
        r: The radius of each particle.
        initial_r: The initial radius of each particle.
        inner_core_radius: The inner core radius.
        critical_ratio: The critical ratio defining the core collapse.
        **kwargs: Additional unused keyword arguments.

    Returns:
        `True` if the simulation should be terminated early, `False` otherwise.
    """
    return core_density_ratio(r=r, initial_r=initial_r, inner_core_radius=inner_core_radius) >= critical_ratio


def core_collapse_scatter_estimate(
    t: Quantity['time'],
    scatter_times: Quantity['time'],
    n_scatters: NDArray[np.int64],
    time_binning: Quantity['time'] = Quantity(100, 'Myr'),
    cutoff: int | float = 1e5,
    kind: str = 'linear',
    bounds_error: bool = False,
    fill_value: float | Literal['extrapolate'] = np.inf,
) -> Quantity['time']:
    """Calculate the time at which the halo starts major core collapse.

    Defined as the time at which the halo first reaches `cutoff` scatters per `time_binning` time.

    Parameters:
        time_binning: The binning resolution to aggregate the number of scattering events.
        cutoff: The number of scatters per `time_binning` time at which the core collapse is considered to have started.
        kind: The kind of interpolation to use.
        bounds_error: Raise error if the time is outside the bounds.
        fill_value: The value to use if outside the bounds, see `scipy.interpolate.interp1d`.

    Returns:
        The core collapse start time
    """
    t = Quantity(np.arange(0, t.value, 10), t.unit)
    scatters = np.bincount(
        np.digitize(scatter_times, t) - 1,
        weights=n_scatters,
        minlength=len(t) - 1,
    )
    return Quantity(
        scipy.interpolate.interp1d(
            *utils.joint_clean([scatters, t]),
            kind=kind,
            bounds_error=bounds_error,
            fill_value=fill_value,
        )(cutoff),
        t.unit,
    )


def core_collapse_core_density_estimate(
    snapshots: table.QTable,
    initial_r: Quantity['length'],
    inner_core_radius: Quantity['length'],
    critical_ratio: float = 2,
) -> Quantity['time']:
    """Calculate the time at which the halo starts major core collapse.

    Defined as the time at which the inner core density first exceeds `critical_ratio` times the initial density.

    Parameters:
        snapshots: The snapshots of the halo.
        initial_r: The initial radius of the particles in the halo.
        inner_core_radius: The radius of the inner core.
        critical_ratio: The critical ratio defining the core collapse.

    Returns:
        The core collapse start time
    """
    groups = snapshots.group_by('time').groups
    t = groups.keys['time']

    ratio = np.array(
        [core_density_ratio(r=group['r'], initial_r=initial_r, inner_core_radius=inner_core_radius) for group in groups]
    )

    if (ratio >= critical_ratio).any():
        i = np.argmax(ratio > critical_ratio)
        return (t[i] - t[i - 1]) / (ratio[i] - ratio[i - 1]) * (critical_ratio - ratio[i - 1]) + t[i - 1]
    return Quantity(np.inf, t.unit)


def core_collapse_estimate(
    method: Literal['core density', 'scatters'] = 'core density', **kwargs: Any
) -> Quantity['time']:
    """Calculate the time at which the halo starts major core collapse."""
    if method == 'core density':
        return core_collapse_core_density_estimate(**kwargs)
    else:
        return core_collapse_scatter_estimate(**kwargs)
