"""Helper types for the halo object"""

from typing import TypedDict
from pathlib import Path

from astropy.units import Quantity


class EarlyQuitParams(TypedDict, total=False):
    """Parameter dictionary for the early quit check.

    Attributes:
        inner_core_radius: The inner core radius. If None, use the current inner core radius.
        critical_ratio: The critical ratio defining the core collapse.
    """

    inner_core_radius: Quantity['length'] | None
    critical_ratio: float


class SaveParams(TypedDict, total=False):
    """Parameter dictionary for halo saving.

    Attributes:
        path: Save path. If `path` is `None` attempts to use the internal save path.
        two_steps: If `True` saves the simulation state in two steps, to avoid rewriting the existing file with data that can be stopped midway (leaving just the 1 corrupted file). This means that for the duration of the saving the disk size used is doubled.
        keep_last_backup: If `True` keeps a full backup of the previous save, otherwise overwrite it based on `two_steps` rules. This option _always_ uses twice the disk space.
        split_snapshots: If `True` saves the snapshots QTable as separate files.
        verbose: If `True` prints progress information.
    """

    path: str | Path | None
    two_steps: bool
    keep_last_backup: bool
    split_snapshots: bool
    verbose: bool


class OptimizeDtParams(TypedDict, total=False):
    """Parameter dictionary for optimizing the time step `dt`.

    Attributes:
        max_factor (20): Maximum factor to divide the initial `dt` by.
        min_factor (1): Minimum factor to divide the initial `dt` by.
        factor_steps (30): Number of factors tested between 1 and `max_factor`.
        factor_steps_down (4): Number of factors tested between 1 and `min_factor`. If `None` use the same value as `factor_steps`.
        min_dt (None): Minimum allowed `dt` value.
        max_dt (None): Maximum allowed `dt` value.
        test_steps (100): Number of steps to take when testing `dt`.
        include_scatters (False): Include scatters in the optimization, otherwise optimize only over the leapfrog integrator.
        verbose (True): Use a tqdm-style progress bar for the optimization process.
        tqdm_leave (False): Leave the progress bar after the optimization is complete.
        reoptimize_rate: How often should the iteration loop be paused to reoptimize the time step. Split the evolution loop into chunks of this duration and reoptimize the time step (`dt`) at the start of each chunk.
    """

    min_factor: int
    max_factor: int
    factor_steps: int
    factor_steps_down: int | None
    min_dt: Quantity['time'] | None
    max_dt: Quantity['time'] | None
    test_steps: int
    include_scatters: bool
    verbose: bool
    tqdm_leave: bool
    reoptimize_rate: Quantity['time']
