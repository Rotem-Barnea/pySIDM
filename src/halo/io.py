"""IO operations on the Halo class"""

import pickle
import shutil
from typing import Any, Literal, TypedDict, cast, overload
from pathlib import Path
from collections import deque

import numpy as np
import regex
from astropy import table
from numpy.typing import NDArray
from astropy.units import Quantity

from src import physics
from src.tqdm import tqdm
from src.background import BackgroundDistribution
from src.distribution.distribution import Distribution


class Metadata(TypedDict, total=False):
    """Metadata for a simulation.

    Attributes:
        time: The current time of the simulation.
        steps: The current step of the simulation.
        dt: The current timestep of the simulation.
        unoptimized_dt: The unoptimized timestep of the simulation.
        save_every_n_steps: The number of steps between saves.
        save_every_time: The time between saves.
        dynamics_params: The parameters for the dynamics.
        scatter_params: The parameters for the scattering.
        last_saved_time: The last time the simulation was saved.
        hard_save: Whether to save the halo to memory at every snapshot save, or just keep in RAM.
        save_path: The path to save the simulation to.
        r_max: Maximum radius of the halo, particles outside of this radius get killed off.
        inner_core_radius: Inner core radius of the halo, used for estimating the collapse.
        critical_ratio: The critical ratio defining the core collapse.
        cleanup_nullish_particles: Whether to clean-up nullish particles.
        cleanup_particles_by_radius: Whether to clean-up particles by radius.
        reached_core_collapse: Whether the halo has reached core collapse.
        seed: The seed for the random number generator.
        generator_state: The state of the random number generator.
        n_particles: The number of particles in the simulation.
        name: The name of the simulation.
    """

    time: Quantity['time']
    steps: int
    dt: Quantity['time']
    unoptimized_dt: Quantity['time']
    save_every_n_steps: int | None
    save_every_time: Quantity['time'] | None
    dynamics_params: physics.leapfrog.Params
    scatter_params: physics.sidm.Params
    last_saved_time: Quantity['time']
    hard_save: bool
    save_path: Path
    r_max: Quantity['length']
    inner_core_radius: Quantity['length']
    critical_ratio: float
    cleanup_nullish_particles: bool
    cleanup_particles_by_radius: bool
    reached_core_collapse: bool
    seed: Any
    generator_state: Any
    n_particles: int
    name: str | list[str]


class HeavyPayload(TypedDict, total=False):
    """Heavy payload metadata for a simulation.

    Attributes:
        ministep_size: The size of the mini-step used for each mini-step (to track changes in them).
        scatter_track_time: The time for each scatter track round, must match `scatter_track_index` and `scatter_track_radius` in shape.
        scatter_track_index: The interacting particles (particle index) at every timestep.
        scatter_track_radius: The location of the interacting particles at every timestep.
        scatter_rounds: Number of scatter rounds the halo had every timestep.
        scatter_rounds_underestimated: Number of underestimated scatter rounds the halo had every timestep (due to `max_allowed_rounds` in `physics.sidm.scatter()`).
        runtime_realtime_track: The time at the start of each step.
        runtime_track_sort: The time taken to sort the particles.
        runtime_track_cleanup: The time taken to clean-up the particles.
        runtime_track_sidm: The time taken to perform SIDM calculations.
        runtime_track_leapfrog: The time taken to perform leapfrog calculations.
        runtime_track_full_step: The time taken to perform a full step.
    """

    ministep_size: deque[float] | None
    scatter_track_time: deque[float] | None
    scatter_track_index: deque[NDArray[np.int64]] | None
    scatter_track_radius: deque[NDArray[np.float64]] | None
    scatter_rounds: deque[int] | None
    scatter_rounds_underestimated: deque[int] | None
    runtime_realtime_track: deque[float] | None
    runtime_track_sort: deque[float] | None
    runtime_track_cleanup: deque[float] | None
    runtime_track_sidm: deque[float] | None
    runtime_track_leapfrog: deque[float] | None
    runtime_track_full_step: deque[float] | None
    runtime_track_simulation_time: deque[float] | None


def metadata_keys() -> list[str]:
    """Return the keys of the metadata payload dictionary, used for saving and loading halos."""
    return list(Metadata.__annotations__.keys())


def heavy_payload_keys() -> list[str]:
    """Return the keys of the metadata payload dictionary, used for saving and loading halos."""
    return list(HeavyPayload.__annotations__.keys())


def save_table(data: table.QTable, path: str | Path, **kwargs: Any) -> None:
    """Save a QTable to a file, splitting the strings from the Quantity data, and saving into `{}_strings.csv` and `{}.fits`."""
    data[[column for column in data.colnames if data[column].dtype != np.dtype('O')]].write(
        path.with_name(f'{path.stem}.fits'), **kwargs
    )
    data[[column for column in data.colnames if data[column].dtype == np.dtype('O')]].write(
        path.with_name(f'strings_{path.stem}.csv'), **kwargs
    )


def load_table(path: str | Path) -> table.QTable:
    """Load a QTable saved via `save_table()`."""
    fits_table = table.QTable.read(path.with_name(f'{path.stem}.fits'))
    csv_table = table.QTable.read(path.with_name(f'strings_{path.stem}.csv'))
    for col in fits_table.colnames:
        fits_table[col] = fits_table[col].astype(fits_table[col].dtype.newbyteorder('='), copy=False)
    for col in csv_table.colnames:
        csv_table[col] = np.array(csv_table[col]).astype('O')
    return cast(table.QTable, table.hstack([fits_table, csv_table]))


def save_pickle(
    path: str | Path, stem: str, payload: dict[str, Any] | Metadata | HeavyPayload, verbose: bool = False
) -> None:
    """Save the simulation's metadata"""
    if verbose:
        print(f'Saving {stem}.pkl')
    with open(Path(path) / f'{stem}.pkl', 'wb') as f:
        pickle.dump(payload, f)


@overload
def load_pickle(path: str | Path, stem: Literal['metadata'], verbose: bool = False) -> Metadata: ...


@overload
def load_pickle(path: str | Path, stem: Literal['heavy_payload'], verbose: bool = False) -> HeavyPayload: ...


def load_pickle(
    path: str | Path, stem: str | Literal['metadata', 'heavy_payload'] = 'metadata', verbose: bool = False
) -> dict[str, Any] | Metadata | HeavyPayload:
    """Load a pickled simulation file"""
    if verbose:
        print(f'Loading {stem}.pkl')
    with open(Path(path) / f'{stem}.pkl', 'rb') as f:
        data = pickle.load(f)
    if stem == 'metadata':
        return Metadata(**data)
    elif stem == 'heavy_payload':
        return HeavyPayload(**data)
    return data


def load_distributions(path: str | Path, verbose: bool = False) -> list[Distribution]:
    """Load the distributions"""
    if verbose:
        print('Loading distributions')
    return [Distribution.load(path.parent, path.stem) for path in (Path(path) / 'distributions').glob('*.pkl')]


def save(
    path: str | Path | None,
    static_tables: dict[str, table.QTable] = {},
    splitable_table: dict[str, table.QTable] = {},
    metadata_payload: Metadata = {},
    heavy_payload: HeavyPayload = {},
    distributions: list[Distribution] = [],
    background: BackgroundDistribution | None = None,
    two_steps: bool = False,
    keep_last_backup: bool = False,
    split_tables: bool = True,
    verbose: bool = False,
) -> None:
    """Save the simulation state to a directory.

    Parameters:
        path: Save path.
        static_tables: Static tables to save (cannot be split).
        splitable_table: Splitable tables to save (if `split_tables` is True).
        metadata_payload: Metadata payload to save.
        heavy_payload: Heavy payload to save.
        distributions: Distributions to save.
        background: Background distribution to save.
        two_steps: If `True` saves the simulation state in two steps, to avoid rewriting the existing file with data that can be stopped midway (leaving just the 1 corrupted file). This means that for the duration of the saving the disk size used is doubled.
        keep_last_backup: If `True` keeps a full backup of the previous save, otherwise overwrite it based on `two_steps` rules. This option _always_ uses twice the disk space.
        split_tables: If `True` saves the `splitable_table` QTables as separate files.
        verbose: If `True` prints progress information.

    Returns:
        None
    """
    assert path is not None, 'Save path must be provided'
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    if keep_last_backup:
        for file in tqdm(
            (files := list(path.glob('*'))), desc='backing up existing data', disable=not verbose or len(files) == 0
        ):
            if '_backup.' in file.name:
                continue
            if file.is_dir():
                shutil.copytree(file, file.with_stem(f'{file.stem}_backup'), dirs_exist_ok=True)
            else:
                shutil.copyfile(file, file.with_stem(f'{file.stem}_backup'))
    tables = static_tables
    if not split_tables:
        tables.update(**splitable_table)
    tag = '_' if two_steps else ''
    save_pickle(path, f'metadata{tag}', metadata_payload, verbose=verbose)
    save_pickle(path, f'heavy_payload{tag}', heavy_payload, verbose=verbose)
    for distribution in tqdm(
        distributions, desc='Saving distributions', disable=not verbose or len(distributions) == 0
    ):
        (path / 'distributions').mkdir(exist_ok=True)
        distribution.save(path / 'distributions', f'{distribution.name}_{distribution.title}{tag}', verbose=verbose)
    if background is not None:
        background.save(path, f'background{tag}', verbose=verbose)
    for name, data in tqdm(tables.items(), desc='Saving tables', disable=not verbose or len(tables) == 0):
        save_table(data, path / f'{name}{tag}.fits', overwrite=True)
    for file in tqdm(
        (files := list(path.glob('*_.*'))), desc='overwriting backup', disable=not verbose or len(files) == 0
    ):
        file.rename(file.with_stem(file.stem[:-1]))
    if split_tables:
        for stem, table in splitable_table.items():
            (path / f'split_{stem}').mkdir(exist_ok=True)
            if len(table) > 0:
                for i, group in tqdm(
                    (tables := list(enumerate(table.group_by('time').groups))),
                    desc=f'Saving split data for {stem}',
                    disable=not verbose or len(tables) == 0,
                ):
                    save_table(group, path / f'split_{stem}/{stem}_{i}.fits', overwrite=True)


def load_tables(
    path: str | Path,
    ensure_keys: list[str] = ['particles', 'initial_particles', 'snapshots'],
    undersample: dict[str, int | None] = {},
    verbose: bool = False,
) -> dict[str, table.QTable | None]:
    """Load the simulation tables.

    Parameters:
        path: Save path to load from.
        ensure_keys: List of keys to ensure are present in the loaded tables (set with `None` value).
        undersample: If provided, undersample loading split tables by the given factor (i.e. load every 10th table, etc.).
        verbose: Whether to print progress information.

    Returns:
        The loaded tables
    """
    path = Path(path)
    tables = {}
    for splitted_path in list(path.glob('split_*')):
        name = regex.sub(r'split_', '', splitted_path.stem)
        files = sorted(list(splitted_path.glob('*.fits')), key=lambda x: int(regex.findall(r'_(\d+)$', x.stem)[0]))
        if name in undersample and undersample[name] is not None:
            files = files[:: undersample[name]]
        table_list = [
            load_table(file) for file in tqdm(files, desc=f'Loading split tables for {name}', disable=not verbose)
        ]
        if len(table_list) > 0:
            tables[name] = cast(table.QTable, table.vstack(table_list))
    for unsplitted_path in list(path.glob('*.fits')):
        tables[unsplitted_path.stem] = load_table(unsplitted_path)

    for key in ensure_keys:
        if key not in tables:
            tables[key] = None

    return tables


def load_background(path: str | Path, stem: str = 'background', verbose: bool = True) -> BackgroundDistribution | None:
    """Load the background from a file."""
    if verbose:
        print('Loading background')
    path = Path(path)
    if (Path(path) / f'{stem}.pkl').exists() or (Path(path) / f'{stem}_distribution.pkl').exists():
        return BackgroundDistribution.load(path=path, stem=stem)
    return None
