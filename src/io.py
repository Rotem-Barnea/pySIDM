import pickle
import shutil
from typing import Any, cast
from pathlib import Path

import numpy as np
import regex
from astropy import table

from .tqdm import tqdm


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


def save_pickle(path: str | Path, stem: str, payload: dict[str, Any]) -> None:
    """Save the simulation's metadata"""
    with open(Path(path) / f'{stem}.pkl', 'wb') as f:
        pickle.dump(payload, f)


def load_pickle(path: str | Path, stem: str) -> dict[str, Any]:
    """Load a pickled simulation file"""
    with open(Path(path) / f'{stem}.pkl', 'rb') as f:
        return pickle.load(f)


def save(
    path: str | Path | None,
    static_tables: dict[str, table.QTable] = {},
    splitable_table: dict[str, table.QTable] = {},
    metadata_payload: dict[str, Any] = {},
    heavy_payload: dict[str, Any] = {},
    two_steps: bool = False,
    keep_last_backup: bool = False,
    split_tables: bool = True,
) -> None:
    """Save the simulation state to a directory.

    Parameters:
        path: Save path.
        static_tables: Static tables to save (cannot be split).
        splitable_table: Splitable tables to save (if `split_tables` is True).
        metadata_payload: Metadata payload to save.
        heavy_payload: Heavy payload to save.
        two_steps: If `True` saves the simulation state in two steps, to avoid rewriting the existing file with data that can be stopped midway (leaving just the 1 corrupted file). This means that for the duration of the saving the disk size used is doubled.
        keep_last_backup: If `True` keeps a full backup of the previous save, otherwise overwrite it based on `two_steps` rules. This option _always_ uses twice the disk space.
        split_tables: If `True` saves the `splitable_table` QTables as separate files.

    Returns:
        None
    """
    assert path is not None, 'Save path must be provided'
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    if keep_last_backup:
        for file in path.glob('*'):
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
    save_pickle(path, f'metadata{tag}', metadata_payload)
    save_pickle(path, f'heavy_payload{tag}', heavy_payload)
    for name, data in tables.items():
        save_table(data, path / f'{name}{tag}.fits', overwrite=True)
    for file in path.glob('*_.*'):
        file.rename(file.with_stem(file.stem[:-1]))
    if split_tables:
        for stem, table in splitable_table.items():
            (path / f'split_{stem}').mkdir(exist_ok=True)
            if len(table) > 0:
                for i, group in enumerate(table.group_by('time').groups):
                    save_table(group, path / f'split_{stem}/{stem}_{i}.fits', overwrite=True)


def load_tables(
    path: str | Path,
    ensure_keys: list[str] = ['particles', 'initial_particles', 'snapshots'],
    undersample: dict[str, int | None] = {},
) -> dict[str, table.QTable | None]:
    """Load the simulation tables.

    Parameters:
        path: Save path to load from.
        ensure_keys: List of keys to ensure are present in the loaded tables (set with `None` value).
        undersample: If provided, undersample loading split tables by the given factor (i.e. load every 10th table, etc.).

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
        table_list = [load_table(file) for file in tqdm(files, desc=f'Loading split tables for {name}')]
        if len(table_list) > 0:
            tables[name] = cast(table.QTable, table.vstack(table_list))
    for unsplitted_path in list(path.glob('*.fits')):
        tables[unsplitted_path.stem] = load_table(unsplitted_path)

    for key in ensure_keys:
        if key not in tables:
            tables[key] = None

    return tables
