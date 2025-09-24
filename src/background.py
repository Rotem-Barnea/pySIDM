import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Self, Unpack, cast
from astropy import units
from . import nsphere
from .spatial_approximation import Lattice


class Mass_Distribution:
    def __init__(self, lattice: Lattice, M: units.Quantity['mass'], time: units.Quantity['time']) -> None:
        self.lattice = lattice
        self.M = M
        self.time = time

    @classmethod
    def from_files(
        cls, lattice: Lattice, Mtot: units.Quantity['mass'], files: pd.DataFrame | None = None, **kwargs: Unpack[nsphere.File_params]
    ) -> Self:
        if files is None:
            files = nsphere.gather_files(**kwargs)
        data = np.vstack([nsphere.load_file(path, dtype)['R'] for path, dtype in tqdm(files[['path', 'record_dtype']].to_numpy(), desc='Load files')])
        lattice.update(data.ravel())
        M = np.vstack([lattice.lattice_to_density_cumsum(lattice(d)) for d in data]) * Mtot / data.shape[1]
        time = units.Quantity(files.time.tolist())
        return cls(lattice, M, time)

    def at_time(self, t: units.Quantity['time']) -> units.Quantity['mass']:
        mask = self.time == t
        if mask.any():
            return cast(units.Quantity['mass'], self.M[mask][0])
        mask = self.time < t
        if not mask.any():
            return cast(units.Quantity['mass'], self.M[0])
        elif mask.all():
            return cast(units.Quantity['mass'], self.M[-1])

        after = np.argmin(mask)
        before = after - 1
        f = (t - self.time[before]) / (self.time[after] - self.time[before])
        return cast(units.Quantity['mass'], self.M[before] * f + self.M[after] * (1 - f))

    def M_at_time(self, r: units.Quantity['length'], time: units.Quantity['time']) -> units.Quantity['mass']:
        return cast(units.Quantity['mass'], self.at_time(time)[self.lattice(r.value).clip(min=0, max=len(self.lattice) - 1).astype(np.int64)])
