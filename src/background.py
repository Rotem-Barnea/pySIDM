import numpy as np
from numpy.typing import NDArray
import pandas as pd
from tqdm import tqdm
from typing import Self,Unpack
from . import nsphere
from .spatial_approximation import Lattice

class Mass_Distribution:
    def __init__(self,lattice:Lattice,M:NDArray[np.float64],time:NDArray[np.float64]) -> None:
        self.lattice:Lattice = lattice
        self.M = M
        self.time = time

    @classmethod
    def from_files(cls,lattice:Lattice,Mtot:float,files:pd.DataFrame|None=None,**kwargs:Unpack[nsphere.File_params]) -> Self:
        if files is None:
            files = nsphere.gather_files(**kwargs)
        data = np.vstack([nsphere.load_file(path,dtype)['R'] for path,dtype in tqdm(files[['path','record_dtype']].to_numpy(),desc='Load files')])
        lattice.update(data.ravel())
        M = np.vstack([lattice.lattice_to_density_cumsum(lattice(d)) for d in data]) * Mtot/data.shape[1]
        time = files.time.to_numpy()
        return cls(lattice,M,time)

    def at_time(self,t:float) -> NDArray[np.float64]:
        mask:NDArray[np.bool_] = (self.time == t)
        if mask.any():
            return self.M[mask][0]
        mask = self.time < t
        if not mask.any():
            return self.M[0]
        elif mask.all():
            return self.M[-1]

        after = np.argmin(mask)
        before = after - 1
        f = (t - self.time[before])/(self.time[after]-self.time[before])
        return self.M[before]*f + self.M[after]*(1-f)
