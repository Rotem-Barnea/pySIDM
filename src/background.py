import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm
from typing import Self,Optional,Unpack
from . import nsphere
from .spatial_approximation import Lattice

class Potential:
    def __init__(self,lattice:Lattice,interpolants:list[interp1d],M:np.ndarray,time:np.ndarray) -> None:
        self.lattice:Lattice = lattice
        self.interpolants:list[interp1d] = interpolants
        self.M:np.ndarray = M
        self.time:np.ndarray = time

    @classmethod
    def from_files(cls,lattice:Lattice,Mtot:float,files:Optional[pd.DataFrame]=None,**kwargs:Unpack[nsphere.File_params]) -> Self:
        if files is None:
            files = nsphere.gather_files(**kwargs)
        data = np.vstack([nsphere.load_file(path,dtype)['R'] for path,dtype in tqdm(files[['path','record_dtype']].to_numpy(),desc='Load files')])
        lattice.update(data.ravel())
        M = np.vstack([lattice.lattice_to_density_cumsum(lattice(d)) for d in data]) * Mtot/data.shape[1]
        time = files.time.to_numpy()
        interpolants = [interp1d(time,bin_m,kind='cubic',bounds_error=False,fill_value=0) for bin_m in M.T]
        return cls(lattice,interpolants,M,time)

    def __call__(self,x) -> interp1d:
        return self.interpolants[self.lattice(x)]

    def at_time(self,t:float) -> np.ndarray:
        return np.array([interp(t) for interp in self.interpolants])

    def quick_at_time(self,t:float) -> np.ndarray:
        mask = self.time == t
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
