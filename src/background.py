import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Self,Optional,Unpack,Callable
from . import nsphere
from .spatial_approximation import Lattice

class Mass_Distribution:
    def __init__(self,lattice:Lattice,M:np.ndarray,time:np.ndarray) -> None:
        self.lattice:Lattice = lattice
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
        return cls(lattice,M,time)

    def at_time(self,t:float) -> np.ndarray:
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

# class Velocity_Distribution(Distribution):
#     def __init__(self,lattice:Lattice,data:np.ndarray,cumulative_data:np.ndarray,time:np.ndarray) -> None:
#         super().__init__(lattice,data,cumulative_data,time)

#     @classmethod
#     def from_files(cls,lattice:Lattice,files:Optional[pd.DataFrame]=None,**kwargs:Unpack[nsphere.File_params]) -> Self:
#         return super().from_files(lattice=lattice,files=files,process_fn=lambda x:np.sqrt(x['Vrad']**2+(x['L']/x['R'])**2),**kwargs)
