import numpy as np
from numba import njit
from typing import Any
from astropy import units
from .density import Density
from ..types import FloatOrArray

class NFW(Density):
    def __init__(self,Rs:units.Quantity['length'],c:float,**kwargs:Any) -> None:
        super().__init__(Rs=Rs,Rvir=c*Rs,**kwargs)
        self.title = 'NFW'
        self.c = c

    def __repr__(self):
        return f"""NFW density
  - Rs = {self.Rs:.4f}
  - c = {self.c:.1f}
  - Mtot = {self.Mtot:.3e}
  - Mdm = {self.unit_mass:.3e}
  - Rvir = {self.Rvir:.4f}
  - rho_s = {self.rho_s:.4f}
  - Tdyn = {self.Tdyn:.4f}

  - Rmin = {self.Rmin:.4f}
  - Rmax = {self.Rmax:.4f}
  - space_steps = {self.space_steps:.0e}"""

    @staticmethod
    @njit
    def calculate_rho(r:FloatOrArray,rho_s:float=1,Rs:float=1,Rvir:float=1) -> FloatOrArray:
        return rho_s/((r/Rs)*(1+(r/Rs))**2)/(1+(r/Rvir)**10)

    def calculate_theoretical_M(self,r:units.Quantity['length']) -> units.Quantity['mass']:
        x = self(r)
        return 4*np.pi*self.rho_s*self.Rs**3*(np.log(1+x)-x/(1+x))
