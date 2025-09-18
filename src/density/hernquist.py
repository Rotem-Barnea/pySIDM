import numpy as np
from numba import njit
from typing import Any
from ..types import FloatOrArray
from .density import Density

class Hernquist(Density):
    def __init__(self,**kwargs:Any) -> None:
        super().__init__(**kwargs)
        self.title = 'Hernquist'
        self.rho_s = self.calculate_rho_scale();

    def __repr__(self):
        return f"""Hernquist density
  - Mtot = {self.Mtot/self.print_mass_units['value']:.3e} {self.print_mass_units['name']}
  - rho_s = {self.rho_s/self.print_density_units['value']:.4f} {self.print_density_units['name']}
  - Tdyn = {self.Tdyn/self.print_time_units['value']:.4f} {self.print_time_units['name']}

  - Rmin = {self.Rmin/self.print_length_units['value']:.4f} {self.print_length_units['name']}
  - Rmax = {self.Rmax/self.print_length_units['value']:.4f} {self.print_length_units['name']}
  - space_steps = {self.space_steps:.0e}"""

    @staticmethod
    @njit
    def calculate_rho(r:FloatOrArray,rho_s:float=1,Rs:float=1,Rvir:float=1) -> FloatOrArray:
        return rho_s/(2*np.pi*(r/Rs)*(1+(r/Rs)**3))
