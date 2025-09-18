import numpy as np
from numba import njit
from typing import Any,cast
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from .density import Density
from ..types import FloatOrArray
from ..constants import kpc,default_units,Unit

class NFW(Density):
    def __init__(self,Rs:float,c:float,**kwargs:Any) -> None:
        super().__init__(Rs=Rs,Rvir=c*Rs,**kwargs)
        self.title = 'NFW'
        self.c = c

    def __repr__(self):
        return f"""NFW density
  - Rs = {self.Rs/self.print_length_units['value']:.4f} {self.print_length_units['name']}
  - c = {self.c:.1f}
  - Mtot = {self.Mtot/self.print_mass_units['value']:.3e} {self.print_mass_units['name']}
  - Mdm = {self.unit_mass/self.print_mass_units['value']:.3e} {self.print_mass_units['name']}
  - Rvir = {self.Rvir/self.print_length_units['value']:.4f} {self.print_length_units['name']}
  - rho_s = {self.rho_s/self.print_density_units['value']:.4f} {self.print_density_units['name']}
  - Tdyn = {self.Tdyn/self.print_time_units['value']:.4f} {self.print_time_units['name']}

  - Rmin = {self.Rmin/self.print_length_units['value']:.4f} {self.print_length_units['name']}
  - Rmax = {self.Rmax/self.print_length_units['value']:.4f} {self.print_length_units['name']}
  - space_steps = {self.space_steps:.0e}"""

    @staticmethod
    @njit
    def calculate_rho(r:FloatOrArray,rho_s:float=1,Rs:float=1,Rvir:float=1) -> FloatOrArray:
        return rho_s/((r/Rs)*(1+(r/Rs))**2)/(1+(r/Rvir)**10)

    def calculate_theoretical_M(self,r:FloatOrArray) -> FloatOrArray:
        x = self(r)
        return cast(FloatOrArray,4*np.pi*self.rho_s*self.Rs**3*(np.log(1+x)-x/(1+x)))

##Plots
    def plot_radius_distribution(self,r_start:float|None=1e-4*kpc,r_end:float|None=None,cumulative:bool=False,
                                 units:Unit=default_units('length'),fig:Figure|None=None,ax:Axes|None=None) -> tuple[Figure,Axes]:
        fig,ax = super().plot_radius_distribution(r_start,r_end or 2*self.Rvir,cumulative=cumulative,units=units,fig=fig,ax=ax)
        r = np.geomspace(r_start or 1e-4*kpc,r_end or 2*self.Rvir,self.space_steps)
        if cumulative:
            ymax = self.mass_cdf(r).max()
        else:
            ymax = self.mass_pdf(r).max()
        ax.vlines(x=[self.Rs/units['value'],self.Rvir/units['value']],ymin=0,ymax=ymax,linestyles='dashed',colors='black')
        ax.text(x=self.Rs/units['value'],y=ymax,s='Rs')
        ax.text(x=self.Rvir/units['value'],y=ymax,s='Rvir')
        return fig,ax
