import numpy as np
from numba import njit
from typing import Optional
from .density import Density
from ..constants import kpc,default_units,Unit

class NFW(Density):
    def __init__(self,Rs,c,**kwargs):
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
    def calculate_rho(r,rho_s=1,Rs=1,Rvir=1):
        return rho_s/((r/Rs)*(1+(r/Rs))**2)/(1+(r/Rvir)**10)

##Plots
    def plot_radius_distribution(self,r_start:Optional[float]=1e-4*kpc,r_end:Optional[float]=None,cumulative=False,
                                 units:Unit=default_units('length'),fig=None,ax=None):
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
