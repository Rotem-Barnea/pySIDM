import numpy as np
import scipy
from numba import njit,prange
from .density import Density
from ..constants import G

class Hernquist(Density):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.title = 'Hernquist'
        self.rho_s = self.Mtot/self.calcualte_M(np.array([self.Rmax]),1)[0]

    def __repr__(self):
        return f"""Hernquist density
  - Mtot = {self.Mtot/self.print_mass_units['value']:.3e} {self.print_mass_units['name']}
  - rho_s = {self.rho_s/self.print_density_units['value']:.4f} {self.print_density_units['name']}
  - Tdyn = {self.Tdyn/self.print_time_units['value']:.4f} {self.print_time_units['name']}

  - Rmin = {self.Rmin/self.print_length_units['value']:.4f} {self.print_length_units['name']}
  - Rmax = {self.Rmax/self.print_length_units['value']:.4f} {self.print_length_units['name']}
  - space_steps = {self.space_steps:.0e}"""

    @staticmethod
    @njit(parallel=True)
    def calcualte_M(r,rho_s,num=10000):
        M_below = np.empty_like(r)
        for i in prange(len(r)):
            x = np.linspace(0,r[i],num)[1:]
            J = 4*np.pi*x**2
            ys = rho_s/(2*np.pi*x*(1+x)**3)
            M_below[i] = np.trapezoid(y=ys*J,x=x)
        return M_below

    def M(self,r):
        scalar_input = np.isscalar(r)
        if scalar_input:
            r = np.array([r])
        M = self.calcualte_M(r,self.rho_s)
        if scalar_input:
            M = M[0]
        return M

    def rho(self,r):
        return self.rho_s/(2*np.pi*r*(1+r)**3)

    def Phi(self,r):
        if 'Phi' not in self.memoization:
            r_grid = self.geomspace_grid
            M_grid = self.M(r_grid)
            Phi_grid = -G*scipy.integrate.cumulative_trapezoid(y=M_grid/r_grid**2,x=r_grid,initial=0)
            Phi_grid -= Phi_grid[-1]
            Phi_grid *= -1
            self.memoization['Phi'] = scipy.interpolate.interp1d(r_grid,Phi_grid,kind='cubic',bounds_error=False,fill_value=(0,0))
        return self.memoization['Phi'](r)
