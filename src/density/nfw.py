import numpy as np
import scipy
from numba import njit,prange
from .density import Density
from ..constants import G,kpc,default_units,Unit

class NFW(Density):
    def __init__(self,Rs,c,**kwargs):
        super().__init__(**kwargs)
        self.title = 'NFW'
        self.Rs = Rs
        self.c = c

        self.Rvir = self.c*self.Rs
        self.rho_s = self.Mtot/self.calcualte_M(np.array([self.Rmax]),self.Rs,self.Rvir,1)[0]

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

    @property
    def Tdyn(self):
        if 'Tdyn' not in self.memoization:
            self.memoization['Tdyn'] = np.sqrt(self.Rs**3/(G*self.Mtot))
        return self.memoization['Tdyn']

    @Tdyn.setter
    def Tdyn(self,value):
        self.memoization['Tdyn'] = value


    def to_scale(self,x):
        return x/self.Rs

    @staticmethod
    @njit(parallel=True)
    def calcualte_M(r,Rs,Rvir,rho_s,num=10000):
        M_below = np.empty_like(r)
        for i in prange(len(r)):
            x = np.linspace(0,r[i],num)[1:]
            J = 4*np.pi*x**2
            ys = rho_s/((x/Rs)*(1+(x/Rs))**2)/(1+(x/Rvir)**10)
            M_below[i] = np.trapezoid(y=ys*J,x=x)
        return M_below

    def M(self,r):
        scalar_input = np.isscalar(r)
        if scalar_input:
            r = np.array([r])
        M = self.calcualte_M(r,self.Rs,self.Rvir,self.rho_s)
        if scalar_input:
            M = M[0]
        return M

    def rho(self,r):
        x = self(r)
        density = self.rho_s/(x*(1+x)**2)/(1+(r/self.Rvir)**10)
        return density

    def Phi(self,r):
        # x = self(r)
        # return -4*np.pi*G*self.rho_s*self.Rs**2/x*np.log(1+x)
        if 'Phi' not in self.memoization:
            r_grid = self.geomspace_grid
            M_grid = self.M(r_grid)
            Phi_grid = -G*scipy.integrate.cumulative_trapezoid(y=M_grid/r_grid**2,x=r_grid,initial=0)
            Phi_grid -= Phi_grid[-1]
            Phi_grid *= -1
            self.memoization['Phi'] = scipy.interpolate.interp1d(r_grid,Phi_grid,kind='cubic',bounds_error=False,fill_value=(0,0))
        return self.memoization['Phi'](r)

##Plots

    def plot_rho(self,r_start=1e-4,r_end=1e4,density_units:Unit=default_units('density'),length_units:Unit=default_units('length'),fig=None,ax=None):
        fig,ax = super().plot_rho(r_start,r_end,density_units,length_units,fig,ax)
        ymax = self.rho(r_start)/length_units['value']
        ax.vlines(x=[self.Rs/length_units['value'],self.Rvir/length_units['value']],ymin=0,ymax=ymax,linestyles='dashed',colors='black')
        ax.text(x=self.Rs/length_units['value']*1.3,y=ymax*0.5,s='Rs')
        ax.text(x=self.Rvir/length_units['value']*1.3,y=ymax*0.5,s='Rvir');
        return fig,ax

    def plot_radius_distribution(self,r_start:None|float=1e-4*kpc,r_end=None,cumulative=False,units:Unit=default_units('length'),fig=None,ax=None):
        if r_end is None:
            r_end = self.Rvir*2
        fig,ax = super().plot_radius_distribution(r_start,r_end,cumulative=cumulative,units=units,fig=fig,ax=ax)

        r = self.geomspace(r_start,r_end)
        if cumulative:
            ymax = self.mass_cdf(r).max()
        else:
            ymax = self.mass_pdf(r).max()
        ax.vlines(x=[self.Rs/units['value'],self.Rvir/units['value']],ymin=0,ymax=ymax,linestyles='dashed',colors='black')
        ax.text(x=self.Rs/units['value'],y=ymax,s='Rs')
        ax.text(x=self.Rvir/units['value'],y=ymax,s='Rvir')
        return fig,ax
