import numpy as np
import scipy
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numba import njit,prange
from numpy.typing import NDArray
from typing import cast,Any
from astropy import units,constants
from astropy.units.typing import UnitLike
from .. import utils,run_units
from ..types import FloatOrArray
# from ..run_units import time,length,velocity,mass,density,G_units,specific_energy,f_units

class Density:
    def __init__(self,Rmin:units.Quantity['length']=1e-4*units.kpc,Rmax:units.Quantity['length']|None=None,Rs:units.Quantity['length']=1*units.kpc,
                 Rvir:units.Quantity['length']=1*units.kpc,Mtot:units.Quantity['mass']=1*units.Msun,unit_mass:units.Quantity['mass']=1*units.Msun,
                 space_steps:float|int=1e4) -> None:
        self.space_steps:int = int(space_steps)
        self.Mtot:units.Quantity['mass'] = Mtot.to(run_units.mass)
        self.title = 'Density'
        self.unit_mass:units.Quantity['mass'] = unit_mass.to(run_units.mass)
        self.Rs:units.Quantity['length'] = Rs.to(run_units.length)
        self.Rvir:units.Quantity['length'] = Rvir.to(run_units.length)
        self.Rmin:units.Quantity['length'] = Rmin.to(run_units.length)
        self.Rmax:units.Quantity['length'] = (Rmax or 85*self.Rs).to(run_units.length)
        self.rho_s:units.Quantity['mass density'] = self.calculate_rho_scale();

        self.memoization = {}

    def __repr__(self):
            return f"""General mass density function
  - Rmin = {self.Rmin:.4f}
  - Rmax = {self.Rmax:.4f}
  - space_steps = {self.space_steps:.0e}
  - Mtot = {self.Mtot:.3e}
  - particle mass = {self.unit_mass:.3e}"""

    def __call__(self,x:FloatOrArray) -> FloatOrArray:
        return self.to_scale(x)

    def to_scale(self,x:FloatOrArray) -> FloatOrArray:
        return (x/self.Rs).value

    @property
    def Tdyn(self) -> units.Unit:
        if 'Tdyn' not in self.memoization:
            self.memoization['Tdyn'] = units.def_unit('Tdyn',np.sqrt(self.Rs**3/(constants.G.to(run_units.G_units)*self.Mtot)).to(run_units.time),
                                                      doc=f'{self.title} dynamic time')
        return self.memoization['Tdyn']

    @Tdyn.setter
    def Tdyn(self,Tdyn:units.Unit) -> None:
        self.memoization['Tdyn'] = Tdyn

    @property
    def geomspace_grid(self) -> units.Quantity['length']:
        if 'geomspace_grid' not in self.memoization:
            self.memoization['geomspace_grid'] = cast(units.Quantity['length'],np.geomspace(self.Rmin,self.Rmax,self.space_steps))
        return self.memoization['geomspace_grid']

    @property
    def linspace_grid(self) -> units.Quantity['length']:
        if 'linspace_grid' not in self.memoization:
            self.memoization['linspace_grid'] = cast(units.Quantity['length'],np.linspace(start=self.Rmin,stop=self.Rmax,num=self.space_steps))
        return self.memoization['linspace_grid']

    @staticmethod
    @njit
    def calculate_rho(r:FloatOrArray,rho_s:float=1,Rs:float=1,Rvir:float=1) -> FloatOrArray:
        return r

    def rho(self,r:units.Quantity['length']) -> units.Quantity['mass density']:
        return units.Quantity(self.calculate_rho(r.to(run_units.length).value,self.rho_s.value,self.Rs.value,self.Rvir.value),run_units.density)

    @property
    def rho_grid(self) -> units.Quantity['mass density']:
        if 'rho_grid' not in self.memoization:
            self.memoization['rho_grid'] = self.rho(self.geomspace_grid)
        return self.memoization['rho_grid']

    def rho_r2(self,r:units.Quantity['length']) -> units.Quantity['linear density']:
        return self.rho(r)*r.to(run_units.length)**2

    @property
    def rho_r2_grid(self) -> units.Quantity['linear density']:
        if 'rho_r2_grid' not in self.memoization:
            self.memoization['rho_r2_grid'] = self.rho_r2(self.geomspace_grid)
        return self.memoization['rho_r2_grid']

    def spherical_rho_integrate(self,r:units.Quantity['length'],use_rho_s:bool=True) -> units.Quantity['mass']:
        rho_s = self.rho_s.value if use_rho_s else 1
        integral = utils.fast_spherical_rho_integrate(np.atleast_1d(r.to(run_units.length).value),self.calculate_rho,rho_s,self.Rs.value,self.Rvir.value)
        return units.Quantity(integral,run_units.mass)

    def M(self,r:units.Quantity['length']) -> units.Quantity['mass']:
        scalar_input = np.isscalar(r)
        M = self.spherical_rho_integrate(r)
        if scalar_input:
            return units.Quantity(np.array(M)[0],run_units.mass)
        return M

    @property
    def M_grid(self) -> units.Quantity['mass']:
        if 'M_grid' not in self.memoization:
            self.memoization['M_grid'] = self.M(self.geomspace_grid)
        return self.memoization['M_grid']

    def calculate_rho_scale(self) -> units.Quantity['mass density']:
        return self.Mtot/(self.spherical_rho_integrate(self.Rmax,False)[0]*run_units.length**3)

    def Phi(self,r:units.Quantity['length']) -> units.Quantity['specific energy']:
        if 'Phi' not in self.memoization:
            r_grid = self.geomspace_grid
            M_grid = self.M(r_grid)
            Phi_grid = -constants.G.to(run_units.G_units).value*scipy.integrate.cumulative_trapezoid(y=M_grid.value/r_grid.value**2,x=r_grid.value,initial=0)
            Phi_grid -= Phi_grid[-1]
            Phi_grid *= -1
            self.memoization['Phi'] = scipy.interpolate.interp1d(r_grid.value,Phi_grid,kind='cubic',bounds_error=False,fill_value=(0,0))
        return units.Quantity(self.memoization['Phi'](r.to(run_units.length).value),run_units.specific_energy)

    @property
    def Phi_grid(self) -> units.Quantity['specific energy']:
        if 'Phi_grid' not in self.memoization:
            self.memoization['Phi_grid'] = self.Phi(self.geomspace_grid)
        return self.memoization['Phi_grid']

    @property
    def Phi0(self) -> units.Quantity['specific energy']:
        if 'Phi0' not in self.memoization:
            self.memoization['Phi0'] = self.Phi(self.Rmax)
        return self.memoization['Phi0']

    def Psi(self,r:units.Quantity['length']) -> units.Quantity['specific energy']:
        return cast(units.Quantity['specific energy'],self.Phi0 - self.Phi(r))

    @property
    def Psi_grid(self) -> units.Quantity['specific energy']:
        if 'Psi_grid' not in self.memoization:
            self.memoization['Psi_grid'] = self.Psi(self.geomspace_grid)
        return self.memoization['Psi_grid']

    def mass_pdf(self,r:units.Quantity['length']) -> FloatOrArray:
        mass_pdf = self.rho_r2(r).value
        mass_pdf /= np.trapezoid(mass_pdf,r.value)
        return mass_pdf

    def mass_cdf(self,r:units.Quantity['length']) -> FloatOrArray:
        return (self.M(r)/self.Mtot).value

    def pdf(self,r:units.Quantity['length']) -> FloatOrArray:
        if 'pdf' not in self.memoization:
            self.memoization['pdf'] = scipy.interpolate.interp1d(self.geomspace_grid.value,self.mass_pdf(self.geomspace_grid),
                                                                 kind='cubic',bounds_error=False,fill_value=(0,1))
        return self.memoization['pdf'](r.to(run_units.length).value)

    def cdf(self,r:units.Quantity['length']) -> FloatOrArray:
        if 'cdf' not in self.memoization:
            self.memoization['cdf'] = scipy.interpolate.interp1d(self.geomspace_grid.value,self.mass_cdf(self.geomspace_grid),
                                                                 kind='cubic',bounds_error=False,fill_value=(0,1))
        return self.memoization['cdf'](r.to(run_units.length).value)

    def quantile_function(self,p:FloatOrArray) -> units.Quantity['length']:
        if 'quantile_function' not in self.memoization:
            rs,cdf = utils.joint_clean([self.geomspace_grid.value,self.mass_cdf(self.geomspace_grid)],['rs','cdf'],'cdf')
            self.memoization['quantile_function'] = scipy.interpolate.interp1d(cdf,rs,kind='cubic',bounds_error=False,
                                                                               fill_value=(self.Rmin.value,self.Rmax.value))
        return units.Quantity(self.memoization['quantile_function'](p),run_units.length)

    def Psi_to_r(self,Psi:units.Quantity['specific energy']) -> units.Quantity['length']:
        if 'Psi_to_r' not in self.memoization:
            r_grid,Psi_grid = utils.joint_clean([self.geomspace_grid.value,self.Psi_grid.value],['r','Psi'],'Psi')
            self.memoization['Psi_to_r'] = scipy.interpolate.interp1d(Psi_grid,r_grid,kind='cubic',bounds_error=False,
                                                                      fill_value=(self.Rmin.value,self.Rmax.value))
        return units.Quantity(self.memoization['Psi_to_r'](Psi.to(run_units.specific_energy).value),run_units.length)

    def Psi_to_rho(self,Psi:units.Quantity['specific energy']) -> units.Quantity['mass density']:
        return self.rho(self.Psi_to_r(Psi))

    def drhodPsi(self,Psi:units.Quantity['specific energy']) -> units.Quantity:
        return utils.quantity_derivate(Psi,self.Psi_to_rho)

    def drho2dPsi2(self,Psi:units.Quantity['specific energy']) -> units.Quantity:
        return utils.quantity_derivate2(Psi,self.Psi_to_rho)

    def calculate_f(self,E:units.Quantity['specific energy']) -> units.Quantity[run_units.f_units]:
        scalar_input = np.isscalar(E)
        Psi = cast(units.Quantity['specific energy'],np.linspace(self.Psi_grid.min(),E.to(run_units.specific_energy).max(),1000))
        drho2dPsi2 = self.drho2dPsi2(Psi)
        integral = units.Quantity(np.zeros_like(np.atleast_1d(E.value)),drho2dPsi2.unit*np.sqrt(1*run_units.specific_energy))
        for i,e in enumerate(np.atleast_1d(E)):
            mask = (Psi < e)
            integral[i] = scipy.integrate.trapezoid(x=Psi[mask].value,y=(drho2dPsi2[mask]/np.sqrt(e-Psi[mask])).value)*integral.unit
        if scalar_input:
            integral = integral[0]
        return 1/(self.unit_mass*np.sqrt(8)*np.pi**2)*(self.drhodPsi(0*run_units.specific_energy)/np.sqrt(E)+integral)

    def E(self,r:units.Quantity['length'],v:units.Quantity['velocity']) -> units.Quantity['specific energy']:
        return cast(units.Quantity['specific energy'],self.Psi(r)-v**2/2)

    def f(self,E:units.Quantity['specific energy']) -> units.Quantity[run_units.f_units]:
        if 'f' not in self.memoization:
            E_grid = cast(units.Quantity['specific energy'],np.linspace(0,self.Psi_grid.max(),int(1e3))[1:])
            f_grid = self.calculate_f(E_grid)
            self.memoization['f'] = scipy.interpolate.interp1d(E_grid,f_grid,kind='cubic',bounds_error=False,fill_value=(0,0))
        return units.Quantity(self.memoization['f'](E.to(run_units.specific_energy).value),run_units.f_units)

## Roll initial setup

    def roll_r(self,n_particles:int|float) -> units.Quantity['length']:
        rolls = np.random.rand(int(n_particles))
        return self.quantile_function(rolls)

    @staticmethod
    @njit(parallel=True)
    def roll_v_fast(Psi:NDArray[np.float64],E_grid:NDArray[np.float64],f_grid:NDArray[np.float64],num:int=100000) -> NDArray[np.float64]:
        output = np.empty_like(Psi)
        for particle in prange(len(Psi)):
            vs_grid = np.linspace(0,np.sqrt(2*Psi[particle]),num=num)
            vs = np.empty_like(vs_grid,dtype=np.float64)
            vs[:] = vs_grid
            pdf = np.zeros_like(vs)
            for i,v in enumerate(vs):
                pdf[i] = v**2*utils.linear_interpolation(E_grid,f_grid,Psi[particle]-v**2/2)
            pdf /= pdf.sum()
            cdf = np.cumsum(pdf)
            p = np.random.rand()
            i = np.searchsorted(cdf,p) - 1
            if i < 0:
                i = 0
            elif i >= len(cdf)-1:
                i = len(cdf)-2
            output[particle] = vs[i]
        return output

    def roll_v(self,r:units.Quantity['length'],num:int=1000) -> units.Quantity['velocity']:
        Psi = self.Psi(r).to(run_units.specific_energy)
        E_grid = cast(units.Quantity['specific energy'],np.linspace(0,self.Psi_grid.max(),int(num))[1:])
        return units.Quantity(self.roll_v_fast(Psi.value,E_grid.value,f_grid=self.calculate_f(E_grid).value,num=num),run_units.velocity)

    def roll_v_3d(self,r:units.Quantity['length'],num:int=1000) -> units.Quantity['velocity']:
        return cast(units.Quantity['velocity'],np.vstack(utils.split_3d(self.roll_v(r))).T)

    def roll_initial_angle(self,n_particles:int) -> NDArray[np.float64]:
        theta = np.acos(np.random.rand(n_particles)*2-1)
        return theta

##Plots

    def plot_phase_space(self,r_range:units.Quantity['length']=np.linspace(1e-2,50,200)*units.kpc,
                         v_range:units.Quantity['velocity']=np.linspace(0,100,200)*units.Unit('km/second'),**kwargs:Any) -> tuple[Figure,Axes]:
        r,v = cast(tuple[units.Quantity['length'],units.Quantity['velocity']],np.meshgrid(r_range,v_range))
        f = self.f(self.E(r,v))
        grid = np.asarray((16*np.pi*r**2*v**2*f).value)
        fig,ax = utils.plot_phase_space(grid,r_range,v_range,**kwargs)
        return fig,ax

    def add_plot_R_markers(self,ax:Axes,ymax:float,x_units:UnitLike='kpc') -> Axes:
        ax.vlines(x=[self.Rs.to(x_units).value,self.Rvir.to(x_units).value],ymin=0,ymax=ymax,linestyles='dashed',colors='black')
        ax.text(x=self.Rs.to(x_units).value,y=ymax,s='Rs')
        ax.text(x=self.Rvir.to(x_units).value,y=ymax,s='Rvir')
        return ax

    def plot_rho(self,r_start:units.Quantity['length']|None=None,r_end:units.Quantity['length']|None=1e4*units.kpc,density_units:UnitLike='Msun/kpc^3',
                 length_units:UnitLike='kpc',fig:Figure|None=None,ax:Axes|None=None) -> tuple[Figure,Axes]:
        fig,ax = utils.setup_plot(fig,ax,title='Density distribution (rho)',xlabel=utils.add_label_unit('Radius',length_units),
                                  ylabel=utils.add_label_unit('Density',density_units))

        r = cast(units.Quantity['length'],np.geomspace(r_start or self.Rmin,r_end or self.Rmax,self.space_steps))
        rho = self.rho(r)
        sns.lineplot(x=r.to(length_units).value,y=rho.to(density_units).value,ax=ax)
        ax.set(xscale='log',yscale='log')

        ax = self.add_plot_R_markers(ax,ymax=rho.max().to(density_units).value,x_units=length_units)
        return fig,ax

    def plot_radius_distribution(self,r_start:units.Quantity['length']|None=None,r_end:units.Quantity['length']|None=None,cumulative:bool=False,
                                 plot_units:UnitLike='kpc',fig:Figure|None=None,ax:Axes|None=None) -> tuple[Figure,Axes]:
        title = 'Particle cumulative range distribution (cdf)' if cumulative else 'Particle range distribution (pdf)'
        fig,ax = utils.setup_plot(fig,ax,title=title,xlabel=utils.add_label_unit('Radius',plot_units),ylabel='Density')

        r = cast(units.Quantity['length'],np.geomspace(r_start or self.Rmin,r_end or self.Rmax,self.space_steps))
        y = self.mass_cdf(r) if cumulative else self.mass_pdf(r)
        sns.lineplot(x=r.to(plot_units).value,y=y,color='r',ax=ax)
        ax = self.add_plot_R_markers(ax,ymax=y.max(),x_units=plot_units)
        return fig,ax
