import numpy as np
import scipy
from functools import partial
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
from numba import njit,prange
from . import utils
from .constants import G,kpc,default_units,km,second

class Density:
    def __init__(self,Rmin,Rmax,Mtot=1,unit_mass=1,space_steps=1e4):
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.space_steps = int(space_steps)
        self.Mtot = Mtot
        self.title = 'Density'
        self.unit_mass = unit_mass

        self.memoization = {}
        self.print_length_units = default_units('length')
        self.print_mass_units = default_units('mass')
        self.print_time_units = default_units('time')
        self.print_density_units = default_units('density')

    def __repr__(self):
            return f"""General mass density function
  - Rmin = {self.Rmin/self.print_length_units['value']:.4f} {self.print_length_units['name']}
  - Rmax = {self.Rmax/self.print_length_units['value']:.4f} {self.print_length_units['name']}
  - space_steps = {self.space_steps:.0e}
  - Mtot = {self.Mtot/self.print_mass_units['value']:.3e} {self.print_mass_units['name']}
  - Mdm = {self.unit_mass/self.print_mass_units['value']:.3e} {self.print_mass_units['name']}"""

    def __call__(self,x):
        return self.to_scale(x)

    def to_scale(self,x):
        return x

    @property
    def Tdyn(self):
        return 0

    def geomspace(self,start=None,end=None,space_steps=None):
        if start is None:
            start = self.Rmin
        if end is None:
            end = self.Rmax
        if space_steps is None:
            space_steps = int(self.space_steps)
        return np.geomspace(start,end,space_steps)

    @property
    def geomspace_grid(self):
        if 'geomspace_grid' not in self.memoization:
            self.memoization['geomspace_grid'] = self.geomspace()
        return self.memoization['geomspace_grid']

    def linspace(self,start=None,end=None,space_steps=None):
        if start is None:
            start = self.Rmin
        if end is None:
            end = self.Rmax
        if space_steps is None:
            space_steps = int(self.space_steps)
        return np.linspace(start=start,stop=end,num=space_steps)

    @property
    def linspace_grid(self):
        if 'linspace_grid' not in self.memoization:
            self.memoization['linspace_grid'] = self.linspace()
        return self.memoization['linspace_grid']

    def rho(self,r):
        return r

    @property
    def rho_grid(self):
        if 'rho_grid' not in self.memoization:
            self.memoization['rho_grid'] = self.rho(self.geomspace_grid)
        return self.memoization['rho_grid']

    def rho_r2(self,r):
        return self.rho(r)*r**2

    @property
    def rho_r2_grid(self):
        if 'rho_r2_grid' not in self.memoization:
            self.memoization['rho_r2_grid'] = self.rho_r2(self.geomspace_grid)
        return self.memoization['rho_r2_grid']

    def M(self,r):
        return r

    @property
    def M_grid(self):
        if 'M_grid' not in self.memoization:
            self.memoization['M_grid'] = self.M(self.geomspace_grid)
        return self.memoization['M_grid']

    def Phi(self,r):
        return r

    @property
    def Phi_grid(self):
        if 'Phi_grid' not in self.memoization:
            self.memoization['Phi_grid'] = self.Phi(self.geomspace_grid)
        return self.memoization['Phi_grid']

    @property
    def Phi0(self):
        if 'Phi0' not in self.memoization:
            self.memoization['Phi0'] = self.Phi(self.Rmax)
        return self.memoization['Phi0']

    def Psi(self,r):
        return self.Phi0 - self.Phi(r)

    @property
    def Psi_grid(self):
        if 'Psi_grid' not in self.memoization:
            self.memoization['Psi_grid'] = self.Psi(self.geomspace_grid)
        return self.memoization['Psi_grid']

    def mass_pdf(self,r):
        mass_pdf = self.rho_r2(r)
        mass_pdf /= np.trapezoid(mass_pdf,r)
        return mass_pdf

    def mass_cdf(self,r):
        return self.M(r)/self.Mtot

    @property
    def pdf(self):
        if 'pdf' not in self.memoization:
            self.memoization['pdf'] = scipy.interpolate.interp1d(self.geomspace_grid,self.mass_pdf(self.geomspace_grid),
                                                                 kind='cubic',bounds_error=False,fill_value=(0,1))
        return self.memoization['pdf']

    @property
    def cdf(self):
        if 'cdf' not in self.memoization:
            self.memoization['cdf'] = scipy.interpolate.interp1d(self.geomspace_grid,self.mass_cdf(self.geomspace_grid),
                                                                 kind='cubic',bounds_error=False,fill_value=(0,1))
        return self.memoization['cdf']

    @property
    def quantile_function(self):
        if 'quantile_function' not in self.memoization:
            r,cdf = utils.joint_clean([self.geomspace_grid,self.mass_cdf(self.geomspace_grid)],['r','cdf'],'cdf')
            self.memoization['quantile_function'] = scipy.interpolate.interp1d(cdf,r,kind='cubic',bounds_error=False,fill_value=(self.Rmin,self.Rmax))
        return self.memoization['quantile_function']

    @property
    def Psi_to_r(self):
        if 'Psi_to_r' not in self.memoization:
            r,Psi = utils.joint_clean([self.geomspace_grid,self.Psi_grid],['r','Psi'],'Psi')
            self.memoization['Psi_to_r'] = scipy.interpolate.interp1d(Psi,r,kind='cubic',bounds_error=False,fill_value=(self.Rmin,self.Rmax))
        return self.memoization['Psi_to_r']

    def Psi_to_rho(self,Psi):
        return self.rho(self.Psi_to_r(Psi))

    @property
    def drhodPsi(self):
        if 'drhodPsi' not in self.memoization:
            self.memoization['drhodPsi'] = partial(utils.derivate,y_fn=self.Psi_to_rho)
        return self.memoization['drhodPsi']

    @property
    def drho2dPsi2(self):
        if 'drho2dPsi2' not in self.memoization:
            self.memoization['drho2dPsi2'] = partial(utils.derivate2,y_fn=self.Psi_to_rho)
        return self.memoization['drho2dPsi2']

    def calculate_f(self,E,disable=True):
        scalar_input = np.isscalar(E)
        if scalar_input:
            E = np.array([E])
        Psi = np.linspace(self.Psi_grid.min(),E.max(),1000)
        drho2dPsi2 = self.drho2dPsi2(Psi)
        integral = np.zeros_like(E)
        for i,e in enumerate(tqdm(E,disable=disable)):
            mask = (Psi < e)
            integral[i] = scipy.integrate.trapezoid(x=Psi[mask],y=drho2dPsi2[mask]/np.sqrt(e-Psi[mask]))
        if scalar_input:
            integral = integral[0]
            E = E[0]
        return 1/(self.unit_mass*np.sqrt(8)*np.pi**2)*(self.drhodPsi(0)/np.sqrt(E)+integral)

    def E(self,r,v):
        return self.Psi(r)-v**2/2

    @property
    def f(self):
        if 'f' not in self.memoization:
            E = np.linspace(0,self.Psi_grid.max(),int(1e3))[1:]
            fs = self.calculate_f(E)
            self.memoization['f'] = scipy.interpolate.interp1d(E,fs,kind='cubic',bounds_error=False,fill_value=(0,0))
        return self.memoization['f']

## Roll initial setup

    def roll_r(self,n_particles):
        rolls = np.random.rand(n_particles)
        return self.quantile_function(rolls)

    def roll_v_slow(self,r,num=1000):
        Psi = self.Psi(r)
        vs = np.linspace(np.zeros_like(r),np.sqrt(2*Psi),num=num)
        pdf = vs**2*self.f(Psi-vs**2/2)
        cdf = np.cumsum(pdf,axis=0)
        cdf /= cdf[-1]
        indices = np.repeat([np.arange(num)],len(r),0).T
        x = np.random.rand(len(r))
        indices = indices*(cdf <= x)
        indices = indices.max(axis=0)
        return vs[indices,np.arange(len(r))]

    @staticmethod
    @njit(parallel=True)
    def roll_v_fast(Psi,E_grid,f_grid,num=100000):
        output = np.empty_like(Psi)
        for particle in prange(len(Psi)):
            vs = np.linspace(0,np.sqrt(2*Psi[particle]),num=num)
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

    def roll_v(self,r,num=1000):
        E_grid = np.linspace(0,self.Psi_grid.max(),int(num))[1:]
        f_grid = self.calculate_f(E_grid)
        return self.roll_v_fast(self.Psi(r),E_grid,f_grid,num=num)

    def roll_initial_angle(self,n_particles):
        theta = np.acos(np.random.rand(n_particles)*2-1)
        return theta

##Plots

    def plot_phase_space(self,r_range=None,v_range=None,length_units=default_units('length'),velocity_units=default_units('velocity'),fig=None,ax=None):
        if r_range is None:
            r_range = np.linspace(1e-2,50,200)*kpc
        if v_range is None:
            v_range = np.linspace(0,100,200)*(km/second)
        r,v = np.meshgrid(r_range,v_range)
        f = self.f(self.E(r,v))
        grid = 16*np.pi*r**2*v**2*f
        fig,ax = utils.plot_phase_space(grid,r_range/length_units['value'],v_range/velocity_units['value'],length_units,velocity_units,fig,ax)
        return fig,ax

    def plot_rho(self,r_start,r_end,density_units=default_units('density'),length_units=default_units('length'),fig=None,ax=None):
        if fig is None or ax is None:
            fig,ax = plt.subplots(figsize=(6,5))
        fig.tight_layout()
        ax.grid(True)
        ax.set_title('Density distribution (rho)')
        ax.set_xlabel('r [{name}]'.format(**length_units))
        ax.set_ylabel('Density [{name}]'.format(**density_units))

        r = self.geomspace(r_start,r_end)
        rho = self.rho(r)
        sns.lineplot(x=r/length_units['value'],y=rho/density_units['value'],ax=ax)
        ax.set(xscale='log',yscale='log')
        return fig,ax

    def plot_radius_distribution(self,r_start:None|float=None,r_end:None|float=None,cumulative=False,units=default_units('length'),fig=None,ax=None):
        if fig is None or ax is None:
            fig,ax = plt.subplots(figsize=(6,5))
        fig.tight_layout()
        ax.grid(True)
        if cumulative:
            ax.set_title('Particle cumulative range distribution (cdf)')
        else:
            ax.set_title('Particle range distribution (pdf)')
        ax.set_xlabel('radius [{name}]'.format(**units))
        ax.set_ylabel('Density')

        if r_start is None:
            r_start = self.Rmin
        if r_end is None:
            r_end = self.Rmax

        r = self.geomspace(r_start,r_end)
        if cumulative:
            sns.lineplot(x=r/units['value'],y=self.mass_cdf(r),color='r',ax=ax)
        else:
            sns.lineplot(x=r/units['value'],y=self.mass_pdf(r),color='r',ax=ax)
        return fig,ax

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
        return np.sqrt(self.Rs**3/(G*self.Mtot))

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

    def plot_rho(self,r_start=1e-4,r_end=1e4,density_units=default_units('density'),length_units=default_units('length'),fig=None,ax=None):
        fig,ax = super().plot_rho(r_start,r_end,density_units,length_units,fig,ax)
        ymax = self.rho(r_start)/length_units['value']
        ax.vlines(x=[self.Rs/length_units['value'],self.Rvir/length_units['value']],ymin=0,ymax=ymax,linestyles='dashed',colors='black')
        ax.text(x=self.Rs/length_units['value']*1.3,y=ymax*0.5,s='Rs')
        ax.text(x=self.Rvir/length_units['value']*1.3,y=ymax*0.5,s='Rvir');
        return fig,ax

    def plot_radius_distribution(self,r_start:None|float=1e-4*kpc,r_end=None,cumulative=False,units=default_units('length'),fig=None,ax=None):
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
