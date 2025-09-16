import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
from typing import Optional
from .spatial_approximation import Lattice
from .density.density import Density
from .background import Mass_Distribution
from . import utils
from .physics import sidm,leapfrog
from .physics.utils import Mass_calculation_methods,get_default_mass_method,M_below,orbit_circular_velocity
from .constants import kpc,km,second,default_units,Unit

class Halo:
    def __init__(self,dt:float,r:np.ndarray,v:np.ndarray,density:Density,time=0,n_interactions=0,background:Optional[Mass_Distribution]=None,
                 save_steps:Optional[np.ndarray|list[int]]=None,mass_calculation_method:Optional[Mass_calculation_methods]=None,
                 default_dynamics_params:leapfrog.Params={},default_scatter_params:sidm.Params={'sigma':0},
                 sigma:Optional[float]=None,scatter_live_only:bool=False):
        self.r = r
        self.v = v
        self.particle_index = np.arange(len(r))
        self.live_particles = np.full(len(r),True)
        self.initial_particles = self.particles.copy()
        self.n_interactions = n_interactions
        self.time = time
        self.dt = dt
        self.save_steps = save_steps
        self.density:Density = density
        self.lattice:Lattice = Lattice.from_density(self.density)
        self.dynamics_params:leapfrog.Params = default_dynamics_params
        self.scatter_params:sidm.Params = default_scatter_params
        if sigma is not None:
            self.scatter_params['sigma'] = sigma
        self.scatter_live_only = scatter_live_only
        self.mass_calculation_method:Mass_calculation_methods = get_default_mass_method(mass_calculation_method,self.scatter_params['sigma'])
        self.interactions_track = []
        self.background:Optional[Mass_Distribution] = background

    @classmethod
    def setup(cls,density:Density,steps_per_Tdyn,n_particles,save_steps:Optional[np.ndarray|list[int]]=None,save_every=None,total_run_time=None,**kwargs):
        r = density.roll_r(n_particles)
        v = np.vstack(utils.split_3d(density.roll_v(r))).T
        dt = density.Tdyn/steps_per_Tdyn
        if save_steps is None and save_every is not None and total_run_time is not None:
            save_steps = cls.calculate_save_steps(save_every,dt,total_run_time)
        return cls(r=r,v=v,dt=dt,density=density,save_steps=save_steps,**kwargs)

    def add_background(self,background:Mass_Distribution):
        self.background = background
        self.lattice = background.lattice

    @staticmethod
    def calculate_save_steps(save_every,dt,total_run_time):
        return np.arange(0,int(total_run_time/dt),int(save_every/dt))

    def reset(self):
        self.time = 0
        self.n_interactions = 0
        self.particle_index = self.initial_particles.index.to_numpy()
        self.r = self.initial_particles.r.to_numpy()
        self.live_particles = self.initial_particles.live.to_numpy()
        self.v = self.initial_particles[['vx','vy','vr']].to_numpy()
        self.interactions_track = []
        self.reset_saved_states()

    @property
    def particles(self):
        return pd.DataFrame({'r':self.r,'vx':self.vx,'vy':self.vy,'vr':self.vr,'vp':self.vp,'v_norm':self.v_norm,'live':self.live_particles},
                            index=self.particle_index)

##Physical properties

    @property
    def unit_mass(self):
        return self.density.Mtot/len(self.r)

    @property
    def Tdyn(self):
        return self.density.Tdyn

    @property
    def M(self):
        halo_mass = M_below(self.r,unit_mass=self.unit_mass,lattice=self.lattice,density=self.density,method=self.mass_calculation_method)
        if self.background is not None:
            background_mass = self.background.quick_at_time(self.time)[self.lattice(self.r).clip(min=0,max=len(self.lattice)-1)]
        else:
            background_mass = 0
        return halo_mass + background_mass

    @property
    def orbit_circular_velocity(self):
        return orbit_circular_velocity(self.r,self.M)

    @property
    def vx(self):
        return self.v[:,0]

    @property
    def vy(self):
        return self.v[:,1]

    @property
    def vr(self):
        return self.v[:,2]

    @property
    def vp(self):
        return np.sqrt(self.vx**2+self.vy**2)

    @property
    def v_norm(self):
        return np.sqrt(self.vr**2+self.vp**2)

    @property
    def ranks(self):
        return utils.rank_array(self.r)

    def sort_particles(self):
        indices = np.argsort(self.r)
        self.r = self.r[indices]
        self.v = self.v[indices]
        self.particle_index = self.particle_index[indices]

##Dynamic evolution

    def step(self,current_time_step:Optional[int]=None):
        if current_time_step in self.save_steps:
            self.save_state(current_time_step)
        if self.scatter_params['sigma'] > 0 or self.mass_calculation_method == 'rank presorted':
            self.sort_particles()
        if self.scatter_params['sigma'] > 0:
            blacklist = np.arange(len(self.r))[~self.live_particles] if self.scatter_live_only else []
            n_interactions,indices = sidm.scatter(r=self.r,v=self.v,blacklist=blacklist,dt=self.dt,m=self.unit_mass,**self.scatter_params)
            self.n_interactions += n_interactions
            self.interactions_track += [self.r[indices]]
        leapfrog.step(r=self.r,v=self.v,M=self.M,live=self.live_particles,dt=self.dt,**self.dynamics_params)
        self.time += self.dt

    def evolve(self,n_time_steps:Optional[int]=None,n_Tdyn:Optional[int]=None,disable_tqdm=False,**kwargs):
        if n_time_steps is None:
            if n_Tdyn is not None:
                n_time_steps = int(n_Tdyn*self.Tdyn/self.dt)
            else:
                raise ValueError("Either n_time_steps or n_Tdyn must be specified")
        for time_step in tqdm(range(int(n_time_steps)),disable=disable_tqdm):
            self.step(time_step,**kwargs)

##Manage save states

    @property
    def save_steps(self):
        return self._save_steps

    @save_steps.setter
    def save_steps(self,save_steps):
        self._save_steps = save_steps if save_steps is not None else np.array([])
        self.reset_saved_states()

    @property
    def saved_states(self):
        data = pd.concat([state for state,saved in zip(self._saved_states,self.save_state_indices) if saved])
        data = data.reset_index().rename(columns={'index':'particle_index'})
        return data

    @property
    def saved_states_map(self):
        return {step:i for i,step in enumerate(self.save_steps)}

    def reset_saved_states(self):
        self._saved_states = [self.initial_particles.copy()]*len(self.save_steps)
        self.save_state_indices = np.full(len(self.save_steps),False)

    def save_state(self,current_time_step):
        index = self.saved_states_map[current_time_step]
        data = self.particles.copy()
        data['step'] = current_time_step
        data['time'] = self.time
        self._saved_states[index] = data
        self.save_state_indices[index] = True

##Plots

    @property
    def default_plot_text(self):
        return {
            'vr':{'title':'radial velocity distribution','xlabel':'radial velocity [{name}]','ylabel':'density'},
            'vx':{'title':'pendicular velocity distribution','xlabel':'pendicular velocity [{name}]','ylabel':'density'},
            'vy':{'title':'pendicular velocity distribution','xlabel':'pendicular velocity [{name}]','ylabel':'density'},
            'vp':{'title':'pendicular velocity distribution','xlabel':'pendicular velocity [{name}]','ylabel':'density'},
            'v_norm':{'title':'velocity distribution','xlabel':'velocity [{name}]','ylabel':'density'},
            'r':{'title':'radius distribution','xlabel':'radius [{name}]','ylabel':'density'}
        }

    def plot_unit_type(self,key):
        if key == 'r':
            return 'length'
        elif key in ['vr','vx','vy','vp','v_norm']:
            return 'velocity'
        return ''

    def plot_r_density_over_time(self,clip=None):
        steps = np.array(list(self.saved_states_map.keys()))[self.save_state_indices]
        times = steps * self.dt / self.Tdyn
        fig,ax = plt.subplots(figsize=(6,5))
        for _,group in self.saved_states.groupby('time'):
            sns.kdeplot(group.r/kpc,ax=ax,clip=clip)
        ax.set_title('Density progression over time')
        ax.set_xlabel('radius [kpc]')
        fig.legend([f'{t:.1f} Tdyn' for t in times], loc='outside center right');
        fig.tight_layout()
        return fig,ax

    def plot_distribution(self,key,data,cumulative=False,absolute=False,title='',xlabel='',ylabel='',fig=None,ax=None):
        x = data[key].copy()
        units = default_units(self.plot_unit_type(key))
        x /= units['value']
        if absolute:
            x = np.abs(x)
        if fig is None or ax is None:
            fig,ax = plt.subplots(figsize=(6,5))
        fig.tight_layout()
        ax.grid(True)
        if not title:
            title = self.default_plot_text.get(key,{}).get('title','').format(**units)
        if not xlabel:
            xlabel = self.default_plot_text.get(key,{}).get('xlabel','').format(**units)
        if not ylabel:
            ylabel = self.default_plot_text.get(key,{}).get('ylabel','').format(**units)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        sns.histplot(x,cumulative=cumulative,ax=ax,stat='density')
        return fig,ax

    def plot_r_distribution(self,data,cumulative=False,add_density=True,**kwargs):
        fig,ax = self.plot_distribution(key='r',data=data,cumulative=cumulative,**kwargs)
        units = default_units(self.plot_unit_type('r'))
        if add_density:
            return self.density.plot_radius_distribution(cumulative=cumulative,units=units,fig=fig,ax=ax)
        return fig,ax

    def plot_phase_space(self,data,r_range=None,v_range=None,length_units:Unit=default_units('length'),velocity_units:Unit=default_units('velocity'),
                         fig=None,ax=None):
        if r_range is None:
            r_range = np.linspace(1e-2,50,200)*kpc/length_units['value']
        if v_range is None:
            v_range = np.linspace(0,100,200)*km/second/velocity_units['value']
        r_lattice = Lattice(len(r_range),r_range.min(),r_range.max(),log=False)
        v_lattice = Lattice(len(v_range),v_range.min(),v_range.max(),log=False)
        grid = np.zeros((len(v_range),len(r_range)))

        r = data.r/length_units['value']
        v_norm = np.sqrt(data.vr**2+data.vp**2)/velocity_units['value']

        mask = r_lattice.in_lattice(r)*v_lattice.in_lattice(v_norm)
        data = pd.DataFrame({'r':r_lattice(r[mask].to_numpy()),'v_norm':v_lattice(v_norm[mask].to_numpy())})
        data['count'] = 1
        data = data.groupby(['r','v_norm']).agg('count').reset_index()
        grid[data['v_norm'].to_numpy(),data['r'].to_numpy()] = data['count']

        fig,ax = utils.plot_phase_space(grid,r_range,v_range,length_units,velocity_units,fig=fig,ax=ax)
        return fig,ax

    def plot_inner_core_density(self,radius=0.2*kpc,time_units:Unit=default_units('time'),xlabel='time [{name}]',ylabel='#particles',
                                title='#particles in inner density',fig=None,ax=None):
        data = self.saved_states.copy()
        data['time'] /= self.Tdyn
        data['in_radius'] = data['r'] < radius
        agg_data = data.groupby('time').in_radius.agg('sum')
        xlabel = xlabel.format(**time_units)
        if fig is None or ax is None:
            fig,ax = plt.subplots(figsize=(6,5))
        fig.tight_layout()
        ax.grid(True)
        sns.lineplot(agg_data,ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return fig,ax

    @staticmethod
    def prep_2d_data(data,radius_cutoff,x_units:Unit,time_units:Unit,agg_fn='count',n_posts=100):
        data['time'] /= time_units['value']
        data = data[data['r'] < radius_cutoff].copy()
        lattice = Lattice(n_posts=n_posts,start=data.r.min(),end=data.r.max()*1.1,log=False)
        data['bin'] = lattice.posts[lattice(data.r.to_numpy())]
        agg_data = data.groupby(['time','bin']).output.agg(agg_fn).reset_index()
        r,time = np.meshgrid(lattice.posts,data.time.unique())
        pad = pd.DataFrame({'time':time.ravel(),'bin':r.ravel()})
        pad['output'] = np.nan
        agg_data = pd.concat([agg_data,pad]).drop_duplicates(['time','bin']).sort_values(['time','bin'])
        agg_data['bin'] /= x_units['value']
        extent=(r.min()/kpc,r.max()/kpc,time.min(),time.max())
        return agg_data.output.to_numpy().reshape(r.shape),extent

    def plot_density_evolution(self,radius_cutoff=40*kpc,length_units:Unit=default_units('length'),time_units:Unit=default_units('Tdyn'),fig=None,ax=None):
        data = self.saved_states.copy()
        data['output'] = data.r
        if time_units['name'] == 'Tdyn':
            time_units['value'] = self.Tdyn
        grid,extent = self.prep_2d_data(data,radius_cutoff,length_units,time_units,agg_fn='count')

        return utils.plot_2d(grid,extent=extent,x_units=length_units,y_units=time_units,fig=fig,ax=ax,xlabel='Radius [{name}]',
                             ylabel='Time [{name}]',cbar_label='#Particles')

    def plot_temperature(self,radius_cutoff=40*kpc,velocity_units:Unit=default_units('velocity'),time_units:Unit=default_units('Tdyn'),fig=None,ax=None):
        data = self.saved_states.copy()
        data['output'] = data.v_norm**2
        if time_units['name'] == 'Tdyn':
            time_units['value'] = self.Tdyn
        grid,extent = self.prep_2d_data(data,radius_cutoff,velocity_units,time_units,agg_fn='mean')

        return utils.plot_2d(grid,extent=extent,fig=fig,ax=ax,cbar_units={'value':1,'name':f'{velocity_units['name']}^2'},x_units=velocity_units,
                             y_units=time_units,xlabel='Radius [{name}]',ylabel='Time [{name}]',cbar_label='mean temperature (v^2 [{name}])')
