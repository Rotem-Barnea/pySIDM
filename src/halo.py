import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy.typing import NDArray
from typing import Any,Self,Callable
from .spatial_approximation import Lattice
from .density.density import Density
from .background import Mass_Distribution
from . import utils
from .physics import sidm,leapfrog
from .physics.utils import Mass_calculation_methods,get_default_mass_method,M_below,orbit_circular_velocity
from .constants import kpc,km,second,default_units,Unit

class Halo:
    def __init__(self,dt:float,r:NDArray[np.float64],v:NDArray[np.float64],density:Density,time:float=0,n_interactions:int=0,
                 background:Mass_Distribution|None=None,save_steps:NDArray[np.int64]|list[int]|None=None,dynamics_params:leapfrog.Params={},
                 scatter_params:sidm.Params={'sigma':0},sigma:float|None=None,scatter_live_only:bool=False,
                 mass_calculation_method:Mass_calculation_methods|None=None) -> None:
        self.time = time
        self.dt = dt
        self.r = r
        self.v = v
        self.particle_index = np.arange(len(r))
        self.live_particles = np.full(len(r),True)
        self.initial_particles = self.particles.copy()
        self.n_interactions = n_interactions
        self._save_steps:NDArray[np.int64] = np.array([])
        self._saved_states:list[pd.DataFrame] = []
        self.save_steps = np.asarray(save_steps) if save_steps is not None else np.array([])
        self.save_state_indices = np.full(len(self.save_steps),False)
        self.density:Density = density
        self.lattice:Lattice = Lattice.from_density(self.density)
        self.dynamics_params:leapfrog.Params = dynamics_params
        self.scatter_params:sidm.Params = scatter_params
        if sigma is not None:
            self.scatter_params['sigma'] = sigma
        self.scatter_live_only = scatter_live_only
        self.mass_calculation_method:Mass_calculation_methods = get_default_mass_method(mass_calculation_method,self.scatter_params['sigma'])
        self.interactions_track = []
        self.background:Mass_Distribution|None = background

    @classmethod
    def setup(cls,density:Density,steps_per_Tdyn:int,n_particles:int,save_steps:NDArray[np.int64]|list[int]|None=None,save_every:int|None=None,
              total_run_time:float|None=None,**kwargs:Any) -> Self:
        r = density.roll_r(n_particles)
        v = np.vstack(utils.split_3d(density.roll_v(r))).T
        dt:float = density.Tdyn/steps_per_Tdyn
        if save_steps is None and save_every is not None and total_run_time is not None:
            save_steps = cls.calculate_save_steps(save_every,dt,total_run_time)
        return cls(r=r,v=v,dt=dt,density=density,save_steps=save_steps,**kwargs)

    def add_background(self,background:Mass_Distribution) -> None:
        self.background = background
        self.lattice = background.lattice

    @staticmethod
    def calculate_save_steps(save_every:int,dt:float,total_run_time:float) -> NDArray[np.int64]:
        return np.arange(0,int(total_run_time/dt),int(save_every/dt))

    def reset(self) -> None:
        self.time = 0
        self.n_interactions = 0
        self.particle_index = self.initial_particles.index.to_numpy()
        self.r = self.initial_particles.r.to_numpy()
        self.live_particles = self.initial_particles.live.to_numpy()
        self.v = self.initial_particles[['vx','vy','vr']].to_numpy()
        self.interactions_track = []
        self.reset_saved_states()

    @property
    def particles(self) -> pd.DataFrame:
        return pd.DataFrame({'r':self.r,'vx':self.vx,'vy':self.vy,'vr':self.vr,'vp':self.vp,'v_norm':self.v_norm,'live':self.live_particles,
                             'time':self.time},index=self.particle_index)

##Physical properties

    @property
    def unit_mass(self) -> float:
        return self.density.Mtot/len(self.r)

    @property
    def Tdyn(self) -> float:
        return self.density.Tdyn

    @property
    def M(self) -> NDArray[np.float64]:
        halo_mass = M_below(self.r,unit_mass=self.unit_mass,lattice=self.lattice,density=self.density,method=self.mass_calculation_method)
        if self.background is not None:
            background_mass = self.background.at_time(self.time)[self.lattice(self.r).clip(min=0,max=len(self.lattice)-1).astype(np.int64)]
        else:
            background_mass = 0
        return halo_mass + background_mass

    @property
    def orbit_circular_velocity(self):
        return orbit_circular_velocity(self.r,self.M)

    @property
    def vx(self) -> NDArray[np.float64]:
        return self.v[:,0]

    @property
    def vy(self) -> NDArray[np.float64]:
        return self.v[:,1]

    @property
    def vr(self) -> NDArray[np.float64]:
        return self.v[:,2]

    @property
    def vp(self) -> NDArray[np.float64]:
        return np.sqrt(self.vx**2+self.vy**2)

    @property
    def v_norm(self) -> NDArray[np.float64]:
        return np.sqrt(self.vr**2+self.vp**2)

    @property
    def ranks(self) -> NDArray[np.int64]:
        return utils.rank_array(self.r)

    def sort_particles(self) -> None:
        indices = np.argsort(self.r)
        self.r = self.r[indices]
        self.v = self.v[indices]
        self.particle_index = self.particle_index[indices]

##Dynamic evolution

    def step(self,current_time_step:int|None=None) -> None:
        if type(current_time_step) is int and current_time_step in self.save_steps:
            self.save_state(current_time_step)
        if self.scatter_params['sigma'] > 0 or self.mass_calculation_method == 'rank presorted':
            self.sort_particles()
        if self.scatter_params['sigma'] > 0:
            blacklist = list(np.arange(len(self.r))[~self.live_particles]) if self.scatter_live_only else []
            n_interactions,indices = sidm.scatter(r=self.r,v=self.v,blacklist=blacklist,dt=self.dt,m=self.unit_mass,**self.scatter_params)
            self.n_interactions += n_interactions
            self.interactions_track += [self.r[indices]]
        leapfrog.step(r=self.r,v=self.v,M=self.M,live=self.live_particles,dt=self.dt,**self.dynamics_params)
        self.time += self.dt

    def evolve(self,n_time_steps:int|None=None,n_Tdyn:int|None=None,disable_tqdm:bool=False,**kwargs:Any) -> None:
        if n_time_steps is None:
            if n_Tdyn is not None:
                n_time_steps = int(n_Tdyn*self.Tdyn/self.dt)
            else:
                raise ValueError("Either n_time_steps or n_Tdyn must be specified")
        for time_step in tqdm(range(int(n_time_steps)),disable=disable_tqdm):
            self.step(time_step,**kwargs)

##Manage save states

    @property
    def saved_states(self):
        data = pd.concat([state for state,saved in zip(self._saved_states,self.save_state_indices) if saved])
        data = data.reset_index().rename(columns={'index':'particle_index'})
        return data

    @property
    def saved_states_map(self):
        return {step:i for i,step in enumerate(self.save_steps)}

    def reset_saved_states(self) -> None:
        self._saved_states = [self.initial_particles.copy()]*len(self.save_steps)
        self.save_state_indices = np.full(len(self.save_steps),False)

    def save_state(self,current_time_step:int) -> None:
        index = self.saved_states_map[current_time_step]
        data = self.particles.copy()
        data['step'] = current_time_step
        self._saved_states[index] = data
        self.save_state_indices[index] = True

##Plots

    @property
    def default_plot_text(self) -> dict[str, dict[str, str]]:
        return {
            'vr':{'title':'radial velocity distribution','xlabel':'radial velocity [{name}]','ylabel':'density'},
            'vx':{'title':'pendicular velocity distribution','xlabel':'pendicular velocity [{name}]','ylabel':'density'},
            'vy':{'title':'pendicular velocity distribution','xlabel':'pendicular velocity [{name}]','ylabel':'density'},
            'vp':{'title':'pendicular velocity distribution','xlabel':'pendicular velocity [{name}]','ylabel':'density'},
            'v_norm':{'title':'velocity distribution','xlabel':'velocity [{name}]','ylabel':'density'},
            'r':{'title':'radius distribution','xlabel':'radius [{name}]','ylabel':'density'}
        }

    def plot_unit_type(self,key:str) -> str:
        if key == 'r':
            return 'length'
        elif key in ['vr','vx','vy','vp','v_norm']:
            return 'velocity'
        return ''

    def plot_r_density_over_time(self,clip:tuple[float,float]|None=None,fig:Figure|None=None,ax:Axes|None=None) -> tuple[Figure,Axes]:
        steps = np.array(list(self.saved_states_map.keys()))[self.save_state_indices]
        times = steps * self.dt / self.Tdyn
        if fig is None or ax is None:
            fig,ax = plt.subplots(figsize=(6,5))
        for _,group in self.saved_states.groupby('time'):
            sns.kdeplot(group.r.to_numpy()/kpc,ax=ax,clip=clip)
        ax.set_title('Density progression over time')
        ax.set_xlabel('radius [kpc]')
        fig.legend([f'{t:.1f} Tdyn' for t in times], loc='outside center right');
        fig.tight_layout()
        return fig,ax

    def plot_distribution(self,key:str,data:pd.DataFrame,cumulative:bool=False,absolute:bool=False,title:str|None=None,xlabel:str|None=None,
                          ylabel:str|None=None,fig:Figure|None=None,ax:Axes|None=None) -> tuple[Figure,Axes]:
        x = data[key].to_numpy()
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
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        sns.histplot(x,cumulative=cumulative,ax=ax,stat='density')
        return fig,ax

    def plot_r_distribution(self,data:pd.DataFrame,cumulative:bool=False,add_density:bool=True,**kwargs:Any) -> tuple[Figure,Axes]:
        fig,ax = self.plot_distribution(key='r',data=data,cumulative=cumulative,**kwargs)
        units = default_units(self.plot_unit_type('r'))
        if add_density:
            return self.density.plot_radius_distribution(cumulative=cumulative,units=units,fig=fig,ax=ax)
        return fig,ax

    def plot_phase_space(self,data:pd.DataFrame,r_range:NDArray[np.float64]=np.linspace(1e-2,50,200)*kpc,
                         v_range:NDArray[np.float64]=np.linspace(0,100,200)*(km/second),length_units:Unit=default_units('length'),
                         velocity_units:Unit=default_units('velocity'),fig:Figure|None=None,ax:Axes|None=None) -> tuple[Figure,Axes]:
        r_lattice = Lattice(len(r_range),r_range.min()/length_units['value'],r_range.max()/length_units['value'],log=False)
        v_lattice = Lattice(len(v_range),v_range.min()/velocity_units['value'],v_range.max()/velocity_units['value'],log=False)
        grid = np.zeros((len(v_range),len(r_range)))

        r = data.r.to_numpy()/length_units['value']
        v_norm = np.sqrt(data.vr.to_numpy()**2+data.vp.to_numpy()**2)/velocity_units['value']

        mask = r_lattice.in_lattice(r)*v_lattice.in_lattice(v_norm)
        data = pd.DataFrame({'r':r_lattice(r[mask]),'v_norm':v_lattice(v_norm[mask])})
        data['count'] = 1
        data = data.groupby(['r','v_norm']).agg('count').reset_index()
        grid[data['v_norm'].to_numpy(),data['r'].to_numpy()] = data['count']

        fig,ax = utils.plot_phase_space(grid,r_range,v_range,length_units,velocity_units,fig=fig,ax=ax)
        return fig,ax

    def plot_inner_core_density(self,radius:float=0.2*kpc,time_units:Unit=default_units('time'),xlabel:str='time [{name}]',ylabel:str='#particles',
                                title:str='#particles in inner density',fig:Figure|None=None,ax:Axes|None=None) -> tuple[Figure,Axes]:
        data = self.saved_states.copy()
        data['time'] /= self.Tdyn
        data['in_radius'] = data['r'] < radius
        agg_data = data.groupby('time').in_radius.agg('sum')
        xlabel = xlabel.format(**time_units)
        if fig is None or ax is None:
            fig,ax = plt.subplots(figsize=(6,5))
        fig.tight_layout()
        ax.grid(True)
        sns.lineplot(pd.DataFrame(agg_data),ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return fig,ax

    @staticmethod
    def prep_2d_data(data:pd.DataFrame,radius_range:tuple[float,float],time_range:tuple[float,float]|None=None,x_units:Unit=default_units('length'),
                     time_units:Unit=default_units('Tdyn'),agg_fn:str|Callable[[Any],Any]='count',n_posts:int=100):
        data = data[(data['r'] >= radius_range[0]) * (data['r'] <= radius_range[1])].copy()
        if time_range is not None:
            data = data[(data['time'] >= time_range[0]) * (data['time'] <= time_range[1])].copy()
        data['time'] /= time_units['value']
        lattice = Lattice(n_posts=n_posts,start=data.r.min(),end=data.r.max()*1.1,log=False)
        data['bin'] = lattice.posts[lattice(data.r.to_numpy())]
        agg_data = data.groupby(['time','bin']).output.agg(agg_fn).reset_index()
        r,time = np.meshgrid(lattice.posts,data.time.unique().astype(np.float64))
        pad = pd.DataFrame({'time':time.ravel(),'bin':r.ravel()})
        pad['output'] = np.nan
        agg_data = pd.concat([agg_data,pad]).drop_duplicates(['time','bin']).sort_values(['time','bin'])
        agg_data['bin'] /= x_units['value']
        extent=(r.min()/kpc,r.max()/kpc,time.min(),time.max())
        return agg_data.output.to_numpy().reshape(r.shape),extent

    def plot_density_evolution(self,radius_range:tuple[float,float]=(0,40*kpc),time_range:tuple[float,float]|None=None,
                               length_units:Unit=default_units('length'),time_units:Unit=default_units('Tdyn'),xlabel:str='Radius [{name}]',
                               ylabel:str='Time [{name}]',cbar_label:str='#Particles',**kwargs:Any) -> tuple[Figure,Axes]:
        data = self.saved_states.copy()
        data['output'] = data.r
        if time_units['name'] == 'Tdyn':
            time_units['value'] = self.Tdyn
        grid,extent = self.prep_2d_data(data,radius_range,time_range,length_units,time_units,agg_fn='count')

        return utils.plot_2d(grid=grid,extent=extent,x_units=length_units,y_units=time_units,xlabel=xlabel,ylabel=ylabel,cbar_label=cbar_label,**kwargs)

    def plot_temperature(self,radius_range:tuple[float,float]=(0,40*kpc),time_range:tuple[float,float]|None=None,
                         velocity_units:Unit=default_units('velocity'),time_units:Unit=default_units('Tdyn'),
                         xlabel:str='Radius [{name}]',ylabel:str='Time [{name}]',**kwargs:Any) -> tuple[Figure,Axes]:
        data = self.saved_states.copy()
        data['output'] = data.v_norm**2
        if time_units['name'] == 'Tdyn':
            time_units['value'] = self.Tdyn
        grid,extent = self.prep_2d_data(data,radius_range,time_range,time_units,agg_fn='mean')

        return utils.plot_2d(grid=grid,extent=extent,cbar_units={'value':1,'name':f'{velocity_units['name']}^2'},x_units=velocity_units,
                             y_units=time_units,xlabel=xlabel,ylabel=ylabel,cbar_label='mean temperature (v^2 [{name}])',**kwargs)
