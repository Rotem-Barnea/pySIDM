import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy.typing import NDArray
from typing import Any,Self,Callable,cast
from astropy import units,table
from astropy.units.typing import UnitLike
from .spatial_approximation import Lattice
from .density.density import Density
from .background import Mass_Distribution
from . import utils
from .physics import sidm,leapfrog
from .physics.utils import Mass_calculation_methods,get_default_mass_method,M_below,orbit_circular_velocity
from .constants import time,length,velocity,mass

class Halo:
    def __init__(self,dt:units.Quantity['time'],r:units.Quantity['length'],v:units.Quantity['velocity'],density:Density,time:units.Quantity['time']=0*time,
                 n_interactions:int=0,background:Mass_Distribution|None=None,save_steps:NDArray[np.int64]|list[int]|None=None,
                 dynamics_params:leapfrog.Params={},scatter_params:sidm.Params={'sigma':units.Quantity(0,'cm^2/gram')},
                 sigma:units.Quantity['opacity']|None=None,scatter_live_only:bool=False,mass_calculation_method:Mass_calculation_methods|None=None) -> None:
        self.time:units.Quantity['time'] = time
        self.dt:units.Quantity['time'] = dt
        self._r:NDArray[np.float64] = r.to(length).value
        self._v:NDArray[np.float64] = v.to(velocity).value
        self.particle_index = np.arange(len(r))
        self.live_particles = np.full(len(r),True)
        self.initial_particles = self.particles.copy()
        self.n_interactions = n_interactions
        self.snapshots:table.QTable = table.QTable()
        self.save_steps = list(save_steps) if save_steps is not None else []
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
    def setup(cls,density:Density,steps_per_Tdyn:int|float,n_particles:int|float,save_steps:NDArray[np.int64]|list[int]|None=None,
              save_every:units.Quantity['time']|None=None,total_run_time:units.Quantity['time']|None=None,**kwargs:Any) -> Self:
        r = density.roll_r(int(n_particles))
        v = density.roll_v_3d(r)
        dt:units.Quantity = cast(units.Quantity,(density.Tdyn/int(steps_per_Tdyn)))
        if save_steps is None and save_every is not None and total_run_time is not None:
            save_steps = cls.calculate_save_steps(save_every,dt,total_run_time)
        return cls(r=r,v=v,dt=dt,density=density,save_steps=save_steps,**kwargs)

    def add_background(self,background:Mass_Distribution) -> None:
        self.background = background
        self.lattice = background.lattice

    @staticmethod
    def calculate_save_steps(save_every:units.Quantity['time'],dt:units.Quantity['time'],total_run_time:units.Quantity['time']) -> NDArray[np.int64]:
        return np.arange(0,int((total_run_time/dt).value),int((save_every/dt).value))

    def reset(self) -> None:
        self.time = 0*time
        self.n_interactions = 0
        self.particle_index = self.initial_particles.index.to_numpy()
        self._r = self.initial_particles.r.to_numpy().value
        self.live_particles = self.initial_particles.live.to_numpy()
        self._v = self.initial_particles[['vx','vy','vr']].to_numpy().value
        self.interactions_track = []
        self.snapshots = table.QTable()

    @property
    def particles(self) -> table.QTable:
        data = table.QTable({'r':self.r,'vx':self.vx,'vy':self.vy,'vr':self.vr,'vp':self.vp,'v_norm':self.v_norm,'live':self.live_particles,
                             'time':[self.time]*len(self.r),'particle_index':self.particle_index})
        data.add_index('particle_index')
        return data

##Physical properties

    @property
    def unit_mass(self) -> units.Quantity['mass']:
        return self.density.unit_mass/len(self.r)

    @property
    def Tdyn(self) -> units.Unit:
        return self.density.Tdyn

    @property
    def Tdyn_value(self) -> float:
        return cast(float,self.density.Tdyn.to(time))

    @property
    def _M(self) -> NDArray[np.float64]:
        halo_mass = M_below(self.r,unit_mass=self.unit_mass.value,lattice=self.lattice,density=self.density,method=self.mass_calculation_method)
        if self.background is not None:
            background_mass = self.background.at_time(self.time)[self.lattice(self.r).clip(min=0,max=len(self.lattice)-1).astype(np.int64)]
        else:
            background_mass = 0
        return halo_mass + background_mass

    @property
    def M(self) -> units.Quantity['mass']:
        return self._M*mass

    @property
    def v(self) -> units.Quantity['velocity']:
        return self._v*velocity

    @property
    def r(self) -> units.Quantity['length']:
        return self._r*length

    @property
    def orbit_circular_velocity(self) -> units.Quantity['velocity']:
        return orbit_circular_velocity(self.r_Q,self.M_Q)

    @property
    def vx(self) -> units.Quantity['velocity']:
        return cast(units.Quantity['velocity'],self.v[:,0])

    @property
    def vy(self) -> units.Quantity['velocity']:
        return cast(units.Quantity['velocity'],self.v[:,1])

    @property
    def vr(self) -> units.Quantity['velocity']:
        return cast(units.Quantity['velocity'],self.v[:,2])

    @property
    def vp(self) -> units.Quantity['velocity']:
        return cast(units.Quantity['velocity'],np.sqrt(self.vx**2+self.vy**2))

    @property
    def v_norm(self) -> units.Quantity['velocity']:
        return cast(units.Quantity['velocity'],np.sqrt(self.vr**2+self.vp**2))

    @property
    def ranks(self) -> NDArray[np.int64]:
        return utils.rank_array(self._r)

    def sort_particles(self) -> None:
        indices = np.argsort(self._r)
        self._r = self._r[indices]
        self._v = self._v[indices]
        self.particle_index = self.particle_index[indices]

##Dynamic evolution

    def save_snapshot(self,current_time_step:int) -> None:
        data = self.particles.copy()
        data['step'] = current_time_step
        self.snapshots = table.vstack([self.snapshots,data])

    def step(self,current_time_step:int|None=None) -> None:
        if current_time_step in self.save_steps:
            self.save_snapshot(current_time_step)
        if self.scatter_params['sigma'] > 0 or self.mass_calculation_method == 'rank presorted':
            self.sort_particles()
        if self.scatter_params['sigma'] > 0:
            blacklist = list(np.arange(len(self._r))[~self.live_particles]) if self.scatter_live_only else []
            n_interactions,indices = sidm.scatter(r=self._r,v=self._v,blacklist=blacklist,dt=self.dt,m=self.unit_mass,**self.scatter_params)
            self.n_interactions += n_interactions
            self.interactions_track += [self._r[indices]]
        leapfrog.step(r=self._r,v=self._v,M=self._M,live=self.live_particles,dt=self.dt.value,**self.dynamics_params)
        self.time += self.dt

    def evolve(self,n_steps:int|None=None,t:units.Quantity['time']|None=None,disable_tqdm:bool=False) -> None:
        if n_steps is None:
            if t is not None:
                n_steps = int((t/self.dt).value)
            else:
                raise ValueError("Either n_steps or t must be specified")
        for time_step in tqdm(range(int(n_steps)),disable=disable_tqdm):
            self.step(time_step)

##Plots

    def default_plot_text(self,key:str,x_units:UnitLike) -> dict[str,str|None]:
        return {
            'vr':{'title':'Radial velocity distribution','xlabel':utils.add_label_unit('Radial velocity',x_units),'ylabel':'Density'},
            'vx':{'title':'Pendicular velocity distribution','xlabel':utils.add_label_unit('Pendicular velocity',x_units),'ylabel':'Density'},
            'vy':{'title':'Pendicular velocity distribution','xlabel':utils.add_label_unit('Pendicular velocity',x_units),'ylabel':'Density'},
            'vp':{'title':'Pendicular velocity distribution','xlabel':utils.add_label_unit('Pendicular velocity',x_units),'ylabel':'Density'},
            'v_norm':{'title':'Velocity distribution','xlabel':utils.add_label_unit('Velocity',x_units),'ylabel':'Density'},
            'r':{'title':'Radius distribution','xlabel':utils.add_label_unit('Radius',x_units),'ylabel':'Density'}
        }.get(key,{})

    def plot_unit_type(self,key:str,plot_unit:UnitLike|None=None) -> UnitLike:
        if plot_unit is not None:
            return plot_unit
        if key == 'r':
            return units.kpc
        elif key in ['vr','vx','vy','vp','v_norm']:
            return units.Unit('km/second')
        return ''

    def plot_r_density_over_time(self,clip:tuple[float,float]|None=None,x_units:UnitLike='kpc',time_units:UnitLike='Tdyn',
                                 title:str|None='Density progression over time',xlabel:str|None='radius',ylabel:str|None=None,
                                 fig:Figure|None=None,ax:Axes|None=None) -> tuple[Figure,Axes]:
        x_units = units.Unit(cast(str,x_units))
        time_units = self.Tdyn if time_units == 'Tdyn' else units.Unit(cast(str,time_units))
        fig,ax = utils.setup_plot(fig,ax,**utils.drop_None(title=title,xlabel=f'{xlabel} [{x_units:latex}]',ylabel=ylabel))
        legend = []
        for group in self.snapshots.group_by('time'):
            sns.kdeplot(group['r'].to(x_units),ax=ax,clip=clip)
            legend += [group['time'][0].to(time_units).to_string(format="latex",formatter=".1f")]
        fig.legend(legend,loc='outside center right')
        return fig,ax

    def plot_distribution(self,key:str,data:table.QTable,cumulative:bool=False,absolute:bool=False,title:str|None=None,xlabel:str|None=None,
                          x_units:UnitLike|None=None,ylabel:str|None=None,grid:bool=True,fig:Figure|None=None,ax:Axes|None=None) -> tuple[Figure,Axes]:
        x_units = self.plot_unit_type(key,x_units)
        x = data[key].to(x_units)
        if absolute:
            x = np.abs(x)
        fig,ax = utils.setup_plot(fig,ax,grid=grid,**{**self.default_plot_text(key,x_units),**utils.drop_None(title=title,xlabel=xlabel,ylabel=ylabel)})
        sns.histplot(x,cumulative=cumulative,ax=ax,stat='density')
        return fig,ax

    def plot_r_distribution(self,data:table.QTable,cumulative:bool=False,add_density:bool=True,x_units:UnitLike|None=None,**kwargs:Any) -> tuple[Figure,Axes]:
        fig,ax = self.plot_distribution(key='r',data=data,cumulative=cumulative,x_units=x_units,**kwargs)
        if add_density:
            return self.density.plot_radius_distribution(cumulative=cumulative,plot_units=self.plot_unit_type('r',x_units),fig=fig,ax=ax)
        return fig,ax

    def plot_phase_space(self,data:table.QTable,r_range:units.Quantity['length']=np.linspace(1e-2,50,200)*units.kpc,
                         v_range:units.Quantity['velocity']=np.linspace(0,100,200)*units.Unit('km/second'),length_units:UnitLike='kpc',
                         velocity_units:UnitLike='km/second',fig:Figure|None=None,ax:Axes|None=None) -> tuple[Figure,Axes]:
        r_lattice = Lattice(len(r_range),r_range.min().to(length_units).value,r_range.max().to(length_units).value,log=False)
        v_lattice = Lattice(len(v_range),v_range.min().to(velocity_units).value,v_range.max().to(velocity_units).value,log=False)
        grid = np.zeros((len(v_range),len(r_range)))

        r = data['r'].to(length_units).value
        v_norm = data['v_norm'].to(velocity_units).value

        mask = r_lattice.in_lattice(r)*v_lattice.in_lattice(v_norm)
        data_table = pd.DataFrame({'r':r_lattice(r[mask]),'v_norm':v_lattice(v_norm[mask])})
        data_table['count'] = 1
        data_table = data_table.groupby(['r','v_norm']).agg('count').reset_index()
        grid[data_table['v_norm'].to_numpy(),data_table['r'].to_numpy()] = data_table['count']

        return utils.plot_phase_space(grid,r_range,v_range,length_units,velocity_units,fig=fig,ax=ax)

    def plot_inner_core_density(self,radius:units.Quantity['length']=0.2*units.kpc,time_units:UnitLike='Tdyn',xlabel:str|None='time',
                                ylabel:str|None='#particles',title:str|None='#particles in inner density',fig:Figure|None=None,ax:Axes|None=None) -> tuple[Figure,Axes]:
        data = self.snapshots.copy()
        time_units = self.Tdyn if time_units == 'Tdyn' else units.Unit(cast(str,time_units))
        data['time'] = data['time'].to(time_units)
        data['in_radius'] = data['r'] <= radius

        agg_data = utils.aggregate_QTable(data,groupby='time',keys=['in_radius'],agg_fn='sum',final_units={'time':time_units})

        xlabel = f'{xlabel} [{time_units}]' if xlabel is not None else None
        fig,ax = utils.setup_plot(fig,ax,**utils.drop_None(title=title,xlabel=xlabel,ylabel=ylabel))
        sns.lineplot(agg_data.to_pandas(index='time'),ax=ax)
        return fig,ax

    @staticmethod
    def prep_2d_data(data:table.QTable,radius_range:tuple[units.Quantity['length'],units.Quantity['length']],
                     time_range:tuple[units.Quantity['time'],units.Quantity['time']]|None=None,length_units:UnitLike='kpc',
                     time_units:UnitLike='Myr',agg_fn:str|Callable[[Any],Any]='count',n_posts:int=100) -> tuple[NDArray[Any],tuple[units.Quantity['length'],units.Quantity['length'],units.Quantity['time'],units.Quantity['time']]]:
        data = data.copy()
        data['r'] = data['r'].to(length_units)
        data['time'] = data['time'].to(time_units)
        radius_mask = (data['r'] >= radius_range[0])*(data['r'] <= radius_range[1])
        time_mask = (data['time'] >= time_range[0]) * (data['time'] <= time_range[1]) if time_range is not None else np.full_like(radius_mask,True)
        data = cast(table.QTable,data[radius_mask*time_mask])
        lattice = Lattice(n_posts=n_posts,start=data['r'].min().value,end=data['r'].max().value*1.1,log=False)
        data['bin'] = lattice.posts[lattice(data['r'].value)]
        agg_data = pd.DataFrame(data.to_pandas().groupby(['time','bin'])['output'].agg(agg_fn)).reset_index()
        r,time = np.meshgrid(lattice.posts,np.unique(data['time'].value))
        pad = pd.DataFrame({'time':time.ravel(),'bin':r.ravel()})
        pad['output'] = np.nan
        agg_data = pd.concat([agg_data,pad]).drop_duplicates(['time','bin']).sort_values(['time','bin'])
        extent=(units.Quantity(r.min(),length_units),units.Quantity(r.max(),length_units),units.Quantity(time.min(),time_units),units.Quantity(time.max(),time_units))
        return agg_data.output.to_numpy().reshape(r.shape),extent

    def plot_density_evolution(self,radius_range:tuple[units.Quantity['length'],units.Quantity['length']]=(0*units.kpc,40*units.kpc),
                               time_range:tuple[units.Quantity['time'],units.Quantity['time']]|None=None,length_units:UnitLike='kpc',
                               time_units:UnitLike='Tdyn',xlabel:str|None='Radius',ylabel:str|None='Time',
                               cbar_label:str|None='#Particles',**kwargs:Any) -> tuple[Figure,Axes]:
        if time_units == 'Tdyn':
            time_units = self.Tdyn
        data = self.snapshots.copy()
        data['output'] = data['r']
        grid,extent = self.prep_2d_data(data,radius_range,time_range,length_units,time_units,agg_fn='count')

        return utils.plot_2d(grid=grid,extent=extent,x_units=length_units,y_units=time_units,xlabel=utils.add_label_unit(xlabel,length_units),
                             ylabel=utils.add_label_unit(ylabel,time_units),cbar_label=cbar_label,**kwargs)

    def plot_temperature(self,radius_range:tuple[units.Quantity['length'],units.Quantity['length']]=(0*units.kpc,40*units.kpc),
                         time_range:tuple[units.Quantity['time'],units.Quantity['time']]|None=None,velocity_units:UnitLike='km/second',
                         length_units:UnitLike='kpc',time_units:UnitLike='Tdyn',xlabel:str|None='Radius',ylabel:str|None='Time',
                         cbar_label:str|None='Temperature (velocity std)',**kwargs:Any) -> tuple[Figure,Axes]:
        if time_units == 'Tdyn':
            time_units = self.Tdyn
        data = self.snapshots.copy()
        data['output'] = data['v_norm']
        grid,extent = self.prep_2d_data(data,radius_range,time_range,length_units,time_units,agg_fn='std')

        return utils.plot_2d(grid=grid,extent=extent,x_units=length_units,y_units=time_units,xlabel=utils.add_label_unit(xlabel,velocity_units),
                             ylabel=utils.add_label_unit(ylabel,time_units),cbar_label=utils.add_label_unit(cbar_label,velocity_units),**kwargs)
