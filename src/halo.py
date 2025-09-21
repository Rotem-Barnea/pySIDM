import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
import pickle
import scipy
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy.typing import NDArray
from typing import Any,Self,Callable,cast
from astropy import units,table,constants
from astropy.units.typing import UnitLike
from .spatial_approximation import Lattice
from .density.density import Density
from .background import Mass_Distribution
from . import utils,run_units
from .physics import sidm,leapfrog
from .physics.utils import Mass_calculation_methods,get_default_mass_method,M_below,orbit_circular_velocity

class Halo:
    def __init__(self,dt:units.Quantity['time'],r:units.Quantity['length'],v:units.Quantity['velocity'],density:Density,n_interactions:int=0,
                 time:units.Quantity['time']=0*run_units.time,background:Mass_Distribution|None=None,save_steps:NDArray[np.int64]|list[int]|None=None,
                 dynamics_params:leapfrog.Params={},scatter_params:sidm.Params={},sigma:units.Quantity['opacity']=units.Quantity(0,'cm^2/gram'),
                 ensure_energy_conservation:bool=True,scatter_live_only:bool=False,mass_calculation_method:Mass_calculation_methods|None=None,
                 local_density_n_posts:int=5000,interactions_track:list[NDArray[np.float64]]=[],snapshots:table.QTable=table.QTable(),
                 lattice:Lattice|None=None) -> None:
        self.time:units.Quantity['time'] = time.to(run_units.time)
        self.dt:units.Quantity['time'] = dt
        self.density:Density = density
        self.lattice:Lattice = lattice if lattice is not None else Lattice.from_density(self.density)
        self._r:NDArray[np.float64] = r.to(run_units.length).value
        self._v:NDArray[np.float64] = v.to(run_units.velocity).value
        self.particle_index = np.arange(len(r))
        self.live_particles = np.full(len(r),True)
        self.n_interactions = n_interactions
        self.snapshots:table.QTable = snapshots
        self.save_steps = list(save_steps) if save_steps is not None else []
        self.dynamics_params:leapfrog.Params = dynamics_params
        self.scatter_params:sidm.Params = {'sigma':sigma,**scatter_params}
        self.scatter_live_only = scatter_live_only
        self.local_density_n_posts:int = local_density_n_posts
        self.ensure_energy_conservation = ensure_energy_conservation
        self.mass_calculation_method:Mass_calculation_methods = get_default_mass_method(mass_calculation_method,self.scatter_params['sigma'])
        self.interactions_track = interactions_track
        self.background:Mass_Distribution|None = background
        self.initial_particles = self.particles.copy()

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
        self.time = 0*run_units.time
        self.n_interactions = 0
        self.particle_index = np.array(self.initial_particles['particle_index'])
        self._r = self.initial_particles['r'].to(run_units.length).value
        self.live_particles = np.array(self.initial_particles['live'])
        self._v = np.vstack([np.array(self.initial_particles[i]) for i in ['vx','vy','vr']]).T
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
        return self.density.unit_mass

    @property
    def time_step(self) -> units.Unit:
        return units.def_unit('time step',self.dt.to(run_units.time),format={'latex':r'time\ step'})

    @property
    def Tdyn(self) -> units.Unit:
        return self.density.Tdyn

    @property
    def Tdyn_value(self) -> float:
        return cast(float,self.density.Tdyn.to(run_units.time))

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
        return self._M*run_units.mass

    @property
    def v(self) -> units.Quantity['velocity']:
        return self._v*run_units.velocity

    @property
    def r(self) -> units.Quantity['length']:
        return self._r*run_units.length

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
        return units.Quantity(utils.fast_norm(self._v[:,:2]),run_units.velocity)

    @property
    def v_norm(self) -> units.Quantity['velocity']:
        return units.Quantity(utils.fast_norm(self._v),run_units.velocity)

    @property
    def kinetic_energy(self) -> units.Quantity['specific energy']:
        return 0.5*self.v_norm**2

    @property
    def Phi(self) -> units.Quantity['specific energy']:
        indices = np.argsort(self._r)
        integral = scipy.integrate.cumulative_trapezoid((self._M/self._r**2)[indices],self._r[indices],initial=0)[indices]
        return -(constants.G*units.Quantity(integral,'Msun/kpc')).to(run_units.specific_energy)

    @property
    def Psi(self) -> units.Quantity['specific energy']:
        return (self.density.Phi0-self.Phi).to(run_units.specific_energy)

    @property
    def E(self) -> units.Quantity['specific energy']:
        return (self.Psi-self.kinetic_energy).to(run_units.specific_energy)

    @property
    def _local_density(self) -> NDArray[np.float64]:
        return Lattice(self.local_density_n_posts,self._r.min()*0.9,self._r.max()*1.1,log=False).assign_spatial_density(self._r,self.unit_mass.value)

    @property
    def local_density(self) -> units.Quantity['mass density']:
        return units.Quantity(self._local_density,run_units.density)

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
        if self.scatter_params.get('sigma',0) > 0 or self.mass_calculation_method == 'rank presorted':
            self.sort_particles()
        if self.scatter_params.get('sigma',0) > 0:
            blacklist = np.arange(len(self._r))[~self.live_particles] if self.scatter_live_only else np.array([],dtype=np.int64)
            self._v,n_interactions,indices = sidm.scatter(r=self._r,v=self._v,blacklist=blacklist,density=self._local_density,dt=self.dt,m=self.unit_mass,
                                                          **self.scatter_params)
            self.n_interactions += n_interactions
            self.interactions_track += [self._r[indices]]
        Ein = self.E.copy() if self.ensure_energy_conservation else None
        self._r,self._v = leapfrog.step(r=self._r,v=self._v,M=self._M,live=self.live_particles,dt=self.dt,**self.dynamics_params)
        if self.ensure_energy_conservation and Ein is not None:
            self._v *= utils.fast_v_correction(self.Psi.value,Ein.value,self.v_norm.value)
        self.time += self.dt

    def evolve(self,n_steps:int|None=None,t:units.Quantity['time']|None=None,disable_tqdm:bool=False) -> None:
        if n_steps is None:
            if t is not None:
                n_steps = int((t/self.dt).value)
            else:
                raise ValueError("Either n_steps or t must be specified")
        for step in tqdm(range(int(n_steps)),disable=disable_tqdm):
            self.step(step)

##Save/Load

    def to_payload(self) -> tuple[dict[str,Any],dict[str,Any]]:
        payload =  {
            'time':self.time,
            'dt':self.dt,
            'density':self.density,
            'lattice':self.lattice,
            'n_interactions':self.n_interactions,
            'save_steps':self.save_steps,
            'dynamics_params':self.dynamics_params,
            'scatter_params':self.scatter_params,
            'scatter_live_only':self.scatter_live_only,
            'local_density_n_posts':self.local_density_n_posts,
            'ensure_energy_conservation':self.ensure_energy_conservation,
            'mass_calculation_method':self.mass_calculation_method,
            'interactions_track':self.interactions_track,
            'background':self.background,
        }
        tables = {
            'particles':self.particles,
            'initial_particles':self.initial_particles,
            'snapshots':self.snapshots,
        }
        return payload,tables

    def save(self,path:str|Path=Path('halo_state')) -> None:
        path = Path(path)
        path.mkdir(exist_ok=True)
        payload,tables = self.to_payload()
        with open(path/'halo_payload.pkl','wb') as f:
            pickle.dump(payload,f)
        for name,data in tables.items():
            data.write(path/f'{name}.fits',overwrite=True)

    @classmethod
    def load(cls,path:str|Path=Path('halo_state')) -> Self:
        path = Path('halo_state')
        with open(path/'halo_payload.pkl','rb') as f:
            payload = pickle.load(f)
        tables = {name:table.QTable.read(path/f'{name}.fits') for name in ['particles','initial_particles','snapshots']}
        r = tables['particles']['r']
        v = cast(units.Quantity['velocity'],np.vstack([tables['particles']['vx'],tables['particles']['vy'],tables['particles']['vr']]).T)
        output = cls(r=r,v=v,**payload,snapshots=tables['snapshots'])
        output.initial_particles = tables['initial_particles']
        return output

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

    def fill_time_unit(self,unit:UnitLike) -> UnitLike:
        if unit == 'Tdyn':
            return self.Tdyn
        elif unit == 'time step':
            return self.time_step
        return unit

    def plot_r_density_over_time(self,clip:units.Quantity['length']|None=None,x_units:UnitLike='kpc',time_units:UnitLike='Tdyn',
                                 title:str|None='Density progression over time',xlabel:str|None='Radius',ylabel:str|None=None,
                                 fig:Figure|None=None,ax:Axes|None=None) -> tuple[Figure,Axes]:
        time_units = self.fill_time_unit(time_units)
        fig,ax = utils.setup_plot(fig,ax,**utils.drop_None(title=title,xlabel=utils.add_label_unit(xlabel,x_units),ylabel=ylabel))
        legend = []
        clip_tuple = tuple(clip.to(x_units).value) if clip is not None else None
        for group in self.snapshots.group_by('time').groups:
            sns.kdeplot(group['r'].to(x_units).value,ax=ax,clip=clip_tuple)
            legend += [group['time'][0].to(time_units).to_string(format="latex",formatter=".1f")]
        fig.legend(legend,loc='outside center right')
        return fig,ax

    def plot_distribution(self,key:str,data:table.QTable,cumulative:bool=False,absolute:bool=False,title:str|None=None,xlabel:str|None=None,
                          x_range:units.Quantity|None=None,x_plot_range:units.Quantity|None=None,stat:str='density',x_units:UnitLike|None=None,
                          ylabel:str|None=None,fig:Figure|None=None,ax:Axes|None=None,**kwargs:Any) -> tuple[Figure,Axes]:
        x_units = self.plot_unit_type(key,x_units)
        #
        if x_range is not None:
            x = data[key][(data[key]>x_range[0])*(data[key]<x_range[1])].to(x_units)
        else:
            x = data[key].to(x_units)
        if absolute:
            x = np.abs(x)
        params = {**self.default_plot_text(key,x_units),**utils.drop_None(title=title,xlabel=xlabel,ylabel=ylabel)}
        fig,ax = utils.setup_plot(fig,ax,**params,**kwargs)
        sns.histplot(x,cumulative=cumulative,ax=ax,stat=stat)
        if x_plot_range is not None:
            ax.set_xlim(*x_plot_range.to(x_units).value)
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
                                ylabel:str|None='#particles',title:str|None='',fig:Figure|None=None,ax:Axes|None=None) -> tuple[Figure,Axes]:
        data = self.snapshots.copy()
        time_units = self.fill_time_unit(time_units)
        data['time'] = data['time'].to(time_units)
        data['in_radius'] = data['r'] <= radius

        agg_data = utils.aggregate_QTable(data,groupby='time',keys=['in_radius'],agg_fn='sum',final_units={'time':time_units})

        if title == '':
            title = f'#particles in inner core ({radius.to_string(format="latex",formatter=".2f")})'

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
        time_units = self.fill_time_unit(time_units)
        data = self.snapshots.copy()
        data['output'] = data['r']
        grid,extent = self.prep_2d_data(data,radius_range,time_range,length_units,time_units,agg_fn='count')

        return utils.plot_2d(grid=grid,extent=extent,x_units=length_units,y_units=time_units,xlabel=utils.add_label_unit(xlabel,length_units),
                             ylabel=utils.add_label_unit(ylabel,time_units),cbar_label=cbar_label,**kwargs)

    def plot_temperature(self,radius_range:tuple[units.Quantity['length'],units.Quantity['length']]=(0*units.kpc,40*units.kpc),
                         time_range:tuple[units.Quantity['time'],units.Quantity['time']]|None=None,velocity_units:UnitLike='km/second',
                         length_units:UnitLike='kpc',time_units:UnitLike='Tdyn',xlabel:str|None='Radius',ylabel:str|None='Time',
                         cbar_label:str|None='Temperature (velocity std)',**kwargs:Any) -> tuple[Figure,Axes]:
        time_units = self.fill_time_unit(time_units)
        data = self.snapshots.copy()
        data['output'] = data['v_norm']
        grid,extent = self.prep_2d_data(data,radius_range,time_range,length_units,time_units,agg_fn='std')

        return utils.plot_2d(grid=grid,extent=extent,x_units=length_units,y_units=time_units,xlabel=utils.add_label_unit(xlabel,velocity_units),
                             ylabel=utils.add_label_unit(ylabel,time_units),cbar_label=utils.add_label_unit(cbar_label,velocity_units),**kwargs)

    def plot_scattering_location(self,title:str|None='Scattering location distribution within the first {time}, total of {n_interactions} events',
                                 xlabel:str|None='Radius',length_units:UnitLike='kpc',time_units:UnitLike='Gyr',time_format:str='.1f',
                                 figsize:tuple[int,int]=(12,6),fig:Figure|None=None,ax:Axes|None=None) -> tuple[Figure,Axes]:
        xlabel = utils.add_label_unit(xlabel,length_units)
        time_units = self.fill_time_unit(time_units)
        if title is not None:
            title = title.format(time=self.time.to(time_units).to_string(format="latex",formatter=time_format),n_interactions=self.n_interactions)
        fig,ax = utils.setup_plot(fig,ax,figsize=figsize,minorticks=True,**utils.drop_None(title=title,xlabel=xlabel))
        sns.histplot(units.Quantity(np.hstack(self.interactions_track),run_units.length).to(length_units),ax=ax)
        return fig,ax

    def plot_before_after_histogram(self,time_units:UnitLike='Tdyn',time_format:str='.1f',**kwargs:Any) -> tuple[Figure,Axes]:
        time_units = self.fill_time_unit(time_units)
        fig,ax = self.plot_distribution(data=self.initial_particles,**kwargs)
        fig,ax = self.plot_distribution(data=self.particles,fig=fig,ax=ax,**kwargs)
        ax.legend(['start',f'after {self.time.to(time_units).to_string(format="latex",formatter=time_format)}'])
        return fig,ax
