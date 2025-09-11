import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
from .spatial_approximation import Lattice
from . import utils,physics
from .constants import kpc,km,second,default_units,Unit

class Halo:
    def __init__(self,dt,r,v,dm_density,b_density=None,is_dm=None,sigma=0,save_steps=None,time=0,simple_radius=1*kpc,n_interactions=0,regulator=1e-10,
                 leapfrog_params={'max_ministeps':1000,'consider_all':True,'kill_divergent':False},scatter_live_only=False,
                 interaction_params={'max_radius_j':10,'rounds':4,'method':'rounds'},
                 mass_calculation_method:physics.utils.Mass_calcualtion_methods='rank presorted'):
        self.r = r
        self.v = v
        if is_dm is None:
            is_dm = np.full(len(r),True)
        self.is_dm = is_dm
        self.particle_index = np.arange(len(r))
        self.live_particles = np.full(len(r),True)
        self.mass_calculation_method:physics.utils.Mass_calcualtion_methods = mass_calculation_method
        self.initial_particles = self.particles.copy()
        self.dt = dt
        self.save_steps = save_steps
        self.lattice = Lattice.from_density(dm_density)
        self.dm_density = dm_density
        self.b_density = b_density
        self.simple_radius = simple_radius
        self.leapfrog_params = leapfrog_params
        self.interaction_params = interaction_params
        self.sigma = sigma
        self.n_interactions = n_interactions
        self.time = time
        self.regulator = regulator
        self.scatter_live_only = scatter_live_only

    @classmethod
    def setup(cls,dm_density,steps_per_Tdyn,n_particles_dm,b_density=None,n_particles_b=0,save_steps=None,save_every=None,total_run_time=None,**kwargs):
        r_dm = dm_density.roll_r(n_particles_dm)
        v_dm = np.vstack(utils.split_3d(dm_density.roll_v(r_dm))).T
        r_b = dm_density.roll_r(n_particles_b)
        v_b = np.vstack(utils.split_3d(dm_density.roll_v(r_b))).T
        r = np.hstack([r_dm,r_b])
        v = np.vstack([v_dm,v_b])
        is_dm = np.hstack([np.full(len(r_dm),True),np.full(len(r_b),False)])
        dt = dm_density.Tdyn/steps_per_Tdyn
        if save_steps is None and save_every is not None and total_run_time is not None:
            save_steps = cls.calculate_save_steps(save_every,dt,total_run_time)
        return cls(r=r,v=v,is_dm=is_dm,dt=dt,dm_density=dm_density,b_density=b_density,save_steps=save_steps,**kwargs)

    @staticmethod
    def calculate_save_steps(save_every,dt,total_run_time):
        return np.arange(0,int(total_run_time/dt),int(save_every/dt))

    def reset(self):
        self.time = 0
        self.n_interactions = 0
        self.particle_index = self.initial_particles.index.to_numpy()
        self.r = self.initial_particles.r.to_numpy()
        self.live_particles = self.initial_particles.live.to_numpy()
        self.is_dm = self.initial_particles.is_dm.to_numpy()
        self.v = self.initial_particles[['vx','vy','vr']].to_numpy()
        self.reset_saved_states()

    @property
    def leapfrog_kwargs(self):
        return {'simple_radius':self.simple_radius,'dt':self.dt,'regulator':self.regulator,**self.leapfrog_params}

    @property
    def interaction_kwargs(self):
        return {'dt':self.dt,'unit_mass':self.unit_mass,'sigma':self.sigma,'regulator':self.regulator,**self.interaction_params}

    @property
    def particles(self):
        return pd.DataFrame({'r':self.r,'vx':self.vx,'vy':self.vy,'vr':self.vr,'vp':self.vp,'v_norm':self.v_norm,'live':self.live_particles,'is_dm':self.is_dm},
                            index=self.particle_index)

##Physical properties

    @property
    def unit_mass(self):
        return self.dm_density.Mtot/len(self.r)

    @property
    def Tdyn(self):
        return self.dm_density.Tdyn

    @property
    def M(self):
        return physics.utils.M_below(self.r,unit_mass=self.unit_mass,lattice=self.lattice,density=self.dm_density,method=self.mass_calculation_method)

    @property
    def orbit_cicular_velocity(self):
        return physics.utils.orbit_cicular_velocity(self.r,self.M)

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

    def step(self,current_time_step=None):
        if current_time_step in self.save_steps:
            self.save_state(current_time_step)
        self.sort_particles()
        if self.sigma > 0:
            blacklist = np.arange(len(self.r))[~self.live_particles] if self.scatter_live_only else []
            self.n_interactions += physics.SIDM.scatter(r=self.r,v=self.v,blacklist=blacklist,**self.interaction_kwargs)
        physics.leapfrog.step(r=self.r,v=self.v,M=self.M,live=self.live_particles,**self.leapfrog_kwargs)
        self.time += self.dt

    def evolve(self,n_time_steps=None,n_Tdyn=None,disable_tqdm=False,**kwargs):
        if n_time_steps is None:
            n_time_steps = int(n_Tdyn*self.Tdyn/self.dt)
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

    def plot_r_distribution(self,data,cumulative=False,add_dm_density=True,**kwargs):
        fig,ax = self.plot_distribution(key='r',data=data,cumulative=cumulative,**kwargs)
        units = default_units(self.plot_unit_type('r'))
        if add_dm_density:
            return self.dm_density.plot_radius_distribution(cumulative=cumulative,units=units,fig=fig,ax=ax)
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
        data = pd.DataFrame({'r':r_lattice(r[mask]),'v_norm':v_lattice(v_norm[mask])})
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

    def plot_density_evolution(self,radius_cutoff=40*kpc,length_units:Unit=default_units('length'),time_units:Unit=default_units('Tdyn'),fig=None,ax=None):
        data = self.saved_states.copy()
        if time_units['name'] == 'Tdyn':
            data['time'] /= self.Tdyn
        else:
            data['time'] /= time_units['value']
        data = data[data['r'] < radius_cutoff]
        data['r_bin'] = data.r
        lattice = Lattice(n_posts=30,start=data.r.min(),end=data.r.max()*1.1,log=False)
        data['bin'] = lattice.posts[lattice(data.r)]
        agg_data = data.groupby(['time','bin']).r_bin.agg('count').reset_index()
        r,time = np.meshgrid(lattice.posts,data.time.unique())
        pad = pd.DataFrame({'time':time.ravel(),'bin':r.ravel()})
        pad['r_bin'] = np.nan
        agg_data = pd.concat([agg_data,pad]).drop_duplicates(['time','bin']).sort_values(['time','bin'])
        agg_data['bin'] /= length_units['value']
        grid = agg_data.r_bin.to_numpy().reshape(r.shape)

        return utils.plot_2d(grid,extent=(r.min()/kpc,r.max()/kpc,time.min(),time.max()),x_units=length_units,y_units=time_units,fig=fig,ax=ax,
                             x_nbins=None,y_nbins=None,xlabel='Radius [{name}]',ylabel='Time [{name}]',cbar_label='#Particles')

    def plot_temperature(self,radius_cutoff=40*kpc,velocity_units:Unit=default_units('velocity'),time_units:Unit=default_units('Tdyn'),fig=None,ax=None):
        data = self.saved_states.copy()
        if time_units['name'] == 'Tdyn':
            data['time'] /= self.Tdyn
        else:
            data['time'] /= time_units['value']
        data = data[data['r'] < radius_cutoff]
        data['temperature'] = data.v_norm**2
        lattice = Lattice(n_posts=30,start=data.r.min(),end=data.r.max()*1.1,log=False)
        data['bin'] = lattice.posts[lattice(data.r)]
        agg_data = data.groupby(['time','bin']).temperature.agg('count').reset_index()
        r,time = np.meshgrid(lattice.posts,data.time.unique())
        pad = pd.DataFrame({'time':time.ravel(),'bin':r.ravel()})
        pad['temperature'] = np.nan
        agg_data = pd.concat([agg_data,pad]).drop_duplicates(['time','bin']).sort_values(['time','bin'])
        agg_data['bin'] /= velocity_units['value']
        grid = agg_data.r_bin.to_numpy().reshape(r.shape)

        return utils.plot_2d(grid,extent=(r.min()/kpc,r.max()/kpc,time.min(),time.max()),x_units=velocity_units,y_units=time_units,fig=fig,ax=ax,
                             cbar_units={'value':1,'name':f'{velocity_units['name']}^2'},x_nbins=None,y_nbins=None,
                             xlabel='Radius [{name}]',ylabel='Time [{name}]',cbar_label='log mean temperature (log v^2 [{name}])')
