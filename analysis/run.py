## To be run in a REPL setup!

# %% Imports + fix path parameter

import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import seaborn as sns
from matplotlib import pyplot as plt

from src import utils,physics
from src.density.nfw import NFW
from src.density.hernquist import Hernquist
from src.halo import Halo
from src.constants import G,Msun,kpc,Myr,cross_section,km,second
from astropy import units as u

# %% Define parameters

# Mtot = 1.0e10 * Msun #Default halo mass in solar masses (Msun).
# Rs = 2.68441 * kpc #kpc
# c = 17
Mtot_dm = 1.15e9 * Msun
Mtot_b = 1e5 * Msun
Rs = 1.18 * kpc
c = 19
sigma = 50 * cross_section

# %% Setup system

n_particles_dm = int(1e5)
dm_density = NFW(Rs=Rs,c=c,Mtot=Mtot_dm,Rmin=1e-4*kpc,Rmax=85*Rs,unit_mass=Mtot_dm/n_particles_dm)
n_particles_b = 0
# n_particles_b = int(Mtot_b/dm_density.unit_mass)
# b_density = Hernquist(Mtot=Mtot_b,Rmin=1e-4*kpc,Rmax=85*Rs,unit_mass=Mtot_b/n_particles_b)
steps_per_Tdyn = 1000
total_run_time = 1100 * dm_density.Tdyn
save_every = 10*dm_density.Tdyn

halo = Halo.setup(dm_density=dm_density,steps_per_Tdyn=steps_per_Tdyn,n_particles_dm=n_particles_dm,n_particles_b=n_particles_b,sigma=sigma,
                  save_every=save_every,total_run_time=total_run_time)
# %% Run

halo.evolve(n_Tdyn=1100)
halo.saved_states.to_csv('saved_states.csv')

print(halo.n_interactions)

# %% Plot run results

fig,ax = halo.plot_inner_core_density()
fig,ax = halo.plot_density_evolution()
fig,ax = halo.plot_temperature()

# %% Plot initial distributions
fig,ax = halo.dm_density.plot_rho()
fig,ax = halo.plot_r_distribution(halo.initial_particles,cumulative=False)
fig,ax = halo.plot_distribution('v_norm',halo.initial_particles)
fig,ax = halo.dm_density.plot_phase_space()
fig,ax = halo.plot_phase_space(halo.initial_particles);
