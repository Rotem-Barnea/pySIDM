## To be run in a REPL setup!

# %% Imports + fix path parameter

import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import seaborn as sns
from matplotlib import pyplot as plt

from src import utils,physics
from src.density import NFW
from src.halo import Halo
from src.constants import G,Msun,kpc,Myr,cross_section,km,second

# %% Define parameters

# Mtot = 1.0e10 * Msun #Default halo mass in solar masses (Msun).
# Rs = 2.68441 * kpc #kpc
# c = 17
Mtot = 1.15e9 * Msun #Default halo mass in solar masses (Msun).
Rs = 1.18 * kpc #kpc
c = 19
sigma = 50 * cross_section

# %% Setup system

n_particles = int(1e5)
nfw = NFW(Rs=Rs,c=c,Mtot=Mtot,Rmin=1e-4*kpc,Rmax=85*Rs,unit_mass=Mtot/n_particles)
steps_per_Tdyn = 1000
total_run_time = 1100 * nfw.Tdyn
save_every = 10*nfw.Tdyn

halo = Halo.setup(n_particles=n_particles,initial_density=nfw,sigma=sigma,steps_per_Tdyn=steps_per_Tdyn,save_every=save_every,total_run_time=total_run_time)

# %% Run

halo.evolve(n_Tdyn=1)

# %% Plot run results

fig,ax = halo.plot_inner_core_density()
fig,ax = halo.plot_2d_density_time()
fig,ax = halo.plot_2d_temperature_time()

# %% Plot initial distributions
fig,ax = halo.initial_density.plot_rho()
fig,ax = halo.plot_r_distribution(halo.initial_particles,cumulative=False)
fig,ax = halo.plot_distribution('v_norm',halo.initial_particles)
fig,ax = halo.initial_density.plot_phase_space()
fig,ax = halo.plot_phase_space(halo.initial_particles);
