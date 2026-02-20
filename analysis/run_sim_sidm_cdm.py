import os
import sys

sys.path.append(os.path.join(os.getcwd(), '..'))

import numpy as np
import seaborn as sns
from astropy import constants, cosmology
from astropy.units import Quantity

import agama_wrappers
from src import GravothermalSIDM, plot, types, units, utils, distribution, gravothermal_fluid
from src.halo.halo import Halo
from src.background import BackgroundDistribution
from src.phase_space import PhaseSpace

sigma = Quantity(50, 'cm**2/gram')
# bundle = distribution.bundle.MixedCSIDM(total_mass=Quantity(2e11, 'Msun'), cdm_factor=1)
dist = distribution.NFW(
    total_mass=Quantity(2e11, 'Msun'), c='From mass', r_vir='From mass', name='CDM+SIDM', agama_truncation_power=4
)

dist = distribution.NFW(
    total_mass=Quantity(2e11, 'Msun'),
    c='From mass',
    r_vir='From mass',
    name='CDM+SIDM',
    # agama_truncation_power=4,
    # truncation_power=1,
    backend='python',
)

dist3 = distribution.NFW.from_example('Draco', agama_truncation_power=4)

dist4 = distribution.NFW.from_example('Draco', backend='python')


halo = Halo.setup(distributions=[dist])
halo2 = Halo.setup(distributions=[dist2])
halo.plot.phase_space()

halo2.plot.phase_space()


out = dist2.full_sample(1e5, phase_space_kwargs={'v_range': Quantity(np.linspace(0, 300, 500), 'km/second')})

np.linalg.norm(out[1], axis=1).max().to('km/s')

halo4.plot.phase_space()

#################

fraction = 0

halo = Halo.setup(
    # distributions=bundle.distributions,
    distributions=[dist],
    save_path=f'../run results/{dist.name} single fraction=0 new',
    scatter_params={'sigma': sigma},
    sample_kwargs={'switch_particle_type': ['cdm', fraction]},
    save_every_time=Quantity(50, 'Myr'),
    bootstrap_steps=10000,
)


halo.evolve(
    # until_t=Quantity(500, 'Myr'),
    until_t=Quantity(1, 'Gyr'),
    # optimize_dt_kwargs={'min_factor': 2, 'max_dt': Quantity(17e-3, 'Myr')},
    # early_quit_kwargs={'critical_ratio': 7.8},
)
