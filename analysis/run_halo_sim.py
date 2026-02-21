import os
import sys

sys.path.append(os.path.join(os.getcwd(), '..'))

import numpy as np
import seaborn as sns
from astropy import constants, cosmology
from astropy.units import Quantity

import agama_wrappers
from src import plot, types, units, utils, distribution, gravothermal_fluid
from src.halo.halo import Halo
from src.background import BackgroundDistribution
from src.phase_space import PhaseSpace
from distribution.distribution import Distribution

sigma = Quantity(50, 'cm**2/gram')

# single = distribution.bundle.Bundle(
#     [distribution.NFW(total_mass=Quantity(8e7, 'Msun'), c='From mass', r_vir='From mass', name='Draco')]
# )
single = distribution.bundle.Bundle([distribution.NFW.from_example('default')])

bundle = distribution.bundle.HaloBaryonBundle.from_distributions(
    dm_kwargs={'total_mass': Quantity(8e7, 'Msun'), 'c': 'From mass', 'r_vir': 'From mass'},
    b_kwargs={'total_mass': Quantity(6.026e5, 'Msun'), 'r_s': Quantity(0.1709, 'kpc')},
    name='Draco',
)

halo = Halo.setup(
    distributions=single.distributions,
    save_path=f'../run results/{single.name}',
    scatter_params={'sigma': sigma},
    save_every_time=Quantity(50, 'Myr'),
    bootstrap_steps=10000,
)

halo = Halo.load('../run results/2026-02-21/CDM+SIDM single fraction=0.4 [11127224]')
halo.plot.cumulative_scattering()

halo.evolve(
    until_t=Quantity(40, 'Gyr'),
    optimize_dt_kwargs={'min_factor': 2, 'max_dt': Quantity(17e-3, 'Myr')},
)

halo.save()

from src.halo import run_optimization
from src.tqdm import tqdm

groups = halo.get_particle_states(filter_particle_type='dm', now=False).group_by('time').groups
t = utils.utils.get_columns(groups.keys, ['time'])[0].to('Gyr')
ratio = np.empty(len(t), dtype=np.float64)
for i, group in enumerate(tqdm(groups)):
    ratio[i] = run_optimization.core_density_ratio(
        group['r'],
        utils.utils.slice_closest(halo.initial_particles, 'dm', 'particle_type')['r'],
        inner_core_radius=halo.inner_core_radius,
    )

fig, ax = plot.setup(xlabel='Time', x_unit=t.unit, ylabel=r'$\rho_c / \rho_{c,0}$', yscale='log')
sns.lineplot(x=t, y=ratio, ax=ax)

from typing import Any

import scipy


def fit_params(t, ratio, **kwargs: Any) -> dict[str, Quantity]:
    def equation(params: tuple[float, float]) -> float:
        """Helper function for the optimizer"""
        t_c, theta = np.exp(params)
        return ratio - (1 - t.value / t_c) ** (-theta)

    t_c, theta = np.exp(
        scipy.optimize.least_squares(
            equation,
            np.log(np.array([10, 0.5])),
            bounds=(
                np.log(np.array([t.value.max() * 1.01, 0.01])),
                np.log(np.array([t.value.max() * 100, 10])),
            ),
            **kwargs,
        ).x
    )
    return {'t_c': Quantity(t_c, t.unit), 'theta': -theta}


fit_params(t=t[t > Quantity(2, 'Gyr')], ratio=ratio[t > Quantity(2, 'Gyr')])

c1 = 1.903e-3
c2 = 8.092e-4
alpha = 6 * (c1 - c2) / (2 * c1 - c2)
alpha
zeta = (5 - 2 * alpha) / (3 - alpha)
zeta
theta = 2 / (11 - 7 * zeta)
theta
xi = 3 / 2 * c2 * b
t_c = theta / xi * t_r
t_c / t_r

c1 = 0
c2 = 1
alpha = 6 * (c1 - c2) / (2 * c1 - c2)
alpha
zeta = (5 - 2 * alpha) / (3 - alpha)
theta = 2 / (11 - 7 * zeta)
b = 25 * np.pi / (32 * np.sqrt(6))
xi = 3 / 2 * c2 * b

a = np.sqrt(16 / np.pi)
t_r = 1 / (
    a
    * halo.distributions[1].density_grid[0]
    * halo.distributions[1].velocity_dispersion_grid[0]
    * halo.scatter_params['sigma']
)

theta / xi

t_c = theta / xi * t_r

t[t > Quantity(2, 'Gyr')].value

halo.plot_cumulative_scattering_amount_over_time(undersample=10)

###########################################################################

sigma = Quantity(50, 'cm**2/gram')
single = distribution.bundle.Bundle(
    [distribution.NFW(total_mass=Quantity(1.01e9, 'Msun'), c='From mass', r_vir='From mass', name='Fornax shp')]
)
bundle = distribution.bundle.HaloBaryonBundle.from_distributions(
    dm_kwargs={'total_mass': Quantity(1.01e9, 'Msun'), 'c': 'From mass', 'r_vir': 'From mass'},
    b_kwargs={'total_mass': Quantity(3.89e7, 'Msun'), 'r_s': Quantity(0.615, 'kpc')},
    name='Fornax Shp',
)

halo = Halo.setup(
    # distributions=bundle.distributions,
    distributions=single.distributions,
    # distribution_as_background=1,
    # save_path=f'../run results/{bundle.name} static baryons 2',
    # save_path=f'../run results/{bundle.name} dynamic baryons 2',
    save_path=f'../run results/{single.name} dm only 2',
    scatter_params={'sigma': sigma},
)

# halo = Halo.load('../run results/Draco static baryons 3')

halo.evolve(
    until_t=Quantity(15, 'Gyr'),
    optimize_dt_kwargs={'min_factor': 2, 'max_dt': Quantity(17e-3, 'Myr')},
    early_quit_kwargs={},
    # early_quit_kwargs=None,
)

###########################################################################

sigma = Quantity(50, 'cm**2/gram')
single = distribution.bundle.Bundle([distribution.NFW.from_example(name='Daneng2024:DM11+baryon')])
bundle = distribution.bundle.HaloBaryonBundle.from_example('Daneng2024:DM11+baryon')

halo = Halo.setup(
    # distributions=bundle.distributions,
    distributions=single.distributions,
    # distribution_as_background=1,
    # save_path=f'../run results/{bundle.name} static baryons',
    # save_path=f'../run results/{bundle.name} dynamic baryons 2',
    save_path=f'../run results/{single.name} dm only',
    scatter_params={'sigma': sigma},
)

# halo = Halo.load('../run results/Draco static baryons 3')

halo.evolve(
    until_t=Quantity(15, 'Gyr'),
    optimize_dt_kwargs={'min_factor': 2, 'max_dt': Quantity(17e-3, 'Myr')},
    early_quit_kwargs={},
    # early_quit_kwargs=None,
)
halo.plot_cumulative_scattering_amount_over_time()
