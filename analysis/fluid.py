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

# distribution.bundle.HaloBaryonBundle.from_distributions(
#     # dm_kwargs={'total_mass': Quantity(8e7, 'Msun'), 'c': 'From mass', 'r_vir': 'From mass', 'agama_truncation_power': 1},
#     dm_kwargs={'r_s': Quantity(0.3310, 'kpc'), 'rho_s': Quantity(0.0736, 'Msun/pc^3'), 'c': 1000},
#     b_kwargs={'total_mass': Quantity(6.026e5, 'Msun'), 'r_s': Quantity(0.1709, 'kpc')},
#     name='Draco',
# )[0]

fluid = gravothermal_fluid.ode.GravothermalFluid(
    distribution=distribution.NFW(
        # dm_kwargs={'total_mass': Quantity(8e7, 'Msun'), 'c': 'From mass', 'r_vir': 'From mass', 'agama_truncation_power': 1},
        r_s=Quantity(0.3310, 'kpc'),
        rho_s=Quantity(0.0736, 'Msun/pc^3'),
        c=1000,
        agama_truncation_power=20,
        name='Draco',
    ),
    sigma=Quantity(50, 'cm**2/gram'),
    dt=Quantity(1 / 1000, 'Myr') * 10,
    relaxation_params={'relaxation_threshold': 0, 'relaxation_dt_factor': 1 / 1000, 'max_relaxation_iterations': 50},
    # radius=Quantity(np.geomspace(2e-3, 1e2, 400), 'kpc'),
    radius=Quantity(np.geomspace(0.00331, 33.1, 400), 'kpc'),
    CFL=0.3,
)
fluid.reset()
import warnings

with warnings.catch_warnings():
    warnings.simplefilter('error')
    # fluid.evolve(n_steps=5_000_000, save_every_n_steps=1000)
    fluid.evolve(n_steps=1)

fluid.luminosity_gradient[:5]

fluid.pressure[:20]
len(fluid.snapshots)

# fluid.reset()

# fluid.scale.cross_section

# fluid.transfer_heat()

fig, ax = fluid.plot('internal energy', minorticks=True, label='auto time', time_unit='Myr')

# fig, ax = fluid.plot('internal energy', minorticks=True, undersample_snapshots=10, label='auto time', time_unit='Myr')


import pickle

with open('temp.pkl', 'rb') as f:
    d = pickle.load(f)

(fluid.luminosity[2] - fluid.luminosity[1]) / (fluid.enclosed_mass[2] - fluid.enclosed_mass[1]) / (
    (d['L_'][2] - d['L_'][1]) / (d['m_'][2] - d['m_'][1])
)

fluid.luminosity[0] / fluid.enclosed_mass[0]

((d['L_'][2] - d['L_'][1]) / (d['m_'][2] - d['m_'][1]))
d['L_grad_'][:4]

fluid.luminosity[:10] / d['L_'][:10]
d['L_'][:5]
# One thing to do
fluid.luminosity[0] / fluid.enclosed_mass[0] / d['L_grad_'][0]

fluid.luminosity_gradient[:10] / d['L_grad_'][:10]
d['L_grad_'][:10]

# Kinv_smfp = Quantity(50, 'cm^2/g') * self.F_elastic_smfp(self.v / self.w) / (self.b * self.v)
# Kinv_lmfp = 1.0 / (self.a * self.C * self.v * self.p * self.sigma_m * self.F_elastic_lmfp(self.v / self.w))
Keff = 1.0 / (d['Kinv_smfp'] + d['Kinv_lmfp'])
k = 1 / fluid.heat_conduction

# Keff - k

L = np.empty(len(d['r_']))
L[1:-1] = (
    -(d['r_'][1:-1] ** 2)
    * (Keff[1:-1] + Keff[2:])
    / 2.0
    * (d['u_'][2:] - d['u_'][1:-1])
    / ((d['r_'][2:] - d['r_'][:-2]) / 2.0)
)

L[0] = -(d['r_'][0] ** 2) * (Keff[0] + Keff[1]) / 2.0 * (d['u_'][1] - d['u_'][0]) / (d['r_'][1] / 2.0)
L[-1] = 0

L[:5]
f = (d['L'] / d['L_'])[2] / fluid.scale.luminosity
f

f / np.sqrt(3)

(d['scale_t'] / fluid.scale.time).decompose()


d['L'][:5]
fluid.scale(fluid.luminosity, 'luminosity')[:5]
fluid.luminosity[:5]

grad_u = np.zeros(len(fluid.radius))
grad_u[1:-1] = np.diff(fluid.internal_energy) / ((fluid.shell_mass[:-1] + fluid.shell_mass[1:]) / 2)

grad_u[0] = (fluid.internal_energy[1] - fluid.internal_energy[0]) / ((fluid.shell_mass[0] + fluid.shell_mass[1]) / 2)
L = -(fluid.radius**4 * utils.utils.to_edge(fluid.pressure) * grad_u) * utils.safe.inverse(
    utils.utils.to_edge(fluid.heat_conduction) * utils.utils.to_edge(fluid.velocity_dispersion)
)

fluid.scale(L[:5], 'luminosity')

fluid.luminosity_gradient.shape

# fig, ax = plot.setup(xscale='log', yscale='linear', minorticks=True)
fig, ax = fluid.plot('luminosity gradient', minorticks=True)
# sns.lineplot(x=fluid.scale(fluid.radius, 'length'), y=fluid.scale(L, 'luminosity'), ax=ax, label='new')
sns.lineplot(x=d['r'], y=fluid.scale(d['L_grad_'], 'luminosity gradient'), ax=ax, label='other code base')
# sns.lineplot(x=d['r'], y=L, ax=ax, label='kim2')

fig, ax = plot.setup(
    xscale='log',
    ylabel='dL/dM',
    xlabel='Radius',
    x_unit=fluid.scale.length.unit,
    y_unit=fluid.scale.luminosity.unit / fluid.scale.mass.unit,
)
for i, (x, y) in enumerate(zip(fluid.snapshots.shell_center, fluid.snapshots.luminosity_gradient)):
    # mask = fluid.scale(x, 'length') > Quantity(2e-3, 'kpc')
    mask = np.full(len(x), True)
    sns.lineplot(
        x=fluid.scale(x, 'length')[mask],
        y=fluid.scale(y, 'luminosity')[mask] / fluid.scale.mass,
        ax=ax,
        label=f'{i} heat+relaxation steps' if i > 0 else 'initial',
    )


fig, ax = plot.setup(
    xscale='log', ylabel='L', xlabel='Radius', x_unit=fluid.scale.length.unit, y_unit=fluid.scale.luminosity.unit
)
for i, (x, y) in enumerate(zip(fluid.snapshots.radius, fluid.snapshots.luminosity)):
    # mask = fluid.scale(x, 'length') > Quantity(2e-3, 'kpc')
    mask = np.full(len(x), True)
    sns.lineplot(
        x=fluid.scale(x, 'length')[mask],
        y=fluid.scale(y, 'luminosity')[mask],
        ax=ax,
        label=f'{i + 1} heat+relaxation steps' if i > 0 else 'initial',
    )
