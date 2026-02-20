import os
import sys

sys.path.append(os.path.join(os.getcwd(), '..'))

import numpy as np
import seaborn as sns
from astropy import constants, cosmology
from astropy.units import Quantity

from src import GravothermalSIDM, plot, types, units, physics, distribution, gravothermal_fluid
from src.utils import utils
from src.halo.halo import Halo
from src.background import BackgroundDistribution
from src.phase_space import PhaseSpace
from src.distribution import example_db

params = example_db.get_db_parameters(name='Draco')

params

dist = distribution.NFW(total_mass=Quantity(8e7, 'Msun'), c='From mass', r_vir='From mass', name='Draco')

dist

g = (params['mass_stellar'] / (4 / 3 * np.pi * (params['r_half_light']) ** 3)).decompose(units.system)
f'{g:.3e}, {dist.rho_s:.3e}'

(g - dist.rho_s) / dist.rho_s


from functools import partial

import scipy


def calculate_from_half_light(
    r_half_light: Quantity['length'],
    mass_half_light: Quantity['mass'],
    projection_factor=1,
    delta=200,
    **kwargs,
):
    def calculate_c(M: float) -> float:
        """Calculate the concentration parameter `c` from the total mass `M` based on Dutton & Maccio (2014) arXiv:1402.7073v2."""
        return 10 ** (1.025 - 0.097 * np.log10((M * cosmology.Planck18.h / Quantity(1e12, 'Msun')).value))

    def mass_eq(rho_s, r_s, c):
        return 4 * np.pi * rho_s * r_s**3 * (np.log(1 + c) - 1 / (1 + 1 / c))

    def crit_mass(r_s, c):
        return 4 / 3 * np.pi * r_s**3 * c**3 * delta * cosmology.Planck18.critical_density0.to(units.density).value

    def equation(
        params: tuple[float, float, float], M_half_light: float, r_half_light: float
    ) -> tuple[float, float, float]:
        """Helper function for the optimizer"""
        r_s, rho_s, total_mass = np.array(params)
        x = projection_factor * r_half_light / r_s
        c = calculate_c(total_mass)

        mass = np.array([total_mass, crit_mass(r_s, c), mass_eq(rho_s, r_s, c)])

        eq1 = (M_half_light - 4 * np.pi * rho_s * r_s**3 * (np.log(1 + x) - 1 / (1 + 1 / x))) / M_half_light
        eq2 = (mass[2] - mass[0]) / total_mass
        eq3 = (mass[1] - mass[0]) / total_mass

        return eq1, eq2

    r_s, rho_s, total_mass = scipy.optimize.least_squares(
        partial(
            equation,
            M_half_light=(m0 := mass_half_light.decompose(units.system).value),
            r_half_light=(r0 := r_half_light.decompose(units.system).value),
        ),
        [r0, m0 / (4 / 3 * np.pi * r0**3), m0],
        **kwargs,
    ).x
    return {
        'r_s': Quantity(r_s, units.length),
        'total_mass': Quantity(total_mass, units.mass),
        'rho_s': Quantity(rho_s, units.density),
        'c': calculate_c(total_mass),
    }


output = calculate_from_half_light(
    r_half_light=params['r_half_light'], mass_half_light=params['mass_half_light'], delta=200, projection_factor=3
)
distribution.NFW(**output)

dist

f'{params["mass_half_light"]:.2e}'

x = (params['r_half_light'] / output['r_s']).decompose()
m = 4 * np.pi * output['rho_s'] * output['r_s'] ** 3 * (np.log(1 + x) - 1 / (1 + 1 / x))

m - params['mass_half_light']
(dist.enclosed_mass(3 * params['r_half_light']) - params['mass_half_light']) / params['mass_half_light']
