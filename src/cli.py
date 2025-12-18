import argparse
from typing import Any
from pathlib import Path
from functools import partial

import regex
from astropy.units import Unit, Quantity

from . import utils, distribution


def parse_unit(x: str, whitelist: list[str] | None = None, required_physical_type: str | None = None) -> Unit | str:
    """Checks if the value is a valid unit"""
    if whitelist is not None and x in whitelist:
        return x
    try:
        unit = Unit(str(x))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f'Value is not a valid unit, got {x}, error:\n{e}')
    if required_physical_type is not None and unit.physical_type != required_physical_type:
        raise argparse.ArgumentTypeError(
            f'Value is not of the right unit type, got {x} with physical type {unit.physical_type}, required {required_physical_type}'
        )
    return unit


def parse_from_args(
    whitelist_prefix: str | None = None,
    blacklist_prefix: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Cleanup keyword arguments read through `parser.parse_args()`."""
    params = utils.drop_None(**kwargs)
    if blacklist_prefix is not None:
        for prefix in blacklist_prefix:
            params = {k: v for k, v in params.items() if not k.startswith(prefix)}
    if whitelist_prefix is not None:
        params = {
            regex.sub(f'^{whitelist_prefix}_?', '', k): v
            for k, v in params.items()
            if k.startswith(f'{whitelist_prefix}_') or k == whitelist_prefix
        }
    unit_prarms = [regex.sub('_unit$', '', k) for k in params if k.endswith('_unit')]
    for k in unit_prarms:
        unit = params.pop(f'{k}_unit')
        if k in params:
            params[k] = Quantity(params[k], unit)
    return params


def setup_distribution(i: int, **kwargs: Any) -> tuple[distribution.distribution.Distribution, int]:
    """Setup a distribution from keyword arguments read through `parser.parse_args()`"""
    params = parse_from_args(whitelist_prefix=f'distribution_{i}', **kwargs)
    if 'n_particles' in params:
        n_particles = params.pop('n_particles')
    else:
        n_particles = 0
    if 'scale_radius' in params:
        params['Rs'] = params.pop('scale_radius')
    if 'virial_radius' in params:
        params['Rvir'] = params.pop('virial_radius')
    if 'total_mass' in params:
        params['Mtot'] = params.pop('total_mass')
    if 'density_scale' in params:
        params['rho_s'] = params.pop('density_scale')
    if 'concentration' in params:
        params['c'] = params.pop('concentration')

    distribution_type = params.pop('')

    if distribution_type == 'NFW':
        dist = distribution.nfw.NFW(**params)
    elif distribution_type == 'Hernquist':
        dist = distribution.hernquist.Hernquist(**params)
    elif distribution_type == 'Cored':
        dist = distribution.cored.Cored(**params)
    else:
        raise argparse.ArgumentTypeError('Cannot recognize distribution type')
    return dist, n_particles


parser = argparse.ArgumentParser(description='SIDM simulation')

#######################
##Distributions
#######################

default_distributions = [
    {'distribution_type': 'NFW', 'particle_type': 'dm', 'n_particles': 1e5},
    {'distribution_type': 'Henrquist', 'particle_type': 'baryon', 'n_particles': 0},
]

for i, default_distribution in enumerate(default_distributions):
    parser.add_argument(
        f'--distribution_{i + 1}',
        metavar=f'-d{i + 1}',
        help=f'Type for distribution {i + 1}',
        default=default_distribution['distribution_type'],
        type=str,
        choices=['NFW', 'Henrquist', 'Cored'],
    )
for i, default_distribution in enumerate(default_distributions):
    parser.add_argument(
        f'--distribution_{i + 1}_particle_type',
        metavar=f'-p{i + 1}',
        help=f'Particle type for distribution {i + 1}',
        default=default_distribution['particle_type'],
        type=str,
        choices=['dm', 'baryon'],
    )
for i, default_distribution in enumerate(default_distributions):
    parser.add_argument(
        f'--distribution_{i + 1}_n_particles',
        metavar=f'-n{i + 1}',
        help=f'Number of particles from distribution {i + 1}',
        default=default_distribution['n_particles'],
        type=float,
    )
for i, default_distribution in enumerate(default_distributions):
    parser.add_argument(
        f'--distribution_{i + 1}_scale_radius',
        metavar=f'-rc{i + 1}',
        help=f'Scale radius for distribution {i + 1}. Uneeded if both `virial_radius_{i + 1}` and `concentration_{i + 1}` are provided',
        type=float,
    )
for i, default_distribution in enumerate(default_distributions):
    parser.add_argument(
        f'--distribution_{i + 1}_scale_radius_unit',
        metavar=f'-rc_u{i + 1}',
        help=f'Units for `scale_radius_{i + 1}`. Must be acceptable by astropy.units',
        type=partial(parse_unit, required_physical_type='kpc'),
    )
for i, default_distribution in enumerate(default_distributions):
    parser.add_argument(
        f'--distribution_{i + 1}_virial_radius',
        metavar=f'-rvir{i + 1}',
        help=f'Virial radius for distribution {i + 1}. Uneeded if both `scale_radius_{i + 1}` and `concentration_{i + 1}` are provided',
        type=float,
    )
for i, default_distribution in enumerate(default_distributions):
    parser.add_argument(
        f'--distribution_{i + 1}_virial_radius_unit',
        metavar=f'-rvir_u{i + 1}',
        help=f'Units for `virial_radius_{i + 1}`. Must be acceptable by astropy.units',
        type=partial(parse_unit, required_physical_type='kpc'),
    )
for i, default_distribution in enumerate(default_distributions):
    parser.add_argument(
        f'--distribution_{i + 1}_total_mass',
        metavar=f'-mtot{i + 1}',
        help=f'Total mass for distribution {i + 1}. Uneeded if `rho_s_{i + 1}` is provided',
        type=float,
    )
for i, default_distribution in enumerate(default_distributions):
    parser.add_argument(
        f'--distribution_{i + 1}_total_mass_unit',
        metavar=f'-mtot_u{i + 1}',
        help=f'Units for `total_mass_{i + 1}`. Must be acceptable by astropy.units',
        type=partial(parse_unit, required_physical_type='mass'),
    )
for i, default_distribution in enumerate(default_distributions):
    parser.add_argument(
        f'--distribution_{i + 1}_density_scale',
        metavar=f'-rho_s{i + 1}',
        help=f'Density scale for distribution {i + 1}. Uneeded if `total_mass_{i + 1}` is provided',
        type=float,
    )
for i, default_distribution in enumerate(default_distributions):
    parser.add_argument(
        f'--distribution_{i + 1}_density_scale_unit',
        metavar=f'-rho_s_u{i + 1}',
        help=f'Units for `density_scale_{i + 1}`. Must be acceptable by astropy.units',
        type=partial(parse_unit, required_physical_type='mass density'),
    )
for i, default_distribution in enumerate(default_distributions):
    parser.add_argument(
        f'--distribution_{i + 1}_concentration',
        metavar=f'-c{i + 1}',
        help=f'Concentration parameter for distribution {i + 1}. Uneeded if both `scale_radius_{i + 1}` and `virial_radius_{i + 1}` are provided',
        type=float,
    )

#######################
##Simulation parameters
#######################

parser.add_argument('--dt', metavar='-dt', help='Time step for the simulation', default=1e-3, type=float)
parser.add_argument(
    '--dt_unit',
    metavar='-dt_u',
    help="Units for the time step provided, defaults to the first distribution's dynamical time. Must be acceptable by astropy.units or 'Tdyn'",
    type=partial(parse_unit, whitelist=['Tdyn'], required_physical_type='time'),
)
parser.add_argument(
    '--cleanup_nullish_particles',
    metavar='-clean1',
    help='Whether to remove particles from the halo after each interaction if they are nullish',
    default=True,
    type=bool,
)
parser.add_argument(
    '--cleanup_particles_by_radius',
    metavar='-clean2',
    help='Whether to remove particles from the halo based on their radius (r >= Rmax)',
    default=True,
    type=bool,
)
parser.add_argument(
    '--cleanup_Rmax',
    metavar='-clean2_value',
    help='Maximum radius of the halo, particles outside of this radius get killed off by `cleanup_particles_by_radius`',
    default=300,
    type=float,
)
parser.add_argument(
    '--cleanup_Rmax_unit',
    metavar='-clean2_value_u',
    help='Units for `cleanup_Rmax`. Must be acceptable by astropy.units',
    default='kpc',
    type=partial(parse_unit, required_physical_type='length'),
)
parser.add_argument(
    '--hard_save',
    metavar='-s',
    help='Save simulation to hard drive during the run (and not just at the end)',
    default=True,
    type=bool,
)
parser.add_argument('--save_path', metavar='-path', help='Save path for the simulation', type=Path, required=True)
parser.add_argument(
    '--save_every_n_steps',
    metavar='-save_n',
    help='How often should a snapshot be saved, in `dt` time-step units (integer)',
    type=int,
)
parser.add_argument(
    '--save_every_time',
    metavar='-save_t',
    help='How often should a snapshot be saved, in time units',
    type=float,
)
parser.add_argument(
    '--save_every_time_unit',
    metavar='-save_t_u',
    help='Units for `save_every_time`. Must be acceptable by astropy.units',
    default='Myr',
    type=partial(parse_unit, required_physical_type='time'),
)
parser.add_argument(
    '--bootstrap_steps',
    metavar='-btsrp',
    help='Number of bootstrap rounds to perform before scattering begins. Time only begins counting after the bootstrap steps',
    default=100,
    type=float,
)
parser.add_argument(
    '--seed',
    metavar='-seed',
    help='Seed for the random number generator',
    type=int,
)
parser.add_argument(
    '--max_allowed_subdivisions',
    metavar='-max_subdivs',
    help='Maximum number of subdivisions allowed in each step. Set to 1 to avoid all logic',
    default=1,
    type=int,
)
parser.add_argument(
    '--subdivide_on_scatter_chance',
    metavar='-subdiv_sidm',
    help='Whether to subdivide based on the scatter chance. Only relevant if `max_allowed_subdivision` is not `1`. If multiple subdivisions logic are provided, take the greater constraint',
    default=False,
    type=bool,
)
parser.add_argument(
    '--subdivide_on_gravitational_step',
    metavar='-subdiv_dyn',
    help='Whether to subdivide based on the ratio of vr*dt to the spacing to the nearest neighbor. Only relevant if `max_allowed_subdivision` is not `1`. If multiple subdivisions logic are provided, take the greater constraint',
    default=True,
    type=bool,
)
parser.add_argument(
    '--subdivide_on_startup',
    metavar='-subdiv_start',
    help='Whether to subdivide to the maximum allowed subdivisions until the 1 Gyr mark. Only relevant if `max_allowed_subdivision` is not `1`. If multiple subdivisions logic are provided, take the greater constraint',
    default=False,
    type=bool,
)

#######################
##Leapfrog
#######################
parser.add_argument(
    '--dynamics_max_minirounds',
    metavar='-dyn_max_rnds',
    help='Maximum number of mini-rounds to perform in the adaptive leapfrog integrator',
    type=int,
)
parser.add_argument(
    '--dynamics_first_mini_round',
    metavar='-dyn_fst_rnds',
    help='First mini-round to perform (i.e. start with a non-trivial subdivision)',
    type=int,
)
parser.add_argument(
    '--dynamics_r_convergence_threshold',
    metavar='-dyn_r_thr',
    help='Convergence threshold for the radius in the adaptive leapfrog integrator',
    type=float,
)
parser.add_argument(
    '--dynamics_vr_convergence_threshold',
    metavar='-dyn_vr_thr',
    help='Convergence threshold for the radial velocity in the adaptive leapfrog integrator',
    type=float,
)
parser.add_argument(
    '--dynamics_richardson_extrapolation',
    metavar='-dyn_richardson',
    help='Whether to use Richardson extrapolation',
    type=bool,
)
parser.add_argument(
    '--dynamics_adaptive',
    metavar='-dyn_adaptive',
    help='Whether to use adaptive mini-rounds',
    type=bool,
)
parser.add_argument(
    '--dynamics_grid_window_radius',
    metavar='-dyn_grid',
    help='Radius of the grid window for updating the enclosed mass during the run, in indices',
    type=int,
)
parser.add_argument(
    '--dynamics_raise_warning',
    metavar='-dyn_warn',
    help='Whether to raise a warning if the integrator fails to converge',
    type=bool,
)
parser.add_argument(
    '--dynamics_levi_civita_mode',
    metavar='-dyn_lc_m',
    help='Mode for the Levi-Civita correction',
    type=str,
    choices=['always', 'never', 'adaptive'],
)
parser.add_argument(
    '--dynamics_levi_civita_condition_coefficient',
    metavar='-dyn_lc_cc',
    help='Coefficient for the Levi-Civita condition',
    type=float,
)

#######################
##SIDM
#######################

parser.add_argument(
    '--sidm_sigma',
    metavar='-sidm_s',
    help='Cross section for the SIDM. Set to 0 to disable SIDM',
    default=50,
    type=float,
)
parser.add_argument(
    '--sidm_sigma_unit',
    metavar='-sidm_s_u',
    help='Units for the cross section provided, defaults to cm^2/gram. Must be acceptable by astropy.units',
    default=Unit('cm^2/gram'),
    type=partial(parse_unit, required_physical_type='opacity'),
)
parser.add_argument(
    '--sidm_max_radius_j',
    metavar='-sidm_max_j',
    help='Maximum index radius for partners for scattering',
    type=int,
)
parser.add_argument(
    '--sidm_max_allowed_rounds',
    metavar='-sidm_max_rounds',
    help='Maximum number of allowed rounds for scattering, used to prevent stalling in case of high density',
    type=int,
)
parser.add_argument(
    '--sidm_max_allowed_scatters',
    metavar='-sidm_max_scatters',
    help='Maximum number of allowed scatter events per particle per `dt` time step',
    type=int,
)
parser.add_argument(
    '--sidm_kappa',
    metavar='-sidm_k',
    help='The maximum allowed scattering probability per particle in the `dt` time step. Particles with a higher scattering rate (due to high density mostly) will instead perform `N` scattering rounds over a time step `dt/N` to lower the rate in each round to match `kappa`',
    type=float,
)
parser.add_argument(
    '--sidm_disable_tqdm',
    metavar='-sidm_dsbl_tqdm',
    help='Whether to disable tqdm progress bar',
    type=bool,
)
parser.add_argument(
    '--sidm_tqdm_cutoff',
    metavar='-sidm_tqdm_ctf',
    help='Disable the tqdm progress bar if the number of scattering rounds is less than this value',
    type=int,
)
parser.add_argument(
    '--sidm_tqdm_cutoff_ratio',
    metavar='-sidm_tqdm_ctf_rat',
    help='Disable the tqdm progress bar if the number of scattering rounds is less than the maximum allowed times by this value',
    type=float,
)
