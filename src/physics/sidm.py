import numpy as np
from numba import njit, prange
from typing import TypedDict
from numpy.typing import NDArray
from astropy import units
from .. import utils, run_units, physics


class Params(TypedDict, total=False):
    max_radius_j: int
    max_interactions_per_mini_timestep: int
    max_allowed_rounds: int | None
    kappa: float
    density_accuracy_coef: float
    sigma: units.Quantity[run_units.cross_section]


default_params: Params = {
    'max_radius_j': 10,
    'max_interactions_per_mini_timestep': 10000,
    'max_allowed_rounds': None,
    'kappa': 0.002,
    'density_accuracy_coef': 0.1,
    'sigma': units.Quantity(0, 'cm^2/gram').to(run_units.cross_section),
}


def normalize_params(params: Params, add_defaults: bool = False) -> Params:
    if add_defaults:
        params = {**default_params, **params}
    if 'sigma' in params:
        params['sigma'] = params['sigma'].to(run_units.cross_section)
    return params


def t_scatter(
    local_density: units.Quantity['mass density'],
    sigma: units.Quantity[run_units.cross_section] | None,
    v_norm: units.Quantity['velocity'],
) -> units.Quantity['time']:
    if sigma is None or sigma.value == 0:
        return units.Quantity(np.full(len(local_density), np.inf), run_units.time)
    return (1 / (local_density * sigma * v_norm)).to(run_units.time)


@njit
def scatter_pair_kinematics(v0: NDArray[np.float64], v1: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # Calculate relative velocity and center of mass velocity
    v_rel = v0 - v1
    v_cm = (v0 + v1) / 2
    v_rel_norm = np.linalg.norm(v_rel)

    # Skip if particles have same velocity (no collision)
    if v_rel_norm < 1e-10:
        return v0, v1

    # Generate random scattering angles (isotropic)
    cos_theta = np.random.rand() * 2 - 1
    sin_theta = np.sqrt(1 - cos_theta**2)
    phi = np.random.rand() * 2 * np.pi
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Create orthonormal basis vectors
    # axis1: along relative velocity direction
    axis1 = v_rel / v_rel_norm

    # axis2: first perpendicular vector
    if np.abs(axis1[2]) < 0.9:  # if relative velocity not too close to radial (z-axis)
        temp = np.array([0.0, 0.0, 1.0])  # use radial direction as reference
    else:  # if relative velocity close to radial, use x-axis
        temp = np.array([1.0, 0.0, 0.0])
    axis2 = np.cross(axis1, temp)  # Cross product to get perpendicular vector
    axis2 = axis2 / np.linalg.norm(axis2)
    # axis3: complete the orthonormal basis
    axis3 = np.cross(axis1, axis2)

    # New relative velocity direction
    v_rel_new_dir = axis1 * cos_theta + (axis2 * cos_phi + axis3 * sin_phi) * sin_theta

    # Keep same magnitude (elastic collision)
    v_rel_new = v_rel_new_dir * v_rel_norm

    # Transform back to lab frame
    v0_new = v_cm + v_rel_new / 2
    v1_new = v_cm - v_rel_new / 2

    return v0_new, v1_new


@njit
def cross_section_term(v_rel: NDArray[np.float64], sigma: float) -> float:
    return sigma * np.sqrt((v_rel**2).sum())


@njit(parallel=True)
def roll_scattering_pairs(
    v: NDArray[np.float64],
    dt: NDArray[np.float64],
    density: NDArray[np.float64],
    sigma: float,
    max_radius_j: int,
    whitelist_mask: NDArray[np.bool_],
    blacklist: NDArray[np.int64] = np.array([], dtype=np.int64),
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    pairs = np.empty((len(dt), 2), dtype=np.int64)
    pair_found = np.full(len(dt), False)
    if sigma > 0:
        whitelist = np.arange(len(dt))[whitelist_mask]
        blacklist = blacklist[blacklist != -1]
        p = np.random.rand(len(dt))
        for i in prange(len(whitelist)):
            particle = whitelist[i]
            if (particle == blacklist).any():
                continue
            probability = 0
            spatial_term = dt[particle] / 2 * density[particle] / max_radius_j
            for offset in range(1, max_radius_j + 1):
                partner = particle + offset
                if partner >= len(dt) or (blacklist == partner).any():
                    continue
                v_rel = v[partner] - v[particle]
                probability += spatial_term * cross_section_term(v_rel, sigma)
                if p[particle] <= probability:
                    pairs[particle] = [particle, partner]
                    pair_found[particle] = True
                    break
    return pairs, pair_found


@njit(parallel=True)
def scatter_unique_pairs(v: NDArray[np.float64], pairs: NDArray[np.int64]) -> NDArray[np.float64]:
    output = v.copy()
    for pair_index in prange(len(pairs)):
        i0, i1 = pairs[pair_index]
        if i0 == -1 or i1 == -1:
            continue
        output[i0], output[i1] = scatter_pair_kinematics(v0=v[i0], v1=v[i1])
    return output


def scatter_found_pairs(
    v: NDArray[np.float64],
    found_pairs: NDArray[np.int64],
    memory_allocated: int,
) -> NDArray[np.float64]:
    if len(found_pairs) == 0:
        return v
    pairs = np.full((memory_allocated, 2), -1, dtype=np.int64)
    pairs[: len(found_pairs)] = found_pairs
    return scatter_unique_pairs(v=v, pairs=pairs)


def calculate_scatter_rounds(
    v: units.Quantity['velocity'],
    dt: units.Quantity['time'],
    sigma: units.Quantity[run_units.cross_section],
    local_density: units.Quantity['mass density'],
    kappa: float = default_params['kappa'],
    max_allowed_rounds: int | None = None,
) -> NDArray[np.int64]:
    time_scale = t_scatter(
        local_density=local_density,
        sigma=sigma,
        v_norm=utils.fast_quantity_norm(v),
    ).to(run_units.time)
    scatter_rounds = np.ceil(dt / (time_scale * kappa)).value.astype(np.int64)
    return scatter_rounds.clip(min=1, max=max_allowed_rounds)


def scatter(
    r: units.Quantity['length'],
    v: units.Quantity['velocity'],
    dt: units.Quantity['time'],
    m: units.Quantity['mass'],
    sigma: units.Quantity[run_units.cross_section],
    blacklist: NDArray[np.int64] = np.array([], dtype=np.int64),
    max_radius_j: int = default_params['max_radius_j'],
    density_accuracy_coef: float = default_params['density_accuracy_coef'],
    kappa: float = default_params['kappa'],
    max_allowed_rounds: int | None = default_params['max_allowed_rounds'],
    max_interactions_per_mini_timestep: int = default_params['max_interactions_per_mini_timestep'],
) -> tuple[units.Quantity['velocity'], int, NDArray[np.int64], int]:
    if sigma == 0:
        return v, 0, np.array([], dtype=np.int64), 0
    v_output = v.value.copy()
    n_interactions = 0
    interacted: list[NDArray[np.int64]] = []
    sigma_value: float = sigma.value
    local_density = m * physics.utils.local_density(r, max_radius_j=max_radius_j, accuracy_cutoff=density_accuracy_coef)
    local_density_value = local_density.to(run_units.density).value
    scatter_rounds = calculate_scatter_rounds(v, dt, sigma, local_density, kappa, max_allowed_rounds)
    round_dt = dt.value / scatter_rounds
    for round in range(1, scatter_rounds.max() + 1):
        pairs, pair_found = roll_scattering_pairs(
            v=v_output,
            dt=round_dt,
            density=local_density_value,
            sigma=sigma_value,
            max_radius_j=max_radius_j,
            blacklist=blacklist,
            whitelist_mask=(scatter_rounds >= round),
        )
        found_pairs = utils.clean_pairs(pairs[pair_found], blacklist=blacklist)
        v_output = scatter_found_pairs(v=v_output, found_pairs=found_pairs, memory_allocated=max_interactions_per_mini_timestep)
        n_interactions += len(found_pairs)
        interacted += [found_pairs.ravel()]
    return units.Quantity(v_output, v.unit), n_interactions, np.hstack(interacted).astype(np.int64), scatter_rounds.max()
