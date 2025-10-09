import numpy as np
import pandas as pd
from numba import njit, prange
from typing import TypedDict, cast
from numpy.typing import NDArray
from astropy.units import Quantity
from .. import utils, run_units, physics


class Params(TypedDict, total=False):
    """Parameter dictionary for the SIDM calculation."""

    max_radius_j: int
    max_interactions_per_mini_timestep: int
    max_allowed_rounds: int | None
    kappa: float
    sigma: Quantity[run_units.cross_section]


default_params: Params = {
    'max_radius_j': 10,
    'max_interactions_per_mini_timestep': 10000,
    'max_allowed_rounds': None,
    'kappa': 0.002,
    'sigma': Quantity(0, 'cm^2/gram').to(run_units.cross_section),
}


def normalize_params(params: Params, add_defaults: bool = False) -> Params:
    """Normalize Quantity parameters to the run units.

    Parameters:
        params: Dictionary of parameters.
        add_defaults: Whether to add default parameters (under the input params).

    Returns:
        Normalized parameters.
    """
    if add_defaults:
        params = {**default_params, **params}
    if 'sigma' in params:
        params['sigma'] = params['sigma'].to(run_units.cross_section)
    return params


@njit
def scatter_pair_kinematics(v0: NDArray[np.float64], v1: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate the new velocities of two particles after a scattering event.

    Calculates a random isotropic scattering direction from a uniform cosine distribution off of v_rel.

    Args:
        v0: Velocity of the first particle. Array of shape (3,) representing the velocity in 3D space as (vx,vy,vr).
        v1: Velocity of the second particle. Array of shape (3,) representing the velocity in 3D space as (vx,vy,vr).

    Returns:
        Tuple of new velocities for the two particles (two arrays of shape (3,)).
    """
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
def particle_v_rel(v: NDArray[np.float64], particle: int, max_radius_j: int) -> NDArray[np.float64]:
    """TODO"""
    return np.sqrt(((v[particle + 1 : particle + 1 + max_radius_j] - v[particle]) ** 2).sum(1))


@njit(parallel=True)
def v_rel(v: NDArray[np.float64], max_radius_j: int, whitelist_mask: NDArray[np.bool_]):
    """Calculate the relative velocity between all neighboring particles.

    Parameters:
        v: Array of particle velocities, shape (n_particles, 3), with components (vx,vy,vr).
        max_radius_j: Maximum index radius for partners for scattering.
        whitelist_mask: Mask for particles to consider in this round. Used to maintain constant input shape to utilize the njit cache.

    Returns:
        Array of relative velocities between particles, shape (n_particles, max_radius_j).
    """
    output = np.empty((len(v), max_radius_j), np.float64)
    whitelist = np.arange(len(v))[whitelist_mask]
    for i in prange(len(whitelist)):
        particle = whitelist[i]
        output[particle] = particle_v_rel(v, particle, max_radius_j)
    return output


@njit
def particle_scatter_chance(v_rel: NDArray[np.float64], dt: float, sigma: float, density_term: float) -> float:
    """TODO"""
    return 1 / 2 * dt * density_term * v_rel.sum() * sigma


@njit(parallel=True)
def scatter_chance(
    v_rel: NDArray[np.float64],
    whitelist_mask: NDArray[np.bool_],
    dt: NDArray[np.float64],
    sigma: float,
    density_term: NDArray[np.float64],
):
    """Calculate the scattering chance for each particle

    Parameters:
        v_rel: relative velocities of neighboring particles. An array of shape (n_particles, max_radius_j), where each row holds the norm of the relative velocity (the 3-vector difference). For particles too close to the edge (less than max_radius_j places from the end), the overflow cells hold 0.
        whitelist_mask: Mask for particles to consider in this round. Used to maintain constant input shape to utilize the njit cache.
        dt: Time step for each particle, adjusted by 1/number_of_rounds to allow parallelized calculation for particles with a different number of rounds.
        sigma: Scattering cross-section.
        density_term: Density term of particles, in the form m/(2*pi*r^2*dr).

    Returns:
        Scattering chance for each particle (shape (n_particles,))
    """
    output = np.empty(len(v_rel), np.float64)
    whitelist = np.arange(len(v_rel))[whitelist_mask]
    for i in prange(len(whitelist)):
        particle = int(whitelist[i])
        output[particle] = particle_scatter_chance(v_rel[particle], dt[particle], sigma, density_term[particle])
    return output


@njit(parallel=True)
def roll_particle_scattering(probability_array: NDArray[np.float64], rolls: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Check which of the particles scatter"""
    output = np.empty(len(probability_array), np.bool_)
    for i in prange(len(probability_array)):
        output[i] = probability_array[i] > rolls[i]
    return output


@njit(parallel=True)
def pick_scatter_partner(v_rel: NDArray[np.float64], scatter_mask: NDArray[np.bool_], rolls: NDArray[np.float64]):
    """Choose the scattering partner for an interacting particle.

    For every particle that was deemed to have scattered in this round, the relative velocity is used to calculate the probability cdf for the scattering partner, and the random roll input is used to pick the partner from the cdf.

    Parameters:
        v_rel: relative velocities of neighboring particles. An array of shape (n_particles, max_radius_j), where each row holds the norm of the relative velocity (the 3-vector difference). For particles too close to the edge (less than max_radius_j places from the end), the overflow cells hold 0.
        scatter_mask: mask indicating which particles are scattering. Used to maintain constant input shape to utilize the njit cache.
        rolls: random numbers between 0 and 1 for each particle (shape (n_particles,)).

    Returns:
        pairs of interacting particles.
    """
    pairs = np.empty((len(v_rel), 2), dtype=np.int64)
    indices = np.arange(len(v_rel))[scatter_mask]
    for i in prange(len(indices)):
        particle = indices[i]
        cumsum = v_rel[particle].cumsum()
        cumsum /= cumsum[-1]
        partner = particle + np.searchsorted(cumsum, rolls[particle]) + 1
        pairs[particle] = [particle, partner]
    return pairs


@njit(parallel=True)
def scatter_unique_pairs(v: NDArray[np.float64], pairs: NDArray[np.int64]) -> NDArray[np.float64]:
    """Loop over all unique pairs found and calculate the new velocities. pairs MUST be unique, otherwise numba will fail the race condition check."""
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
    """Scatter particles that have been found to interact. Handles memory allocation and wraps around scatter_unique_pairs()."""
    if len(found_pairs) == 0:
        return v
    pairs = np.full((memory_allocated, 2), -1, dtype=np.int64)
    pairs[: len(found_pairs)] = found_pairs
    return scatter_unique_pairs(v=v, pairs=pairs)


def scatter(
    r: Quantity['length'] | NDArray[np.float64] | pd.Series,
    vx: Quantity['velocity'] | NDArray[np.float64] | pd.Series,
    vy: Quantity['velocity'] | NDArray[np.float64] | pd.Series,
    vr: Quantity['velocity'] | NDArray[np.float64] | pd.Series,
    dt: Quantity['time'],
    m: Quantity['mass'] | NDArray[np.float64] | pd.Series,
    sigma: Quantity[run_units.cross_section],
    blacklist: NDArray[np.int64] = np.array([], dtype=np.int64),
    max_radius_j: int = default_params['max_radius_j'],
    kappa: float = default_params['kappa'],
    max_allowed_rounds: int | None = default_params['max_allowed_rounds'],
    max_interactions_per_mini_timestep: int = default_params['max_interactions_per_mini_timestep'],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int, NDArray[np.int64], int]:
    """Perform SIDM scatter events.

    Every time step the particle density is calculated, the overall scattering rate is estimated, and each particle is given N scattering rounds where in each round the scatter probability is at most kappa. Specifically:
        N = ceil(estimated scattering rate / kappa), and at most max_allowed_rounds to prevent stalling.
        dt -> dt / N
        So scatter chance in every round (which is proportional to dt) is ~kappa.
    In every round, the relative velocity is recalculated for any particle that scattered in the previous round to re-estimate the scattering change.
    Scattering is allowed with the next max_radius_j particles (one sided to avoid double counting), with the probability:
        p = dt/2 * m/(2*pi*r^2*dr) * sigma * sum(v_rel_j)
        With the sum over indices i to i+max_radius_j, and dr the distance between the particle (i) and the i+max_radius_j one.
        v_rel is the norm of the relative velocity vector between the particle and the potential partner (from i to i+max_radius_j).
    If a particle is rolled to have scattered, the partner is chosen randomly weighted by the relative velocity.

    Parameters:
        r: Particles position.
        vx: The first pernpendicular component (to the radial direction) of the velocity of the particles.
        vy: The second pernpendicular component (to the radial direction) of the velocity of the particles.
        vr: The radial velocity of the particles.
        dt: Time step.
        m: The mass of the particles.
        sigma: Cross-section for scattering. At the moment, it is assumed to be constant (TODO to make it velocity dependent).
        blacklist: Indices of particles to exclude from scattering. Used to allow multiple particle types in the simulation where only some perform the scattering (i.e. the rest are baryonic, other dm particles, etc.).
        max_radius_j: Maximum index radius for partners for scattering.
        kappa: The maximum allowed scattering probability. Particles with a higher scattering rate (due to high density mostly) will instead perform N scattering rounds over a time step dt/N to lower the rate in each round to match kappa.
        max_allowed_rounds: Maximum number of allowed rounds for scattering, used to prevent stalling in case of high density.
        max_interactions_per_mini_timestep: Internal memory buffer size parameter. No need to change it. Sets the maximum allowed number of interactions per round.

    Returns:
        post scattering vx, vy, vz, n_interactions, indices of the particles that interacted, The maximum number of rounds performed.
    """
    _r, _vx, _vy, _vr, _m = np.array(r), np.array(vx), np.array(vy), np.array(vr), np.array(m)
    if sigma == 0:
        return _vx, _vy, _vr, 0, np.array([], dtype=np.int64), 0
    _v = np.vstack([_vx, _vy, _vr]).T
    v_output = _v.copy()
    n_interactions = 0
    interacted: NDArray[np.int64] = np.empty(0, dtype=np.int64)
    _sigma = sigma.value
    local_density = cast(NDArray[np.float64], physics.utils.local_density(_r, _m, max_radius_j))
    v_rel_array = v_rel(v=_v, max_radius_j=max_radius_j, whitelist_mask=np.full(len(_v), True))
    scatter_base_chance = scatter_chance(
        v_rel=v_rel_array,
        whitelist_mask=np.full(len(_v), True),
        dt=np.full(len(_v), dt.value),
        sigma=_sigma,
        density_term=local_density,
    )
    scatter_rounds = np.nan_to_num(np.ceil(scatter_base_chance / kappa)).clip(min=1, max=max_allowed_rounds).astype(np.int64)
    round_dt = dt.value / scatter_rounds
    interacted_particles = np.empty(0, dtype=np.int64)
    for round in range(1, scatter_rounds.max() + 1):
        mask = scatter_rounds >= round
        if len(interacted_particles) > 0:
            mask *= utils.expand_mask_back(utils.indices_to_mask(interacted_particles, len(_v)), n=max_radius_j)
        v_rel_array[mask] = v_rel(v=v_output, max_radius_j=max_radius_j, whitelist_mask=mask)[mask]
        scatter_base_chance[mask] = scatter_chance(
            v_rel=v_rel_array,
            whitelist_mask=mask,
            dt=round_dt,
            sigma=_sigma,
            density_term=local_density,
        )[mask]
        rolls = np.random.random(len(_v))
        events = roll_particle_scattering(probability_array=scatter_base_chance, rolls=rolls)
        pairs = pick_scatter_partner(v_rel=v_rel_array, scatter_mask=events, rolls=rolls)[events]
        pairs = utils.clean_pairs(pairs, blacklist)
        interacted_particles = pairs.ravel()
        v_output = scatter_found_pairs(v=v_output, found_pairs=pairs, memory_allocated=max_interactions_per_mini_timestep)
        n_interactions += len(pairs)
        interacted = np.hstack([interacted, interacted_particles])
    return *v_output.T, n_interactions, interacted, scatter_rounds.max()
