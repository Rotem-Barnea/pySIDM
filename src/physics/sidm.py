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
    max_allowed_rounds: int | None
    kappa: float
    sigma: Quantity[run_units.cross_section]


default_params: Params = {
    'max_radius_j': 10,
    'max_allowed_rounds': 100,
    'kappa': 0.02,
    'sigma': Quantity(0, 'cm^2/gram').to(run_units.cross_section),
}


def normalize_params(params: Params, add_defaults: bool = False) -> Params:
    """Normalize Quantity parameters to the run units.

    Parameters:
        params: Dictionary of parameters.
        add_defaults: Whether to add default parameters (under the input `params`).

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

    Calculates a random isotropic scattering direction from a uniform cosine distribution off of `v_rel`.

    Parameters:
        v0: Velocity of the first particle. Array of shape `(3,)` representing the velocity in 3D space as `(vx,vy,vr)`.
        v1: Velocity of the second particle. Array of shape `(3,)` representing the velocity in 3D space as `(vx,vy,vr)`.

    Returns:
        Tuple of new velocities for the two particles (two arrays of shape `(3,)`).
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


@njit(parallel=True)
def update_v_rel(v_rel: NDArray[np.float64], v: NDArray[np.float64], max_radius_j: int, whitelist_mask: NDArray[np.bool_]) -> None:
    """Calculate the relative velocity between all neighboring particles.

    The result is an array of relative velocity norms `sqrt(sum((v[partner]-v[particle])^2))`, and the partners are taken from: `[particle + 1 : particle + 1 + max_radius_j]`

    Parameters:
        v: Array of particle velocities, shape `(n_particles, 3)`, with components `(vx,vy,vr)`.
        max_radius_j: Maximum index radius for partners for scattering.
        whitelist_mask: Mask for particles to consider in this round. Used to maintain constant input shape to utilize the njit cache.

    Returns:
        Array of relative velocities between particles, shape `(n_particles, max_radius_j)`.
    """
    for particle in prange(len(v_rel)):
        if whitelist_mask[particle]:
            v_rel_array = np.sqrt(((v[particle + 1 : particle + 1 + max_radius_j] - v[particle]) ** 2).sum(1))
            v_rel[particle, : len(v_rel_array)] = v_rel_array


@njit(parallel=True)
def fast_scatter_chance(
    v_rel: NDArray[np.float64],
    dt: NDArray[np.float64],
    sigma: float,
    density_term: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calculate the scattering chance for each particle.

    Parameters:
        v_rel: relative velocities of neighboring particles. An array of shape `(n_particles, max_radius_j)`, where each row holds the norm of the relative velocity (the 3-vector difference). For particles too close to the edge (less than max_radius_j places from the end), the overflow cells hold 0.
        dt: Time step for each particle, adjusted by `1/number_of_rounds` to allow parallelized calculation for particles with a different number of rounds.
        sigma: Scattering cross-section.
        density_term: Density term of particles, in the form `m/(2*pi*r^2*dr)`.

    Returns:
        Scattering chance for each particle (shape `(n_particles,)`)
    """
    output = np.empty(len(v_rel), np.float64)
    for particle in prange(len(v_rel)):
        output[particle] = 1 / 2 * dt[particle] * density_term[particle] * v_rel[particle].sum() * sigma
    return output


@njit(parallel=True)
def fast_scatter_rounds(scatter_chance: NDArray[np.float64], kappa: float, max_allowed_rounds: int) -> NDArray[np.int64]:
    """Calculate the number of scattering rounds required for each particle to comply with a per-round scattering chance <= `kappa`.

    Parameters:
        scatter_chance: Base scattering chance for each particle, without subdivision (shape `(n_particles,)`).
        kappa: Maximum allowed scattering chance per round.
        max_allowed_rounds: Maximum allowed number of scattering rounds. If negative no limit is applied.

    Returns:
        Number of scattering rounds required for each particle (shape `(n_particles,)`)
    """
    output = np.empty(len(scatter_chance), dtype=np.int64)
    for particle in prange(len(scatter_chance)):
        ratio = np.ceil(scatter_chance[particle] / kappa)
        if np.isnan(ratio):
            output[particle] = 0
        elif ratio < 1:
            output[particle] = 1
        elif max_allowed_rounds > 0 and ratio > max_allowed_rounds:
            output[particle] = max_allowed_rounds
        else:
            output[particle] = int(ratio)
    return output


@njit(parallel=True)
def pick_scatter_partner(v_rel: NDArray[np.float64], scatter_mask: NDArray[np.bool_], rolls: NDArray[np.float64]):
    """Choose the scattering partner for an interacting particle.

    For every particle that was deemed to have scattered in this round, the relative velocity is used to calculate the probability cdf for the scattering partner, and the random roll input is used to pick the partner from the cdf.

    Parameters:
        v_rel: relative velocities of neighboring particles. An array of shape `(n_particles, max_radius_j)`, where each row holds the norm of the relative velocity (the 3-vector difference). For particles too close to the edge (less than max_radius_j places from the end), the overflow cells hold 0.
        scatter_mask: mask indicating which particles are scattering. Used to maintain constant input shape to utilize the njit cache.
        rolls: random numbers between 0 and 1 for each particle (shape `(n_particles,)`).

    Returns:
        pairs of interacting particles.
    """
    pairs = np.empty((len(v_rel), 2), dtype=np.int64)
    for particle in prange(len(v_rel)):
        if scatter_mask[particle]:
            cumsum = v_rel[particle].cumsum()
            cumsum /= cumsum[-1]
            partner = particle + np.searchsorted(cumsum, rolls[particle]) + 1
            pairs[particle] = [particle, partner]
    return pairs[scatter_mask]


@njit(parallel=True)
def scatter_unique_pairs(v: NDArray[np.float64], pairs: NDArray[np.int64]) -> None:
    """Loop over all unique pairs found and calculate the new velocities inplace. pairs MUST be unique, otherwise numba will fail the race condition check."""
    for pair_index in prange(len(pairs)):
        i0, i1 = pairs[pair_index]
        v[i0], v[i1] = scatter_pair_kinematics(v0=v[i0], v1=v[i1])


def scatter_chance_shortcut(
    r: Quantity['length'] | NDArray[np.float64] | pd.Series,
    vx: Quantity['velocity'] | NDArray[np.float64] | pd.Series,
    vy: Quantity['velocity'] | NDArray[np.float64] | pd.Series,
    vr: Quantity['velocity'] | NDArray[np.float64] | pd.Series,
    dt: Quantity['time'],
    m: Quantity['mass'] | NDArray[np.float64] | pd.Series,
    sigma: Quantity[run_units.cross_section],
    max_radius_j: int = default_params['max_radius_j'],
) -> NDArray[np.float64]:
    """Calculate the scattering change of particles. A shortcut calculation used for calculating the number of underestimated scattering events.

    Parameters:
        r: Particles position.
        vx: The first pernpendicular component (to the radial direction) of the velocity of the particles.
        vy: The second pernpendicular component (to the radial direction) of the velocity of the particles.
        vr: The radial velocity of the particles.
        dt: Time step.
        m: The mass of the particles.
        sigma: Cross-section for scattering.
        max_radius_j: Maximum index radius for partners for scattering.

    Returns:
        The scattering change for each particle over the entire `dt` time step (without division).
    """
    v = np.vstack([np.array(vx), np.array(vy), np.array(vr)]).T
    v_rel = np.empty((len(v), max_radius_j), dtype=np.float64)
    update_v_rel(v_rel=v_rel, v=v, max_radius_j=max_radius_j, whitelist_mask=np.full(len(v), True))
    return fast_scatter_chance(
        v_rel=v_rel,
        dt=np.full(len(v), dt.value),
        sigma=sigma.value,
        density_term=np.array(physics.utils.local_density(r, m, max_radius_j, volume_kind='shell', mass_kind='single')),
    )


def scatter_underestimate_shortcut(
    r: Quantity['length'] | NDArray[np.float64] | pd.Series,
    vx: Quantity['velocity'] | NDArray[np.float64] | pd.Series,
    vy: Quantity['velocity'] | NDArray[np.float64] | pd.Series,
    vr: Quantity['velocity'] | NDArray[np.float64] | pd.Series,
    dt: Quantity['time'],
    m: Quantity['mass'] | NDArray[np.float64] | pd.Series,
    sigma: Quantity[run_units.cross_section],
    max_radius_j: int = default_params['max_radius_j'],
    kappa: float = default_params['kappa'],
    max_allowed_rounds: int | None = default_params['max_allowed_rounds'],
    overestimated_by_more_than: int = 0,
) -> int:
    """Calculate the number of particles with underestimated amount of scattering events.

    The scattering probability is calculated for particle, and particles with `P/max_allowed_rounds > kappa` are considered to have underestimated amount of scattering events (because in the regular run these particles will not have the `dt` time step divided enough to reach a scattering chance below `kappa`).

    Parameters:
        r: Particles position.
        vx: The first pernpendicular component (to the radial direction) of the velocity of the particles.
        vy: The second pernpendicular component (to the radial direction) of the velocity of the particles.
        vr: The radial velocity of the particles.
        dt: Time step.
        m: The mass of the particles.
        sigma: Cross-section for scattering.
        max_radius_j: Maximum index radius for partners for scattering.
        kappa: The maximum allowed scattering probability. Particles with a higher scattering rate (due to high density mostly) will instead perform `N` scattering rounds over a time step `dt/N` to lower the rate in each round to match `kappa`.
        max_allowed_rounds: Maximum number of allowed rounds for scattering, used to prevent stalling in case of high density.
        overestimated_by_more_than: Only consider particles as "underestimated" if they are underestimated by more than this value.

    Returns:
        Number of particles with underestimated amount of scattering events.
    """
    P = scatter_chance_shortcut(r, vx, vy, vr, dt, m, sigma, max_radius_j)
    scatter_rounds_preclip = np.nan_to_num(np.ceil(P / kappa)).astype(np.int64)
    scatter_rounds_postclip = scatter_rounds_preclip.clip(min=1, max=max_allowed_rounds)
    return (scatter_rounds_preclip > scatter_rounds_postclip + overestimated_by_more_than).sum()


def scatter(
    r: Quantity['length'] | NDArray[np.float64] | pd.Series,
    vx: Quantity['velocity'] | NDArray[np.float64] | pd.Series,
    vy: Quantity['velocity'] | NDArray[np.float64] | pd.Series,
    vr: Quantity['velocity'] | NDArray[np.float64] | pd.Series,
    dt: Quantity['time'],
    m: Quantity['mass'] | NDArray[np.float64] | pd.Series,
    sigma: Quantity[run_units.cross_section],
    max_radius_j: int = default_params['max_radius_j'],
    kappa: float = default_params['kappa'],
    max_allowed_rounds: int | None = default_params['max_allowed_rounds'],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], int]:
    """Perform SIDM scatter events.

    Every time step the particle density is calculated, the overall scattering rate is estimated, and each particle is given `N` scattering rounds where in each round the scatter probability is at most `kappa`. Specifically:
        `N = ceil(estimated scattering rate / kappa)`, and at most max_allowed_rounds to prevent stalling.
        `dt -> dt / N`
        So the scatter chance in every round (which is proportional to `dt`) is ~`kappa`.
    In every round, the relative velocity is recalculated for any particle that scattered in the previous round to re-estimate the scattering change.
    Scattering is allowed with the next max_radius_j particles (one sided to avoid double counting), with the probability:
        `p = dt/2 * m/(2*pi*r^2*dr) * sigma * sum(v_rel_j)`
        With the sum over indices `i` to `i+max_radius_j`, and `dr` the distance between the `particle` (`i`) and the `i+max_radius_j` one.
        `v_rel` is the norm of the relative velocity vector between the `particle` and the potential `partner` (from `i` to `i+max_radius_j`).
    If a particle is rolled to have scattered, the partner is chosen randomly weighted by the relative velocity.

    Parameters:
        r: Particles position.
        vx: The first pernpendicular component (to the radial direction) of the velocity of the particles.
        vy: The second pernpendicular component (to the radial direction) of the velocity of the particles.
        vr: The radial velocity of the particles.
        dt: Time step.
        m: The mass of the particles.
        sigma: Cross-section for scattering. At the moment, it is assumed to be constant (TODO to make it velocity dependent).
        max_radius_j: Maximum index radius for partners for scattering.
        kappa: The maximum allowed scattering probability. Particles with a higher scattering rate (due to high density mostly) will instead perform `N` scattering rounds over a time step `dt/N` to lower the rate in each round to match `kappa`.
        max_allowed_rounds: Maximum number of allowed rounds for scattering, used to prevent stalling in case of high density.

    Returns:
        post scattering vx, vy, vz, indices of the particles that interacted, and the maximum number of rounds performed.
    """
    if max_allowed_rounds is None:
        max_allowed_rounds = -1
    _r, _vx, _vy, _vr, _m = np.array(r), np.array(vx).copy(), np.array(vy).copy(), np.array(vr).copy(), np.array(m)
    interacted: NDArray[np.int64] = np.empty(0, dtype=np.int64)
    if sigma == 0:
        return _vx, _vy, _vr, interacted, 0
    v_output = np.vstack([_vx, _vy, _vr]).T
    _sigma = sigma.value
    local_density = cast(NDArray[np.float64], physics.utils.local_density(_r, _m, max_radius_j, volume_kind='shell', mass_kind='single'))
    v_rel = np.zeros((len(v_output), max_radius_j), dtype=np.float64)
    update_v_rel(v_rel=v_rel, v=v_output, max_radius_j=max_radius_j, whitelist_mask=np.full(len(v_output), True))
    scatter_chance = fast_scatter_chance(
        v_rel=v_rel,
        dt=np.full(len(v_output), dt.value),
        sigma=_sigma,
        density_term=local_density,
    )
    scatter_rounds = fast_scatter_rounds(scatter_chance=scatter_chance, kappa=kappa, max_allowed_rounds=max_allowed_rounds)
    round_dt = dt.value / scatter_rounds
    scatter_chance /= scatter_rounds
    interacted_particles = np.empty(0, dtype=np.int64)
    for round in range(1, scatter_rounds.max() + 1):
        mask = scatter_rounds >= round
        if len(interacted_particles) > 0:
            # Only update the relative velocities and scattering chance for particles that scattered in the past round or in the neighborhood of scattering particles (i.e. only particles that would have a change in their v_rel values, otherwise the probability is the same and we don't need to recalculate it)
            mask *= utils.expand_mask_back(utils.indices_to_mask(interacted_particles, len(v_output)), n=max_radius_j)
            update_v_rel(v_rel=v_rel, v=v_output, max_radius_j=max_radius_j, whitelist_mask=mask)
            scatter_chance[mask] = fast_scatter_chance(
                v_rel=v_rel[mask],
                dt=round_dt[mask],
                sigma=_sigma,
                density_term=local_density[mask],
            )
        rolls = np.random.random(len(v_output))
        events = scatter_chance >= rolls  # TODO - slice with mask
        pairs = utils.clean_pairs(pairs=pick_scatter_partner(v_rel=v_rel, scatter_mask=events, rolls=rolls))
        interacted_particles = pairs.ravel()
        interacted = np.hstack([interacted, interacted_particles])
        scatter_unique_pairs(v=v_output, pairs=pairs)
    return *v_output.T, interacted, scatter_rounds.max()
