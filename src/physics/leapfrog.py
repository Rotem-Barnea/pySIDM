import warnings
from typing import Literal, TypedDict, cast

import numpy as np
import pandas as pd
from numba import njit, prange
from astropy import constants
from numpy.typing import NDArray
from astropy.units import Quantity

from .. import utils, run_units

G = constants.G.decompose(run_units.system).value


class Params(TypedDict, total=False):
    """Parameter dictionary for the leapfrog integrator.

    Attributes:
        max_minirounds: Maximum number of mini-rounds to perform.
        r_convergence_threshold: Convergence threshold for the radius.
        vr_convergence_threshold: Convergence threshold for the radial velocity.
        richardson_extrapolation: Whether to use Richardson extrapolation.
        adaptive: Whether to use adaptive mini-rounds.
        grid_window_radius: Radius of the grid window for updating the enclosed mass during the run.
        raise_warning: Whether to raise a warning if the integrator fails to converge.
        levi_civita_mode: Mode for the Levi-Civita correction.
        levi_civita_condition_coefficient: Coefficient for the Levi-Civita condition.
    """

    max_minirounds: int
    r_convergence_threshold: float
    vr_convergence_threshold: float
    richardson_extrapolation: bool
    adaptive: bool
    grid_window_radius: int
    raise_warning: bool
    levi_civita_mode: Literal['always', 'never', 'adaptive']
    levi_civita_condition_coefficient: float
    guess_dt_factor: bool


default_params: Params = {
    'max_minirounds': 20,
    'r_convergence_threshold': 1e-7,
    'vr_convergence_threshold': 1e-7,
    'richardson_extrapolation': True,
    'adaptive': True,
    'grid_window_radius': 50,
    'raise_warning': False,
    'levi_civita_mode': 'adaptive',
    'levi_civita_condition_coefficient': 1 / 20,
    'guess_dt_factor': False,
}


def normalize_params(params: Params | None, add_defaults: bool = False) -> Params:
    """Normalize Quantity parameters to the run units.

    Parameters:
        params: Dictionary of parameters.
        add_defaults: Whether to add default parameters (under the input `params`).

    Returns:
        Normalized parameters.
    """
    params = cast(Params, utils.handle_default(params, Params({})))
    if add_defaults:
        params = {**default_params, **params}
    return params


class FactorGuessKwargs(TypedDict, total=False):
    """Parameters for the factor guess function. See `guess_factor()` for details."""

    scale: float | Literal['mean']
    rounding: float
    base: float | None


@njit
def get_grid(
    r: NDArray[np.float64], M: NDArray[np.float64], window_radius: int, particle_index: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the grid of masses and positions around a particle, without it.

    Parameters:
        r: Array of particle positions.
        M: Array of particle masses.
        window_radius: Radius (in indexes) of the grid window, i.e. the window will span from window_radius in each direction (slice `[particle_index - window_radius : particle_index + window_radius + 1]`).
        particle_index: Index of the particle for which the grid is being generated.

    Returns:
        (M_grid, r_grid): Tuple containing the grid of masses and positions.
    """
    if window_radius == 0:
        return np.empty(0, np.float64), np.empty(0, np.float64)
    M_grid = np.concatenate(
        (M[particle_index - window_radius : particle_index], M[particle_index + 1 : particle_index + window_radius + 1])
    )
    r_grid = np.concatenate(
        (r[particle_index - window_radius : particle_index], r[particle_index + 1 : particle_index + window_radius + 1])
    )
    return M_grid, r_grid


@njit
def adjust_M(
    r: float,
    m: float,
    r_grid: NDArray[np.float64],
    M_grid: NDArray[np.float64],
    backup_M: float,
    count_self: bool = False,
) -> float:
    """Calculate the mass cdf (`M(<=r)`) for the particle.

    If the input grid is empty (i.e. the subprocess is disabled by setting `grid_window_radius`=0), returns the backup mass cdf value.
    Otherwise, calculate the position (`index`) the particle should be inserted into the grid in, and return the mass cdf of the particle before + the self mass (`m`).

    **Assumes r_grid is sorted!**

    Parameters:
        r: Position of the particle.
        m: Mass of the particle.
        r_grid: Array of positions for the particles pre-step.
        M_grid: Array of mass cdf values for the particles pre-step.
        backup_M: The mass cdf value for the particle pre-step.
        count_self: Whether to include the self mass in the enclosed mass.

    Returns:
        Mass cdf at the given position.
    """
    if len(r_grid) == 0:
        return backup_M
    M = M_grid[np.searchsorted(r_grid, r)]
    if count_self:
        M += m
    return M


@njit
def acceleration(r: float, L: float, M: float) -> float:
    """Calculate the acceleration of the particle."""
    return -G * M / r**2 + L**2 / r**3


@njit
def levi_civita_criteria(
    r: float,
    vx: float,
    vy: float,
    M: float,
    alpha: float = 1 / 20,
    override_always: bool = False,
    override_never: bool = False,
) -> bool:
    """Check if the particle is in the Levi-Civita regime.

    The innermost particle (M=0) is always considered to be in the Levi-Civita regime.

    Parameters:
        r: The position of the particle.
        vx: The first perpendicular component (to the radial direction) of the velocity of the particle.
        vy: The second perpendicular component (to the radial direction) of the velocity of the particle.
        M: The mass cdf (`M(<=r)`) of the particle at the start of the step.
        alpha: Threshold parameter.
        override_always: If `True`, particle is always in the Levi-Civita regime. Considered before `override_never`. Equivalent to `levi_civita_condition_coefficient=infinity`.
        override_never: If `True`, the particle is never in the Levi-Civita regime. Considered after `override_always`. Equivalent to `levi_civita_condition_coefficient=0`.

    Returns:
        Bool.
    """
    if override_always:
        return True
    if override_never:
        return False
    if M == 0:
        return True
    return r <= alpha * (r**2 * (vx**2 + vy**2)) / (G * M)


@njit
def particle_levi_civita_step(
    r: float,
    vx: float,
    vy: float,
    vr: float,
    m: float,
    M: float,
    M_grid: NDArray[np.float64],
    r_grid: NDArray[np.float64],
    dt: float,
    N: int = 1,
) -> tuple[float, float, float, float]:
    """Perform a simple leapfrog step in the radius (1D). See `particle_step()` for more details on the leapfrog implementation, this function will only detail the Levi Civita aspect.

    The integration is done in the Levi-Civita coordinates `r_ = sqrt(r)`, with physical velocity and a Sundman time reparameterization `dt/dτ = r = r_^2`.
        The initial fictitious time step is given by `dτ = dt/r_^2`, and subdivided further by `N`.
        Physical velocity equation of motion: `dv/dτ = -GM/r_^2 + L^2/r_^4`, with M selected using `adjust_M()`.
        Levi-Civita coordinate equation of motion: `dr_/dτ = 1/2 * r_* vr`.
        Integrate until the physical time overshoots, then take a step back and perform a fractional step forward to hit exactly `dt`.

    Parameters:
        r: The position of the particle.
        vx: The first perpendicular component (to the radial direction) of the velocity of the particle.
        vy: The second perpendicular component (to the radial direction) of the velocity of the particle.
        vr: The radial velocity of the particle.
        M: The mass cdf (`M(<=r)`) of the particle at the start of the step. Used only if `M_grid` is empty.
        m: The mass of the particle.
        M_grid: Array of mass cdf values for the particles pre-step to re-estimate the mass cdf when the position changes.
        r_grid: Array of position values for the particles pre-step to re-estimate the mass cdf when the position changes.
        dt: The time step.
        N: The number of mini-steps to perform. Should be an odd number, where the actual step size is adjusted by `dt -> dt/(N-1)`.

    Returns:
        The new position and velocity of the particle.
    """
    Lx, Ly = r * vx, r * vy
    L = np.sqrt(Lx**2 + Ly**2)

    r_ = np.sqrt(r)
    t_step = 0
    dtau = dt / (N * r_**2)
    a = -G * adjust_M(r_**2, m, r_grid, M_grid, M) / r_**2 + L**2 / r_**4
    vr += a * dtau / 2
    for ministep in range(N):
        r_ += 0.5 * r_ * vr * dtau
        if r_ < 0:
            r_ = -r_
            vr = -vr
        dt_physical = r_**2 * dtau
        t_step += dt_physical

        if t_step >= dt and ministep < N - 1:  # Check if we've overshot the target time
            fraction = 1 - (t_step - dt) / dt_physical  # Fractional step to hit exactly dt

            # Backtrack and take fractional step
            r_ -= 0.5 * r_ * vr * dtau
            r_ += 0.5 * r_ * vr * dtau * fraction

            # Recalculate acceleration and take final half-step
            a = -G * adjust_M(r_**2, m, r_grid, M_grid, M) / r_**2 + L**2 / r_**4
            vr += a * dtau * fraction / 2
            break  # Exit early since we've reached target time

        a = -G * adjust_M(r_**2, m, r_grid, M_grid, M) / r_**2 + L**2 / r_**4
        if ministep < N - 1:
            vr += a * dtau
        else:
            vr += a * dtau / 2

    # Transform back to physical coordinates
    r = r_**2

    return r, Lx / r, Ly / r, vr


@njit
def particle_normal_step(
    r: float,
    vx: float,
    vy: float,
    vr: float,
    m: float,
    M: float,
    M_grid: NDArray[np.float64],
    r_grid: NDArray[np.float64],
    dt: float,
    N: int = 1,
) -> tuple[float, float, float, float]:
    """Perform a simple leapfrog step in the radius (1D).

    Splits the step into `N` mini-steps, each over a time interval of `dt/N`, and then integrates the velocity and position using a leapfrog algorithm:
        `velocity half step` -> [`position step` -> `velocity step` -> ...] -> `velocity half step`
    The acceleration is re-calculated every time the position changes. Angular momentum is explicitly conserved by recalculating the velocity in the non-radial directions at the end to conform the same angular momentum with the new position.

    Parameters:
        r: The position of the particle.
        vx: The first perpendicular component (to the radial direction) of the velocity of the particle.
        vy: The second perpendicular component (to the radial direction) of the velocity of the particle.
        vr: The radial velocity of the particle.
        M: The mass cdf (`M(<=r)`) of the particle at the start of the step. Used only if `M_grid` is empty.
        m: The mass of the particle.
        M_grid: Array of mass cdf values for the particles pre-step to re-estimate the mass cdf when the position changes.
        r_grid: Array of position values for the particles pre-step to re-estimate the mass cdf when the position changes.
        dt: The time step.
        N: The number of mini-steps to perform. Should be an odd number, where the actual step size is adjusted by `dt -> dt/(N-1)`.

    Returns:
        The new position and velocity of the particle.
    """
    Lx, Ly = r * vx, r * vy
    L = np.sqrt(Lx**2 + Ly**2)
    step_size = dt / N
    a = acceleration(r, L, adjust_M(r, m, r_grid, M_grid, M))
    vr += a * step_size / 2
    for ministep in range(N):
        r += vr * step_size
        if r < 0:
            r *= -1
            vr *= -1
        a = acceleration(r, L, adjust_M(r, m, r_grid, M_grid, M))
        if ministep < N - 1:
            vr += a * step_size
        else:
            vr += a * step_size / 2
    return r, Lx / r, Ly / r, vr


@njit
def particle_step(
    r: float,
    vx: float,
    vy: float,
    vr: float,
    m: float,
    M: float,
    M_grid: NDArray[np.float64],
    r_grid: NDArray[np.float64],
    dt: float,
    N: int = 1,
    levi_civita_override_always: bool = False,
    levi_civita_override_never: bool = False,
    levi_civita_condition_coefficient: float = 1 / 20,
) -> tuple[float, float, float, float]:
    """Perform a simple leapfrog step in the radius (1D).

    Handler function for performing either a normal or a Levi-Civita step.

    Parameters:
        r: The position of the particle.
        vx: The first perpendicular component (to the radial direction) of the velocity of the particle.
        vy: The second perpendicular component (to the radial direction) of the velocity of the particle.
        vr: The radial velocity of the particle.
        M: The mass cdf (`M(<=r)`) of the particle at the start of the step. Used only if `M_grid` is empty.
        m: The mass of the particle.
        M_grid: Array of mass cdf values for the particles pre-step to re-estimate the mass cdf when the position changes.
        r_grid: Array of position values for the particles pre-step to re-estimate the mass cdf when the position changes.
        dt: The time step.
        N: The number of mini-steps to perform. Should be an odd number, where the actual step size is adjusted by `dt -> dt/(N-1)`.
        levi_civita_override_always: If `True`, particle is always in the Levi-Civita regime. Equivalent to `levi_civita_condition_coefficient=infinity`.
        levi_civita_override_never: If `True`, the particle is never in the Levi-Civita regime. Equivalent to `levi_civita_condition_coefficient=0`.
        levi_civita_condition_coefficient: Threshold parameter for the Levi-Civita condition.

    Returns:
        The new position and velocity of the particle.
    """
    if levi_civita_criteria(
        r=r,
        vx=vx,
        vy=vy,
        M=M,
        alpha=levi_civita_condition_coefficient,
        override_always=levi_civita_override_always,
        override_never=levi_civita_override_never,
    ):
        return particle_levi_civita_step(r=r, vx=vx, vy=vy, vr=vr, m=m, M=M, M_grid=M_grid, r_grid=r_grid, dt=dt, N=N)
    return particle_normal_step(r=r, vx=vx, vy=vy, vr=vr, m=m, M=M, M_grid=M_grid, r_grid=r_grid, dt=dt, N=N)


@njit
def particle_adaptive_step(
    r: float,
    vx: float,
    vy: float,
    vr: float,
    m: float,
    M: float,
    M_grid: NDArray[np.float64],
    r_grid: NDArray[np.float64],
    dt: float,
    max_minirounds: int = 20,
    r_convergence_threshold: float = 1e-7,
    vr_convergence_threshold: float = 1e-7,
    first_mini_round: int = 0,
    richardson_extrapolation: bool = True,
    levi_civita_override_always: bool = False,
    levi_civita_override_never: bool = False,
    levi_civita_condition_coefficient: float = 1 / 20,
) -> tuple[float, float, float, float, bool, int]:
    """Perform an adaptive leapfrog step for a particle.

    Parameters:
        r: Particle position.
        vx: The first perpendicular component (to the radial direction) of the velocity of the particle.
        vy: The second perpendicular component (to the radial direction) of the velocity of the particle.
        vr: The radial velocity of the particle.
        m: The mass of the particle.
        M: The mass cdf (`M(<=r)`) of the particle at the start of the step. Used only if `M_grid` is empty.
        M_grid: Array of mass cdf values for the particles pre-step to re-estimate the mass cdf when the position changes.
        r_grid: Array of position values for the particles pre-step to re-estimate the mass cdf when the position changes.
        dt: Time step.
        max_minirounds: Maximum number of mini-rounds to perform.
        r_convergence_threshold: Convergence threshold for position.
        vr_convergence_threshold: Convergence threshold for radial velocity.
        first_mini_round: The first mini-round to perform.
        richardson_extrapolation: Use Richardson extrapolation.
        levi_civita_override_always: If `True`, particle is always in the Levi-Civita regime. Equivalent to `levi_civita_condition_coefficient=infinity`.
        levi_civita_override_never: If `True`, the particle is never in the Levi-Civita regime. Equivalent to `levi_civita_condition_coefficient=0`.
        levi_civita_condition_coefficient: Threshold parameter for the Levi-Civita condition.

    Returns:
        Updated position and velocity. The final two values are the convergence status and the number of rounds to convergence.
    """
    converged = False
    r_coarse, vx_coarse, vy_coarse, vr_coarse = r, vx, vy, vr
    r_fine, vx_fine, vy_fine, vr_fine = r_coarse, vx_coarse, vy_coarse, vr_coarse
    rounds_to_convergence = first_mini_round
    for mini_round in range(first_mini_round, max(max_minirounds, first_mini_round + 1)):
        r_coarse, vx_coarse, vy_coarse, vr_coarse = particle_step(
            r=r,
            vx=vx,
            vy=vy,
            vr=vr,
            m=m,
            M=M,
            M_grid=M_grid,
            r_grid=r_grid,
            dt=dt,
            N=2**mini_round,
            levi_civita_override_always=levi_civita_override_always,
            levi_civita_override_never=levi_civita_override_never,
            levi_civita_condition_coefficient=levi_civita_condition_coefficient,
        )
        r_fine, vx_fine, vy_fine, vr_fine = particle_step(
            r=r,
            vx=vx,
            vy=vy,
            vr=vr,
            m=m,
            M=M,
            M_grid=M_grid,
            r_grid=r_grid,
            dt=dt,
            N=2**mini_round * 2,
            levi_civita_override_always=levi_civita_override_always,
            levi_civita_override_never=levi_civita_override_never,
            levi_civita_condition_coefficient=levi_civita_condition_coefficient,
        )
        r_relative_error = np.abs(r_fine - r_coarse) / r_fine
        vr_relative_error = np.abs(vr_fine - vr_coarse) / vr_fine
        if (r_relative_error < r_convergence_threshold) and (vr_relative_error < vr_convergence_threshold):
            converged = True
            rounds_to_convergence = mini_round
            break
    if richardson_extrapolation:
        r_result = 4 * r_fine - 3 * r_coarse
        vx_result = 4 * vx_fine - 3 * vx_coarse
        vy_result = 4 * vy_fine - 3 * vy_coarse
        vr_result = 4 * vr_fine - 3 * vr_coarse

        if r_result < 0:
            r_result *= -1
            vr_result *= -1
    else:
        r_result, vx_result, vy_result, vr_result = r_fine, vx_fine, vy_fine, vr_fine
    return r_result, vx_result, vy_result, vr_result, converged, rounds_to_convergence


@njit
def particle_step_wrapper(
    r: float,
    vx: float,
    vy: float,
    vr: float,
    m: float,
    M: float,
    dt: float,
    first_mini_round: int,
    M_grid: NDArray[np.float64],
    r_grid: NDArray[np.float64],
    max_minirounds: int = 20,
    r_convergence_threshold: float = 1e-7,
    vr_convergence_threshold: float = 1e-7,
    richardson_extrapolation: bool = True,
    levi_civita_override_always: bool = False,
    levi_civita_override_never: bool = False,
    levi_civita_condition_coefficient: float = 1 / 20,
    adaptive: bool = True,
    grid_window_radius: int = 2,
) -> tuple[
    float,
    float,
    float,
    float,
    bool,
    int,
]:
    """Perform a leapfrog step (single or adaptive) for a particle.

    Parameters:
        r: Particle position.
        vx: The first perpendicular component (to the radial direction) of the velocity of the particle.
        vy: The second perpendicular component (to the radial direction) of the velocity of the particle.
        vr: The radial velocity of the particle.
        m: The mass of the particle.
        M: The mass cdf (`M(<=r)`) of the particle at the start of the step. Used only if `M_grid` is empty.
        dt: Time step.
        M_grid: Array of mass cdf values for the particles pre-step to re-estimate the mass cdf when the position changes.
        r_grid: Array of position values for the particles pre-step to re-estimate the mass cdf when the position changes.
        max_minirounds: Maximum number of mini-rounds to perform.
        r_convergence_threshold: Convergence threshold for position.
        vr_convergence_threshold: Convergence threshold for radial velocity.
        first_mini_round: The first mini-round to perform for each particle.
        richardson_extrapolation: Use Richardson extrapolation.
        levi_civita_override_always: If `True`, particle is always in the Levi-Civita regime. Equivalent to `levi_civita_condition_coefficient=infinity`.
        levi_civita_override_never: If `True`, the particle is never in the Levi-Civita regime. Equivalent to `levi_civita_condition_coefficient=0`.
        levi_civita_condition_coefficient: Threshold parameter for the Levi-Civita condition.
        adaptive: Use adaptive step size - iterate over mini-rounds until convergence. If `False` performs a single mini-round at `first_mini_round`.
        grid_window_radius: Radius of the grid window. Allows recalculating the mass cdf (`M(<=r)`) during the step to account for the particle's motion, by changing position with upto grid_window_radius places in either direction. Assumes the rest of the particles are static. If 0, avoids recalculating the mass cdf (`M(<=r)`) during the step (use the value pre-step for all acceleration calculations).

    Returns:
        Updated position and velocity. final two arrays are the convergence status and the number of rounds to convergence for each particle.
    """
    if not adaptive:
        return (
            *particle_step(
                r=r,
                vx=vx,
                vy=vy,
                vr=vr,
                m=m,
                M=M,
                M_grid=M_grid,
                r_grid=r_grid,
                dt=dt,
                N=2**first_mini_round,
                levi_civita_override_always=levi_civita_override_always,
                levi_civita_override_never=levi_civita_override_never,
                levi_civita_condition_coefficient=levi_civita_condition_coefficient,
            ),
            True,
            first_mini_round,
        )
    else:
        return particle_adaptive_step(
            r=r,
            vx=vx,
            vy=vy,
            vr=vr,
            m=m,
            M=M,
            M_grid=M_grid,
            r_grid=r_grid,
            dt=dt,
            max_minirounds=max_minirounds,
            r_convergence_threshold=r_convergence_threshold,
            vr_convergence_threshold=vr_convergence_threshold,
            first_mini_round=first_mini_round,
            richardson_extrapolation=richardson_extrapolation,
            levi_civita_override_always=levi_civita_override_always,
            levi_civita_override_never=levi_civita_override_never,
            levi_civita_condition_coefficient=levi_civita_condition_coefficient,
        )


@njit(parallel=True)
def fast_step(
    r: NDArray[np.float64],
    vx: NDArray[np.float64],
    vy: NDArray[np.float64],
    vr: NDArray[np.float64],
    m: NDArray[np.float64],
    M: NDArray[np.float64],
    dt: float,
    first_mini_round: NDArray[np.int64],
    factor: NDArray[np.int64],
    max_minirounds: int = 20,
    r_convergence_threshold: float = 1e-7,
    vr_convergence_threshold: float = 1e-7,
    richardson_extrapolation: bool = True,
    levi_civita_override_always: bool = False,
    levi_civita_override_never: bool = False,
    levi_civita_condition_coefficient: float = 1 / 20,
    adaptive: bool = True,
    grid_window_radius: int = 2,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
    NDArray[np.int64],
]:
    """TODO"""
    output_r, output_vx, output_vy, output_vr = (
        r.copy(),
        vx.copy(),
        vy.copy(),
        vr.copy(),
    )
    converged = np.ones(len(r), dtype=np.bool_)
    convergence_rounds = first_mini_round.copy()
    for particle in prange(len(r)):
        M_grid, r_grid = get_grid(r=r, M=M, window_radius=grid_window_radius, particle_index=particle)
        for _ in range(factor[particle]):
            (
                output_r[particle],
                output_vx[particle],
                output_vy[particle],
                output_vr[particle],
                converged_,
                convergence_rounds_,
            ) = particle_step_wrapper(
                r=output_r[particle],
                vx=output_vx[particle],
                vy=output_vy[particle],
                vr=output_vr[particle],
                m=m[particle],
                M=M[particle],
                dt=dt / factor[particle],
                M_grid=M_grid,
                r_grid=r_grid,
                max_minirounds=max_minirounds,
                r_convergence_threshold=r_convergence_threshold,
                vr_convergence_threshold=vr_convergence_threshold,
                first_mini_round=first_mini_round[particle],
                richardson_extrapolation=richardson_extrapolation,
                levi_civita_override_always=levi_civita_override_always,
                levi_civita_override_never=levi_civita_override_never,
                levi_civita_condition_coefficient=levi_civita_condition_coefficient,
                adaptive=adaptive,
                grid_window_radius=grid_window_radius,
            )
            if not converged_:
                converged[particle] = False
            else:
                convergence_rounds[particle] = max(convergence_rounds[particle], convergence_rounds_)
    return output_r, output_vx, output_vy, output_vr, converged, convergence_rounds


def guess_factor(
    r: Quantity['length'] | NDArray[np.float64] | pd.Series,
    vx: Quantity['velocity'] | NDArray[np.float64] | pd.Series,
    vy: Quantity['velocity'] | NDArray[np.float64] | pd.Series,
    vr: Quantity['velocity'] | NDArray[np.float64] | pd.Series,
    dt: Quantity['time'] | float,
    scale: float | Literal['mean'] = 2.5e3,
    rounding: float = 1,
    base: float | None = None,
) -> NDArray[np.int64]:
    """Guess the factor for adaptive time stepping.

    The factor is used to split a single time step into `factor` steps of size `dt/factor` prior to convergence consideration, to allow the system to more naturally converge without geometric blowouts.

    The factor is calculated based on the `scale * (v * dt) / r`, i.e. the ratio of the movement distance scale in a single time step and the current radius, scaled by some factor.
    This gives large factors for orbits where the radius is likely to change rapidly, and thus the acceleration will be sensitive to the time step and require many iterations to converge.
    To simplify the calculation, the factor is rounded to give full divisors (to allow using simple for loops over all the steps without needing "partial" steps).

    If `base` is provided, the factor is further grouped into powers by taking a log10 and rounding up (i.e. the number of decimal digits required + 1), and then returning `base` to the power of that value.

    If `scale='mean'`, instead of `scale * (v * dt) / r`, the factor is calculated relatively to the sample as `(r/v)/(r/v).mean()`.

    Parameters:
        r: Radius.
        vx: The first perpendicular component (to the radial direction) of the velocity.
        vy: The second perpendicular component (to the radial direction) of the velocity.
        vr: The radial velocity.
        dt: Time step.
        scale: The scaling factor parameter, controls the spacing of the factors between different inputs (see explanation above). If `mean`, instead set `factor` to be `(r/v)/(r/v).mean()`.
        rounding: When performing the np.ceil operation, first divide by `rounding` and then multiply by it after the rounding. This changes the rounding from rounding to the nearest full integer to rounding to the nearest multiple of `rounding`.
        base: The base factor for scaling. If `None` return the base factor as is, otherwise must be an integer and the returned factor will be `base**np.ceil(np.log10(factor))`.

    Returns:
        The calculated factor.
    """
    if isinstance(r, Quantity):
        r = r.to(run_units.length).value
    v = np.sqrt(vx**2 + vy**2 + vr**2)
    if isinstance(v, Quantity):
        v = v.to(run_units.velocity).value
    if isinstance(dt, Quantity):
        dt = dt.to(run_units.time).value
    if scale == 'mean':
        factor = (r / v) / (r / v).mean()
    else:
        factor = (v * dt * scale) / r
    if base is None:
        factor = np.ceil(factor / rounding) * rounding
    else:
        factor = base ** (np.ceil(np.log10(factor) / rounding) * rounding)
    return factor.clip(min=1).astype(np.int64)


def step(
    r: Quantity['length'] | NDArray[np.float64] | pd.Series,
    vx: Quantity['velocity'] | NDArray[np.float64] | pd.Series,
    vy: Quantity['velocity'] | NDArray[np.float64] | pd.Series,
    vr: Quantity['velocity'] | NDArray[np.float64] | pd.Series,
    m: Quantity['mass'] | NDArray[np.float64] | pd.Series,
    M: Quantity['mass'] | NDArray[np.float64] | pd.Series,
    dt: Quantity['time'] | float,
    max_minirounds: int = default_params['max_minirounds'],
    r_convergence_threshold: float = default_params['r_convergence_threshold'],
    vr_convergence_threshold: float = default_params['vr_convergence_threshold'],
    first_mini_round: NDArray[np.int64] | None = None,
    factor: NDArray[np.int64] | None = None,
    richardson_extrapolation: bool = default_params['richardson_extrapolation'],
    levi_civita_mode: Literal['always', 'never', 'adaptive'] = default_params['levi_civita_mode'],
    levi_civita_condition_coefficient: float = default_params['levi_civita_condition_coefficient'],
    adaptive: bool = default_params['adaptive'],
    grid_window_radius: int = default_params['grid_window_radius'],
    raise_warning: bool = default_params['raise_warning'],
    guess_dt_factor: bool = default_params['guess_dt_factor'],
    guess_dt_factor_kwargs: FactorGuessKwargs = {},
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Perform an adaptive leapfrog step for a particle.

    Wrapper for the njit `fast_step()` function.

    Parameters:
        r: Particles position.
        vx: The first perpendicular component (to the radial direction) of the velocity of the particles.
        vy: The second perpendicular component (to the radial direction) of the velocity of the particles.
        vr: The radial velocity of the particles.
        m: The mass of the particles.
        M: The mass cdf (`M(<=r)`) of the particles at the start of the step. Used only if `M_grid` is empty.
        dt: Time step.
        max_minirounds: Maximum number of mini-rounds to perform.
        r_convergence_threshold: Convergence threshold for position.
        vr_convergence_threshold: Convergence threshold for radial velocity.
        first_mini_round: The first mini-round to perform for each particle.
        factor: For each particle, perform `factor` steps of `dt/factor` size as an additional smoothing. If `None` perform a single step of constant `dt` for all particles.
        richardson_extrapolation: Use Richardson extrapolation.
        levi_civita_mode: Operation mode for controlling when to use Levi-Civita regularization and when to use a normal leapfrog step.
        alpha: Threshold parameter for the Levi-Civita condition.
        adaptive: Use adaptive step size - iterate over mini-rounds until convergence. If `False` performs a single mini-round at `first_mini_round`.
        grid_window_radius: Radius of the grid window. Allows recalculating the mass cdf (`M(<=r)`) during the step to account for the particle's motion, by changing position with upto grid_window_radius places in either direction. Assumes the rest of the particles are static. If 0, avoids recalculating the mass cdf (`M(<=r)`) during the step (use the value pre-step for all acceleration calculations).
        raise_warning: Raise a warning if a particle fails to converge.
        guess_dt_factor: Guess the value of `factor` for each particle based on `r/v`. Only relevant if `factor` is not `None`.
        guess_dt_factor_kwargs: Keyword arguments for the `guess_factor` function.

    Returns:
        Updated position and velocity. The final array is the number of convergence rounds required for each particle to converge.
    """
    if factor is not None:
        _factor = factor
    elif guess_dt_factor:
        _factor = guess_factor(r=r, vx=vx, vy=vy, vr=vr, dt=dt, **guess_dt_factor_kwargs)
    else:
        _factor = np.ones(len(r), dtype=np.int64)
    _r, _vx, _vy, _vr, converged, convergence_rounds = fast_step(
        r=np.array(r),
        vx=np.array(vx),
        vy=np.array(vy),
        vr=np.array(vr),
        m=np.array(m),
        M=np.array(M),
        dt=dt.value if isinstance(dt, Quantity) else dt,
        max_minirounds=max_minirounds,
        r_convergence_threshold=r_convergence_threshold,
        vr_convergence_threshold=vr_convergence_threshold,
        first_mini_round=first_mini_round if first_mini_round is not None else np.zeros(len(r), dtype=np.int64),
        factor=_factor,
        richardson_extrapolation=richardson_extrapolation,
        levi_civita_override_always=levi_civita_mode == 'always',
        levi_civita_override_never=levi_civita_mode == 'never',
        levi_civita_condition_coefficient=levi_civita_condition_coefficient,
        adaptive=adaptive,
        grid_window_radius=grid_window_radius,
    )
    if raise_warning and not converged.all():
        for index in np.where(~converged)[0]:
            warnings.warn(
                f'Maximum number of mini-rounds reached for particle {index}, starting with r={r[index]}, vx={vx[index]}, vy={vy[index]}, vr={vr[index]}, M={M[index]}'
            )
    return _r, _vx, _vy, _vr, convergence_rounds
