import numpy as np
import pandas as pd
from numba import njit, prange
from numpy.typing import NDArray
from typing import TypedDict
from astropy import constants
from astropy.units import Quantity
from .. import run_units

G = constants.G.to(run_units.G_units).value


class Params(TypedDict, total=False):
    """Parameter dictionary for the leapfrog integrator."""

    max_minirounds: int
    r_convergence_threshold: Quantity['length']
    vr_convergence_threshold: Quantity['velocity']
    first_mini_round: int
    richardson_extrapolation: bool
    adaptive: bool
    adaptive_exponential: bool
    grid_window_radius: int


default_params: Params = {
    'max_minirounds': 30,
    'r_convergence_threshold': Quantity(1e-3, 'pc').to(run_units.length),
    'vr_convergence_threshold': Quantity(1e-3, 'km/second').to(run_units.velocity),
    'first_mini_round': 0,
    'richardson_extrapolation': False,
    'adaptive': True,
    'adaptive_exponential': False,
    'grid_window_radius': 2,
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
    if 'r_convergence_threshold' in params:
        params['r_convergence_threshold'] = params['r_convergence_threshold'].to(run_units.length)
    if 'vr_convergence_threshold' in params:
        params['vr_convergence_threshold'] = params['vr_convergence_threshold'].to(run_units.velocity)
    return params


@njit
def get_grid(
    r: NDArray[np.float64], M: NDArray[np.float64], window_radius: int, particle_index: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the grid of masses and positions around a particle, without it.

    Parameters:
        r: Array of particle positions.
        M: Array of particle masses.
        window_radius: Radius (in indexes) of the grid window, i.e. the window will span from window_radius in each direction (slice [particle_index - window_radius : particle_index + window_radius + 1]).
        particle_index: Index of the particle for which the grid is being generated.

    Returns:
        (M_grid, r_grid): Tuple containing the grid of masses and positions.
    """
    if window_radius == 0:
        return np.empty(0, np.float64), np.empty(0, np.float64)
    M_grid = np.concatenate((M[particle_index - window_radius : particle_index], M[particle_index + 1 : particle_index + window_radius + 1]))
    r_grid = np.concatenate((r[particle_index - window_radius : particle_index], r[particle_index + 1 : particle_index + window_radius + 1]))
    return M_grid, r_grid


@njit
def adjust_M(r: float, m: float, r_grid: NDArray[np.float64], M_grid: NDArray[np.float64], backup_M: float) -> float:
    """Calculate the mass cdf (M(<=r)) for the particle.

    If the input grid is empty (i.e. the subprocess is disabled by setting grid_window_radius=0), returns the backup mass cdf value.
    Otherwise, calculate the position (index) the particle should be inserted into the grid in, and return the mass cdf of the particle before + the self mass (m).
    Assumes r_grid is sorted!

    Parameters:
        r: Position of the particle.
        m: Mass of the particle.
        r_grid: Array of positions for the particles pre-step.
        M_grid: Array of mass cdf values for the particles pre-step.
        backup_M: The mass cdf value for the particle pre-step.

    Returns:
        Mass cdf at the given position.
    """
    if len(r_grid) == 0:
        return backup_M
    return M_grid[np.searchsorted(r_grid, r)] + m


@njit
def acceleration(r: float, L: float, M: float) -> float:
    """Calculate the acceleration of the particle."""
    return -G * M / r**2 + L**2 / r**3


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
) -> tuple[float, NDArray[np.float64]]:
    """Perform a simple leapfrog step in the radius (1D).

    Splits the step into N mini-steps, each over a time interval of dt/N, and then integrates the velocity and position using a leapfrog algorithm:
        velocity half step -> [position step -> velocity step -> ...] -> velocity half step
    The acceleration is re-calculated every time the position changes. Angular momentum is explicitly conserved by recalculating the velocity in the non-radial directions at the end to conform the same angular momentum with the new position.

    Parameters:
        r: The position of the particle.
        vx: The first pernpendicular component (to the radial direction) of the velocity of the particle.
        vy: The second pernpendicular component (to the radial direction) of the velocity of the particle.
        vr: The radial velocity of the particle.
        M: The mass cdf (M(<=r)) of the particle at the start of the step. Used only if M_grid is empty.
        m: The mass of the particle.
        M_grid: Array of mass cdf values for the particles pre-step to re-estimate the mass cdf when the position changes.
        r_grid: Array of position values for the particles pre-step to re-estimate the mass cdf when the position changes.
        dt: The time step.
        N: The number of mini-steps to perform. Should be an odd number.

    Returns:
        The new position and velocity of the particle.
    """
    Lx, Ly = r * vx, r * vy
    L = np.sqrt(Lx**2 + Ly**2)
    a = acceleration(r, L, adjust_M(r, m, r_grid, M_grid, M))
    vr += a * dt / (2 * N)
    for ministep in range(N):
        r += vr * dt / N
        if r < 0:
            r *= -1
            vr *= -1
        a = acceleration(r, L, adjust_M(r, m, r_grid, M_grid, M))
        if ministep < N - 1:
            vr += a * dt / N
        else:
            vr += a * dt / (2 * N)
    return r, np.array([Lx / r, Ly / r, vr])


@njit
def mini_step_to_N(step: int, adaptive_exponential: bool = False, base: int = 2):
    """Convert number of mini-steps to the actual step size (linear / exponential).

    If the base is set to not be 2, ensure elsewhere that the number of steps is odd.
    """
    if adaptive_exponential:
        return base**step + 1
    else:
        return base * step + 1


@njit(parallel=True)
def fast_step(
    r: NDArray[np.float64],
    vx: NDArray[np.float64],
    vy: NDArray[np.float64],
    vr: NDArray[np.float64],
    m: NDArray[np.float64],
    M: NDArray[np.float64],
    dt: float,
    max_minirounds: int = 100,
    r_convergence_threshold: float = 1e-3,
    vr_convergence_threshold: float = 0.001,
    first_mini_round: int = 0,
    richardson_extrapolation: bool = False,
    adaptive: bool = True,
    adaptive_exponential: bool = False,
    grid_window_radius: int = 2,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Perform an adaptive leapfrog step for a particle.

    Parameters:
        r: Particle position.
        vx: The first pernpendicular component (to the radial direction) of the velocity of the particle.
        vy: The second pernpendicular component (to the radial direction) of the velocity of the particle.
        vr: The radial velocity of the particle.
        m: The mass of the particle.
        M: The mass cdf (M(<=r)) of the particle at the start of the step. Used only if M_grid is empty.
        dt: Time step.
        max_mini_rounds: Maximum number of mini-rounds to perform (will be sent to mini_step_to_N() to evaluate the actual number of mini-steps).
        r_convergence_threshold: Convergence threshold for position.
        vr_convergence_threshold: Convergence threshold for radial velocity.
        first_mini_round: The first mini-round to perform (will be sent to mini_step_to_N() to evaluate the actual number of mini-steps).
        richardson_extrapolation: Use Richardson extrapolation.
        adaptive: Use adaptive step size - iterate over mini-rounds until convergence. If False - performs a single mini-round at first_mini_round.
        adaptive_exponential: Use exponential mini-steps growth (sent to mini_step_to_N()).
        grid_window_radius: Radius of the grid window. Allows recalculating the mass cdf (M(<=r)) during the step to account for the particle's motion, by changing position with upto grid_window_radius places in either direction. Assumes the rest of the particles are static. If 0, avoids recalculating the mass cdf (M(<=r)) during the step (use the value pre-step for all acceleration calculations).

    Returns:
        Updated position and velocity.
    """
    output_r, output_vx, output_vy, output_vr = np.empty_like(r), np.empty_like(vx), np.empty_like(vy), np.empty_like(vr)
    for i in prange(len(r)):
        M_grid, r_grid = get_grid(r=r, M=M, window_radius=grid_window_radius, particle_index=i)
        r_coarse, v_coarse = particle_step(
            r=r[i],
            vx=vx[i],
            vy=vy[i],
            vr=vr[i],
            m=m[i],
            M=M[i],
            M_grid=M_grid,
            r_grid=r_grid,
            dt=dt,
            N=mini_step_to_N(first_mini_round, adaptive_exponential),
        )
        r_fine, v_fine = r_coarse, v_coarse
        if adaptive:
            for mini_round in range(first_mini_round, first_mini_round + max_minirounds):
                r_fine, v_fine = particle_step(
                    r=r[i],
                    vx=vx[i],
                    vy=vy[i],
                    vr=vr[i],
                    m=m[i],
                    M=M[i],
                    M_grid=M_grid,
                    r_grid=r_grid,
                    dt=dt,
                    N=mini_step_to_N(mini_round + 1, adaptive_exponential),
                )
                if (np.abs(r_coarse - r_fine) < r_convergence_threshold) and (np.abs(v_coarse[2] - v_fine[2]) < vr_convergence_threshold):
                    break
                r_coarse = r_fine
                v_coarse = v_fine
        if richardson_extrapolation:
            output_r[i] = 4 * r_fine - 3 * r_coarse
            output_vx[i], output_vy[i], output_vr[i] = 4 * v_fine - 3 * v_coarse
        else:
            output_r[i] = r_fine
            output_vx[i], output_vy[i], output_vr[i] = v_fine
    return output_r, output_vx, output_vy, output_vr


def step(
    r: Quantity['length'] | NDArray[np.float64] | pd.Series,
    vx: Quantity['velocity'] | NDArray[np.float64] | pd.Series,
    vy: Quantity['velocity'] | NDArray[np.float64] | pd.Series,
    vr: Quantity['velocity'] | NDArray[np.float64] | pd.Series,
    m: Quantity['mass'] | NDArray[np.float64] | pd.Series,
    M: Quantity['mass'] | NDArray[np.float64] | pd.Series,
    dt: Quantity['time'] | float,
    max_minirounds: int = default_params['max_minirounds'],
    r_convergence_threshold: Quantity['length'] = default_params['r_convergence_threshold'],
    vr_convergence_threshold: Quantity['velocity'] = default_params['vr_convergence_threshold'],
    first_mini_round: int = default_params['first_mini_round'],
    richardson_extrapolation: bool = default_params['richardson_extrapolation'],
    adaptive: bool = default_params['adaptive'],
    adaptive_exponential: bool = default_params['adaptive_exponential'],
    grid_window_radius: int = default_params['grid_window_radius'],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Perform an adaptive leapfrog step for a particle.

    Wrapper for the njit fast_step() function.

    Parameters:
        r: Particles position.
        vx: The first pernpendicular component (to the radial direction) of the velocity of the particles.
        vy: The second pernpendicular component (to the radial direction) of the velocity of the particles.
        vr: The radial velocity of the particles.
        m: The mass of the particles.
        M: The mass cdf (M(<=r)) of the particles at the start of the step. Used only if M_grid is empty.
        dt: Time step.
        max_mini_rounds: Maximum number of mini-rounds to perform (will be sent to mini_step_to_N() to evaluate the actual number of mini-steps).
        r_convergence_threshold: Convergence threshold for position.
        vr_convergence_threshold: Convergence threshold for radial velocity.
        first_mini_round: The first mini-round to perform (will be sent to mini_step_to_N() to evaluate the actual number of mini-steps).
        richardson_extrapolation: Use Richardson extrapolation.
        adaptive: Use adaptive step size - iterate over mini-rounds until convergence. If False - performs a single mini-round at first_mini_round.
        adaptive_exponential: Use exponential mini-steps growth (sent to mini_step_to_N()).
        grid_window_radius: Radius of the grid window. Allows recalculating the mass cdf (M(<=r)) during the step to account for the particle's motion, by changing position with upto grid_window_radius places in either direction. Assumes the rest of the particles are static. If 0, avoids recalculating the mass cdf (M(<=r)) during the step (use the value pre-step for all acceleration calculations).

    Returns:
        Updated position and velocity.
    """
    _r, _vx, _vy, _vr = fast_step(
        r=np.array(r),
        vx=np.array(vx),
        vy=np.array(vy),
        vr=np.array(vr),
        m=np.array(m),
        M=np.array(M),
        dt=dt.value if isinstance(dt, Quantity) else dt,
        max_minirounds=max_minirounds,
        r_convergence_threshold=r_convergence_threshold.value,
        vr_convergence_threshold=vr_convergence_threshold.value,
        first_mini_round=first_mini_round,
        richardson_extrapolation=richardson_extrapolation,
        adaptive=adaptive,
        adaptive_exponential=adaptive_exponential,
        grid_window_radius=grid_window_radius,
    )
    return _r, _vx, _vy, _vr
