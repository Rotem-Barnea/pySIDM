import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from typing import TypedDict
from astropy import units, constants
from .. import run_units

G = constants.G.to(run_units.G_units).value


class Params(TypedDict, total=False):
    max_minirounds: int
    regulator: units.Quantity['length']
    r_convergence_threshold: units.Quantity['length']
    vr_convergence_threshold: units.Quantity['velocity']


default_params: Params = {
    'max_minirounds': 30,
    'regulator': units.Quantity(1e-10, 'kpc').to(run_units.length),
    'r_convergence_threshold': units.Quantity(1e-3, 'pc').to(run_units.length),
    'vr_convergence_threshold': units.Quantity(1e-3, 'km/second').to(run_units.velocity),
}


def normalize_params(params: Params, add_defaults: bool = False) -> Params:
    if add_defaults:
        params = {**default_params, **params}
    if 'r_convergence_threshold' in params:
        params['r_convergence_threshold'] = params['r_convergence_threshold'].to(run_units.length)
    if 'vr_convergence_threshold' in params:
        params['vr_convergence_threshold'] = params['vr_convergence_threshold'].to(run_units.velocity)
    if 'regulator' in params:
        params['regulator'] = params['regulator'].to(run_units.length)
    return params


@njit
def acceleration(r: float, L: float, M: float, regulator: float = 0) -> float:
    return -G * M / (r + regulator) ** 2 + L**2 / (r + regulator) ** 3


@njit
def particle_step(
    r: float, vx: float, vy: float, vr: float, M: float, dt: float, N: int = 1, regulator: float = 0
) -> tuple[float, NDArray[np.float64]]:
    Lx, Ly = r * vx, r * vy
    L = np.sqrt(Lx**2 + Ly**2)
    a = acceleration(r, L, M, regulator)
    vr += a * dt / (2 * N)
    for ministep in range(N):
        r += vr * dt / N
        if r < 0:
            r *= -1
            vr *= -1
        final_N = N if ministep < N - 1 else 2 * N
        a = acceleration(r, L, M, regulator)
        vr += a * dt / final_N
    return r, np.array([Lx / r, Ly / r, vr])


@njit(parallel=True)
def fast_step(
    r: NDArray[np.float64],
    v: NDArray[np.float64],
    M: NDArray[np.float64],
    dt: float,
    regulator: float = 0,
    max_minirounds: int = 100,
    r_convergence_threshold: float = 1e-3,
    vr_convergence_threshold: float = 0.001,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    output_r = np.empty_like(r)
    output_v = np.empty_like(v)
    for i in prange(len(r)):
        r_fine: float = 0.0
        v_fine: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        r_coarse, v_coarse = particle_step(r=r[i], vx=v[i, 0], vy=v[i, 1], vr=v[i, 2], M=M[i], dt=dt, regulator=regulator, N=1)
        for mini_step in range(0, max_minirounds):
            N = 2 ** (mini_step)
            r_fine, v_fine = particle_step(r=r[i], vx=v[i, 0], vy=v[i, 1], vr=v[i, 2], M=M[i], dt=dt, regulator=regulator, N=2 * N)
            if (np.abs(r_coarse - r_fine) < r_convergence_threshold) and (np.abs(v_coarse[2] - v_fine[2]) < vr_convergence_threshold):
                break
            r_coarse = r_fine
            v_coarse = v_fine
        output_r[i] = r_fine
        output_v[i] = v_fine
    return output_r, output_v


def step(
    r: units.Quantity['length'],
    v: units.Quantity['velocity'],
    M: units.Quantity['mass'],
    dt: units.Quantity['time'],
    regulator: units.Quantity['length'] = default_params['regulator'],
    max_minirounds: int = default_params['max_minirounds'],
    r_convergence_threshold: units.Quantity['length'] = default_params['r_convergence_threshold'],
    vr_convergence_threshold: units.Quantity['velocity'] = default_params['vr_convergence_threshold'],
) -> tuple[units.Quantity['length'], units.Quantity['velocity']]:
    _r, _v = fast_step(
        r=r.value,
        v=v.value,
        M=M.value,
        dt=dt.value,
        regulator=regulator.value,
        max_minirounds=max_minirounds,
        r_convergence_threshold=r_convergence_threshold.value,
        vr_convergence_threshold=vr_convergence_threshold.value,
    )
    return units.Quantity(_r, r.unit), units.Quantity(_v, v.unit)
