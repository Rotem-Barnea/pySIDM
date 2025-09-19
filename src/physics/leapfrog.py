import numpy as np
from numba import njit,prange
from numpy.typing import NDArray
from typing import TypedDict
from astropy import units,constants
from .. import run_units

G = constants.G.to(run_units.G_units).value

class Params(TypedDict,total=False):
    max_ministeps: int
    consider_all: bool
    kill_divergent: bool
    regulator: units.Quantity['length']
    simple_radius: units.Quantity['length']
    r_convergence_threshold:units.Quantity['length']
    vr_convergence_threshold:units.Quantity['velocity']

default_step_params:Params={'max_ministeps':1000,'consider_all':True,'kill_divergent':False,'regulator':units.Quantity(1e-10,'kpc').to(run_units.length),
                            'simple_radius':units.Quantity(1,'kpc').to(run_units.length),
                            'r_convergence_threshold':units.Quantity(1e-3,'kpc').to(run_units.length),
                            'vr_convergence_threshold':units.Quantity(5,'km/second').to(run_units.velocity)}

@njit
def acceleration(r:float,L:float,M:float,regulator:float=0) -> float:
    return -G*M/(r**2+regulator) + L**2/(r**3+regulator)

@njit
def particle_step(r:float,vx:float,vy:float,vr:float,M:float,dt:float,N:int=1,regulator:float=0) -> tuple[float,NDArray[np.float64]]:
    Lx,Ly = r*vx,r*vy
    L = np.sqrt(Lx**2+Ly**2)
    a = acceleration(r,L,M,regulator)
    vr += a*dt/(2*N)
    for ministep in range(N):
        r += vr*dt/N
        if r < 0:
            r *= -1
            vr *= -1
        final_N = 2*N if ministep == N-1 else N
        a = acceleration(r,L,M,regulator)
        vr += a*dt/(2*final_N)
    return r,np.array([Lx/r,Ly/r,vr])

@njit(parallel=True)
def fast_step(r:NDArray[np.float64],v:NDArray[np.float64],M:NDArray[np.float64],live:NDArray[np.bool_],dt:float,regulator:float=0,max_ministeps:int=100,
         r_convergence_threshold:float=1e-3,vr_convergence_threshold:float=0.005,simple_radius:float=1,consider_all:bool=True,kill_divergent:bool=False) -> None:
    for i in prange(len(r)):
        if not consider_all and not live[i]:
            continue
        if r[i] > simple_radius:
            r[i],v[i] = particle_step(r=r[i],vx=v[i,0],vy=v[i,1],vr=v[i,2],M=M[i],dt=dt,regulator=regulator)
            continue
        for N in range(1,max_ministeps+1):
            r_coarse,v_coarse = particle_step(r=r[i],vx=v[i,0],vy=v[i,1],vr=v[i,2],M=M[i],dt=dt,regulator=regulator,N=N)
            r_fine,v_fine = particle_step(r=r[i],vx=v[i,0],vy=v[i,1],vr=v[i,2],M=M[i],dt=dt,regulator=regulator,N=2*N)
            if (np.abs(r_coarse-r_fine) < r_convergence_threshold) and (np.abs(v_coarse[2]-v_fine[2]) < vr_convergence_threshold):
                r[i],v[i] = r_fine,v_fine
                break
            if N == max_ministeps:
                if kill_divergent:
                    live[i] = False
                else:
                    r[i],v[i] = r_fine,v_fine

def step(r:NDArray[np.float64],v:NDArray[np.float64],M:NDArray[np.float64],live:NDArray[np.bool_],dt:units.Quantity['time'],
         regulator:units.Quantity['length']=default_step_params['regulator'],max_ministeps:int=default_step_params['max_ministeps'],
         r_convergence_threshold:units.Quantity['length']=default_step_params['r_convergence_threshold'],
         vr_convergence_threshold:units.Quantity['velocity']=default_step_params['vr_convergence_threshold'],
         simple_radius:units.Quantity['length']=default_step_params['simple_radius'],
         consider_all:bool=default_step_params['consider_all'],kill_divergent:bool=default_step_params['kill_divergent']) -> None:
    fast_step(r=r,v=v,M=M,live=live,dt=dt.to(run_units.time).value,regulator=regulator.to(run_units.length).value,max_ministeps=max_ministeps,
              r_convergence_threshold=r_convergence_threshold.to(run_units.length).value,
              vr_convergence_threshold=vr_convergence_threshold.to(run_units.velocity).value,
              simple_radius=simple_radius.to(run_units.length).value,consider_all=consider_all,kill_divergent=kill_divergent)
