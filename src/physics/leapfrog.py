import numpy as np
from numba import njit,prange
from typing import TypedDict
from ..constants import G,kpc

class Params(TypedDict,total=False):
    max_ministeps: int
    consider_all: bool
    kill_divergent: bool

@njit()
def particle_step(r,vx,vy,vr,M,dt,N=1,regulator=0):
    Lx,Ly = r*vx,r*vy
    L = np.sqrt(Lx**2+Ly**2)
    a = -G*M/(r**2+regulator) + L**2/(r**3+regulator)
    vr += a*dt/(2*N)
    for ministep in range(N):
        r += vr*dt/N
        if r < 0:
            r *= -1
            vr *= -1
        final_N = 2*N if ministep == N-1 else N
        a = -G*M/(r**2+regulator) + L**2/(r**3+regulator)
        vr += a*dt/(2*final_N)
    return r,np.array([Lx/r,Ly/r,vr])

@njit(parallel=True)
def step(r,v,M,live,dt,max_ministeps=10,r_convergence_threshold=1e-3*kpc,vr_convergence_threshold=5,regulator=0,simple_radius=0,consider_all=False,
         kill_divergent=True):
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
