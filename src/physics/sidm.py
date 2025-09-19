import numpy as np
from numba import njit,prange
from typing import TypedDict,Required
from numpy.typing import NDArray
from astropy import units
from .. import utils,run_units

class Params(TypedDict,total=False):
    max_radius_j: int
    regulator: float
    rounds: int
    max_n_allowed:int
    sigma: Required[units.Quantity['opacity']]

default_scatter_params:Params={'max_radius_j':10,'rounds':10,'regulator':1e-10,'max_n_allowed':10000,'sigma':units.Quantity(0,'cm^2/gram')}

@njit(parallel=True)
def roll_scattering_pairs(r:NDArray[np.float64],v:NDArray[np.float64],dt:float,m:float,sigma:float,regulator:float=default_scatter_params['regulator'],
                          max_radius_j:int=default_scatter_params['max_radius_j']) -> NDArray[np.int64]:
    if sigma == 0:
        return np.empty((0,2),dtype=np.int64)

    delta_r:NDArray[np.float64] = np.zeros_like(r)
    delta_r[:-max_radius_j] = r[max_radius_j:]
    delta_r[-max_radius_j:] = r[-1]
    delta_r -= r

    pairs = np.empty((len(r),2),dtype=np.int64)
    pair_found = np.full(len(r),False)
    for particle in prange(len(r)):
        if delta_r[particle] == 0:
            continue
        p = np.random.rand()
        cum_p = 0
        for offset in range(1,max_radius_j+1):
            partner = particle+offset
            if partner >= len(r):
                continue
            v_rel = v[partner]-v[particle]
            v_rel_norm = np.sqrt((v_rel**2).sum())
            cross_section_term = sigma*v_rel_norm
            volume = 4*np.pi*r[particle]**2*delta_r[particle]+regulator
            cum_p += dt/2*(m/volume)*cross_section_term
            if p <= cum_p:
                pairs[particle] = [particle,partner]
                pair_found[particle] = True
                break
    return pairs[pair_found]

@njit()
def scatter_pair_kinematics(v0:NDArray[np.float64],v1:NDArray[np.float64]) -> tuple[NDArray[np.float64],NDArray[np.float64]]:
    #Calculate relative velocity and center of mass velocity
    v_rel = v0-v1
    v_cm = (v0+v1)/2
    v_rel_norm = np.linalg.norm(v_rel)

    #Skip if particles have same velocity (no collision)
    if v_rel_norm < 1e-10:
        return v0,v1

    #Generate random scattering angles (isotropic)
    cos_theta = np.random.rand()*2-1
    sin_theta = np.sqrt(1-cos_theta**2)
    phi = np.random.rand()*2*np.pi
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    #Create orthonormal basis vectors
    #axis1: along relative velocity direction
    axis1 = v_rel/v_rel_norm

    #axis2: first perpendicular vector
    if np.abs(axis1[2]) < 0.9: #if relative velocity not too close to radial (z-axis)
        temp = np.array([0.0, 0.0, 1.0])  # use radial direction as reference
    else: #if relative velocity close to radial, use x-axis
        temp = np.array([1.0, 0.0, 0.0])
    axis2 = np.cross(axis1,temp) #Cross product to get perpendicular vector
    axis2 = axis2/np.linalg.norm(axis2)
    #axis3: complete the orthonormal basis
    axis3 = np.cross(axis1,axis2)

    # New relative velocity direction
    v_rel_new_dir = axis1*cos_theta+(axis2*cos_phi+axis3*sin_phi)*sin_theta

    # Keep same magnitude (elastic collision)
    v_rel_new = v_rel_new_dir*v_rel_norm

    # Transform back to lab frame
    v0_new = v_cm+v_rel_new/2
    v1_new = v_cm-v_rel_new/2

    return v0_new,v1_new

@njit(parallel=True)
def scatter_unique_pairs(v:NDArray[np.float64],pairs:NDArray[np.int64]) -> None:
    for pair_index in prange(len(pairs)):
        i0,i1 = pairs[pair_index]
        if i0 == -1 or i1 == -2:
            continue
        v[i0],v[i1] = scatter_pair_kinematics(v0=v[i0],v1=v[i1])

def scatter(r:NDArray[np.float64],v:NDArray[np.float64],dt:units.Quantity['time'],m:units.Quantity['mass'],sigma:units.Quantity['opacity'],
            blacklist:list[int]=[],rounds:int=default_scatter_params['rounds'],regulator:float=default_scatter_params['regulator'],
            max_radius_j:int=default_scatter_params['max_radius_j'],max_n_allowed:int=default_scatter_params['max_n_allowed']) -> tuple[int,NDArray[np.int64]]:
    sigma_value:float = sigma.to(run_units.cross_section).value
    if sigma_value == 0:
        return 0, np.array([],dtype=np.int64)
    n_interactions = 0
    interacted:list[NDArray[np.int64]] = []
    round_dt = (dt/rounds).value
    for _ in range(rounds):
        pairs_buffer = np.full((max_n_allowed,2),-1,dtype=np.int64)
        pairs:NDArray[np.int64] = roll_scattering_pairs(r=r,v=v,dt=round_dt,m=m.value,sigma=sigma_value,regulator=regulator,max_radius_j=max_radius_j)
        pairs = utils.clean_pairs(pairs,blacklist)
        pairs_buffer[:len(pairs)] = pairs
        scatter_unique_pairs(v=v,pairs=pairs_buffer)
        n_interactions += len(pairs)
        interacted += [pairs.ravel()]
    return n_interactions,np.unique(np.hstack(interacted)).astype(np.int64)
