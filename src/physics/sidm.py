import numpy as np
from numba import njit,prange
from typing import TypedDict
from numpy.typing import NDArray
from astropy import units
from .. import utils,run_units

class Zones(TypedDict):
    cutoff: units.Quantity['mass density']
    mini_rounds: NDArray[np.int64]

class Params(TypedDict,total=False):
    max_radius_j: int
    regulator: units.Quantity['length']
    base_rounds: int
    max_interactions_allowed_per_timestep:int
    density_zones:Zones
    sigma: units.Quantity['opacity']

default_density_zones:Zones={
    'cutoff':units.Quantity([10**7,10**8,10**9],'Msun/kpc^3'),
    'mini_rounds':np.array([10,20,50],dtype=np.int64),
}

default_params:Params={
    'max_radius_j':10,
    'regulator':units.Quantity(1e-10,'kpc').to(run_units.length),
    'base_rounds':1,
    'max_interactions_allowed_per_timestep':10000,
    'density_zones':default_density_zones,
    'sigma':units.Quantity(0,'cm^2/gram').to(run_units.cross_section)
}

def prepare_zones(density_zones:Zones,density:NDArray[np.float64],dt:float) -> tuple[NDArray[np.float64],NDArray[np.int64],int]:
    cutoff = density_zones['cutoff'].to(run_units.density).value
    indices = np.argsort(cutoff)
    cutoff = np.hstack([0,cutoff[indices],np.inf])
    rounds = np.hstack([1,density_zones['mini_rounds'][indices]])
    output_dt = np.zeros_like(density)
    mini_rounds = np.zeros_like(density,dtype=np.int64)
    max_mini_rounds = 0
    for low,high,rounds in zip(cutoff[:-1],cutoff[1:],rounds):
        mask = (density >= low)*(density < high)
        output_dt[mask] = dt/rounds
        mini_rounds[mask] = rounds
        if mask.any():
            max_mini_rounds = max(max_mini_rounds,rounds)
    return output_dt,mini_rounds,max_mini_rounds

@njit
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
def roll_scattering_pairs(r:NDArray[np.float64],v:NDArray[np.float64],dt:NDArray[np.float64],m:float,sigma:float,regulator:float,max_radius_j:int,
                          whitelist_mask:NDArray[np.bool_],blacklist:NDArray[np.int64]=np.array([],dtype=np.int64)) -> tuple[NDArray[np.int64],NDArray[np.bool_],NDArray[np.float64]]:
    pairs = np.empty((len(r),2),dtype=np.int64)
    pair_found = np.full(len(r),False)
    probabilities = np.zeros_like(r)
    if sigma > 0:
        delta_r:NDArray[np.float64] = np.zeros_like(r)
        delta_r[:-max_radius_j] = r[max_radius_j:]
        delta_r[-max_radius_j:] = r[-1]
        delta_r -= r

        whitelist = np.arange(len(r))[whitelist_mask]
        blacklist = blacklist[blacklist != -1]
        for i in prange(len(whitelist)):
            particle = whitelist[i]
            if delta_r[particle] == 0 or particle in blacklist:
                continue
            # volume = 4*np.pi*r[particle]**2*delta_r[particle]+regulator
            volume = 4/3*np.pi*((r[particle]+delta_r[particle])**3-r[particle]**3)+regulator
            p = np.random.rand()
            for offset in range(1,max_radius_j+1):
                partner = particle+offset
                if partner >= len(r) or partner in blacklist:
                    continue
                v_rel = v[partner]-v[particle]
                v_rel_norm = np.sqrt((v_rel**2).sum())
                # volume = 4/3*np.pi*(r[partner]**3-r[particle]**3)+regulator
                cross_section_term = sigma*v_rel_norm
                probabilities[particle] += dt[particle]/2*(m/volume)*cross_section_term
                if p <= probabilities[particle]:
                    pairs[particle] = [particle,partner]
                    pair_found[particle] = True
                    break
    return pairs,pair_found,probabilities

@njit(parallel=True)
def scatter_unique_pairs(v:NDArray[np.float64],pairs:NDArray[np.int64]) -> NDArray[np.float64]:
    output = v.copy()
    for pair_index in prange(len(pairs)):
        i0,i1 = pairs[pair_index]
        if i0 == -1 or i1 == -1:
            continue
        output[i0],output[i1] = scatter_pair_kinematics(v0=v[i0],v1=v[i1])
    return output

def scatter_found_pairs(v:NDArray[np.float64],found_pairs:NDArray[np.int64],memory_allocated:int) -> NDArray[np.float64]:
    if len(found_pairs) == 0:
        return v
    pairs = np.full((memory_allocated,2),-1,dtype=np.int64)
    pairs[:len(found_pairs)] = found_pairs
    return scatter_unique_pairs(v=v,pairs=pairs)

def scatter(r:NDArray[np.float64],v:NDArray[np.float64],density:NDArray[np.float64],dt:units.Quantity['time'],m:units.Quantity['mass'],
            sigma:units.Quantity['opacity'],blacklist:NDArray[np.int64]=np.array([],dtype=np.int64),base_rounds:int=default_params['base_rounds'],
            regulator:units.Quantity['length']=default_params['regulator'],max_radius_j:int=default_params['max_radius_j'],
            density_zones:Zones=default_params['density_zones'],max_interactions_allowed_per_timestep:int=default_params['max_interactions_allowed_per_timestep']) -> tuple[NDArray[np.float64],int,NDArray[np.int64]]:
    output = v.copy()
    sigma_value:float = sigma.to(run_units.cross_section).value
    n_interactions = 0
    interacted:list[NDArray[np.int64]] = []
    if sigma_value > 0:
        zone_dt,mini_rounds,max_mini_rounds = prepare_zones(density_zones=density_zones,density=density,dt=dt.to(run_units.time).value/base_rounds)
        values = {'m':m.to(run_units.mass).value,'regulator':regulator.to(run_units.length).value,'sigma':sigma_value}
        for _ in range(base_rounds):
            for mini_round in range(max_mini_rounds):
                whitelist_mask = (mini_rounds >= mini_round)
                pairs,pair_found,_ = roll_scattering_pairs(r=r,v=v,dt=zone_dt,max_radius_j=max_radius_j,whitelist_mask=whitelist_mask,blacklist=blacklist,**values)
                found_pairs = utils.clean_pairs(pairs[pair_found],blacklist=blacklist)
                output = scatter_found_pairs(v=output,found_pairs=found_pairs,memory_allocated=max_interactions_allowed_per_timestep)
                n_interactions += len(found_pairs)
                interacted += [found_pairs.ravel()]
    return output,n_interactions,np.hstack(interacted).astype(np.int64)
