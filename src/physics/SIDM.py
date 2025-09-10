import numpy as np
from numba import njit,prange
from typing import Literal
from .. import utils

# @njit(parallel=True)
# def multi_roll_scattering_pairs_worker(r,vr,vp,indices,dt,unit_mass,sigma,scatter_multiplicity=1,regulator=0,max_radius_j=10,buffer_size=100000):
#     pairs_threads = np.full((scatter_multiplicity,buffer_size,2),-1,dtype=np.int64)
#     mini_dt = dt/scatter_multiplicity
#     for thread in prange(scatter_multiplicity):
#         rolled_pairs = roll_scattering_pairs(r,vr,vp,indices,dt=mini_dt,unit_mass=unit_mass,sigma=sigma,regulator=regulator,max_radius_j=max_radius_j)
#         for j in range(len(rolled_pairs)):
#             if rolled_pairs[j,0] == rolled_pairs[j,1]:
#                 continue
#             pairs_threads[thread,j,0] = rolled_pairs[j,0]
#             pairs_threads[thread,j,1] = rolled_pairs[j,1]
#     return pairs_threads

# def multi_roll_scattering_pairs(r,vr,vp,indices,dt,unit_mass,sigma,scatter_multiplicity=1,regulator=0,max_radius_j=10,buffer_size=100000):
#     output = multi_roll_scattering_pairs_worker(r,vr,vp,indices,dt,unit_mass,sigma,scatter_multiplicity,regulator,max_radius_j,buffer_size)
#     output = np.vstack(output)
#     return indices[output[output[:,0]!=-1]]

@njit()
def interaction_rate(v,particle,sigma,max_radius_j=10):
    v_rel = v[particle]-v[particle+1:particle+max_radius_j+1]
    v_rel_norm = np.sqrt((v_rel**2).sum(axis=1))
    return sigma*v_rel_norm

@njit(parallel=True)
def roll_scattering_lambda(r,v,dt,unit_mass,sigma,regulator=0,max_radius_j=10):
    P = np.zeros(len(r))
    if sigma == 0:
        return P

    delta_r = np.zeros_like(r)
    delta_r[:-max_radius_j] = r[max_radius_j:]
    delta_r[-max_radius_j:] = r[-1]
    delta_r -= r

    for particle in prange(len(r)):
        if delta_r[particle] == 0:
            continue
        cross_section_term = interaction_rate(v=v,particle=particle,sigma=sigma,max_radius_j=max_radius_j).sum()
        volume = 4*np.pi*r[particle]**2*delta_r[particle]+regulator
        P[particle] = dt/2*(unit_mass/volume)*cross_section_term
    return P

@njit(parallel=True)
def roll_scattering_pairs(r,v,dt,unit_mass,sigma,regulator=0,max_radius_j=10):
    if sigma == 0:
        return np.empty((0,2),dtype=np.int64)

    delta_r = np.zeros_like(r)
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
            cum_p += dt/2*(unit_mass/volume)*cross_section_term
            if p <= cum_p:
                pairs[particle] = [particle,partner]
                pair_found[particle] = True
                break
    return pairs[pair_found]

@njit()
def scatter_pair_kinematics(v0,v1):
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
def scatter_unique_pairs(v,pairs):
    for pair_index in prange(len(pairs)):
        i0,i1 = pairs[pair_index]
        if i0 == -1 or i1 == -2:
            continue
        v[i0],v[i1] = scatter_pair_kinematics(v0=v[i0],v1=v[i1])

def scatter(r,v,dt,unit_mass,sigma,blacklist=[],method:Literal['rounds','poisson']='rounds',rounds=1,regulator=0,max_radius_j=10,max_n_allowed=10000):
    n_interactions = 0
    if method == 'rounds':
        for _ in range(rounds):
            pairs_buffer = np.full((max_n_allowed,2),-1,dtype=np.int64)
            pairs = roll_scattering_pairs(r=r,v=v,dt=dt/rounds,unit_mass=unit_mass,sigma=sigma,regulator=regulator,max_radius_j=max_radius_j)
            pairs = utils.clean_pairs(pairs,blacklist)
            pairs_buffer[:len(pairs)] = pairs
            scatter_unique_pairs(v=v,pairs=pairs_buffer)
            n_interactions += len(pairs)
    else:
        n_events = np.random.poisson(roll_scattering_lambda(r,v,dt=dt,unit_mass=unit_mass,sigma=sigma,regulator=regulator,max_radius_j=max_radius_j))
        event_mask = n_events > 0
        indices = np.arange(len(r))[event_mask]
        n_events = n_events[event_mask]
        n_interactions = n_events.sum()

        pairs = []
        for particle,interactions in zip(indices,n_events):
            cdf = np.cumsum(interaction_rate(v,particle,sigma=sigma,max_radius_j=max_radius_j))
            cdf /= cdf[-1]
            partners = particle+np.argmax(np.random.rand(interactions,1) <= cdf,axis=1)
            for partner in partners:
                pairs += [[particle,partner]]

        for batch in utils.split_pairs_to_batches(pairs):
            scatter_unique_pairs(v=v,pairs=np.array(batch))

    return n_interactions
