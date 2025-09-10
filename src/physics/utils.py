import numpy as np
from numba import njit
from typing import Literal
from .. import utils
from ..constants import G

Mass_calcualtion_methods = Literal['lattice','density','rank presorted','rank unsorted']

def M_below(r,unit_mass=1,lattice=None,density=None,count_self=True,method:Mass_calcualtion_methods='lattice'):
    mask = np.isnan(r)
    M = np.zeros_like(r)
    if method == 'lattice' and lattice is not None:
        M[~mask] = (lattice.assign_from_density(r[~mask]) - (not count_self))*unit_mass
    elif method == 'density' and density is not None:
        M[~mask] = density(r[~mask])
    elif method == 'rank presorted':
        M[~mask] = (np.arange(len(r[~mask]))+count_self)*unit_mass
    else:
        M[~mask] = (utils.rank_array(r[~mask])+count_self)*unit_mass
    return M

@njit()
def orbit_cicular_velocity(r,M):
    return np.sqrt(G*M/r)
