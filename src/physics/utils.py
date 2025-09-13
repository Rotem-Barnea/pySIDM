import numpy as np
from numba import njit
from typing import Literal,Optional
from ..spatial_approximation import Lattice
from ..density.density import Density
from .. import utils
from ..constants import G

Mass_calculation_methods = Literal['lattice','density','rank presorted','rank unsorted']

def M_below(r,unit_mass:float=1,lattice:Optional[Lattice]=None,density:Optional[Density]=None,count_self=True,method:Mass_calculation_methods='lattice'):
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
