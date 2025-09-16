import numpy as np
from numba import njit
from typing import Literal,Optional
from ..spatial_approximation import Lattice
from ..density.density import Density
from .. import utils
from ..constants import G

Mass_calculation_methods = Literal['lattice','density','rank presorted','rank unsorted']

def get_default_mass_method(method:Optional[Mass_calculation_methods]=None,sigma:float=0,**kwargs) -> Mass_calculation_methods:
    if method is None and sigma == 0:
        return 'lattice'
    return method or 'rank presorted'

def M_below(r,unit_mass:float=1,lattice:Optional[Lattice]=None,density:Optional[Density]=None,count_self=True,method:Mass_calculation_methods='lattice'):
    if method == 'lattice' and lattice is not None:
        return (lattice.assign_from_density(r) - (not count_self))*unit_mass
    elif method == 'density' and density is not None:
        return density(r)
    elif method == 'rank presorted':
        return (np.arange(len(r))+count_self)*unit_mass
    else:
        return (utils.rank_array(r)+count_self)*unit_mass

@njit()
def orbit_circular_velocity(r,M):
    return np.sqrt(G*M/r)
