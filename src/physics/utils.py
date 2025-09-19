import numpy as np
from numba import njit
from numpy.typing import NDArray
from typing import Literal
from astropy import units,constants
from ..spatial_approximation import Lattice
from ..density.density import Density
from .. import utils

Mass_calculation_methods = Literal['lattice','density','rank presorted','rank unsorted']

def get_default_mass_method(method:Mass_calculation_methods|None=None,sigma:units.Quantity['opacity']=0*units.Unit('cm^2/gram')) -> Mass_calculation_methods:
    if method is None and sigma.value == 0:
        return 'lattice'
    return method or 'rank presorted'

def M_below(r:NDArray[np.float64],unit_mass:float=1,lattice:Lattice|None=None,density:Density|None=None,count_self:bool=True,
            method:Mass_calculation_methods='lattice') -> NDArray[np.float64]:
    if method == 'lattice' and lattice is not None:
        return (lattice.assign_from_density(r) - int(not count_self))*unit_mass
    elif method == 'density' and density is not None:
        return density(r)
    elif method == 'rank presorted':
        return (np.arange(len(r))+count_self)*unit_mass
    else:
        return (utils.rank_array(r)+count_self)*unit_mass

@njit()
def orbit_circular_velocity(r:units.Quantity['length'],M:units.Quantity['mass']) -> units.Quantity['velocity']:
    return np.sqrt(constants.G*M/r)
