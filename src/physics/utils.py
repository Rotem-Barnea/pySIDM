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

@njit
def orbit_circular_velocity(r:units.Quantity['length'],M:units.Quantity['mass']) -> units.Quantity['velocity']:
    return np.sqrt(constants.G*M/r)

def local_density(r:NDArray[np.float64],max_radius_j:int=10,regulator:float=0,accuracy_cutoff:float=0.1) -> NDArray[np.float64]:
    """Assumes the array is sorted"""
    delta_r = np.zeros_like(r)
    delta_r[:-max_radius_j] = r[max_radius_j:]
    delta_r[-max_radius_j:] = r[-1]
    delta_r -= r
    n = np.full(len(r),max_radius_j,dtype=np.int64)
    n[-max_radius_j:] = np.arange(max_radius_j-1,-1,-1)

    volume = np.full_like(r,regulator)
    mask = delta_r/r > accuracy_cutoff
    volume[~mask] += 4*np.pi*r[~mask]**2*delta_r[~mask]
    volume[mask] += 4/3*np.pi*((r[mask]+delta_r[mask])**3-r[mask]**3)
    density = n[:-1]/volume[:-1]
    density = np.hstack([density,density[-1]])
    return density
