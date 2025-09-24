import numpy as np
from numba import njit
from typing import Literal,cast
from astropy import units,constants
from ..spatial_approximation import Lattice
from .. import utils,run_units

Mass_calculation_methods = Literal['lattice','rank presorted','rank unsorted']

def get_default_mass_method(method:Mass_calculation_methods|None=None,sigma:units.Quantity[run_units.cross_section]=0*units.Unit('cm^2/gram')) -> Mass_calculation_methods:
    if method is None and sigma.value == 0:
        return 'lattice'
    return method or 'rank presorted'

def M(r:units.Quantity['length'],m:units.Quantity['mass']=units.Quantity(1,run_units.mass),lattice:Lattice|None=None,
      count_self:bool=True,method:Mass_calculation_methods='lattice') -> units.Quantity['mass']:
    if method == 'lattice' and lattice is not None:
        n_below = np.array((lattice.assign_from_density(r.to(run_units.length).value) - int(not count_self)))
    elif method == 'rank presorted':
        n_below = np.arange(len(r))+count_self
    else:
        n_below = utils.rank_array(r)+count_self
    return cast(units.Quantity['mass'],n_below*m)

@njit
def orbit_circular_velocity(r:units.Quantity['length'],M:units.Quantity['mass']) -> units.Quantity['velocity']:
    return np.sqrt(constants.G*M/r)

def local_density(r:units.Quantity['length'],max_radius_j:int=10,regulator:units.Quantity['length']=units.Quantity(0,'kpc^3'),
                  accuracy_cutoff:float=0.1) -> units.Quantity['number density']:
    """Assumes the array is sorted"""
    x = r.value
    dx = np.zeros_like(x)
    dx[:-max_radius_j] = x[max_radius_j:]
    dx[-max_radius_j:] = x[-1]
    dx -= x
    n = np.full(len(x),max_radius_j,dtype=np.int64)
    n[-max_radius_j:] = np.arange(max_radius_j-1,-1,-1)

    volume = np.full_like(x,regulator.value)
    mask = dx/x > accuracy_cutoff
    volume[~mask] += 4*np.pi*dx[~mask]**2*x[~mask]
    volume[mask] += 4/3*np.pi*((dx[mask]+x[mask])**3-dx[mask]**3)
    density = n[:-1]/volume[:-1]
    density = np.hstack([density,density[-1]])
    return units.Quantity(density,1/cast(units.Unit,r.unit)**3)
