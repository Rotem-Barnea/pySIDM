from typing import cast
from astropy import units

## Base work units
length:units.UnitBase = cast(units.UnitBase,units.kpc)
time:units.UnitBase = cast(units.UnitBase,units.Myr)
mass:units.UnitBase = cast(units.UnitBase,units.Msun)

## Compound units
velocity:units.UnitBase = cast(units.UnitBase,length/time)
acceleration:units.UnitBase = cast(units.UnitBase,length/time**2)
specific_energy:units.UnitBase = cast(units.UnitBase,velocity**2)
density:units.UnitBase = cast(units.UnitBase,mass/length**3)
number_density:units.UnitBase = cast(units.UnitBase,1/length**3)
cross_section:units.UnitBase = cast(units.UnitBase,length**2/mass)
G_units:units.UnitBase = cast(units.UnitBase,length**3/(mass*time**2))
f_units:units.UnitBase = cast(units.UnitBase,density/(mass*specific_energy**(3/2)))
