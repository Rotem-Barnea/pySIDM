from typing import cast

from astropy.units import Unit, UnitBase

## Base work units
length: UnitBase = cast(UnitBase, Unit('kpc'))
time: UnitBase = cast(UnitBase, Unit('Myr'))
mass: UnitBase = cast(UnitBase, Unit('Msun'))
system = [length, time, mass]

## Compound units
velocity: UnitBase = cast(UnitBase, length / time)
acceleration: UnitBase = cast(UnitBase, length / time**2)
energy: UnitBase = cast(UnitBase, mass * velocity**2)
specific_energy: UnitBase = cast(UnitBase, velocity**2)
specific_angular_momentum: UnitBase = cast(UnitBase, length * velocity)
density: UnitBase = cast(UnitBase, mass / length**3)
number_density: UnitBase = cast(UnitBase, 1 / length**3)
cross_section: UnitBase = cast(UnitBase, length**2 / mass)
f_unit: UnitBase = cast(UnitBase, density / (mass * specific_energy ** (3 / 2)))
F_unit: UnitBase = cast(UnitBase, density / (mass * specific_energy ** (1 / 2)))
