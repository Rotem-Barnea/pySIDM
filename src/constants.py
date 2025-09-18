from typing import TypedDict
from astropy import units,constants

class Unit(TypedDict):
    name: str
    value: float

## Work units
length:units.UnitBase = units.kpc
time:units.UnitBase = units.Myr
mass:units.UnitBase = units.Msun

## Compound units
velocity:units.UnitBase = length/time
acceleration:units.UnitBase = length/time**2
energy:units.UnitBase = mass*velocity**2
density:units.UnitBase = mass/length**3

assert isinstance(length,units.UnitBase)
assert isinstance(time,units.UnitBase)
assert isinstance(mass,units.UnitBase)
assert isinstance(velocity,units.UnitBase)
assert isinstance(acceleration,units.UnitBase)
assert isinstance(energy,units.UnitBase)
assert isinstance(density,units.UnitBase)

## SI units redefinition to the work units
m:float = (1*units.m).to(length).value
second:float = (1*units.second).to(time).value
kg:float = (1*units.kg).to(mass).value
kpc:float = (1*units.kpc).to(length).value
year:float = (1*units.year).to(time).value
Myr:float = (1*units.Myr).to(time).value
Gyr:float = (1*units.Gyr).to(time).value
Msun:float = (1*units.Msun).to(mass).value
G:float = constants.G.to(length**3/(mass*time**2)).value
cm:float = (1*units.cm).to(length).value
km:float = (1*units.km).to(length).value
gram:float = (1*units.gram).to(mass).value

cross_section:float = (1*units.Unit('cm^2/gram')).to(length**2/mass).value

## default unit handles
unit_name={
    'length':str(length),
    'time':str(time),
    'mass':str(mass),
    'velocity':str(velocity),
    'acceleration':str(acceleration),
    'energy':str(energy),
    'density':str(density),
}

def default_units(x:str) -> Unit:
    match x:
        case 'length':
            return {'value':kpc,'name':'kpc'}
        case 'velocity':
            return {'value':km/second,'name':'km/second'}
        case 'mass':
            return {'value':Msun,'name':'Msun'}
        case 'time':
            return {'value':Myr,'name':'Myr'}
        case 'Tdyn':
            return {'value':1,'name':'Tdyn'}
        case _:
            return {'value':1,'name':unit_name.get(x,'')}
