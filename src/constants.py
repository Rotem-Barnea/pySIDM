from typing import TypedDict
# from astropy import units,constants
from . import si_units as SI

class Unit(TypedDict):
    name: str
    value: float

## Work units
length_unit = SI.kpc
time_unit = SI.Myr
mass_unit = SI.Msun

# time_unit_ = 'Myr'

## SI units redefinition to the work units
# m = (1*units.m).value/length_unit
# second = (1*units.second).value/time_unit
# kg = (1*units.kg).value/mass_unit
# kpc:float = (1*units.kpc).value/length_unit
m = SI.m/length_unit
second = SI.second/time_unit
kg = SI.kg/mass_unit
kpc = SI.kpc/length_unit
year = SI.year/time_unit
Myr = SI.Myr/time_unit
Gyr = SI.Gyr/time_unit
Msun = SI.Msun/mass_unit
# G:float = constants.G.value/(length_unit**3/(mass_unit*time_unit**2))
G:float = SI.G/(length_unit**3/(mass_unit*time_unit**2))
cm = SI.cm/length_unit
km = SI.km/length_unit
gram = SI.gram/mass_unit

cross_section = SI.cross_section/(length_unit**2/mass_unit)

## default unit handles
unit_name={
    'length':'kpc',
    'time':'Myr',
    'mass':'Msun',
}
unit_name.update({
    'velocity':f'{unit_name['length']}/{unit_name['time']}',
    'acceleration':f'{unit_name['length']}/{unit_name['time']}^2',
    'energy':f'{unit_name['mass']}*({unit_name['length']}/{unit_name['time']})^2',
    'density':f'{unit_name['mass']}/{unit_name['length']}^3',
})

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
