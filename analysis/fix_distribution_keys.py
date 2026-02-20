import os
import sys

sys.path.append(os.path.join(os.getcwd(), '..'))

import pickle
from pathlib import Path


def update_dist(path, stem):
    with open(Path(path) / f'{stem}.pkl', 'rb') as f:
        d = pickle.load(f)

    d._r_s = d._rho_s = d._r_vir = d._c = d._total_mass = None

    d.r_s = d.__dict__.pop('r_s')
    d.rho_s = d.__dict__.pop('rho_s')
    d.r_vir = d.__dict__.pop('r_vir')
    d.c = d.__dict__.pop('c')
    d.total_mass = d.__dict__.pop('total_mass')

    d.save(path=path, stem=stem)


update_dist(path='../run results/Fornax Shp dynamic baryons 2/distributions', stem='Fornax Shp_NFW')
update_dist(path='../run results/Fornax Shp dynamic baryons 2/distributions', stem='Fornax Shp_Hernquist')
# update_dist(path='../run results/Fornax Shp dynamic baryons', stem='background_distribution')
