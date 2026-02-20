import os
import sys

sys.path.append(os.path.join(os.getcwd(), '..'))

import os
import glob
import pickle
import shutil
from pathlib import Path

import numpy as np
import seaborn as sns
import matplotlib
import astropy.units as ut
import matplotlib.pyplot as plt
from astropy import constants, cosmology
from astropy.units import Quantity

import agama_wrappers
from src import GravothermalSIDM, plot, types, units, physics, distribution, gravothermal_fluid
from src.utils import utils
from src.halo.halo import Halo
from src.background import BackgroundDistribution
from src.phase_space import PhaseSpace
from GravothermalSIDM.SourcePy.evolve import Halo
from GravothermalSIDM.SourcePy.record import HaloRecord

if os.path.exists(Path('../analysis/Data')):
    shutil.rmtree(Path('../analysis/Data'))

hrec = HaloRecord('Data/draco run')
hevo = Halo(hrec, flag_hydrostatic_initial=True, r_s=0.3310, rho_s=0.0736, sigma_m_with_units=50)
# hevo.evolve_halo(t_end=1, save_frequency_rate=10)

hevo.get_shell_midpoints().shape

with open('temp.pkl', 'wb') as f:
    pickle.dump(
        dict(
            r=hevo.r * hevo.scale_r,
            r_=hevo.r,
            density=hevo.rho * (hevo.scale_rho).decompose(units.system),
            density_=hevo.rho,
            mass=hevo.m * (hevo.scale_m).decompose(units.system),
            m_=hevo.m,
            L=hevo.L * (hevo.scale_L).decompose(units.system),
            L_=hevo.L,
            pressure=hevo.p * (hevo.scale_p).decompose(units.system),
            u=hevo.u * (hevo.scale_u).decompose(units.system),
            u_=hevo.u,
            v=hevo.v * (hevo.scale_v).decompose(units.system),
            sigma=hevo.sigma_m * hevo.scale_sigma_m.to('cm^2/g'),
            Kinv_smfp=hevo.Kinv_smfp,
            Kinv_lmfp=hevo.Kinv_lmfp,
            scale_r=hevo.scale_r,
            scale_t=hevo.scale_t,
            scale_L=hevo.scale_L,
            scale_m=hevo.scale_m,
            L_grad_=np.hstack([(hevo.L[0] / hevo.m[0]), (hevo.L[1:] - hevo.L[:-1]) / (hevo.m[1:] - hevo.m[:-1])]),
        ),
        f,
    )

fig, ax = plot.setup(xscale='log', yscale='log', xlabel='Radius', ylabel='Density', minorticks=True)
sns.lineplot(
    x=hevo.r * hevo.scale_r,
    y=hevo.rho * (hevo.scale_rho).decompose(units.system),
)

fig, ax = plot.setup(xscale='log')
sns.lineplot(
    x=hevo.r * hevo.scale_r,
    y=hevo.L * (hevo.scale_L).decompose(units.system),
)

fig, ax = plot.setup(xscale='log')
sns.lineplot(
    x=hevo.get_shell_midpoints()[1:] * hevo.scale_r,
    y=(hevo.L[1:] - hevo.L[:-1]) / (hevo.m[1:] - hevo.m[:-1]) * (hevo.scale_L / hevo.scale_m).decompose(units.system),
)
# sns.lineplot(x=hevo.r * hevo.scale_r, y=(hevo.L * hevo.scale_L).decompose(units.system), ax=ax)
