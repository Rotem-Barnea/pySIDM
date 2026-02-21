import os
import sys

sys.path.append(os.path.join(os.getcwd(), '..'))

import re
import json
from pathlib import Path

import numpy as np
import seaborn as sns
from astropy import constants, cosmology
from astropy.units import Quantity

import agama_wrappers
from src import plot, types, units, utils, distribution, gravothermal_fluid
from src.halo import compare, run_optimization
from src.tqdm import tqdm
from src.halo.halo import Halo
from src.background import BackgroundDistribution
from src.phase_space import PhaseSpace

paths = sorted(list(Path('../run results/2026-02-21').glob('*')))

halo = Halo.load(paths[0], verbose=False)

halos = compare.Halos.from_paths(paths)
path_condition = {
    (f := re.findall(r'fraction=([\d.]+)', path.name)[0]): {'label': f'CDM fraction={f}'} for path in paths
}

halos[5].animate_enclosed_mass_ratio(
    'cdm',
    length_unit=halos[0].distributions[0].r_s_unit,
    ylim=(1e-1, 1e1),
    yscale='log',
    reference=float(re.findall(r'fraction=([\d.]+)', path.name)[0]),
    save_kwargs={'save_path': 'test.gif'},
)

halos[5].plot_enclosed_mass_ratio(
    'cdm',
    times=Quantity([0], 'Gyr'),
    length_unit=halos[0].distributions[0].r_s_unit,
    palette='magma',
)

fig, ax = halos.plot_cumulative_scattering(
    time_unit='Gyr',
    path_condition=path_condition,
    palette='magma',
    plot_kwargs={'undersample': 5, 'per_dm_particle': False, 'xscale': 'linear'},
    save_kwargs={'save_path': 'CDM fraction cumulative scatter compare.png'},
)

fig, ax = halos.plot_core_density(
    time_unit='Gyr',
    path_condition=path_condition,
    palette='magma',
    plot_kwargs={'filter_particle_type': 'dm', 'smoothing_sigma': 5},
    save_kwargs={'save_path': 'CDM fraction core density ratio compare.png'},
)


from matplotlib import colors

halo = Halo.load(paths[0], verbose=False)
halos[0].animate.phase_space(
    x_bins=Quantity(np.linspace(0, 250, 200), 'kpc'),
    y_bins=Quantity(np.linspace(0, 250, 200), 'km/second'),
    xscale='linear',
    norm=colors.Normalize(vmin=0, vmax=40),
    save_kwargs={'save_path': 'Phase space f=0.1.gif'},
)

halo = Halo.load(paths[-2])
f_SIDM = 1 - float(re.findall(r'fraction=([\d.]+)', halo.save_path.name)[0])
halo.plot.local_density(
    # times=Quantity([0, 2 / f_SIDM, 11], 'Gyr'),
    times=Quantity(np.arange(0, halo.time.to('Gyr').value, 1), 'Gyr'),
    filter_particle_type='dm',
    title=f'%SIDM = {f_SIDM:.1f}',
    palette='magma',
    xlim=(1e-1, 4e2),
    ylim=(1e0, 1e10),
    max_radius_j=100,
)

halo.animate.local_density(
    filter_particle_type='dm',
    xlim=(1e-1, 4e2),
    ylim=(1e0, 1e10),
    max_radius_j=100,
    save_kwargs={'save_path': 'CDM fraction SIDM density f=0.1.gif'},
)

fig, ax = plot.setup(
    xlabel=r'f $\times$ Time',
    x_unit='Gyr',
    ylabel=r'$f^{-1}$ $\times$ $\rho$',
    y_unit=units.density,
    # yscale='log',
    # xscale='log',
)
for color, halo in plot.color_palette(tqdm(halos), 'magma'):
    f_SIDM = 1 - float(re.findall(r'fraction=([\d.]+)', halo.save_path.name)[0])

    data = halo.get_particle_states(now=False, filter_particle_type='dm').to_pandas()
    data = data[data['r'] < halo.inner_core_radius.value]
    scaleup_factor = (halo.scatter_params['sigma'] / Quantity(50, 'cm^2/g')).decompose()
    t, rho = (
        (data.groupby('time')['m'].agg('sum') / (4 / 3 * np.pi * halo.inner_core_radius.value))
        .reset_index()
        .to_numpy()
        .T
    )
    t = Quantity(t, units.time)
    rho = Quantity(rho, units.density)
    sns.lineplot(
        x=t.to('Gyr').value[::10] * f_SIDM,
        y=utils.utils.gaussian_filter1d(rho[::10], sigma=1).value / f_SIDM,
        ax=ax,
        label=f'CDM fraction={1 - f_SIDM:.1f}',
        color=color,
    )
plot.save(fig=fig, save_path='CDM fraction core density compare scaled.png')

data = halo.get_particle_states().to_pandas()
data = data[data['r'] < halo.inner_core_radius.value]
t, rho = (
    data.groupby('time')['m'].agg('sum') / (4 / 3 * np.pi * halo.inner_core_radius.value).reset_index().to_numpy().T
)
t = Quantity(t, units.time)
rho = Quantity(rho, units.density)


fig, ax = plot.setup(
    xlabel=r'$\text{SIDM fraction}$ $\times$ Time',
    x_unit='Gyr',
    ylabel='Cumulative number of scattering events / SIDM particle',
    # yscale='log',
)
for color, halo in plot.color_palette(tqdm(halos.halos), 'magma'):
    t = halo.scatter_times.to('Gyr')
    y = Quantity(halo.n_scatters.cumsum())
    f_SIDM = 1 - float(re.findall(r'fraction=([\d.]+)', halo.save_path.name)[0])
    sns.lineplot(
        x=t.to('Gyr').value * f_SIDM,
        y=y / halo.n_particles['dm'],
        ax=ax,
        label=f'CDM fraction={1 - f_SIDM:.1f}',
        color=color,
    )
plot.save(fig=fig, save_path='CDM fraction cumulative scatter compare scaled.png')


halos.print_core_collapse_time(
    path_condition={
        (f := re.findall(r'fraction=([\d.]+)', path.name)[0]): {'label': f'CDM fraction={f}'} for path in paths
    }
)


halo = Halo.load('../run results/CDM+SIDM single distribution fraction=0')
# halo = Halo.load('../run results/Draco single tester')
halo = Halo.load('../run results/CDM+SIDM single fraction=0.5 [10960572]')
halo = Halo.load('../run results/CDM+SIDM single fraction=0.1 [10961345]')

ps = halo.phase_space_snapshots

ps.animate_plot(
    'density', text_label_unit='Gyr', save_kwargs={'save_path': 'test.gif'}, xlim=(1e-2, 1e2), ylim=(1e3, 1e12)
)


fig, ax = plot.setup(xscale='log', yscale='log', xlim=(1e-2, 3e2), ylim=(1e1, 1e10))
for color, group in plot.color_palette(list(groups)[:1], 'magma'):
    sns.lineplot(
        x=group['r'],
        y=utils.physics.local_density(
            r=group['r'],
            m=group['m'],
            max_radius_j=100,
            volume_kind='density',
            mass_kind='sum',
        ),
        ax=ax,
        color=color,
        label=group['time'][0],
    )
sns.lineplot(
    x=halo.initial_particles['r'],
    y=utils.physics.local_density(
        r=halo.initial_particles['r'],
        m=halo.initial_particles['m'],
        max_radius_j=100,
        volume_kind='density',
        mass_kind='sum',
    ),
    ax=ax,
    color='tab:green',
    label='setup',
)
ax.legend()
halo.distributions[0].plot_density(fig=fig, ax=ax, add_markers=False)

import scipy

x = halo.initial_particles['r']
y = utils.physics.local_density(
    r=halo.initial_particles['r'],
    m=halo.initial_particles['m'],
    max_radius_j=10,
    volume_kind='density',
    mass_kind='sum',
)
x = group['r']
y = utils.physics.local_density(
    r=group['r'],
    m=group['m'],
    max_radius_j=100,
    volume_kind='density',
    mass_kind='sum',
)

distribution.NFW.fit_data(x=x, y=y, data_mask=(x > Quantity(1e0, 'kpc')) * (x < Quantity(5e1, 'kpc')))
distribution.NFW(
    **distribution.NFW.fit_data(x=x, y=y, data_mask=(x > Quantity(1e0, 'kpc')) * (x < Quantity(5e1, 'kpc')))
)

# 'CDM+SIDM single fraction=0.1 [10961345]'  'CDM+SIDM single fraction=0.6 [10961349]'
# 'CDM+SIDM single fraction=0.2 [10961346]'  'CDM+SIDM single fraction=0.7 [10961350]'
# 'CDM+SIDM single fraction=0.3 [10961347]'  'CDM+SIDM single fraction=0.8 [10961351]'
# 'CDM+SIDM single fraction=0.4 [10961348]'  'CDM+SIDM single fraction=0.9 [10961352]'
# 'CDM+SIDM single fraction=0.5 [10960572]'


halo.distributions[0]
halo.unoptimized_dt

10000 * halo.unoptimized_dt

halo.distributions[0]

halo.plot_core_density_ratio(time_unit='Gyr', include_start=False)
