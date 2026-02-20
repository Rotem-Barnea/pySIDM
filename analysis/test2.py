import os
import sys

sys.path.append(os.path.join(os.getcwd(), '..'))

import numpy as np
import scipy
import seaborn as sns
from astropy import constants, cosmology
from astropy.units import Quantity

from src import plot


def ode_system(t, y, c1, c2):
    logrho, v2, L = y
    M = np.exp(t)

    dv2_dr = -L / (r**2 * rho * v2**1.5)

    # dv2_dt = -L / v2**1.5

    dlogrho_dt = -M / v2 - dv2_dx / v2

    dL_dt = M * v2 * (c1 - c2 * (1.5 * dv2_dt / v2 - dlogrho_dt))

    return [dlogrho_dx, dv2_dx, dL_dt]


c1 = 1.903e-3
c2 = 8.092e-4

r_min, r_max = 1e-4, 1e4
t_span = np.array([np.log(r_min**3 / 3), np.log(r_max**3 / 3)])

y0 = [np.log(1), 1, c1 * r_min**3 / 3]  # [rho, v, L]

sol = scipy.integrate.solve_ivp(
    lambda t, y: ode_system(t, y, c1, c2),
    t_span,
    y0,
    t_eval=np.linspace(*t_span, 1000),
    method='LSODA',
)
logrho, v2, L = sol.y
rho = np.exp(logrho)
M = np.exp(sol.t)
v = np.sqrt(v2)

r_approx = (3 * M) ** (1 / 3)

# Then iterate:
dr_dM = 1 / (r_approx**2 * rho)
r = np.concatenate([[r_min], r_min + scipy.integrate.cumulative_trapezoid(dr_dM, M)])

# Can refine by iterating:
for iteration in range(5):
    dr_dM = 1 / (r**2 * rho)
    r = np.concatenate([[r_min], r_min + scipy.integrate.cumulative_trapezoid(dr_dM, M)])


r = (3 * M) ** (1 / 3)

fig, ax = plot.setup(xscale='log', yscale='log', xlim=(r_min, r_max), ylim=(1e-8, 1e2))
sns.lineplot(x=r, y=rho, ax=ax, label='rho')
sns.lineplot(x=r, y=M, ax=ax, label='M')
sns.lineplot(x=r, y=v, ax=ax, label='v')
sns.lineplot(x=r, y=L, ax=ax, label='L')
