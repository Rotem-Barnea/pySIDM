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
    logrho, logv2, L, M = y
    rho = np.exp(logrho)
    v2 = np.exp(logv2)

    dM_dt = t**2 * rho
    dlogv2_dt = -L / (t**2 * rho * v2**2.5)
    dlogrho_dt = -M / (t**2 * v2) - dlogv2_dt
    dL_dt = t**2 * v2 * (c1 - c2 * (1.5 * dlogv2_dt - dlogrho_dt))

    return [dlogrho_dt, dlogv2_dt, dL_dt, dM_dt]


c1 = 1.903e-3
c2 = 8.092e-4

r_min, r_max = 1e-3, 1e4
# t_span = np.array([r_min, r_max])
t_span = np.array([r_max, r_min])

# y0 = [np.log(1), np.log(1), c1 * r_min**3 / 3, r_min**3 / 3]  # [rho, v, L, M]
y0 = [r_max ** (-2.25), 1, r_max ** (3 - 2.25) / 3, c1 * r_max**3 / 3]

sol = scipy.integrate.solve_ivp(
    lambda t, y: ode_system(t, y, c1, c2),
    t_span,
    y0,
    t_eval=np.linspace(*t_span, 1000),
    method='LSODA',
)
logrho, v2, L, M = sol.y
rho = np.exp(logrho)
r = sol.t
v = np.sqrt(v2)


fig, ax = plot.setup(xscale='log', yscale='log', xlim=(r_min, r_max), ylim=(1e-8, 1e2))
sns.lineplot(x=r, y=rho, ax=ax, label='rho')
sns.lineplot(x=r, y=M, ax=ax, label='M')
sns.lineplot(x=r, y=v, ax=ax, label='v')
sns.lineplot(x=r, y=L, ax=ax, label='L')
