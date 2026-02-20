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
    M, rho, v2, L, r = y

    # eps = 1e-30
    # if r < eps or rho < eps or v2 < eps or M < eps:
    #     return [0, 0, 0, 0]

    dr_dx = M / (r**2 * rho)
    dM_dx = M
    dv2_dr = -L / (r**2 * rho * v2**1.5)
    dv2_dx = dv2_dr * dr_dx
    drho_dr = (-M * rho / r**2 - rho * dv2_dr) / v2
    drho_dx = drho_dr * dr_dx
    d_ln_v3_rho_dx = 1.5 * dv2_dx / v2 - drho_dx / rho
    dL_dx = M * v2 * (c1 - c2 * d_ln_v3_rho_dx)

    return [dM_dx, drho_dx, dv2_dx, dL_dx, dr_dx]


    # dM_dr = r**2 * rho
    # dv2_dr = -L / (r**2 * rho * v2**1.5)
    # drho_dr = (-M * rho / r**2 - rho * dv2_dr) / v2

    # d_ln_v3_rho_dr = 3 / 2 * dv2_dr / v2 - drho_dr / rho
    # d_ln_M_dr = dM_dr / M

    # dL_dr = r**2 * rho * v2 * (c1 - c2 * d_ln_v3_rho_dr / d_ln_M_dr)

    # return [dM_dr, drho_dr, dv2_dr, dL_dr]


c1 = 1.903e-3
c2 = 8.092e-4

r_min,r_max = 1e-4,1e4
x_span = np.array([np.log(r_min**3 / 3), np.log(r_max**3 / 3)])
x_eval = np.linspace(*x_span, 1000)

y0 = [r_min ** 3 / 3, 1, 1, c1 * r_min**3 / 3,r_min]  # [M_*, rho_*, v_*, L_*, r_*]

sol = scipy.integrate.solve_ivp(
    lambda t, y: ode_system(t, y, c1, c2),
    x_span,
    y0,
    t_eval=x_eval,
    method='LSODA',
)
M,rho,v2,L,r = sol.y
# r = sol.t[-100:]
# rho = sol.y[1, -100:]

# alpha_fit = -np.polyfit(np.log(r), np.log(rho), 1)[0]
# alpha_tho = 6 * (c1 - c2) / (2 * c1 - c2)

# alpha_fit, alpha_tho

fig, ax = plot.setup(xscale='log', yscale='log',xlim=r_span,ylim=(1e-8,1e2))
sns.lineplot(x=r, y=rho, ax=ax)
