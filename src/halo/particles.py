"""Internal module for handling halo particles"""

from __future__ import annotations

from typing import TYPE_CHECKING
from functools import cached_property

import numpy as np
import pandas as pd
from astropy import table
from astropy.units import Quantity

from src import types, units, utils

if TYPE_CHECKING:
    from .halo import Halo


class HaloParticles(object):
    """Class managing the halo's particles, allowing easier slicing and gathering"""

    def __init__(self, halo: 'Halo'):
        self.halo = halo

    def invalidate(self, *properties: str):
        """Invalidates the cache for the specified properties, so they are recalculated."""
        for property in properties:
            if property == 'density':
                continue
            try:
                delattr(self, property)
            except AttributeError:
                pass

    @cached_property
    def snapshots(self) -> table.QTable:
        """Snapshots of all particles"""
        return self.halo.get_particle_states()

    @cached_property
    def groups(self) -> table.TableGroups:
        """List of snapshots over time"""
        return self.snapshots.group_by('time').groups

    def get_particle_states(
        self,
        now: bool = True,
        snapshots: bool = True,
        initial: bool = False,
    ) -> table.QTable:
        """Return a table of particle snapshots, potentially including the initial and current states."""
        assert now or snapshots or initial, 'At least one of now, snapshots, or initial must be True'
        data_tables = []
        if now:
            data_tables += [self.particles]
        if snapshots:
            data_tables += [self.snapshots]
        if initial:
            data_tables += [self.initial_particles]
        return table.QTable(table.vstack(data_tables))

    @cached_property
    def now(self) -> table.QTable:
        """Particle's current state"""
        return self.particles

    def __getitem__(self, key: types.ParticleType | Quantity['time']) -> table.QTable:
        if isinstance(key, str):
            return utils.utils.slice_closest(self.snapshots, key, 'particle_type')
        return utils.utils.slice_closest(self.snapshots, key)

    def to_table(self, data: pd.DataFrame) -> table.QTable:
        """Particle data QTable.

        Has the following columns:
            r: Radius.
            vx: The first perpendicular component (to the radial direction) of the velocity.
            vy: The second perpendicular component (to the radial direction) of the velocity.
            vr: The radial velocity.
            vp: Tangential velocity (`np.sqrt(vx**2 + vy**2)`).
            m: Mass.
            v_norm: Velocity norm (`np.sqrt(vx**2 + vy**2 + vr**2)`).
            time: Current simulation time.
            E: Relative energy (`potential-1/2*m*v_norm**2`).
            particle_type: Type of particle.
            particle_index: Index of particle.
            distribution_id: Identifier of the source distribution.
            leapfrog_convergence_rounds: Number of leapfrog convergence rounds in the previous step.
        """
        data = data.copy().sort_values('r', kind=self.sort_kind)
        vx, vy, vr = [Quantity(data[key], units.velocity) for key in ['vx', 'vy', 'vr']]
        return table.QTable(
            {
                'r': Quantity(data['r'], units.length),
                'vx': vx,
                'vy': vy,
                'vr': vr,
                'vp': np.sqrt(vx**2 + vy**2),
                'v_norm': np.sqrt(vx**2 + vy**2 + vr**2),
                'm': Quantity(data['m'], units.mass),
                'time': [self.halo.time] * len(data),
                # 'E': Quantity(data['E'], units.energy),
                'particle_type': data['particle_type'],
                'particle_index': data.index,
                'distribution_id': data['distribution_id'],
                'leapfrog_convergence_rounds': data['leapfrog_convergence_rounds'],
            }
        )
