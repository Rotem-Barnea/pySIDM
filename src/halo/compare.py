"""Module for analysing and comparing multiple halos post run"""

from typing import Any, Literal, cast
from pathlib import Path
from itertools import zip_longest

import numpy as np
import seaborn as sns
from astropy import table
from numpy.typing import NDArray
from astropy.units import Quantity
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from astropy.units.typing import UnitLike

from src import plot, utils, run_units
from src.tqdm import tqdm
from src.types import ParticleType
from src.phase_space import PhaseSpace
from src.distribution import PhysicalProperty

from .halo import Halo
from .types import TimeUnitLike

PlotProperty = PhysicalProperty | Literal['baryon mass ratio']


class Halos:
    """A collection of halos for comparison and analysis."""

    def __init__(self, halos: list[Halo] = []) -> None:
        """A collection of halos for comparison and analysis."""
        self.halos = halos

    def __getitem__(self, index: int) -> Halo:
        return self.halos[index]

    @classmethod
    def from_paths(cls, paths: list[str | Path], verbose: bool = False) -> 'Halos':
        """Create form a list of paths."""
        return cls([Halo.load(path=path, verbose=verbose) for path in tqdm(paths, disable=not verbose)])

    def _plot_baryon_mass_ratio(
        self,
        halo: Halo,
        data: table.QTable,
        x_unit: UnitLike | None = None,
        label: str | None = None,
        initial_particles: bool = False,
        final_particles: bool = True,
        lineplot_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        r = utils.get_columns(data, ['r'])[0]
        if x_unit is None:
            x_unit = str(r.unit)

        enclosed_dm_mass = utils.get_columns(data, ['m'])[0].cumsum()
        if halo.background is not None and halo.background.distribution is not None:
            enclosed_baryon_mass = halo.background.distribution.enclosed_mass(r)
            ratio = enclosed_baryon_mass / (enclosed_baryon_mass + enclosed_dm_mass)
        else:
            _data = utils.slice_closest(
                data=halo.get_particle_states(initial=initial_particles, now=final_particles),
                value=cast(Quantity, data['time'][0]),
            ).to_pandas()
            _data['DM mass'] = (_data['m'] * (_data['particle_type'] == 'dm')).cumsum()
            _data['baryon mass'] = (_data['m'] * (_data['particle_type'] == 'baryon')).cumsum()
            _data['enclosed mass'] = _data['baryon mass'] + _data['DM mass']
            _data['mass ratio'] = _data['baryon mass'] / _data['enclosed mass']
            _data = _data[_data['particle_type'] == 'dm']
            ratio = Quantity(_data['mass ratio'].to_numpy(), data['m'].unit)

        fig, ax = plot.setup(**kwargs)
        ax = sns.lineplot(
            x=r.to(x_unit).value,
            y=ratio,
            label=label,
            ax=ax,
            **lineplot_kwargs,
        )
        return fig, ax

    def plot(
        self,
        y: PlotProperty,
        time: Quantity['time'] | float,
        filter_particle_type: ParticleType = 'dm',
        initial_particles: bool = False,
        final_particles: bool = True,
        x_unit: UnitLike | None = None,
        y_unit: UnitLike | None = None,
        time_unit: UnitLike | None = None,
        labels: list[str] = [],
        labels_suffix: Literal['time'] | str | None = 'time',
        title: str | None | Literal['auto'] = 'auto',
        early_quit: bool = False,
        use_default_params: bool = True,
        plot_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot a physical property of the system accross all halos at time `time`.

        Parameters:
            y: The property to plot (i.e. temperature, pressure, etc.).
            time: The time to plot for each halo. If a `float`, treat as the value in terms of collapse time for each halo (i.e. based on the individual collapse time).
            filter_particle_type: Type of particles to plot.
            include_start: Whether to include the initial halo states.
            include_now: Whether to include the final (current) halo states.
            x_unit: The units for the x-axis.
            y_unit: The units for the y-axis.
            time_unit: The time units for labels.
            label: The labels to add to each halo's plot.
            labels_suffix: Suffix to add to the labels. If `auto` uses the tempate `t={time}` where `time` in either `time_unit` (if `time` is a float), or in the collapse time units for each individual halo (if `time` is a Quantity).
            title: Title for the plot. If `auto` uses the template `{filter_particle_type} {y} at time t={time}` where `time` in either `time_unit` (if `time` is a Quantity), or in the collapse time units (if `time` is a float).
            early_quit: If `True` force the plot on a predefined `fig` and `ax`. If `False` allows the plotting function (`phase_space.plot()`) to define axis parameters (title, axis labels, etc.).
            use_default_params: Whether to use the default plot parameters defined in `Halos.defualt_plot_params()`.
            plot_kwargs: Keyword arguments to pass to `phase_space.plot()`.
            **kwargs: Additional keyword arguments to pass to the plot function (`plot.setup()`).

        Returns:
            fig, ax.
        """
        if use_default_params:
            kwargs = kwargs.copy()
            kwargs = {**kwargs, **self.default_plot_params(y=y)}
            x_unit = kwargs.pop('x_unit', x_unit)
            y_unit = kwargs.pop('y_unit', y_unit)
            time_unit = kwargs.pop('time_unit', time_unit)
        time_unit = time_unit or run_units.time
        fig, ax = plot.setup(x_unit=x_unit, y_unit=y_unit, **kwargs)
        aboslute_time_mode = isinstance(time, Quantity)
        for halo, label in zip_longest(self.halos, labels):
            _time = time if aboslute_time_mode else Quantity(time, halo.core_collapse_time)
            plot_kwargs = plot_kwargs.copy()
            lineplot_kwargs = plot_kwargs.pop('lineplot_kwargs', {})
            if labels_suffix == 'time':
                time_label = (
                    f'{_time.to(halo.core_collapse_time):.2f}' if aboslute_time_mode else f'{_time.to(time_unit):.1f}'
                )
                label = f'{label} t={time_label}'
            elif labels_suffix is not None:
                label = f'{label} {labels_suffix}'
            if title == 'auto':
                time_label = (
                    f'{_time.to(time_unit):.1f}' if aboslute_time_mode else f'{_time.to(halo.core_collapse_time):.2f}'
                )
                if y == 'baryon mass ratio':
                    title = f'Stellar enclosed mass ratio at time t={time_label}'
                else:
                    title = f'{filter_particle_type} {y} at time t={time_label}'

            data = utils.slice_closest(
                data=utils.slice_closest(
                    data=halo.get_particle_states(initial=initial_particles, now=final_particles), value=_time
                ),
                value=filter_particle_type,
                key='particle_type',
            )

            if y == 'baryon mass ratio':
                fig, ax = self._plot_baryon_mass_ratio(
                    halo=halo,
                    data=data,
                    x_unit=x_unit,
                    label=label,
                    initial_particles=initial_particles,
                    final_particles=final_particles,
                    fig=fig,
                    ax=ax,
                    early_quit=early_quit,
                    lineplot_kwargs=lineplot_kwargs,
                    title=title,
                    **kwargs,
                )
            else:
                fig, ax = PhaseSpace.from_particles(distribution=halo.distributions[0], data=data, verbose=False).plot(
                    y,
                    lineplot_kwargs={'label': label, **lineplot_kwargs},
                    **plot_kwargs,
                    **kwargs,
                    x_unit=x_unit,
                    y_unit=y_unit,
                    title=title,
                    early_quit=early_quit,
                    fig=fig,
                    ax=ax,
                )
        return fig, ax

    def animate(
        self,
        y: PlotProperty,
        times: Quantity['time'] | NDArray[np.float64] | list[float],
        save_path: str | Path,
        **kwargs: Any,
    ) -> None:
        """Animate a physical property of the system accross all halos.

        Parameters:
            y: The property to plot (i.e. temperature, pressure, etc.).
            times: The times to plot for each halo. If an array (i.e. not Quantity), treat as the value in terms of collapse time for each halo (i.e. based on the individual collapse time).
            save_path: The path to save the animation to.
            **kwargs: Additional keyword arguments to pass to the plot function (`Halos.plot()`).
        """
        plot.save_images(
            plot.to_images(
                iterator=times,
                plot_fn=lambda x: self.plot(time=x, y=y, **kwargs),
            ),
            save_path=save_path,
        )

    @staticmethod
    def default_plot_params(y: PlotProperty) -> dict[str, Any]:
        """Default plotting parameters."""
        params = {'xscale': 'log', 'yscale': 'log', 'xlim': (1e-3, 5e1)}
        if y == 'temperature':
            return {**params, 'xlim': (1e-3, 9e0), 'ylim': (1e0, 1e3), 'y_unit': 'km^2/second^2'}
        elif y == 'density':
            return {**params, 'ylim': (1e2, 2e12)}
        elif y == 'pressure':
            return {**params, 'ylim': (1e-3, 2e6)}
        elif y == 'baryon mass ratio':
            return {**params, 'xlim': (2e-4, 5e2), 'ylim': (5e-3, 1.1)}
        return {}

    @staticmethod
    def default_path_condition() -> dict[str, dict[str, Any]]:
        """Default path condition for plotting."""
        return {
            'dm only': {'label': 'No baryons', 'color': 'tab:blue'},
            'static baryons': {'label': 'Static baryons', 'color': 'tab:orange'},
            'dynamic baryons': {'label': 'Dynamic baryons', 'color': 'tab:green'},
        }

    def plot_cumulative_scattering(
        self,
        time_unit: TimeUnitLike = 'Gyr',
        path_condition: dict[str, dict[str, Any]] | Literal['default'] = 'default',
        palette: str | None = None,
        lineplot_kwargs: dict[str, Any] = {},
        plot_kwargs: dict[str, Any] = {},
        save_kwargs: dict[str, Any] = {},
        early_quit: bool = False,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the cumulative scattering halos over time.

        Parameters
            time_unit: The unit of time to use for the x-axis. If 'time step', 'dynamical time', or 'core collapse time', each halo will use its own value for its plot.
            path_condition: A dictionary of conditions to apply to the plot of each halo. If 'default', use the default set in `Halos.default_path_condition()`.
            palette: The color palette to use for the plot. If None, use the default palette.
            lineplot_kwargs: Additional keyword arguments to pass to `sns.lineplot()`
            plot_kwargs: Additional keyword arguments to pass to the plot function (`Halo.plot_cumulative_scattering_amount_over_time()`).
            save_kwargs: Additional keyword arguments to pass to `plot.save_plot()`.
            early_quit: If `True` force the plot on a predefined `fig` and `ax`. If `False` allows the plotting function (`phase_space.plot()`) to define axis parameters (title, axis labels, etc.).
            **kwargs: Additional keyword arguments to pass to the plot function (`plot.setup()`).

        Returns:
            fig, ax.
        """
        fig, ax = plot.setup(**kwargs)
        plot_kwargs = plot_kwargs.copy()
        if path_condition == 'default':
            path_condition = self.default_path_condition()

        used_labels = []

        for color, halo in plot.color_palette(tqdm(self.halos), palette=palette):
            assert halo.save_path is not None
            plot_kwargs.pop('save_kwargs', None)
            plot_kwargs.pop('lineplot_kwargs', None)
            plot_kwargs.pop('early_quit', None)
            _lineplot_kwargs = lineplot_kwargs.copy()
            for key, value in path_condition.items():
                if key in halo.save_path.name:
                    _lineplot_kwargs = {'color': color, **_lineplot_kwargs, **value}
                    if 'label' in _lineplot_kwargs:
                        if _lineplot_kwargs['label'] in used_labels:
                            _lineplot_kwargs.pop('label')
                        else:
                            used_labels += [_lineplot_kwargs['label']]
            fig, ax = halo.plot_cumulative_scattering_amount_over_time(
                lineplot_kwargs=_lineplot_kwargs,
                fig=fig,
                ax=ax,
                early_quit=early_quit,
                time_unit=time_unit,
                **plot_kwargs,
            )
        plot.save(fig=fig, **save_kwargs)
        return fig, ax

    def plot_core_density_ratio(
        self,
        time_unit: TimeUnitLike = 'Gyr',
        path_condition: dict[str, dict[str, Any]] | Literal['default'] = 'default',
        palette: str | None = None,
        lineplot_kwargs: dict[str, Any] = {},
        plot_kwargs: dict[str, Any] = {},
        save_kwargs: dict[str, Any] = {},
        early_quit: bool = False,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the inner core density ratio over time.

        Parameters
            time_unit: The unit of time to use for the x-axis. If 'time step', 'dynamical time', or 'core collapse time', each halo will use its own value for its plot.
            path_condition: A dictionary of conditions to apply to the plot of each halo. If 'default', use the default set in `Halos.default_path_condition()`.
            palette: The color palette to use for the plot. If None, use the default palette.
            lineplot_kwargs: Additional keyword arguments to pass to `sns.lineplot()`
            plot_kwargs: Additional keyword arguments to pass to the plot function (`Halo.plot_cumulative_scattering_amount_over_time()`).
            save_kwargs: Additional keyword arguments to pass to `plot.save_plot()`.
            early_quit: If `True` force the plot on a predefined `fig` and `ax`. If `False` allows the plotting function (`phase_space.plot()`) to define axis parameters (title, axis labels, etc.).
            **kwargs: Additional keyword arguments to pass to the plot function (`plot.setup()`).

        Returns:
            fig, ax.
        """

        fig, ax = plot.setup(**kwargs)
        plot_kwargs = plot_kwargs.copy()
        if path_condition == 'default':
            path_condition = self.default_path_condition()

        used_labels = []

        for color, halo in plot.color_palette(tqdm(self.halos), palette=palette):
            assert halo.save_path is not None
            plot_kwargs.pop('save_kwargs', None)
            plot_kwargs.pop('lineplot_kwargs', None)
            plot_kwargs.pop('early_quit', None)
            _lineplot_kwargs = lineplot_kwargs.copy()
            for key, value in path_condition.items():
                if key in halo.save_path.name:
                    _lineplot_kwargs = {'color': color, **_lineplot_kwargs, **value}
                    if 'label' in _lineplot_kwargs:
                        if _lineplot_kwargs['label'] in used_labels:
                            _lineplot_kwargs.pop('label')
                        else:
                            used_labels += [_lineplot_kwargs['label']]
            fig, ax = halo.plot_core_density_ratio(
                lineplot_kwargs=_lineplot_kwargs,
                fig=fig,
                ax=ax,
                early_quit=early_quit,
                time_unit=time_unit,
                **plot_kwargs,
            )
        plot.save(fig=fig, **save_kwargs)
        return fig, ax

    def print_core_collapse_time(
        self,
        time_unit: TimeUnitLike = 'Gyr',
        time_format: str = '.1f',
        path_condition: dict[str, dict[str, Any]] | Literal['default'] = 'default',
    ) -> None:
        """Print the core collapse time for each halo.

        Parameters:
            time_unit: The unit of time to use for the x-axis. If 'time step', 'dynamical time', or 'core collapse time', each halo will use its own value for its plot.
            time_format: Format for the core collapse time.
            path_condition: A dictionary of conditions to apply to the print of each halo. If 'default', use the default set in `Halos.default_path_condition()`.
        """
        if path_condition == 'default':
            path_condition = self.default_path_condition()
        for halo in self:
            assert halo.save_path is not None
            label = ''
            for key, value in path_condition.items():
                if key in halo.save_path.name:
                    label = value.get('label', label)
            print(f'{label}: {halo.core_collapse_time.to(time_unit):{time_format}}')
