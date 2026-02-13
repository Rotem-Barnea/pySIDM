"""Internal module for plotting halo data"""

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import scipy
import seaborn as sns
from astropy import table
from astropy.units import Quantity
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from astropy.units.typing import UnitLike

from src import plot, types, physics
from src.utils import utils

if TYPE_CHECKING:
    from .halo import Halo


class HaloPlotter:
    """Class for plotting halo data."""

    def __init__(self, halo: 'Halo'):
        self.halo: 'Halo' = halo

    @property
    def results_path(self):
        """The path to the results directory."""
        return self.halo.results_path

    def fill_time_unit(self, unit: types.TimeUnitLike) -> UnitLike:
        """If `unit` is a halo-related time parameter return its unit, otherwise return `unit`."""
        if unit == 'dynamical time':
            return self.halo.units.dynamical_time
        elif unit == 'core collapse':
            return self.halo.units.core_collapse
        elif unit == 't_c':
            return self.halo.units.t_c
        elif unit == 'time step':
            return self.halo.units.time_step
        return unit

    def save(self, fig: Figure, save_kwargs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Saves the plot."""
        if save_kwargs is None:
            return
        save_kwargs = deepcopy(save_kwargs)
        if 'name' in save_kwargs:
            save_kwargs['save_path'] = self.results_path / save_kwargs.pop('name')
        plot.save(fig=fig, **save_kwargs)

    def enclosed_mass_ratio(
        self,
        particle_type: types.ParticleType,
        times: Quantity['time'] | None = None,
        data: table.QTable | None = None,
        stable_first_index: bool = True,
        reference: float = 1,
        xlabel: str | None = 'Radius',
        ylabel: str | Literal['auto'] | None = 'auto',
        length_unit: UnitLike = 'kpc',
        xscale: plot.Scale = 'log',
        palette: str | None = None,
        labels: list[str] | Literal['auto'] | None = 'auto',
        time_unit: types.TimeUnitLike = 'Gyr',
        time_format: str = '.1f',
        lineplot_kwargs: dict[str, Any] = {},
        save_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the fraction of the enclosed mass that is contributed by the specified particle type, as a function of the radius.

        Parameters:
            particle_type: The particle type to calculate the fraction distribution for.
            times: The times to plot the fraction distribution in. If not provided use all times.
            data: The data to pull from. If not provided use all snapshots.
            stable_first_index: If `True` only start plotting from a range that has at least one `particle_type` particle and at least one that's not.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis. If `'auto'` use the template `{particle_type} enclosed mass fraction`
            length_unit: The units for the x-axis.
            reference: Divide the y-axis by this amount. This is used for a halo with a homogeneous mix of types, allowing quantifying the relative change in ratio (from the original mixed fraction `reference`), rather than the absolute mass fraction.
            xscale: The scale for the x-axis.
            palette: The color palette to use for the plot. If `None`, use the default palette.
            labels: Labels for each plot. If a list is provided, it must match the length of `time`. If '`auto`', use the template `Time={time.to(time_unit):{time_format}}`.
            time_unit: The time units to use for the labels. Only relevant if `labels` is `'auto'`.
            time_format: Format string for time to use in the labels. Only relevant if `labels` is `'auto'`.
            lineplot_kwargs: Additional keyword arguments to pass to `sns.lineplot()`.
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.
            kwargs: Additional keyword arguments passed to `plot.setup()`.

        Returns:
            fig, ax
        """
        if ylabel == 'auto':
            ylabel = f'{particle_type} enclosed mass fraction'
        fig, ax = plot.setup(
            xlabel=xlabel,
            ylabel=ylabel,
            x_unit=length_unit,
            xscale=xscale,
            **kwargs,
        )
        if data is None:
            data = self.halo.get_particle_states()
        time_unit = self.fill_time_unit(time_unit)
        if times is None:
            times = cast(Quantity, np.unique(utils.get_columns(data, ['time'])[0]))
        times = cast(Quantity, np.atleast_1d(times))
        used_times = []
        if labels is None:
            labels = ['auto-disable'] * len(times)
        elif labels == 'auto':
            labels = ['auto'] * len(times)
        for color, (t, label) in plot.color_palette(list(zip(times, labels)), palette=palette):
            if t in used_times:
                continue
            used_times += [t]
            group = utils.slice_closest(data, t)
            r, m, types = utils.get_columns(group, ['r', 'm', 'particle_type'])
            y = (m * (types == particle_type)).cumsum() / m.cumsum()
            min_i = (
                max(np.argmax(types == particle_type), np.argmax(types != particle_type)) if stable_first_index else 0
            )
            if label == 'auto-disable':
                label = None
            elif label == 'auto':
                label = f'Time={t.to(time_unit):{time_format}}'
            if len(r[min_i:]) < 2:
                continue
            sns.lineplot(
                x=r[min_i:].to(length_unit).value,
                y=y[min_i:] / reference,
                ax=ax,
                color=color,
                label=label,
                **lineplot_kwargs,
            )
        self.save(fig=fig, save_kwargs=save_kwargs)
        return fig, ax

    def animate_enclosed_mass_ratio(
        self,
        particle_type: types.ParticleType,
        save_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> None:
        """Animate the fraction of the enclosed mass that is contributed by the specified particle type, as a function of the radius, over time.

        Parameters:
            particle_type: The particle type to calculate the fraction distribution for.
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`.
            kwargs: Additional keyword arguments passed to the plotting function `self.enclosed_mass_ratio()`.
        """
        plot.save_images(
            plot.to_images(
                iterator=list(self.get_particle_states().group_by('time').groups)[:-1],
                plot_fn=lambda x: self.plot_enclosed_mass_ratio(data=x, particle_type=particle_type, **kwargs),
            ),
            **save_kwargs,
        )

    def core_density(
        self,
        inner_core_radius: Quantity['length'] | None = None,
        filter_particle_type: types.ParticleType | None = 'dm',
        stat: Literal['density', 'density ratio', 'count', 'fraction'] = 'density ratio',
        time_unit: types.TimeUnitLike = 'Gyr',
        density_unit: UnitLike = 'Msun/kpc^3',
        xlabel: str | None = 'Time',
        ylabel: str | Literal['auto'] | None = 'auto',
        title: str | None = 'Inner core density ratio over time',
        lineplot_kwargs: dict[str, Any] = {},
        save_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the inner core density ratio over time.

        Parameters:
            inner_core_radius: The inner core radius. If `None`, use the current inner core radius.
            filter_particle_type: Whether to filter to only plot the specified particle type.
            stat: The type of statistic to calculate. `density ratio` returns the density divided by the initial density (post bootstrap).
            time_unit: Units to use for x-axis.
            density_unit: Units to use for y-axis. Only used if `stat` is `'density ratio'`.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            title: The title of the plot.
            lineplot_kwargs: Additional keyword arguments to pass to `sns.lineplot()`.
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.
            kwargs: Additional keyword arguments passed to `plot.setup()`.

        Returns:
            fig, ax.
        """
        if ylabel == 'auto':
            if stat == 'count':
                ylabel = '#Particles'
            elif stat == 'density':
                ylabel = r'$\rho_c$'
            elif stat == 'density ratio':
                ylabel = r'$\rho_c$/$\rho_{c,0}$'
            elif stat == 'fraction':
                ylabel = '%Particles'
        time_unit = self.fill_time_unit(time_unit)
        fig, ax = plot.setup(
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            **kwargs,
        )
        t, ratio = self.halo.inner_core_density(
            inner_core_radius=inner_core_radius,
            filter_particle_type=filter_particle_type,
            stat=stat,
        )
        t = t.to(time_unit)
        ratio = Quantity(ratio)
        if stat == 'density':
            ratio = ratio.to(density_unit)
        sns.lineplot(x=t.value, y=ratio.value, ax=ax, **lineplot_kwargs)
        ax = plot.update_units(
            ax=ax, x_unit=str(t.unit), y_unit=str(ratio.unit)
        )  # to-do - Decide if we want this sort of thing.
        self.save(fig=fig, save_kwargs=save_kwargs)
        return fig, ax

    def local_density(
        self,
        times: Quantity['time'] | None = None,
        data: table.QTable | None = None,
        filter_particle_type: types.ParticleType | None = None,
        max_radius_j: int = 10,
        smooth_sigma: float = 0,
        x_range: Quantity['length'] | None = None,
        xlabel: str | None = 'Radius',
        ylabel: str | None = r'$\rho$',
        x_unit: UnitLike = 'kpc',
        y_unit: UnitLike = 'Msun/kpc**3',
        xscale: plot.Scale = 'log',
        yscale: plot.Scale = 'log',
        palette: str | None = None,
        labels: list[str] | Literal['auto'] | None = 'auto',
        time_unit: types.TimeUnitLike = 'Gyr',
        time_format: str = '.1f',
        lineplot_kwargs: dict[str, Any] = {},
        save_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the local density profile as a function of the radius.

        Parameters:
            times: The times to plot the fraction distribution in. If not provided use all times.
            data: The data to pull from. If not provided use all snapshots.
            filter_particle_type: Only relevant if `data` is not provided. Filter the snapshots to only include the specified particle type.
            max_radius_j: The maximum neighbor radius to consider for the cell volume when computing the local density.
            smooth_sigma: Smoothing factor for the density plot (sigma for a 1d Gaussian kernel). Ignore if 0.
            radius_range: Range of radius to consider (filters the data).
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            title: Title for the plot.
            x_unit: Units to use for the x-axis.
            y_unit: Units to use for the y-axis.
            xscale: Scale for the x-axis.
            yscale: Scale for the y-axis.
            palette: The color palette to use for the plot. If `None`, use the default palette.
            labels: Labels for each plot. If a list is provided, it must match the length of `time`. If '`auto`', use the template `Time={time.to(time_unit):{time_format}}`.
            time_unit: The time units to use for the labels. Only relevant if `labels` is `'auto'`.
            time_format: Format string for time to use in the labels. Only relevant if `labels` is `'auto'`.
            lineplot_kwargs: Additional keyword arguments to pass to `sns.lineplot()`.
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.setup()`).

        Returns:
            fig, ax.
        """
        fig, ax = plot.setup(
            xlabel=xlabel, ylabel=ylabel, x_unit=x_unit, y_unit=y_unit, xscale=xscale, yscale=yscale, **kwargs
        )

        if data is None:
            data = self.halo.get_particle_states(filter_particle_type=filter_particle_type)
        time_unit = self.fill_time_unit(time_unit)
        if times is None:
            times = cast(Quantity, np.unique(utils.get_columns(data, ['time'])[0]))
        times = cast(Quantity, np.atleast_1d(times))
        used_times = []
        if labels is None:
            labels = ['auto-disable'] * len(times)
        elif labels == 'auto':
            labels = ['auto'] * len(times)
        for color, (t, label) in plot.color_palette(list(zip(times, labels)), palette=palette):
            if t in used_times:
                continue
            used_times += [t]
            group = utils.slice_closest(data, t)
            r, m = utils.get_columns(group, ['r', 'm'])
            x = cast(Quantity, r.to(x_unit))
            y = cast(
                Quantity,
                physics.utils.local_density(
                    r=r, m=m, max_radius_j=max_radius_j, volume_kind='density', mass_kind='sum'
                ).to(y_unit),
            )
            y = Quantity(scipy.ndimage.gaussian_filter1d(y, sigma=smooth_sigma), y.unit) if smooth_sigma > 0 else y
            if x_range is not None:
                mask = (x > x_range[0]) * (x < x_range[1])
                x, y = cast(tuple[Quantity, Quantity], (x[mask], y[mask]))
            if label == 'auto-disable':
                label = None
            elif label == 'auto':
                label = f'Time={t.to(time_unit):{time_format}}'
            sns.lineplot(x=x.value, y=y.value, ax=ax, label=label, color=color, **lineplot_kwargs)
        self.save(fig=fig, save_kwargs=save_kwargs)
        return fig, ax

    def cumulative_scattering(
        self,
        time_unit: types.TimeUnitLike = 'Gyr',
        undersample: int | None = None,
        xlabel: str | None = 'Time',
        ylabel: str | None = 'Cumulative number of scattering events',
        yscale: plot.Scale = 'log',
        lineplot_kwargs: dict[str, Any] = {},
        save_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the cumulative number of scattering events over time.

        Parameters:
            time_unit: Units for the x-axis.
            undersample: Downsample the data by this factor.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            yscale: The scale of the y-axis.
            lineplot_kwargs: Additional keyword arguments to pass to `sns.lineplot()`.
            save_kwargs: Additional keyword arguments to pass to `plot.save_plot()`.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.setup()`).

        Returns:
            fig, ax.
        """
        time_unit = self.fill_time_unit(time_unit)
        fig, ax = plot.setup(xlabel=xlabel, ylabel=ylabel, x_unit=time_unit, yscale=yscale, **kwargs)
        x = cast(Quantity, self.halo.scatter_times.to(time_unit))
        y = Quantity(self.halo.n_scatters.cumsum())
        if undersample is not None:
            x, y = cast(tuple[Quantity, Quantity], (x[::undersample], y[::undersample]))
        sns.lineplot(x=x.value, y=y.value, ax=ax, **lineplot_kwargs)
        self.save(fig=fig, save_kwargs=save_kwargs)
        return fig, ax

    def binned_scattering(
        self,
        time_binning: Quantity = Quantity(100, 'Myr'),
        time_unit: types.TimeUnitLike = 'Gyr',
        xlabel: str | None = 'Time',
        ylabel: str | None = 'Number of scattering events',
        title: str | Literal['auto'] | None = 'auto',
        time_format: str | None = None,
        title_time_unit: str | None = 'Myr',
        yscale: plot.Scale = 'log',
        lineplot_kwargs: dict[str, Any] = {},
        save_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the number of scattering events over time, binned.

        Parameters:
            time_binning: Binning for the x-axis.
            time_unit: Units for the x-axis.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            title: Title for the plot.
            time_format: Format for the time in the title.
            title_time_unit: Units for the time displayed in the title.
            yscale: The scale of the y-axis.
            lineplot_kwargs: Additional keyword arguments to pass to `sns.lineplot()`.
            save_kwargs: Keyword arguments to pass to `plot.save_plot()`. Must include `save_path`. If `None` ignores saving.
            kwargs: Additional keyword arguments to pass to the plot function (`plot.setup()`).

        Returns:
            fig, ax.
        """
        time_unit = self.fill_time_unit(time_unit)
        t = Quantity(np.arange(0, self.halo.time.value, 10), self.halo.time.unit).to(time_unit)
        scatters = np.bincount(
            np.digitize(self.halo.scatter_times, t.to(self.halo.time.unit)) - 1,
            weights=self.halo.n_scatters,
            minlength=len(t) - 1,
        )

        if title == 'auto':
            title = f'Number of scattering events over time per {time_binning.to(title_time_unit):{time_format}}'

        fig, ax = plot.setup(
            xlabel=xlabel,
            ylabel=ylabel,
            x_unit=time_unit,
            title=title,
            yscale=yscale,
            **kwargs,
        )
        sns.lineplot(x=t[:-1].value, y=scatters[:-1].value, ax=ax, **lineplot_kwargs)
        self.save_plot(fig=fig, save_kwargs=save_kwargs)
        return fig, ax
