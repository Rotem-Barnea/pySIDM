from typing import Any, Literal, cast
from pathlib import Path
from collections.abc import Iterable

import numpy as np
import scipy
import pandas as pd
import seaborn as sns
from PIL import Image
from astropy import table
from matplotlib import colors, ticker
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from astropy.units import Unit, Quantity
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from astropy.units.typing import UnitLike
from matplotlib.transforms import BboxTransformTo
from numba.misc.coverage_support import Callable
from matplotlib.backends.backend_agg import FigureCanvasAgg

from . import utils, run_units
from .tqdm import tqdm


def pretty_ax_text(
    x: float | None = None,
    y: float | None = None,
    transform: BboxTransformTo | str | None = None,
    verticalalignment: str | None = 'top',
    bbox_boxstyle: str | None = 'round',
    bbox_facecolor: str | None = 'wheat',
    bbox_alpha: float | None = 0.5,
    bbox: dict[str, Any] = {},
    **kwargs: Any,
):
    """Pretty text keyword arguments.

    Parameters:
        ax: Axes to plot on.
        x: x-coordinate of the text.
        y: y-coordinate of the text.
        verticalalignment: Vertical alignment of the text.
        bbox_boxstyle: Box style of the text. Gets added to `bbox` with lower priority.
        bbox_facecolor: Face color of the text. Gets added to `bbox` with lower priority.
        bbox_alpha: Alpha of the text. Gets added to `bbox` with lower priority.
        bbox: Additional bounding box properties. Supercedes all other `bbox_` arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Keyword arguments for `ax.text`.
    """
    return utils.drop_None(
        x=x,
        y=y,
        transform=transform,
        verticalalignment=verticalalignment,
        bbox={**utils.drop_None(boxstyle=bbox_boxstyle, facecolor=bbox_facecolor, alpha=bbox_alpha), **bbox},
        **kwargs,
    )


def setup_plot(
    fig: Figure | None = None,
    ax: Axes | None = None,
    grid: bool = True,
    minorticks: bool = False,
    figsize: tuple[int, int] | None = (6, 5),
    ax_set: dict[str, str] | None = None,
    x_axis_percent_formatter: dict[str, Any] | None = None,
    y_axis_percent_formatter: dict[str, Any] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    hlines: list[dict[str, Any]] = [],
    vlines: list[dict[str, Any]] = [],
    texts: list[dict[str, Any]] = [],
    early_quit: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Setup a plot with optional grid, minor ticks, and axis settings.

    If both `fig` and `ax` are provided, they are returned as is (early quit).

    Parameters:
        fig: Figure to plot on. If `None` a new figure is created.
        ax: Axes to plot on. If `None` a new axes is created.
        grid: Whether to add a grid to the plot (major ticks).
        minorticks: Whether to add the grid for the minor ticks.
        figsize: The figure size to create. Ignored if `fig` / `ax` is provided.
        ax_set: Additional keyword arguments to pass to `Axes.set()`. e.g `{'xscale': 'log'}`.
        x_axis_percent_formatter: Format the x-axis as a percentage and pass the arguments to the formatter (`matplotlib.ticker.PercentFormatter(**x_axis_percent_formatter)`). If `None` ignores.
        y_axis_percent_formatter: Format the y-axis as a percentage and pass the arguments to the formatter (`matplotlib.ticker.PercentFormatter(**y_axis_percent_formatter)`). If `None` ignores.
        title: The title of the plot.
        xlabel: The label of the x-axis.
        ylabel: The label of the y-axis.
        xlim: The limits of the x-axis. If `None` ignores.
        ylim: The limits of the y-axis. If `None` ignores.
        hlines: List of horizontal lines to plot. Each element contains the keywords arguments passed to `ax.axhline()`. If the argument `transform` is `'transAxes'`, the transformed is derived from the `ax`.
        texts: List of texts to plot. Each element contains the keywords arguments passed to `ax.text()`. If the argument `transform` is `'transAxes'`, the transformed is derived from the `ax`.
        vlines: List of vertical lines to plot. Each element contains the keywords arguments passed to `ax.axvline()`. If the argument `transform` is `'transAxes'`, the transformed is derived from the `ax`.
        early_quit: If `True`, quits if `fig` and `ax` are provided (and thus doesn't update the labels etc.).
        kwargs: Additional keyword arguments to pass to `plt.subplots()`.

    Returns:
        fig, ax.
    """
    if fig is not None and ax is not None and early_quit:
        return fig, ax
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize, **kwargs)
    assert fig is not None and ax is not None
    fig.tight_layout()
    ax.grid(grid)
    if minorticks:
        ax.minorticks_on()
        ax.grid(True, which='minor', alpha=0.3, linestyle='--')
    if ax_set is not None:
        ax.set(**ax_set)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    for line in hlines:
        if line.get('transform', None) == 'transAxes':
            line['transform'] = ax.transAxes
        ax.axhline(**line)
    for line in vlines:
        if line.get('transform', None) == 'transAxes':
            line['transform'] = ax.transAxes
        ax.axvline(**line)
    for text in texts:
        if text.get('transform', None) == 'transAxes':
            text['transform'] = ax.transAxes
        ax.text(**text)

    if x_axis_percent_formatter is not None:
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(**x_axis_percent_formatter))
    if y_axis_percent_formatter is not None:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(**y_axis_percent_formatter))

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    return fig, ax


def default_plot_text(
    key: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    x_units: UnitLike | None = None,
    y_units: UnitLike | None = None,
    lower: bool = False,
) -> dict[str, str | None]:
    """Return default plot `title`/`xlabel`/`ylabel` for a given key and add the appropriate units.

    If `xlabel`/`ylabel`/`title` are provided, they override the defaults.
    """
    if key == 'vr':
        output = {'title': 'Radial velocity distribution', 'xlabel': 'Radial velocity', 'ylabel': 'Density'}
    elif key in ['vx', 'vy', 'vp']:
        output = {'title': 'Pendicular velocity distribution', 'xlabel': 'Pendicular velocity', 'ylabel': 'Density'}
    elif key == 'v_norm':
        output = {'title': 'Velocity norm distribution', 'xlabel': 'Velocity norm', 'ylabel': 'Density'}
    elif key == 'r':
        output = {'title': 'Radius distribution', 'xlabel': 'Radius', 'ylabel': 'Density'}
    else:
        output = {}
    if lower:
        output = {k: v.lower() for k, v in output.items()}
    output = {**output, **utils.drop_None(xlabel=xlabel, ylabel=ylabel, title=title)}
    return utils.drop_None(
        title=output.get('title', None),
        xlabel=utils.add_label_unit(output.get('xlabel', None), x_units),
        ylabel=utils.add_label_unit(output.get('ylabel', None), y_units),
    )


def default_plot_unit_type(key: str, plot_unit: UnitLike | None = None) -> UnitLike:
    """Return the appropriate unit type for a given `key`. If `plot_unit` is provided, return it instead."""
    if plot_unit is not None:
        return plot_unit
    if key == 'r':
        return 'kpc'
    elif key in ['vr', 'vx', 'vy', 'vp', 'v_norm']:
        return 'km/second'
    return ''


def plot_trace(
    key: str,
    data: table.QTable,
    particle_index: int,
    relative: Literal['relative change', 'change', 'absolute'] = 'absolute',
    xlabel: str | None = 'Time',
    ylabel: str | None = None,
    title: str | None = 'Trace of particle id={particle_index}, initial position={r}',
    label: str | None = 'particle id={particle_index}, initial position={r}',
    time_units: UnitLike = 'Gyr',
    y_units: UnitLike | None = None,
    length_units: UnitLike = 'kpc',
    length_format: str = '.1f',
    save_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot the trace of a particle's property over time.

    Parameters:
        key: The property to plot. Must be a valid column name in the data table.
        data: The data table to plot (i.e. `halo.snapshots`, or an external table from an `NSphere` run).
        particle_index: The index of the particle to trace.
        relative: If `absolute` plot the property as is. If `relative` plot the change in the property relative to the initial value. If `relative change` plot the change in the property relative to the initial value divided by the initial value.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis. If not provided, the label will be automatically generated based on the key and units. To disable this set to `''`.
        title: Title for the plot.
        time_units: Units for the x-axis.
        y_units: Units for the y-axis.
        length_units: Units for the length.
        length_format: Format string for length.
        save_kwargs: Keyword arguments to pass to `save_plot()`. Must include `save_path`. If `None` ignores saving.
        kwargs: Additional keyword arguments to pass to the plot function (`setup_plot()`).

    Returns:
        fig, ax.
    """
    xlabel = utils.add_label_unit(xlabel, time_units)
    particle = data[data['particle_index'] == particle_index].copy()
    x = particle['time'].to(time_units)
    y = particle[key]
    if y_units is not None:
        y = y.to(y_units)
    if relative == 'change':
        y = y - cast(Quantity, y[0])
        if ylabel is None:
            ylabel = rf'$\Delta {key}$'
    elif relative == 'relative change':
        y = (y - cast(Quantity, y[0])) / cast(Quantity, y[0])
        if ylabel is None:
            ylabel = rf'$\%\Delta {key}$'
    elif ylabel is None:
        ylabel = f'${key}$'
    ylabel = utils.add_label_unit(ylabel, y_units)
    if title is not None:
        title = title.format(
            particle_index=particle_index,
            r=particle['r'][0].to(length_units).to_string(format='latex', formatter=length_format),
        )
    if label is not None:
        label = label.format(
            particle_index=particle_index,
            r=particle['r'][0].to(length_units).to_string(format='latex', formatter=length_format),
        )
    fig, ax = setup_plot(**kwargs, **utils.drop_None(title=title, xlabel=xlabel, ylabel=ylabel))
    sns.lineplot(x=x, y=np.array(y), ax=ax, label=label)
    if save_kwargs is not None:
        save_plot(fig=fig, **save_kwargs)
    return fig, ax


def plot_2d(
    grid: Quantity,
    extent: tuple[Quantity, Quantity, Quantity, Quantity] | None = None,
    plot_method: Literal['imshow', 'pcolormesh'] = 'imshow',
    x_range: Quantity | None = None,
    y_range: Quantity | None = None,
    x_units: UnitLike = run_units.length,
    y_units: UnitLike = run_units.velocity,
    cbar_units: UnitLike = '',
    transparent_value: float | None = None,
    transparent_range: tuple[float, float] | None = None,
    x_nbins: int | None = 6,
    y_nbins: int | None = 6,
    x_tick_format: str = '.0f',
    y_tick_format: str = '.0f',
    x_log: bool = False,
    y_log: bool = False,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    cbar_label: str | None = None,
    cbar_label_autosuffix: bool = True,
    cbar_format: str | None = None,
    cbar_log_numticks: int | None = None,
    log_scale: bool = False,
    percentile_clip_scale: tuple[float, float] | None = None,
    grid_row_normalization: Literal['max', 'sum', 'integral'] | float | None = None,
    hlines: list[dict[str, Any]] = [],  # TODO - deprecate
    vlines: list[dict[str, Any]] = [],  # TODO - deprecate
    texts: list[dict[str, Any]] = [],  # TODO - deprecate
    fig: Figure | None = None,
    ax: Axes | None = None,
    setup_kwargs: dict[str, Any] = {},
    save_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot a 2d heatmap, such as a phase space distribution.

    Parameters:
        grid: 2d array of data to plot.
        extent: Range of values to plot, used to define the x and y axes. If `None` derive from `x_range` and `y_range`, otherwise takes priority. Only used if `plot_method` is `imshow`.
        plot_method: Method to use for plotting.
        x_range: Range of x values to plot, used to define the extent. Must be provided if `plot_method` is `pcolormesh` or if `extent` is `None`, otherwise ignored.
        y_range: Range of y values to plot, used to define the extent. Must be provided if `plot_method` is `pcolormesh` or if `extent` is `None`, otherwise ignored.
        x_units: Units to use for the x-axis.
        y_units: Units to use for the y-axis.
        cbar_units: Units to use for the value of each grid cell.
        transparent_value: Grid value to turn transparent (i.e. plot as `NaN`). If `None` ignores.
        transparent_range: Range of values to turn transparent (i.e. plot as `NaN`). If `None` ignores.
        x_nbins: Number of bins to use for the x-axis.
        y_nbins: Number of bins to use for the y-axis.
        x_tick_format: Format string for the x-axis ticks.
        y_tick_format: Format string for the y-axis ticks.
        x_log: Sets the x-axis to a log scale.
        y_log: Sets the y-axis to a log scale.
        title: Title of the plot.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        cbar_label: Label for the colorbar.
        cbar_label_autosuffix: Add a prefix and suffix based on the `row_normalization` selected.
        cbar_format: Format string for the colorbar.
        cbar_log_numticks: Number of ticks for the colorbar when using a log scale. If `None` ignores.
        log_scale: Whether to use a log scale for the colorbar. If `True` overwrites the `norm` argument if provided (in `kwargs`), and sets the `norm` to `colors.LogNorm()`. Ignored if the grid has no variance.
        percentile_clip_scale: Whether to use a percentile clip scale for the colorbar. If `True` overwrites the `norm` argument if provided (in `kwargs`), and sets the `norm` to `colors.PercentileNorm()`.
        grid_row_normalization: Normalization applied to the grid row values, used for the cbar label prefix and suffix.
        hlines: List of horizontal lines to plot. Each element contains the keywords arguments passed to `ax.axhline()`. If the argument `transform` is `'transAxes'`, the transformed is derived from the `ax`.
        vlines: List of vertical lines to plot. Each element contains the keywords arguments passed to `ax.axvline()`. If the argument `transform` is `'transAxes'`, the transformed is derived from the `ax`.
        texts: List of texts to plot. Each element contains the keywords arguments passed to `ax.text()`. If the argument `transform` is `'transAxes'`, the transformed is derived from the `ax`.
        fig: Figure to plot on.
        ax: Axes to plot on.
        setup_kwargs: Additional keyword arguments to pass to `setup_plot()`.
        save_kwargs: Keyword arguments to pass to `save_plot()`. Must include `save_path`. If `None` ignores saving.
        kwargs: Additional keyword arguments to pass to `plt.imshow()` or `plt.pcolormesh()`.

    Returns:
        fig, ax.
    """
    if extent is None:
        assert x_range is not None and y_range is not None, 'x_range and y_range must be provided if extent is None'
        extent = cast(
            tuple[Quantity, Quantity, Quantity, Quantity], utils.to_extent(x_range.to(x_units), y_range.to(y_units))
        )
    extent_value = (
        float(extent[0].to(x_units).value),
        float(extent[1].to(x_units).value),
        float(extent[2].to(y_units).value),
        float(extent[3].to(y_units).value),
    )

    if x_log:
        setup_kwargs['ax_set'] = {**setup_kwargs.get('ax_set', {}), 'xscale': 'log'}
    if y_log:
        setup_kwargs['ax_set'] = {**setup_kwargs.get('ax_set', {}), 'yscale': 'log'}

    fig, ax = setup_plot(
        fig=fig,
        ax=ax,
        grid=False,
        title=title,
        xlabel=utils.add_label_unit(xlabel, x_units),
        ylabel=utils.add_label_unit(ylabel, y_units),
        **setup_kwargs,
    )
    grid = grid.copy()
    if cbar_units != '':
        grid = grid.to(cbar_units)
    cbar_units = Unit(str(grid.unit))

    if log_scale and grid.std() != 0:
        kwargs.update(norm=colors.LogNorm())
    elif percentile_clip_scale is not None:
        kwargs.update(norm=colors.Normalize(*np.nanpercentile(grid.value, percentile_clip_scale)))

    if transparent_value is not None:
        grid[grid.value == transparent_value] = np.nan
    if transparent_range is not None:
        grid[(grid.value >= transparent_range[0]) * (grid.value <= transparent_range[1])] = np.nan

    if plot_method == 'imshow':
        im = ax.imshow(grid.value, origin='lower', aspect='auto', extent=extent_value, **kwargs)
    else:
        assert x_range is not None and y_range is not None, (
            "x_range and y_range must be provided for `plot_method='pcolormesh'`"
        )
        im = ax.pcolormesh(*np.meshgrid(x_range, y_range), grid.value, **kwargs)

    cbar = fig.colorbar(im, ax=ax)

    if cbar_label_autosuffix and cbar_label is not None:
        if grid_row_normalization is None:
            cbar_label = f'#{cbar_label}'
        elif grid_row_normalization == 'max':
            cbar_label = f'{cbar_label} fraction from max bin in row'
        elif type(grid_row_normalization) is float:
            cbar_label = f'{cbar_label} fraction from the {grid_row_normalization:.0%}-th quantile bin in row'
        elif grid_row_normalization == 'sum':
            cbar_label = f'%{cbar_label} per bin in row'
        elif grid_row_normalization == 'integral':
            cbar_label = f'{cbar_label} density per unit length'

    cbar_label = utils.add_label_unit(cbar_label, cbar_units)
    if cbar_label:
        cbar.set_label(cbar_label)

    if cbar_log_numticks is not None:
        cbar.locator = ticker.LogLocator(numticks=cbar_log_numticks)
        cbar.update_ticks()
    if cbar_format is not None:
        cbar.formatter = ticker.StrMethodFormatter(cbar_format)
        cbar.update_ticks()

    if x_nbins is not None:
        if x_log:
            ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=x_nbins))
            ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10))
            ax.xaxis.set_minor_locator(
                ticker.LogLocator(base=10, subs=list(np.arange(2, 10, dtype=float)), numticks=999)
            )
            ax.set_xscale('log')
        else:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=x_nbins))
            ax.xaxis.set_major_formatter(ticker.StrMethodFormatter(f'{{x:{x_tick_format}}}'))
        ax.xaxis.tick_bottom()
        for lab in ax.get_xticklabels():
            lab.set_rotation(0)
            lab.set_horizontalalignment('center')

    if y_nbins is not None:
        if y_log:
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=y_nbins))
            ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10))
            ax.yaxis.set_minor_locator(
                ticker.LogLocator(base=10, subs=list(np.arange(2, 10, dtype=float)), numticks=999)
            )
            ax.set_yscale('log')
        else:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=y_nbins))
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter(f'{{x:{y_tick_format}}}'))

    for line in hlines:
        if line.get('transform', None) == 'transAxes':
            line['transform'] = ax.transAxes
        ax.axhline(**line)
    for line in vlines:
        if line.get('transform', None) == 'transAxes':
            line['transform'] = ax.transAxes
        ax.axvline(**line)
    for text in texts:
        if text.get('transform', None) == 'transAxes':
            text['transform'] = ax.transAxes
        ax.text(**text)
    if save_kwargs is not None:
        save_plot(fig=fig, **save_kwargs)
    return fig, ax


def plot_phase_space(
    grid: Quantity,
    r_range: Quantity['length'] | None = Quantity([1e-2, 50], 'kpc'),
    v_range: Quantity['velocity'] | None = Quantity([0, 100], 'km/second'),
    length_units: UnitLike = run_units.length,
    velocity_units: UnitLike = run_units.velocity,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot the phase space distribution. Wrapper for `plot_2d()` to provide convenient defaults and variable names (i.e. `r` and `v`)."""
    return plot_2d(
        grid,
        xlabel='Radius',
        ylabel='Velocity',
        x_units=length_units,
        y_units=velocity_units,
        **utils.drop_None(x_range=r_range, y_range=v_range),
        **kwargs,
    )


def plot_density(
    data: Quantity['length'],
    bins: int | NDArray[np.float64] | str = 100,
    unit_mass: Quantity['mass'] = Quantity(1, 'Msun'),
    xlabel: str | None = 'Radius',
    ylabel: str | None = r'$\rho$',
    title: str | None = None,
    length_units: UnitLike = 'kpc',
    density_units: UnitLike = 'Msun/kpc^3',
    ax_set: dict[str, Any] = {'xscale': 'log', 'yscale': 'log'},
    minorticks: bool = True,
    label: str | None = None,
    cleanup_nonpositive: bool = True,
    smooth_sigma: float = 1,
    add_J: bool = False,
    save_kwargs: dict[str, Any] | None = None,
    line_kwargs: dict[str, Any] = {},
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot the density distribution per unit volume.

    Parameters:
        data: Data to plot.
        bins: Argument accepted by `np.histogram()`.
        unit_mass: Unit mass to convert the number density to mass density,
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        title: Title for the plot.
        length_units: Units to use for the x-axis.
        density_units: Number of bins to use for the y-axis.
        ax_set: Additional keyword arguments to pass to `Axes.set()`. e.g `{'xscale': 'log'}`.
        minorticks: Whether to add the grid for the minor ticks.
        label: label to add to the plot legend.
        cleanup_nonpositive: drop non-positive values from the plot, to avoid "pits" in the log plot.
        smooth_sigma: sigma for smoothing the density distribution.
        add_J: Multiply the density by the spherical jacobian (4*pi*r^2).
        save_kwargs: Keyword arguments to pass to `save_plot()`. Must include `save_path`. If `None` ignores saving.
        line_kwargs: Additional keyword arguments to pass to `sns.lineplot()`.
        kwargs: Additional keyword arguments to pass to `setup_plot()`.


    Returns:
        fig, ax.
    """
    counts, bin_edges = np.histogram(data, bins=bins)
    histogram_bins = Quantity(list(map(Quantity, zip(bin_edges, bin_edges[1:])))).to(length_units)
    bin_centers = cast(Quantity, histogram_bins.mean(1))
    volume = 4 / 3 * np.pi * (histogram_bins[:, 1] ** 3 - histogram_bins[:, 0] ** 3)

    density = counts / volume * unit_mass
    if add_J:
        density *= 4 * np.pi * bin_centers**2
        density_units = Unit(cast(str, density_units)) * cast(Unit, bin_centers.unit) ** 2
        if ylabel is not None:
            ylabel = rf'{ylabel} $\cdot 4\pi r^2$'
    density = density.to(density_units)
    fig, ax = setup_plot(
        ax_set=ax_set,
        minorticks=minorticks,
        **default_plot_text(
            'r', xlabel=xlabel, ylabel=ylabel, title=title, x_units=length_units, y_units=density_units
        ),
        **kwargs,
    )
    x = np.array(bin_centers)
    if cleanup_nonpositive:
        x = x[density > 0]
        density = density[density > 0]
    if smooth_sigma > 0:
        density = scipy.ndimage.gaussian_filter1d(density, sigma=smooth_sigma)
    sns.lineplot(x=x, y=density, ax=ax, label=label, **line_kwargs)
    if label is not None:
        ax.legend()
    if save_kwargs is not None:
        save_plot(fig=fig, **save_kwargs)
    return fig, ax


def aggregate_evolution_data(
    data: table.QTable | pd.DataFrame,
    radius_bins: Quantity['length'] = Quantity(np.linspace(1e-3, 5, 100), 'kpc'),
    time_range: Quantity['time'] | None = None,
    unit_mass: Quantity['mass'] = Quantity(1, 'Msun'),
    output_type: Literal['density', 'counts', 'temperature', 'specific heat flux'] = 'counts',
    row_normalization: Literal['max', 'sum', 'integral'] | float | None = None,
    v_axis: Literal['vx', 'vy', 'vr'] = 'vr',
    density_units: UnitLike = run_units.density,
    data_time_units: UnitLike = 'Gyr',
    data_length_units: UnitLike = 'kpc',
    data_specific_energy_units: UnitLike = 'kpc^2/Myr^2',
    data_mass_units: UnitLike = 'Msun',
    output_grid_units: UnitLike | None = None,
) -> tuple[Quantity, tuple[Quantity['length'], Quantity['length'], Quantity['time'], Quantity['time']]]:
    """Prepares data to be plotted in a 2d evolution heatmap plot, with radius x-axis and time y-axis. Intended to be passed on to `plot_2d()`.

    Parameters:
        data: The input data, as a table with every row being a particle at a given time (fully raveled). Must contain the columns `r` and `time`. If `output_type='temperature'` or `'specific heat flux'` must also contain the columns `vx`, `vy`, and `vr`.
        radius_bins: The bins for the radius axis. Also used to define the radius range to consider.
        time_range: The range of time to consider.
        unit_mass: The unit of mass. Used for density calculations, only relevant if `output_type='density'`.
        output_type: The type of calculation to fill each bin.
        row_normalization: The normalization to apply to each row. If `None` no normalization is applied. If `float` it must be a percentile value (between 0 and 1), and the normalization will be based on this quantile of each row.
        v_axis: The velocity to calculate the heat flux in. Only relevant if `output_type='specific heat flux'`.
        density_units: The units for the density.
        data_time_units: The units for the time in the data. Only used if `data` doesn't have defined units (i.e. a `pd.DataFrame` input).
        data_length_units: The units for the radius in the data. Only used if `data` doesn't have defined units (i.e. a `pd.DataFrame` input).
        data_specific_energy_units: The units for the specific energy in the data. Only used if `data` doesn't have defined units (i.e. a `pd.DataFrame` input).
        data_mass_units: The units for the mass in the data. Only used if `data` doesn't have defined units (i.e. a `pd.DataFrame` input).

    Returns:
        data, extent.
    """
    keep_columns = ['r', 'time']
    if output_type in ['temperature', 'specific heat flux']:
        keep_columns += ['vx', 'vy', 'vr']
    if isinstance(data, pd.DataFrame):
        sub = data[keep_columns]
    else:
        sub = data[keep_columns].to_pandas()
        data_time_units = data['time'].unit
        data_length_units = data['r'].unit
        if 'vx' in sub:
            data_specific_energy_units = data['vx'].unit ** 2
        if 'm' in sub:
            data_mass_units = data['m'].unit

    if time_range is not None:
        sub = sub[sub['time'].between(*time_range.to(data_time_units).value)]

    radius_bins = radius_bins.to(data_length_units)
    r_bin_volume = 4 / 3 * np.pi * (radius_bins[1:] ** 3 - radius_bins[:-1] ** 3)
    grid = np.empty((len(np.unique(np.array(sub['time']))), len(radius_bins) - 1), dtype=np.float64)

    for i, (_, group) in enumerate(sub.groupby('time')):
        if output_type == 'temperature':
            group_bins = pd.cut(
                Quantity(group['r'], data_length_units).to(radius_bins.unit).value, bins=radius_bins.value
            )
            grid[i] = group.groupby(group_bins, observed=False)[['vx', 'vy', 'vr']].var().sum(axis=1)
        elif output_type == 'specific heat flux':
            group_bins = pd.cut(
                Quantity(group['r'], data_length_units).to(radius_bins.unit).value, bins=radius_bins.value
            )
            centralized_group = group.groupby(group_bins, observed=False)[['vx', 'vy', 'vr']].apply(
                lambda q: q - q.mean()
            )
            velocity_term = centralized_group * np.vstack((centralized_group**2).sum(axis=1))
            grid[i] = (
                1
                / 2
                * np.array(velocity_term.reset_index().groupby('level_0', observed=False)[v_axis].sum())
                / r_bin_volume
            )
        else:
            counts, _ = np.histogram(Quantity(group['r'], data_length_units), bins=radius_bins)
            if output_type == 'density':
                grid[i] = counts / r_bin_volume * unit_mass
            else:
                grid[i] = counts

    if output_type == 'density':
        grid_units = Unit(str(data_mass_units)) / Unit(str(data_length_units))
    elif output_type == 'temperature':
        grid_units = data_specific_energy_units
    elif output_type == 'specific heat flux':
        grid_units = Unit(str(data_specific_energy_units)) / (
            Unit(str(data_length_units)) ** 2 * Unit(str(data_time_units))
        )
    else:
        grid_units = ''

    grid_quantity = Quantity(grid, grid_units)
    if output_grid_units is not None:
        grid_quantity = grid_quantity.to(output_grid_units)

    if row_normalization == 'max':
        grid_quantity /= grid_quantity.max(1, keepdims=True)
    elif type(row_normalization) is float:
        grid_quantity /= np.quantile(grid_quantity, row_normalization, axis=1, keepdims=True)
    elif row_normalization == 'sum':
        grid_quantity /= grid_quantity.sum(1, keepdims=True)
    elif row_normalization == 'integral':
        grid_quantity /= np.expand_dims(
            np.trapezoid(y=grid, x=np.matlib.repmat(radius_bins[:-1], len(grid), 1), axis=1), 1
        )

    extent = (
        radius_bins.min(),
        radius_bins.max(),
        Quantity(sub['time'].min(), data_time_units),
        Quantity(sub['time'].max(), data_time_units),
    )
    return cast(Quantity, grid_quantity), extent


def aggregate_2d_data(
    data: table.QTable | pd.DataFrame,
    x_key: str,
    y_key: str,
    x_bins: Quantity,
    y_bins: Quantity,
    x_adjust_bins_edges_to_data: bool = False,
    y_adjust_bins_edges_to_data: bool = False,
    output_type: Literal['counts', 'value'] = 'counts',
    value_key: str | None = None,
    value_statistic: str | None = 'mean',
    row_normalization: Literal['max', 'sum', 'integral'] | float | None = None,
    data_x_units: UnitLike = '',
    data_y_units: UnitLike = '',
) -> tuple[Quantity, tuple[Quantity, Quantity, Quantity, Quantity]]:
    """Prepares data to be plotted in a 2d heatmap plot, like a phase space plot.

    Parameters:
        data: The input data, as a table with every row being a particle at a given time (fully raveled). Must contain the columns given in `x_key` and `y_key`.
        x_key: The key for the x-axis data.
        y_key: The key for the y-axis data.
        x_bins: The bins for the x-axis. Also used to define the x-axis range to consider.
        y_bins: The bins for the y-axis. Also used to define the y-range to consider.
        x_adjust_bins_edges_to_data: Overwrite `x_bins` edges to match the data range.
        y_adjust_bins_edges_to_data: Overwrite `y_bins` edges to match the data range.
        output_type: The type of calculation to fill each bin.
        row_normalization: The normalization to apply to each row. If `None` no normalization is applied. If `float` it must be a percentile value (between 0 and 1), and the normalization will be based on this quantile of each row.
        data_x_units: The units for the x-column in the data. Only used if `data` doesn't have defined units (i.e. a `pd.DataFrame` input).
        data_y_units: The units for the y-column in the data. Only used if `data` doesn't have defined units (i.e. a `pd.DataFrame` input).

    Returns:
        data, extent.
    """
    if isinstance(data, pd.DataFrame):
        sub = table.QTable.from_pandas(data[[x_key, y_key]], units={x_key: data_x_units, y_key: data_y_units})
    else:
        sub = data[[x_key, y_key]]
        data_x_units = data[x_key].unit
        data_y_units = data[y_key].unit

    if data_x_units is None:
        data_x_units = ''

    if data_y_units is None:
        data_y_units = ''

    if x_adjust_bins_edges_to_data:
        x_bins = Quantity(np.linspace(sub[x_key].min(), sub[x_key].max(), len(x_bins)))
    if y_adjust_bins_edges_to_data:
        y_bins = Quantity(np.linspace(sub[y_key].min(), sub[y_key].max(), len(y_bins)))

    x_bins = x_bins.to(data_x_units)
    y_bins = y_bins.to(data_y_units)
    grid = np.histogram2d(
        cast(NDArray[np.float64], sub[x_key]), cast(NDArray[np.float64], sub[y_key]), (x_bins, y_bins)
    )[0].T

    if output_type == 'counts':
        grid = Quantity(
            np.histogram2d(
                cast(NDArray[np.float64], sub[x_key]), cast(NDArray[np.float64], sub[y_key]), (x_bins, y_bins)
            )[0].T
        )
    else:
        assert value_key is not None and value_statistic is not None, (
            '`value_key` and `value_statistic` must be provided if `output_type=value`'
        )
        grid = Quantity(
            scipy.stats.binned_statistic_2d(
                x=data[x_key].to(x_bins.unit).value.astype(np.float64),
                y=data[y_key].to(y_bins.unit).value.astype(np.float64),
                values=data[value_key].value.astype(np.float64),
                statistic=value_statistic,
                bins=[x_bins.value, y_bins.value],
            )[0].T,
            data[value_key].unit,
        )

    if row_normalization == 'max':
        grid /= grid.max(1, keepdims=True)
    elif type(row_normalization) is float:
        grid /= np.quantile(grid, row_normalization, axis=1, keepdims=True)
    elif row_normalization == 'sum':
        grid /= grid.sum(1, keepdims=True)
    elif row_normalization == 'integral':
        grid /= np.expand_dims(np.trapezoid(y=grid, x=np.matlib.repmat(x_bins[:-1], len(grid), 1), axis=1), 1)

    return cast(Quantity, grid), cast(tuple[Quantity, Quantity, Quantity, Quantity], utils.to_extent(x_bins, y_bins))


def aggregate_phase_space_data(
    data: table.QTable | pd.DataFrame,
    radius_bins: Quantity['length'] = Quantity(np.linspace(1e-3, 5, 100), 'kpc'),
    velocity_bins: Quantity['velocity'] = Quantity(np.linspace(0, 100, 200), 'km/second'),
    row_normalization: Literal['max', 'sum', 'integral'] | float | None = None,
    data_length_units: UnitLike = 'kpc',
    data_velocity_units: UnitLike = 'km/second',
) -> tuple[Quantity, tuple[Quantity['length'], Quantity['length'], Quantity['velocity'], Quantity['velocity']]]:
    """Prepares data to be plotted in a 2d phase space heatmap plot, with radius x-axis and velocity y-axis. Intended to be passed on to `plot_2d()`.

    Parameters:
        data: The input data, as a table with every row being a particle at a given time (fully raveled). Must contain the columns `r` and `v_norm`.
        radius_bins: The bins for the radius axis. Also used to define the radius range to consider.
        velocity_bins: The bins for the velocity axis. Also used to define the velocity range to consider.
        row_normalization: The normalization to apply to each row. If `None` no normalization is applied. If `float` it must be a percentile value (between 0 and 1), and the normalization will be based on this quantile of each row.
        data_length_units: The units for the radius in the data. Only used if `data` doesn't have defined units (i.e. a `pd.DataFrame` input).
        data_velocity_units: The units for the velocity in the data. Only used if `data` doesn't have defined units (i.e. a `pd.DataFrame` input).

    Returns:
        data, extent.
    """
    return aggregate_2d_data(
        data=data,
        x_key='r',
        y_key='v_norm',
        x_bins=radius_bins,
        y_bins=velocity_bins,
        row_normalization=row_normalization,
        data_x_units=data_length_units,
        data_y_units=data_velocity_units,
    )


def plot_cumulative_scattering_amount_over_time(
    cumulative_scatters: pd.Series | NDArray[np.float64],
    time: pd.Series | NDArray[np.float64] | Quantity['time'],
    time_unit: UnitLike = 'Gyr',
    xlabel: str | None = 'Time',
    ylabel: str | None = 'Cumulative number of scattering events',
    label: str | None = None,
    data_time_units: UnitLike = 'Myr',
    lineplot_kwargs: dict[str, Any] = {},
    save_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot the cumulative number of scattering events over time.

    Parameters:
        cumulative_scatters: Cumulative number of scattering events.
        time_unit: Units for the x-axis.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        label: Label for the plot (legend).
        data_time_units: The units for the time in the data. Only used if `data` doesn't have defined units (i.e. a `pd.DataFrame` input).
        lineplot_kwargs: Additional keyword arguments to pass to `sns.lineplot()`.
        save_kwargs: Keyword arguments to pass to `save_plot()`. Must include `save_path`. If `None` ignores saving.
        kwargs: Additional keyword arguments to pass to the plot function (`setup_plot()`).

    Returns:
        fig, ax.
    """
    if not isinstance(time, Quantity):
        time = Quantity(time, data_time_units)
    fig, ax = setup_plot(xlabel=utils.add_label_unit(xlabel, time_unit), ylabel=ylabel, **kwargs)
    sns.lineplot(x=time.to(time_unit).value, y=cumulative_scatters, ax=ax, label=label, **lineplot_kwargs)
    if label is not None:
        ax.legend()
    if save_kwargs is not None:
        save_plot(fig=fig, **save_kwargs)
    return fig, ax


def to_images(
    iterator: Iterable[Any],
    plot_fn: Callable[[Any], tuple[Figure, Axes]],
    tight_layout: dict[str, Any] | None = {'pad': 1.5},
    multiplicity: list[int] | NDArray[np.int64] | None = None,
    tqdm_kwargs: dict[str, Any] = {},
) -> list[Image.Image]:
    """Applies a plotting function over an iterator and produce a set of PIL images.

    Parameters:
        iterator: An iterable object containing the data to be plotted.
        plot_fn: A function that takes an element from the iterator and returns a tuple of `Figure` and `Axes`.
        tight_layout: A dictionary of arguments to be passed to the `tight_layout` method of the `Figure` object.
        multiplicity: Inflate the number of frames by the input amount, to allow different duration to different frames. If `None` ignore.
        tqdm_kwargs: Additional keyword arguments to be passed to the `tqdm` function.

    Returns:
        A list of PIL images.
    """
    images = []
    for i, element in enumerate(tqdm(iterator, **tqdm_kwargs)):
        fig, _ = plot_fn(element)
        if tight_layout:
            fig.tight_layout(**tight_layout)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        raw_data, size = canvas.print_to_buffer()
        multiplicity_factor = 1 if multiplicity is None else multiplicity[i]
        images += [Image.frombytes('RGBA', size=size, data=raw_data).convert('RGB')] * multiplicity_factor
        plt.close()
    return images


def evolution_to_images(data: table.QTable, **kwargs: Any) -> list[Image.Image]:
    """Convert a snapshot data table to a list of PIL images with frames for every saved time.

    Parameters:
        data: Table of evolution data.
        kwargs: Additional keyword arguments to pass to the `to_images()` function.

    Returns:
        A list of PIL images.
    """
    unique_times = np.unique(cast(NDArray[np.float64], data['time']))
    tqdm_kwargs = {'start_time': unique_times.min(), 'dt': unique_times.diff(1).mean()}
    return to_images(iterator=data.group_by('time').groups, tqdm_kwargs=tqdm_kwargs, **kwargs)


def save_images(
    images: list[Image.Image], save_path: str | Path, duration: int = 200, loop: int = 0, **kwargs: Any
) -> None:
    """Save a list of images as a GIF.

    Parameters:
        images: List of images to save.
        save_path: Path to save the GIF.
        duration: Duration of each frame in milliseconds.
        loop: Number of times to loop the GIF. 0 means infinite loop.
        kwargs: Additional keyword arguments to pass to the save function.

    Returns:
        None.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    images[0].save(save_path, save_all=True, append_images=images[1:], duration=duration, loop=loop, **kwargs)


def save_plot(fig: Figure, save_path: str | Path | None = None, bbox_inches: str = 'tight', **kwargs: Any) -> None:
    """Save the figure.

    Parameters:
        fig: Figure to save.
        save_path: Path to save the file to. if 'None', the figure is not saved.
        bbox_inches: see `plt.savefig()`. This just defines the default value.
        kwargs: Additional keyword arguments to pass to `plt.savefig()`.

    Returns:
        None.
    """
    if save_path is None:
        return
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches=bbox_inches, **kwargs)


def plot_phase_space_energy_lines(
    Psi_fn: Callable[[Quantity['length']], Quantity['specific energy']],
    E: Quantity['specific energy'],
    r_range: Quantity['length'] = Quantity([1e-3, 35], 'kpc'),
    v_range: Quantity['velocity'] | None = None,
    steps: int = 100,
    length_units: UnitLike = 'kpc',
    velocity_units: UnitLike = 'km/second',
    autolabel: bool = True,
    autolabel_format: str = '.1f',
    autolabel_units: UnitLike = 'km^2/second^2',
    fig: Figure | None = None,
    ax: Axes | None = None,
    lineplot_kwargs: dict[str, Any] = {},
    save_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot constant-energy lines on an r-v phase space.

    Parameters:
        Psi_fn: Function that computes the potential energy.
        E: Array of energy values to plot.
        r_range: Range of radius values to plot on. Only the minimum and maximum values are used.
        v_range: Range of velocities to limit the plot to. Only the minimum and maximum values are used. If `None` ignores.
        steps: Number of grid points.
        length_units: Units for the radius axis.
        velocity_units: Units for the velocity axis.
        autolabel: Whether to automatically label the lines by the energy.
        autolabel_format: Format string for the labels.
        autolabel_units: Units for the labels.
        fig: Figure to plot on.
        ax: Axes to plot on.
        lineplot_kwargs: Keyword arguments to pass to `sns.lineplot()`.
        save_kwargs: Keyword arguments to pass to `save_plot()`. If `None` ignores saving.
        kwargs: Additional keyword arguments to pass to `setup_plot()`.

    Returns:
        fig, ax.
    """
    r = cast(Quantity, np.linspace(r_range.min(), r_range.max(), steps))
    Psi = Psi_fn(r)
    fig, ax = setup_plot(fig=fig, ax=ax, **kwargs)
    for e in E:
        mask = np.ones(len(Psi), dtype=np.bool_)
        v = Quantity(np.zeros(len(Psi)), velocity_units)
        v = 2 * np.sqrt(Psi - e, where=Psi > e).to(velocity_units)
        if v_range is not None:
            mask = (v >= v_range.min()) * (v <= v_range.max())
        if autolabel:
            lineplot_kwargs['label'] = (
                f'E={e.to(autolabel_units).to_string(format="latex", formatter=autolabel_format)}'
            )
        sns.lineplot(x=r[mask], y=v[mask], ax=ax, **lineplot_kwargs)
    if save_kwargs is not None:
        save_plot(fig=fig, **save_kwargs)
    return fig, ax
