import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from astropy import table
from astropy.units import Quantity
from astropy.units.typing import UnitLike
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.ticker as mtick
from . import run_units
from numpy.typing import NDArray
from typing import Any, Literal, cast
from . import utils


def setup_plot(
    fig: Figure | None = None,
    ax: Axes | None = None,
    grid: bool = True,
    minorticks: bool = False,
    figsize: tuple[int, int] | None = (6, 5),
    ax_set: dict[str, str] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Setup a plot with optional grid, minor ticks, and axis settings.

    Parameters:
        fig: Figure to plot on. If None, a new figure is created.
        ax: Axes to plot on. If None, a new axes is created.
        grid: Whether to add a grid to the plot (major ticks).
        minorticks: Whether to add the grid for the minor ticks.
        figsize: The figure size to create. Ignored if fig/ax is provided.
        ax_set: Additional keyword arguments to pass to Axes.set(). e.g {'xscale': 'log'}.
        title: The title of the plot.
        xlabel: The label of the x axis.
        ylabel: The label of the y axis.
        kwargs: Additional keyword arguments to pass to plt.subplots().

    Returns:
        fig, ax.
    """
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
    return fig, ax


def default_plot_text(
    key: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    x_units: UnitLike | None = None,
    y_units: UnitLike | None = None,
) -> dict[str, str | None]:
    """Return default plot title/xlabel/ylabel for a given key and add the appropriate units.

    If xlabel/ylabel/title are provided, they override the defaults.
    """
    if key == 'vr':
        output = {'title': 'Radial velocity distribution', 'xlabel': 'Radial velocity', 'ylabel': 'Density'}
    elif key in ['vx', 'vy', 'vp']:
        output = {'title': 'Pendicular velocity distribution', 'xlabel': 'Pendicular velocity', 'ylabel': 'Density'}
    elif key == 'v_norm':
        output = {'title': 'Pendicular velocity distribution', 'xlabel': 'Pendicular velocity', 'ylabel': 'Density'}
    elif key == 'r':
        output = {'title': 'Radius distribution', 'xlabel': 'Radius', 'ylabel': 'Density'}
    else:
        output = {}
    output = {**output, **utils.drop_None(xlabel=xlabel, ylabel=ylabel, title=title)}
    return utils.drop_None(
        title=output.get('title', None),
        xlabel=utils.add_label_unit(output.get('xlabel', None), x_units),
        ylabel=utils.add_label_unit(output.get('ylabel', None), y_units),
    )


def default_plot_unit_type(key: str, plot_unit: UnitLike | None = None) -> UnitLike:
    """Return the appropriate unit type for a given key. If plot unit is provided, return it instead."""
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
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot the trace of a particle's property over time.

    Parameters:
        key: The property to plot. Must be a valid column name in the data table.
        data: The data table to plot (i.e. halo.snapshots, or an external table from an NSphere run).
        particle_index: The index of the particle to trace.
        relative: If `absolute`, plot the property as is. If `relative`, plot the change in the property relative to the initial value. If `relative change`, plot the change in the property relative to the initial value divided by the initial value.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis. If not provided, the label will be automatically generated based on the key and units. To disable this, set to ''.
        title: Title for the plot.
        time_units: Units for the x-axis.
        y_units: Units for the y-axis.
        length_units: Units for the length.
        length_format: Format string for length.
        kwargs: Additional keyword arguments to pass to the plot function (setup_plot).

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
        title = title.format(particle_index=particle_index, r=particle['r'][0].to(length_units).to_string(format='latex', formatter=length_format))
    if label is not None:
        label = label.format(particle_index=particle_index, r=particle['r'][0].to(length_units).to_string(format='latex', formatter=length_format))
    fig, ax = setup_plot(**kwargs, **utils.drop_None(title=title, xlabel=xlabel, ylabel=ylabel))
    sns.lineplot(x=x, y=np.array(y), ax=ax, label=label)
    return fig, ax


def plot_2d(
    grid: Quantity,
    extent: tuple[Quantity, Quantity, Quantity, Quantity] | None = None,
    x_range: Quantity | None = None,
    y_range: Quantity | None = None,
    x_units: UnitLike = run_units.length,
    y_units: UnitLike = run_units.velocity,
    cbar_units: UnitLike = '',
    x_nbins: int | None = 6,
    y_nbins: int | None = 6,
    x_tick_format: str = '%.0f',
    y_tick_format: str = '%.0f',
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    cbar_label: str | None = None,
    log_scale: bool = False,
    hlines: list[dict[str, Any]] = [],
    vlines: list[dict[str, Any]] = [],
    fig: Figure | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot a 2d heatmap, such as a phase space distribution.

    Parameters:
        grid: 2d array of data to plot.
        extent: Range of values to plot, used to define the x and y axes. If None, derive from x_range and y_range, otherwise takes priority.
        x_range: Range of x values to plot, used to define the extent. Must be provided if extent is None, otherwise ignored.
        y_range: Range of y values to plot, used to define the extent. Must be provided if extent is None, otherwise ignored.
        x_units: Units to use for the x axis.
        y_units: Units to use for the y axis.
        cbar_units: Units to use for the value of each grid cell.
        x_nbins: Number of bins to use for the x axis.
        y_nbins: Number of bins to use for the y axis.
        x_tick_format: Format string for the x axis ticks.
        y_tick_format: Format string for the y axis ticks.
        title: Title of the plot.
        xlabel: Label for the x axis.
        ylabel: Label for the y axis.
        cbar_label: Label for the colorbar.
        log_scale: Whether to use a log scale for the colorbar. If True, overwrites the "norm" argument if provided (in kwargs), and sets the norm to LogNorm().
        hlines: List of horizontal lines to plot. Each element contains the keywords arguments passed to ax.axhline().
        vlines: List of vertical lines to plot. Each element contains the keywords arguments passed to ax.axvline().
        fig: Figure to plot on.
        ax: Axes to plot on.
        kwargs: Additional keyword arguments to pass to imshow.

    Returns:
        fig, ax.
    """
    if extent is None:
        assert x_range is not None and y_range is not None, 'x_range and y_range must be provided if extent is None'
        extent = (x_range.to(x_units).min(), x_range.to(x_units).max(), y_range.to(y_units).min(), y_range.to(y_units).max())
    extent_value = (
        float(extent[0].to(x_units).value),
        float(extent[1].to(x_units).value),
        float(extent[2].to(y_units).value),
        float(extent[3].to(y_units).value),
    )

    # extent_value = tuple(np.hstack([*extent[:2].to(x_units).value, *extent[2:].to(y_units).value]))

    if log_scale:
        kwargs.update(norm=LogNorm())

    fig, ax = setup_plot(fig=fig, ax=ax, grid=False, title=title, xlabel=xlabel, ylabel=ylabel)
    if cbar_units != '':
        grid = grid.to(cbar_units)
    im = ax.imshow(grid, origin='lower', aspect='auto', extent=extent_value, **kwargs)
    cbar = fig.colorbar(im, ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label)

    if x_nbins is not None:
        ax.xaxis.set_major_locator(mtick.MaxNLocator(nbins=x_nbins))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter(x_tick_format))
        ax.xaxis.tick_bottom()
        for lab in ax.get_xticklabels():
            lab.set_rotation(0)
            lab.set_horizontalalignment('center')

    if y_nbins is not None:
        ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=y_nbins))
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter(y_tick_format))

    for line in hlines:
        ax.axhline(**line)
    for line in vlines:
        ax.axvline(**line)

    return fig, ax


def plot_phase_space(
    grid: Quantity,
    r_range: Quantity['length'] | None = Quantity([1e-2, 50], 'kpc'),
    v_range: Quantity['velocity'] | None = Quantity([0, 100], 'km/second'),
    length_units: UnitLike = run_units.length,
    velocity_units: UnitLike = run_units.velocity,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot the phase space distribution. Wrapper for plot_2d() to provide convenient defaults and variable names (i.e. r and v)."""
    return plot_2d(
        grid,
        xlabel=utils.add_label_unit('Radius', length_units),
        ylabel=utils.add_label_unit('Velocity', velocity_units),
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
    label: str | None = None,
    cleanup_nonpositive: bool = True,
    smooth_sigma: float = 1,
    fig: Figure | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot the density distribution per unit volume.

    Parameters:
        data: Data to plot.
        bins: Argument accepted by np.histogram.
        unit_mass: Unit mass to convert the number density to mass density,
        xlabel: Label for the x axis.
        ylabel: Label for the y axis.
        title: Title for the plot.
        length_units: Units to use for the x axis.
        density_units: Number of bins to use for the y axis.
        label: label to add to the plot legend.
        cleanup_nonpositive: drop non-positive values from the plot, to avoid "pits" in the log plot.
        smooth_sigma: sigma for smoothing the density distribution.
        fig: Figure to plot on.
        ax: Axes to plot on.
        kwargs: Additional keyword arguments to pass to setup_plot().


    Returns:
        fig, ax.
    """
    counts, bin_edges = np.histogram(data, bins=bins)
    histogram_bins = Quantity(list(map(Quantity, zip(bin_edges, bin_edges[1:])))).to(length_units)
    bin_centers = histogram_bins.mean(1)
    volume = 4 / 3 * np.pi * (histogram_bins[:, 1] ** 3 - histogram_bins[:, 0] ** 3)
    density = (counts / volume * unit_mass).to(density_units)
    fig, ax = setup_plot(
        fig=fig,
        ax=ax,
        ax_set={'xscale': 'log', 'yscale': 'log'},
        minorticks=True,
        **default_plot_text('r', xlabel=xlabel, ylabel=ylabel, title=title, x_units=length_units, y_units=density_units),
        **kwargs,
    )
    x = np.array(bin_centers)
    if cleanup_nonpositive:
        x = x[density > 0]
        density = density[density > 0]
    if smooth_sigma > 0:
        density = scipy.ndimage.gaussian_filter1d(density, sigma=smooth_sigma)
    sns.lineplot(x=x, y=density, ax=ax, label=label)
    if label is not None:
        ax.legend()
    return fig, ax


def aggregate_2d_data(
    data: table.QTable | pd.DataFrame,
    radius_bins: Quantity['length'] = Quantity(np.linspace(1e-3, 5, 100), 'kpc'),
    time_range: Quantity['time'] | None = None,
    unit_mass: Quantity['mass'] = Quantity(1, 'Msun'),
    output_type: Literal['density', 'counts', 'temperature'] = 'counts',
    density_units: UnitLike = run_units.density,
    data_time_units: UnitLike = 'Gyr',
    data_length_units: UnitLike = 'kpc',
    data_velocity_units: UnitLike = 'kpc/Myr',
) -> tuple[Quantity, tuple[Quantity['length'], Quantity['length'], Quantity['time'], Quantity['time']]]:
    """Prepares data to be plotted in a 2d heatmap plot, with radius x-axis and time y-axis. Intended to be passed on to plot_2d().

    Parameters:
        data: The input data, as a table with every row being a particle at a given time (fully raveled). Must contain the columns 'r' and 'time'. If `output_type='temperature'`, must also contain the column 'v_norm'.
        radius_bins: The bins for the radius axis. Also used to define the radius range to consider.
        time_range: The range of time to consider.
        unit_mass: The unit of mass. Used for density calculations, only relevant if `output_type='density'`.
        output_type: The type of calculation to fill each bin.
        density_units: The units for the density.
        data_time_units: The units for the time in the data. Only used if `data` doesn't have defined units (i.e. a pd.DataFrame input).
        data_length_units: The units for the radius in the data. Only used if `data` doesn't have defined units (i.e. a pd.DataFrame input).

    Returns:
        data, extent.
    """
    keep_columns = ['r', 'time']
    if output_type == 'temperature':
        keep_columns += ['v_norm']
    if isinstance(data, pd.DataFrame):
        sub = data[keep_columns]
    else:
        sub = data[keep_columns].to_pandas()
        data_time_units = data['time'].unit
        data_length_units = data['r'].unit
        if output_type == 'temperature':
            data_velocity_units = data['v_norm'].unit

    if time_range is not None:
        sub = data[data['time'].between(*time_range.to(data_time_units).value)]

    radius_bins = radius_bins.to(data_length_units)
    r_bin_volume = 4 / 3 * np.pi * (radius_bins[1:] ** 3 - radius_bins[:-1] ** 3)
    grid = np.empty((len(np.unique(np.array(sub['time']))), len(radius_bins) - 1), dtype=np.float64)

    for i, (_, group) in enumerate(sub.groupby('time')):
        if output_type == 'temperature':
            group_bins = pd.cut(Quantity(group['r'], data_length_units).to(radius_bins.units).value, bins=radius_bins.value)
            grid[i] = np.array(group.groupby(group_bins, observed=False)['v_norm'].std())
        else:
            counts, _ = np.histogram(Quantity(group['r'], data_length_units), bins=radius_bins)
            if output_type == 'density':
                grid[i] = counts / r_bin_volume * unit_mass
            else:
                grid[i] = counts

    if output_type == 'density':
        grid_units = unit_mass.unit / r_bin_volume.unit
    elif output_type == 'temperature':
        grid_units = data_velocity_units
    else:
        grid_units = ''

    extent = (radius_bins.min(), radius_bins.max(), sub['time'].min(), sub['time'].max())
    return Quantity(grid, grid_units), extent
