import numpy as np
import seaborn as sns
from astropy import table
from astropy.units import Quantity
from astropy.units.typing import UnitLike
from matplotlib import pyplot as plt
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
        ylabel: Label for the y-axis.
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
        ylabel = utils.add_label_unit(ylabel, y_units)
    if relative == 'change':
        y = y - cast(Quantity, y[0])
    elif relative == 'relative change':
        y = (y - cast(Quantity, y[0])) / cast(Quantity, y[0])
    if title is not None:
        title = title.format(particle_index=particle_index, r=particle['r'][0].to(length_units).to_string(format='latex', formatter=length_format))
    if label is not None:
        label = label.format(particle_index=particle_index, r=particle['r'][0].to(length_units).to_string(format='latex', formatter=length_format))
    fig, ax = setup_plot(**kwargs, **utils.drop_None(title=title, xlabel=xlabel, ylabel=ylabel))
    sns.lineplot(x=x, y=np.array(y), ax=ax, label=label)
    return fig, ax


def plot_2d(
    grid: NDArray[Any],
    extent: tuple[Quantity, Quantity, Quantity, Quantity] | None = None,
    x_range: Quantity | None = None,
    y_range: Quantity | None = None,
    x_units: UnitLike = run_units.length,
    y_units: UnitLike = run_units.velocity,
    x_nbins: int | None = 6,
    y_nbins: int | None = 6,
    x_tick_format: str = '%.0f',
    y_tick_format: str = '%.0f',
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    cbar_label: str | None = None,
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
        x_nbins: Number of bins to use for the x axis.
        y_nbins: Number of bins to use for the y axis.
        x_tick_format: Format string for the x axis ticks.
        y_tick_format: Format string for the y axis ticks.
        title: Title of the plot.
        xlabel: Label for the x axis.
        ylabel: Label for the y axis.
        cbar_label: Label for the colorbar.
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

    fig, ax = setup_plot(fig=fig, ax=ax, grid=False, title=title, xlabel=xlabel, ylabel=ylabel)
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
    return fig, ax


def plot_phase_space(
    grid: NDArray[Any],
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
