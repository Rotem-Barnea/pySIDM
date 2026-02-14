"""Internal module for animating halo data"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, cast
from functools import partial

from PIL import Image
from astropy import table
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from astropy.units.typing import UnitLike

from src import plot, types

if TYPE_CHECKING:
    from .halo import Halo
    from .plot import HaloPlotter


class HaloAnimator:
    """Class for animating halo data."""

    def __init__(self, halo: Halo):
        self.halo: Halo = halo

    @property
    def results_path(self):
        """The path to the results directory."""
        return self.halo.results_path

    @property
    def plot(self) -> HaloPlotter:
        """Plotter object"""
        return self.halo.plot

    def fill_time_unit(self, unit: types.TimeUnitLike) -> UnitLike:
        """If `unit` is a halo-related time parameter return its unit, otherwise return `unit`."""
        return self.plot.fill_time_unit(unit)

    def save(self, images: list[Image.Image], save_kwargs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Saves the plot."""
        if save_kwargs is None:
            return
        save_kwargs = deepcopy(save_kwargs)
        if 'name' in save_kwargs:
            save_kwargs['save_path'] = self.results_path / save_kwargs.pop('name')
        plot.save_images(images=images, **save_kwargs, **kwargs)

    def states_iterator(self, drop_last: bool = False, *args: Any, **kwargs: Any) -> list[table.QTable]:
        """A list of tables containing the particle states at each saved time.

        Parameters:
            drop_last: Whether to drop the last timestep.
            *args: Arguments passed on to `Halo.get_particle_states()`.
            **kwargs: Keyword arguments passed on to `Halo.get_particle_states()`.

        Returns:
            A list of Qtables.
        """
        data = cast(list[table.QTable], list(self.halo.get_particle_states(*args, **kwargs).group_by('time').groups))
        if drop_last:
            data = data[:-1]
        return data

    def animate(
        self,
        plot_fn: Callable[[Any], tuple[Figure, Axes]],
        include_start: bool = False,
        include_now: bool = False,
        drop_last: bool = True,
        image_kwargs: dict[str, Any] = {},
        save_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> None:
        """Animate the fraction of the enclosed mass that is contributed by the specified particle type, as a function of the radius, over time.

        Parameters:
            plot_fn: Plotting function, Must receive the data from `self.states_iterator` (individual groups), and return (fig, ax).
            include_start: Whether to include the initial state.
            include_now: Whether to include the current state.
            drop_last: Whether to drop the last timestep.
            image_kwargs: Additional keyword arguments for transforming the frames to images (`plot.to_images()`).
            save_kwargs: Additional keyword arguments to pass to `save()`. Must include `save_path`.
            kwargs: Additional keyword arguments passed to the plotting function `plot_fn()`.
        """
        assert save_kwargs.get('save_path') is not None, "save_kwargs must be provided and must include 'save_path'"
        self.save(
            plot.to_images(
                iterator=self.states_iterator(initial=include_start, now=include_now, drop_last=drop_last),
                plot_fn=partial(plot_fn, **kwargs),
                **image_kwargs,
            ),
            save_kwargs=save_kwargs,
        )

    def enclosed_mass_ratio(self, particle_type: types.ParticleType, drop_last: bool = True, **kwargs: Any) -> None:
        """Animate the fraction of the enclosed mass that is contributed by the specified particle type, as a function of the radius, over time.

        Parameters:
            particle_type: The particle type to calculate the fraction distribution for.
            drop_last: Whether to drop the last timestep.
            kwargs: Additional keyword arguments passed to the animating function `self.animate()` and from there to the plotting function `halo.plot.enclosed_mass_ratio()`.
        """
        self.animate(
            plot_fn=lambda x, **kwargs: self.plot.enclosed_mass_ratio(data=x, particle_type=particle_type, **kwargs),
            drop_last=drop_last,
            **kwargs,
        )

    def local_density(self, drop_last: bool = True, **kwargs: Any) -> None:
        """Animate the local local density profile as a function of the radius.

        Parameters:
            drop_last: Whether to drop the last timestep.
            kwargs: Additional keyword arguments passed to the animating function `self.animate()` and from there to the plotting function `halo.plot.local_density()`.
        """
        self.animate(
            plot_fn=lambda x, **kwargs: self.plot.local_density(data=x, **kwargs),
            drop_last=drop_last,
            **kwargs,
        )

    def phase_space(self, **kwargs: Any) -> None:
        """animate the phase space evolution of the halo.

        Parameters:
            kwargs: Additional keyword arguments passed to the animating function `self.animate()` and from there to the plotting function `halo.plot.phase_space()`.
        """
        self.animate(
            plot_fn=lambda x, **kwargs: self.halo.plot.phase_space(
                data=x,
                **{k: v for k, v in kwargs.items() if k != 'setup_kwargs'},
                setup_kwargs={
                    'texts': [
                        {
                            's': f'{x["time"][0].to("Gyr"):.2f}',
                            **plot.pretty_ax_text(x=0.95, y=0.95, transform='transAxes', horizontalalignment='right'),
                        }
                    ],
                    **kwargs.get('setup_kwargs', {}),
                },
            ),
            **kwargs,
        )
