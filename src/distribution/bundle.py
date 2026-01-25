"""Managing setting up physical (real world) examples of galactic halos"""

from typing import TYPE_CHECKING, Any, Self, Literal, cast, get_args

import regex
import seaborn as sns
from astropy.units import Quantity
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from astropy.units.typing import UnitLike

if TYPE_CHECKING:
    from src.plot import Scale

from src import plot, report, run_units

from .nfw import NFW
from .hernquist import Hernquist
from .distribution import Distribution

physical_examples = Literal['Sague-1', 'Draco', 'Fornax dSph', 'default', 'Daneng2024:DM11+baryon']
distribution_options = Literal['dm_only', 'b_only', None]

DistributionGrids = Literal['rho', 'velocity dispersion', 'temperature', 'pressure', 'internal energy']


class Bundle:
    """A collection of distributions that represent a multi-component galactic halo."""

    def __init__(self, distributions: list[Distribution], merge: bool = False) -> None:
        self.distributions = distributions
        if merge:
            Distribution.merge_distributions(self.distributions)

    @classmethod
    def by_name(
        cls,
        name: physical_examples = 'default',
        suffix: distribution_options = None,
        r_min: Quantity['length'] = Quantity(1e-5, 'kpc'),
        r_max: Quantity['length'] = Quantity(300, 'kpc'),
        dm_kwargs: dict[str, Any] = {},
        b_kwargs: dict[str, Any] = {},
        verbose: bool = False,
        **kwargs: Any,
    ) -> Self:
        """Return a predefined set of distributions for a physical example (mimicking a real galaxy).

        Parameters:
            name: Name of the physical example.
            r_min: Minimum radius of the distribution. Set for all distributions to match internal grids.
            r_max: Maximum radius of the distribution. Set for all distributions to match internal grids.
            dm_kwargs: Keyword arguments for the dark matter distribution.
            b_kwargs: Keyword arguments for the baryonic distribution.
            verbose: Whether to print information about the distributions.
            **kwargs: Additional keyword arguments passed to both distributions.

        Returns:
            A list of distributions representing the physical example.
        """
        if verbose:
            print('Setup distributions')
            print('running example', name)
        distributions: list[Distribution] = []
        if suffix == 'dm_only' or suffix is None:
            distributions += [
                NFW.from_example(name, r_min=r_min, r_max=r_max, particle_type='dm', **dm_kwargs, **kwargs)
            ]
        if suffix == 'b_only' or suffix is None:
            distributions += [
                Hernquist.from_example(name, r_min=r_min, r_max=r_max, particle_type='baryon', **b_kwargs, **kwargs)
            ]
        return cls(distributions=distributions, merge=True)

    def __len__(self) -> int:
        return len(self.distributions)

    def __getitem__(self, index: int) -> Distribution:
        return self.distributions[index]

    @staticmethod
    def validate_input(name: str) -> tuple[physical_examples, distribution_options]:
        """Validate that the given name is a known physical example."""
        suffix = None
        for option in filter(lambda x: x, get_args(distribution_options)):
            if name.endswith(option):
                name = regex.sub(rf'_{option}$', '', name)
                suffix = option
        assert name in get_args(physical_examples), f'Unknown physical example: {name}'
        return cast(physical_examples, name), suffix

    def plot(
        self,
        method: DistributionGrids,
        x_unit: UnitLike | None | Literal['auto'] = 'auto',
        y_unit: UnitLike | None | Literal['auto'] = 'auto',
        xlabel: str | None | Literal['auto'] = 'auto',
        ylabel: str | None | Literal['auto'] = 'auto',
        xscale: 'Scale' = 'log',
        labels: list[str] | None = None,
        lineplot_kwargs: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot properties of the distribution bundle."""
        if xlabel == 'auto':
            xlabel = 'Radius'
        if ylabel == 'auto':
            ylabel = method.title()
        if x_unit == 'auto':
            x_unit = str(self.distributions[0].geomspace_grid.unit)
        if y_unit == 'auto':
            y_unit = str(getattr(self.distributions[0], f'{method.replace(" ", "_")}_grid').unit)
        fig, ax = plot.setup(**kwargs, xscale=xscale, x_unit=x_unit, y_unit=y_unit, xlabel=xlabel, ylabel=ylabel)
        if x_unit is None:
            x_unit = str(self.distributions[0].geomspace_grid.unit)
        if y_unit is None:
            y_unit = str(getattr(self.distributions[0], f'{method.replace(" ", "_")}_grid').unit)
        for i, distribution in enumerate(self.distributions):
            sns.lineplot(
                x=distribution.geomspace_grid.to(x_unit).value,
                y=getattr(distribution, f'{method.replace(" ", "_")}_grid').to(y_unit),
                ax=ax,
                label=labels[i] if labels is not None else None,
                **(lineplot_kwargs[i] if lineplot_kwargs is not None else {}),
            )
        return fig, ax

    def plot_with(
        self,
        method: DistributionGrids,
        other: Self,
        labels: list[str] | None = None,
        lineplot_kwargs: list[dict[str, Any]] | None = None,
        other_labels: list[str] | None = None,
        other_lineplot_kwargs: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot properties of the paired distributions alongside the bundle."""
        fig, ax = self.plot(method=method, labels=labels, lineplot_kwargs=lineplot_kwargs, **kwargs)
        other.plot(method=method, labels=other_labels, lineplot_kwargs=other_lineplot_kwargs, fig=fig, ax=ax, **kwargs)
        return fig, ax


class HaloBaryonBundle(Bundle):
    """Bundle for DM and Baryon distributions (one of each)"""

    def __init__(
        self,
        distributions: list[Distribution] | None = None,
        dm_distribution: Distribution | None = None,
        stellar_distribution: Distribution | None = None,
        **kwargs: Any,
    ):
        if distributions is None:
            assert dm_distribution is not None and stellar_distribution is not None, (
                'DM and stellar distributions must be provided'
            )
            distributions = [dm_distribution, stellar_distribution]
        super().__init__(distributions=distributions, **kwargs)
        self.dm_distribution, self.baryon_distribution = self.distributions

    @property
    def name(self) -> str:
        """The name of the distribution bundle."""
        if self.dm_distribution.name != '' and self.baryon_distribution.name != '':
            if self.dm_distribution.name != self.baryon_distribution.name:
                return f'DM={self.dm_distribution.name} / Stellar={self.baryon_distribution.name}'
            return self.dm_distribution.name
        elif self.dm_distribution.name != '':
            return self.dm_distribution.name
        elif self.baryon_distribution.name != '':
            return self.baryon_distribution.name
        return ''

    def describe(
        self,
        length_unit: UnitLike = run_units.length,
        mass_unit: UnitLike = run_units.mass,
        density_unit: UnitLike = run_units.density,
    ) -> report.Report:
        """Print a description of a pair distribution."""

        return report.Report(
            body_lines=[
                report.Line(title='DM r_s', value=self.dm_distribution.r_s, unit=length_unit, format='.3f'),
                report.Line(title='DM total_mass', value=self.dm_distribution.total_mass, unit=mass_unit, format='.2e'),
                report.Line(title='DM rho0', value=self.dm_distribution.rho_s, unit=density_unit, format='.2e'),
                report.Line(title='Baryon r_s', value=self.baryon_distribution.r_s, unit=length_unit, format='.3f'),
                report.Line(
                    title='Baryon total_mass', value=self.baryon_distribution.total_mass, unit=mass_unit, format='.2e'
                ),
                report.Line(title='Baryon rho0', value=self.baryon_distribution.rho_s, unit=density_unit, format='.2e'),
                report.Line(
                    title='Tilde r_s', value=self.baryon_distribution.r_s / self.dm_distribution.r_s, format='.2f'
                ),
                report.Line(
                    title='Tilde rho0', value=self.baryon_distribution.rho_s / self.dm_distribution.rho_s, format='.2f'
                ),
            ],
            header=f'Description for {self.name}',
            body_prefix='  - ',
        )

    @property
    def report(self) -> report.Report:
        """Print a description of a pair distribution."""
        return self.describe()

    def plot_pair(
        self,
        method: DistributionGrids,
        dm_label: str = 'DM',
        baryon_label: str = 'Baryons',
        dm_lineplot_kwargs: dict[str, Any] = {},
        baryon_lineplot_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot properties of the paired distributions."""
        return self.plot(
            method=method,
            labels=[dm_label, baryon_label],
            lineplot_kwargs=[dm_lineplot_kwargs, baryon_lineplot_kwargs],
            **kwargs,
        )

    def plot_pair_with(
        self,
        method: DistributionGrids,
        other: Bundle,
        other_labels: list[str] | None = None,
        other_lineplot_kwargs: list[dict[str, Any]] | None = None,
        dm_label: str = 'DM',
        baryon_label: str = 'Baryons',
        dm_lineplot_kwargs: dict[str, Any] = {},
        baryon_lineplot_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot properties of the paired distributions alongside the bundle."""
        fig, ax = self.plot_pair(
            method=method,
            dm_label=dm_label,
            baryon_label=baryon_label,
            dm_lineplot_kwargs=dm_lineplot_kwargs,
            baryon_lineplot_kwargs=baryon_lineplot_kwargs,
            **kwargs,
        )
        other.plot(method=method, labels=other_labels, lineplot_kwargs=other_lineplot_kwargs, fig=fig, ax=ax, **kwargs)
        return fig, ax
