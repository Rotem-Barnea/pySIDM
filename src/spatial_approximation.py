import numpy as np
from . import utils
from numba import njit, prange
from .types import FloatOrArray
from numpy.typing import NDArray
from .distribution.distribution import Distribution
from typing import Any, cast, Self


class Lattice:
    """Lattice used for spatial discretization and approximation."""

    def __init__(self, n_posts: int | float, start: float, end: float, log: bool = True) -> None:
        """Initialize a lattice object.

        Parameters:
            n_posts: Number of posts in the lattice.
            start: Start of the lattice.
            end: End of the lattice.
            log: Whether to use logarithmic spacing, or a linear one.

        Returns:
            The lattice object.
        """
        self.start = float(start)
        self.end = float(end)
        if log:
            self.start_lattice: float = np.log10(self.start)
            self.end_lattice: float = np.log10(self.end)
        else:
            self.start_lattice = self.start
            self.end_lattice = self.end
        self.log = log
        self.n_posts = int(n_posts)
        self.lattice_spacing: float = np.abs(self.end_lattice - self.start_lattice) / self.n_posts

    @classmethod
    def from_distribution(
        cls, distribution: Distribution, start: float = 1e-4, overide_start: bool = True, n_posts: int | float = int(1e4), **kwargs: Any
    ) -> Self:
        """Construct a lattice to match the extent of a density object."""
        return cls(start=distribution.Rmin.value if overide_start else start, end=distribution.Rmax.value, n_posts=n_posts, **kwargs)

    def __len__(self) -> int:
        """Return the number of posts in the lattice."""
        return self.n_posts

    def __call__(self, x: FloatOrArray) -> FloatOrArray:
        """Transforms spatial coordinates to lattice coordinates."""
        return self.to_lattice_coordinates(x)

    @property
    def posts(self) -> NDArray[np.float64]:
        """Calculate the posts of the lattice (i.e. `np.linspace` or `np.geomspace`)."""
        if self.log:
            return np.geomspace(self.start, self.end, self.n_posts)
        return np.linspace(self.start, self.end, self.n_posts)

    @property
    def post_volume(self) -> NDArray[np.float64]:
        """Calculate the volume of each point in the lattice, assuming a spherical geometry, and using a thin shell approximation."""
        if self.log:
            posts = self.posts
            return 4 * np.pi * posts**2 * np.diff(posts, 1, append=posts[-1])
        return 4 * np.pi * self.posts**2 * self.lattice_spacing

    def update(self, r: NDArray[np.float64], n_posts: int | float | None = None) -> None:
        """Update the lattice spacing and posts to ensure the given radius range is contained within."""
        self.end = np.max([r.max(), self.end])
        if self.log:
            self.start_lattice = np.log10(self.start)
            self.end_lattice = np.log10(self.end)
        else:
            self.start_lattice = self.start
            self.end_lattice = self.end
        if n_posts:
            self.n_posts = int(n_posts)
        self.lattice_spacing = np.abs(self.end_lattice - self.start_lattice) / self.n_posts

    @staticmethod
    @njit(parallel=True)
    def fast_augment_to_lattice(
        x: NDArray[np.float64], start_lattice: float, log: bool, clip: bool = False, min_lattice: float = 0, max_lattice: int = 100000
    ) -> NDArray[np.float64]:
        """Transform the given array from spatial coordinates to lattice coordinates. njit accelerated."""
        output = np.empty_like(x)
        for i in prange(len(x)):
            if log:
                new_x = np.log10(x[i]) - start_lattice
            else:
                new_x = x[i] - start_lattice
            if clip:
                if new_x < min_lattice:
                    new_x = min_lattice
                elif new_x > max_lattice:
                    new_x = max_lattice
            output[i] = new_x
        return output

    @staticmethod
    @njit(parallel=True)
    def fast_augment_from_lattice(x: NDArray[np.float64], start_lattice: float, log: bool) -> NDArray[np.float64]:
        """Transform the given array from lattice coordinates to spatial coordinates. njit accelerated."""
        output = np.empty_like(x)
        for i in prange(len(x)):
            if log:
                output[i] = 10 ** (x[i] + start_lattice)
            else:
                output[i] = x[i] + start_lattice
        return output

    def augment_to_lattice(self, x: FloatOrArray, clip: bool = True) -> FloatOrArray:
        """Transform the given array from spatial coordinates to lattice coordinates. Wraps the njit-accelerated `fast_augment_to_lattice()`."""
        output = self.fast_augment_to_lattice(np.atleast_1d(x), self.start_lattice, self.log, clip, 0, len(self))
        if np.isscalar(x):
            output = output[0]
        return cast(FloatOrArray, output)

    def augment_from_lattice(self, x: FloatOrArray) -> FloatOrArray:
        """Transform the given array from lattice coordinates to spatial coordinates. Wraps the njit-accelerated `fast_augment_from_lattice()`."""
        output = self.fast_augment_from_lattice(np.atleast_1d(x), self.start_lattice, self.log)
        if np.isscalar(x):
            output = output[0]
        return cast(FloatOrArray, output)

    def in_lattice(self, x: FloatOrArray) -> FloatOrArray:
        """Check if the given values are within the lattice bounds."""
        return cast(FloatOrArray, (x >= self.start) * (x <= self.end))

    def to_lattice_coordinates(self, x: FloatOrArray, clip: bool = True) -> FloatOrArray:
        """Transform the given array from spatial coordinates to lattice points (i.e. integer values). Wraps `augment_to_lattice()`."""
        x_lattice = self.augment_to_lattice(x, clip) / self.lattice_spacing
        if np.isscalar(x):
            if x_lattice == 0:
                return cast(FloatOrArray, 0)
            else:
                return cast(FloatOrArray, int(x_lattice))
        assert isinstance(x_lattice, np.ndarray)
        x_lattice[x == 0] = 0
        x_lattice = (x_lattice).astype(int)
        return cast(FloatOrArray, x_lattice)

    def to_space_coordinates(self, x: FloatOrArray) -> FloatOrArray:
        """Transform the given array from lattice points (i.e. integer values) to spatial coordinates. Wraps `augment_from_lattice()`."""
        return self.augment_from_lattice(x * self.lattice_spacing)

    def values_on_lattice_point(self, x: NDArray[np.float64]) -> NDArray[np.int64]:
        """Calculate number of values in each point in the lattice.

        Parameters:
            x: Array of lattice points. must be integer values.

        Returns:
            Array of shape `(len(density),)` where every cell is the number of points in `x` that match the corresponding lattice cell.
        """
        return np.bincount(x.clip(min=0), minlength=len(self))

    def values_on_lattice_point_cumsum(self, x: NDArray[np.float64]) -> NDArray[np.int64]:
        """Calculate cumulative sum of number of values in each point in the lattice. See `values_on_lattice_point()`."""
        density = self.values_on_lattice_point(x)
        return np.cumsum(density)

    def assign_from_values_cumsum(self, r: FloatOrArray) -> FloatOrArray:
        """Calculate the cumulative sum for each element in `r`.

        Every element in `r` is converted to the corresponding lattice point, and then assigned the number of values that fall within or below that point.

        Parameters:
            r: Array of spatial points.

        Returns:
            Array of shape `(len(r),)` where every cell is the number of in or below the corresponding lattice cell.
        """
        x = np.atleast_1d(self(r))
        value_cumsum = self.values_on_lattice_point_cumsum(x).astype(np.float64)
        assigned = utils.fast_assign(x, value_cumsum)
        if np.isscalar(r):
            return assigned[0]
        return cast(FloatOrArray, assigned)

    def assign_spatial_density(self, r: FloatOrArray, unit_mass: float) -> FloatOrArray:
        """Calculate the density for each element in `r`.

        Every element in `r` is converted to the corresponding lattice point, and then assigned the number of values that fall within that point divided by the volume of that cell.

        Parameters:
            r: Array of spatial points.
            unit_mass: The mass of each element in the lattice (assumes a single equal mass for all elements).

        Returns:
            Array of shape `(len(r),)` where every cell is the mass density in the corresponding lattice cell for that element value of `r`.
        """
        x = np.atleast_1d(self(r))
        density = unit_mass * self.values_on_lattice_point(x) / self.post_volume
        assigned = utils.fast_assign(x, density)
        if np.isscalar(r):
            return assigned[0]
        return cast(FloatOrArray, assigned)
