from tqdm.auto import tqdm as tqdm_base
from astropy.units import Quantity
from astropy.units.typing import UnitLike
from collections.abc import Iterable, Generator
from typing import Any, cast


class tqdm(tqdm_base):
    """tqdm subclass that also displays the simulation time."""

    def __init__(
        self,
        iterable: Iterable[Any],
        start_time: Quantity['time'] | None = None,
        dt: Quantity['time'] | None = None,
        time_units: UnitLike = 'Gyr',
        time_format: str = '.3f',
        **kwargs: Any,
    ):
        """See `tqdm.tqdm` from the original `tqdm module` for additional arguments.

        Parameters:
            iterable: The iterable to iterate over.
            start_time: The start time of the loop. If `None` defaults to the regular tqdm behavior (not displaying the time).
            dt: The time step of every element in the loop. If `None` defaults to the regular tqdm behavior (not displaying the time).
            time_units: The display units for the time.
            time_format: The format string for the time.
            **kwargs: Additional keyword arguments for `tqdm.tqdm` (from the original `tqdm module`).
        """
        if start_time is not None and dt is not None:
            self.time_mode = True
            self.start_time: Quantity['time'] = start_time.to(time_units)
            self.dt: Quantity['time'] = dt.to(time_units)
            self.time_format = time_format
            iterable = list(iterable)
            self.total_time = cast(Quantity['time'], self.start_time + self.dt * len(iterable))
        else:
            self.time_mode = False

        super().__init__(iterable, **kwargs)

    def __iter__(self) -> Generator[Any, None, None]:
        for i, obj in enumerate(super().__iter__()):
            if self.time_mode:
                current_time = cast(Quantity['time'], i * self.dt + self.start_time)
                self.set_description(f'Time: [{self.cleanup_time(current_time)}]/[{self.cleanup_time(self.total_time, add_units=True)}]')
            yield obj

    def cleanup_time(self, time: Quantity['time'], add_units: bool = False) -> str:
        value = format(time.value, self.time_format).rstrip('0').rstrip('.')
        if add_units:
            return f'{value} {time.unit}'
        return value
