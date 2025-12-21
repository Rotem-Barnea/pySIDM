from copy import deepcopy
from typing import Any, Self

from astropy.units import Quantity
from astropy.units.typing import UnitLike


class Line:
    """A line in a report. Will be printed as `{prefix_buffer}title: {value.to(unit):format}`."""

    def __init__(
        self,
        title: str,
        value: Any,
        format: str = '',
        prefix_buffer: str = '',
        unit: UnitLike | None = None,
        global_prefix: str = '',
        add_thousands_separator: bool = True,
        title_format: str = '',
    ) -> None:
        self.title = title
        self.value = value.to(self.unit) if (isinstance(value, Quantity) and unit is not None) else value
        self.format = format
        self.prefix_buffer = prefix_buffer
        self.global_prefix = global_prefix
        self.add_thousands_separator = add_thousands_separator
        self.title_format = title_format

    def __str__(self) -> str:
        """Prepares a line for the report, by adjusting its formats and constructing it into a string."""
        line_prefix = f'{self.global_prefix}{self.prefix_buffer}'
        title = f'{self.title}:'
        value = self.validate_format(
            self.value,
            format=(',' if self.add_thousands_separator and not isinstance(self.value, bool) else '') + self.format,
            backup_format=self.format,
        )
        return f'{line_prefix}{title:{self.title_format}} {value}'

    @staticmethod
    def validate_format(value: Any, format: str, backup_format: str = ''):
        """Check if the format is applicable to the value."""
        try:
            return f'{value:{format}}'
        except Exception:
            return f'{value:{backup_format}}'

    def __len__(self) -> int:
        return len(self.title)

    def with_updated_parameters(self, **kwargs: Any) -> Self:
        """Update a parameter"""
        output = self.copy()
        for key, value in kwargs.items():
            setattr(output, key, value)
        return output

    def copy(self) -> Self:
        """Deepcopy the object."""
        return deepcopy(self)


class Report:
    """A report object with pretty print capabilities."""

    def __init__(
        self,
        body_lines: list[Line | Self],
        body_prefix: str = '',
        header: str = '',
        body_title_buffer: int = 4,
        add_thousands_separator: bool = True,
    ) -> None:
        """Constructor for a Report object.

        Parameters:
            body_lines: The lines for the body.
            header: The header to add at the start of the report.
            body_title_buffer: Additional buffer to add between the title and the value.
            add_thousands_separator: Whether to add a thousands separator to all values.

        Returns:
            The Report object.
        """
        self.body_lines = body_lines
        self.body_prefix = body_prefix
        self.header = header
        self.add_thousands_separator = add_thousands_separator
        self.body_title_buffer = body_title_buffer

    def __str__(self) -> str:
        return (self.header + '\n' + self.body).strip()

    def __len__(self) -> int:
        return len(self.header)

    def copy(self) -> Self:
        """Deepcopy the object."""
        return deepcopy(self)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        keys: list[str] = [],
        line_kwargs: dict[str, Any] = {},
        rec_prefix: str = '- ',
        **kwargs: Any,
    ) -> 'Report':
        """Create a Report object from a dictionary (handling nested dictionaries)."""
        items = [(key, payload[key]) for key in keys] if keys else payload.items()
        return cls(
            body_lines=[
                Line(title=title, value=value, **line_kwargs.get(title, {}))
                if not isinstance(value, dict)
                else cls.from_dict(
                    header=f'{title}:',
                    payload=value,
                    body_prefix=f'  {rec_prefix}',
                    rec_prefix=f'  {rec_prefix}',
                    **line_kwargs.get(title, {}),
                )
                for title, value in items
            ],
            **kwargs,
        )

    @property
    def body_title_format(self) -> str:
        """Total format for the titles in the body, aligning the `value` section in all lines."""
        return str(max(map(len, self.body_lines)) + self.body_title_buffer)

    @property
    def compiled_body_lines(self) -> list[Line | Self]:
        """Compiled version of the lines (prepared for printing)"""
        return [
            line.with_updated_parameters(
                global_prefix=self.body_prefix,
                add_thousands_separator=self.add_thousands_separator,
                title_format=self.body_title_format,
            )
            if type(line) is Line
            else line
            for line in self.body_lines
        ]

    @property
    def body(self) -> str:
        """Body of the report."""
        return '\n'.join(map(str, self.compiled_body_lines))
