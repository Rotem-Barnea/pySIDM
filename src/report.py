from typing import Any, Required, TypedDict


class Line(TypedDict, total=False):
    """A line in a report. Will be printed as `title: {value:format}`

    Parameters:
        title: The title of the line.
        value: The value of the line.
        format: The format of the line.
    """

    title: Required[str]
    value: Required[Any]
    format: str
    prefix_buffer: str


class Report:
    """A report object with pretty print capabilities."""

    def __init__(
        self,
        body_lines: list[Line],
        body_prefix: str | None = None,
        header: str | None = None,
        body_title_buffer: int = 4,
        add_thousands_separator: bool = True,
    ):
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
        self.body_prefix = body_prefix or ''
        self.header: str = header or ''
        self.body_title_buffer = body_title_buffer
        self.add_thousands_separator = add_thousands_separator

    def __repr__(self) -> str:
        return f"""Report(
        body_lines={self.body_lines},
        header={self.header if self.header else 'None'},
        body_title_buffer={self.body_title_buffer},
        add_thousands_separator={self.add_thousands_separator}
    )"""

    def __str__(self):
        return (self.header + '\n' + self.body).strip()

    @property
    def body_title_format(self) -> str:
        """Total format for the titles in the body, aligning the `value` section in all lines."""
        return str(max([len(line['title']) for line in self.body_lines]) + self.body_title_buffer)

    @property
    def body(self) -> str:
        """Body of the report."""
        return '\n'.join(map(self.prepare_line, self.body_lines))

    def prepare_line(self, line: Line) -> str:
        """Prepares a line for the report, by adjusting its formats and constructing it into a string."""
        line_prefix = f'{self.body_prefix}{line.get("prefix_buffer", "")}'
        title = f'{line["title"]}:'
        add_thousands_separator = self.add_thousands_separator and not isinstance(line['value'], str)
        value_format = (',' if add_thousands_separator else '') + line.get('format', '')
        return f'{line_prefix}{title:{self.body_title_format}} {line["value"]:{value_format}}'
