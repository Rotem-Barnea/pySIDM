from typing import Any, TypedDict


class Line(TypedDict):
    """A line in a report. Will be printed as `title: {value:format}`

    Parameters:
        title: The title of the line.
        value: The value of the line.
        format: The format of the line.
    """

    title: str
    value: Any
    format: str


class Report:
    """A report object with pretty print capabilities."""

    def __init__(
        self,
        body_lines: list[Line],
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
        title = f'{line["title"]}:'
        add_thousands_separator = self.add_thousands_separator and not isinstance(line['value'], str)
        value_format = (',' if add_thousands_separator else '') + line['format']
        return f'{title:{self.body_title_format}} {line["value"]:{value_format}}'


def compile(
    report: list[Line],
    header: str | None = None,
    title_buffer: int = 4,
    add_thousands_separator: bool = True,
) -> str:
    """Compile a report from a list of dictionaries.

    Parameters:
        report: The report to compile.
        header: The header to add at the start of the report.
        title_buffer: The buffer to add between the title and the value.
        add_thousands_separator: Whether to add a thousands separator to all values.

    Returns:
        The compiled report.
    """
    buffer = max([len(line['title']) for line in report]) + title_buffer
    body = '\n'.join(
        [
            f'{line["title"] + ":":{buffer}} {line["value"]:{',' if add_thousands_separator else ''}{line['format']}}'
            for line in report
        ]
    )
    if header is not None:
        return header + '\n' + body
    return body
