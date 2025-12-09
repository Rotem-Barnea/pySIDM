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
