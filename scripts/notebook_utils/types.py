"""Type definitions for notebook manipulation."""

from typing import Literal, TypedDict


class NotebookCell(TypedDict, total=False):
    """Jupyter notebook cell structure."""

    cell_type: Literal["code", "markdown", "raw"]
    execution_count: int | None
    metadata: dict
    outputs: list
    source: str | list[str]


class NotebookStats(TypedDict):
    """Statistics from notebook processing."""

    matplotlib_magic_added: int
    ipython_display_import_added: int
    plt_show_replaced: int
    cells_modified: int
