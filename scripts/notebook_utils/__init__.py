"""Shared utilities for Jupyter notebook manipulation.

This package provides common utilities for reading, writing, and manipulating
Jupyter notebooks. It eliminates code duplication across notebook processing
scripts and provides robust error handling.
"""

from .cells import (
    cell_contains_pattern,
    create_ipython_display_import_cell,
    create_matplotlib_config_cell,
    find_cell_with_pattern,
    find_first_code_cell_index,
    has_ipython_display_import,
    has_matplotlib_magic,
    uses_display,
)
from .core import (
    NotebookError,
    NotebookReadError,
    NotebookValidationError,
    NotebookWriteError,
    read_notebook,
    validate_notebook_structure,
    write_notebook,
)
from .types import NotebookCell, NotebookStats

__all__ = [
    "NotebookCell",
    "NotebookError",
    "NotebookReadError",
    "NotebookStats",
    "NotebookValidationError",
    "NotebookWriteError",
    "cell_contains_pattern",
    "create_ipython_display_import_cell",
    "create_matplotlib_config_cell",
    "find_cell_with_pattern",
    "find_first_code_cell_index",
    "has_ipython_display_import",
    "has_matplotlib_magic",
    "read_notebook",
    "uses_display",
    "validate_notebook_structure",
    "write_notebook",
]

__version__ = "0.1.0"
