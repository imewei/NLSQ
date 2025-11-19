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
    # Types
    "NotebookCell",
    "NotebookStats",
    # Exceptions
    "NotebookError",
    "NotebookReadError",
    "NotebookWriteError",
    "NotebookValidationError",
    # Core I/O
    "read_notebook",
    "write_notebook",
    "validate_notebook_structure",
    # Cell utilities
    "has_matplotlib_magic",
    "has_ipython_display_import",
    "uses_display",
    "find_first_code_cell_index",
    "find_cell_with_pattern",
    "cell_contains_pattern",
    "create_matplotlib_config_cell",
    "create_ipython_display_import_cell",
]

__version__ = "0.1.0"
