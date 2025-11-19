"""Core I/O utilities for notebook manipulation with robust error handling."""

import contextlib
import json
import logging
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class NotebookError(Exception):
    """Base exception for notebook processing errors."""


class NotebookReadError(NotebookError):
    """Error reading notebook file."""


class NotebookWriteError(NotebookError):
    """Error writing notebook file."""


class NotebookValidationError(NotebookError):
    """Invalid notebook structure."""


def validate_notebook_structure(notebook: dict) -> None:
    """Validate notebook has required structure.

    Args:
        notebook: Notebook dictionary to validate

    Raises:
        NotebookValidationError: If notebook structure is invalid
    """
    if not isinstance(notebook, dict):
        raise NotebookValidationError("Notebook must be a dictionary")

    if "cells" not in notebook:
        raise NotebookValidationError("Notebook missing 'cells' key")

    if not isinstance(notebook["cells"], list):
        raise NotebookValidationError("'cells' must be a list")

    # Validate each cell has required fields
    for i, cell in enumerate(notebook["cells"]):
        if not isinstance(cell, dict):
            raise NotebookValidationError(f"Cell {i} is not a dictionary")

        if "cell_type" not in cell:
            raise NotebookValidationError(f"Cell {i} missing 'cell_type'")

        if cell["cell_type"] not in ["code", "markdown", "raw"]:
            raise NotebookValidationError(
                f"Cell {i} has invalid cell_type: {cell['cell_type']}"
            )


def read_notebook(path: Path) -> dict | None:
    """Safely read notebook with comprehensive error handling.

    Args:
        path: Path to notebook file

    Returns:
        Notebook dictionary or None if error occurred

    Logs all errors but doesn't raise exceptions.
    """
    try:
        if not path.exists():
            logger.error(f"Notebook not found: {path}")
            return None

        if not path.is_file():
            logger.error(f"Path is not a file: {path}")
            return None

        with open(path, encoding="utf-8") as f:
            notebook = json.load(f)

        validate_notebook_structure(notebook)
        return notebook

    except PermissionError:
        logger.error(f"Permission denied reading: {path}")
        return None
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error in {path}: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(
            f"Invalid JSON in {path} at line {e.lineno}, column {e.colno}: {e.msg}"
        )
        return None
    except NotebookValidationError as e:
        logger.error(f"Invalid notebook structure in {path}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error reading {path}: {e}")
        return None


def write_notebook(path: Path, notebook: dict, backup: bool = False) -> bool:
    """Safely write notebook with atomic operation and optional backup.

    Args:
        path: Path to notebook file
        notebook: Notebook dictionary to write
        backup: Create .bak file before overwriting

    Returns:
        True if successful, False otherwise

    Uses atomic write pattern:
    1. Write to temporary file
    2. Verify write succeeded
    3. Optionally backup original
    4. Move temp file to final location
    """
    temp_path = None
    backup_path = None

    try:
        # Validate before writing
        validate_notebook_structure(notebook)

        # Create temporary file in same directory (ensures same filesystem)
        temp_fd, temp_path_str = tempfile.mkstemp(
            suffix=".ipynb.tmp", dir=path.parent, text=True
        )
        temp_path = Path(temp_path_str)

        # Write to temp file
        with open(temp_fd, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
            f.write("\n")

        # Create backup if requested and original exists
        if backup and path.exists():
            backup_path = path.with_suffix(".ipynb.bak")
            shutil.copy2(str(path), str(backup_path))

        # Atomic move
        shutil.move(str(temp_path), str(path))
        temp_path = None  # Prevent cleanup of successfully moved file

        return True

    except NotebookValidationError as e:
        logger.error(f"Cannot write invalid notebook to {path}: {e}")
        return False
    except (PermissionError, OSError) as e:
        logger.error(f"Error writing notebook {path}: {e}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error writing {path}: {e}")
        return False
    finally:
        # Clean up temp file if it still exists
        if temp_path and temp_path.exists():
            with contextlib.suppress(Exception):
                temp_path.unlink()  # Best effort cleanup
