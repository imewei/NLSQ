"""Cell manipulation utilities for Jupyter notebooks."""

from .types import NotebookCell


def has_matplotlib_magic(cells: list[NotebookCell]) -> bool:
    """Check if notebook already has %matplotlib inline."""
    for cell in cells:
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if "%matplotlib inline" in source:
                return True
    return False


def has_ipython_display_import(cells: list[NotebookCell]) -> bool:
    """Check if notebook already imports display from IPython.display."""
    for cell in cells:
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if "from IPython.display import display" in source:
                return True
    return False


def uses_display(cells: list[NotebookCell]) -> bool:
    """Check if notebook uses display() function."""
    for cell in cells:
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if "display(" in source:
                return True
    return False


def find_first_code_cell_index(cells: list[NotebookCell]) -> int:
    """Find index of first code cell."""
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            return i
    return 0


def find_cell_with_pattern(cells: list[NotebookCell], pattern: str) -> int | None:
    """Find index of first cell containing pattern.

    Args:
        cells: List of notebook cells
        pattern: String pattern to search for

    Returns:
        Index of first cell containing pattern, or None if not found
    """
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if pattern in source:
                return i
    return None


def create_matplotlib_config_cell() -> NotebookCell:
    """Create cell with matplotlib configuration."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Configure matplotlib for inline plotting in VS Code/Jupyter\n",
            "# MUST come before importing matplotlib\n",
            "%matplotlib inline",
        ],
    }


def create_ipython_display_import_cell() -> NotebookCell:
    """Create cell with IPython.display import."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ["from IPython.display import display"],
    }


def cell_contains_pattern(
    cells: list[NotebookCell], pattern: str, cell_type: str = "code"
) -> bool:
    """Check if any cell contains the specified pattern.

    Args:
        cells: List of notebook cells
        pattern: String pattern to search for
        cell_type: Type of cells to search (default: "code")

    Returns:
        True if pattern found in any matching cell
    """
    for cell in cells:
        if cell.get("cell_type") == cell_type:
            source = "".join(cell.get("source", []))
            if pattern in source:
                return True
    return False
