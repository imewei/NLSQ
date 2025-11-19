"""Matplotlib inline configuration transformer."""

from ..cells import (
    create_matplotlib_config_cell,
    find_first_code_cell_index,
    has_matplotlib_magic,
)
from ..types import NotebookCell
from .base import NotebookTransformer


class MatplotlibInlineTransformer(NotebookTransformer):
    """Adds %matplotlib inline magic before first code cell.

    This transformation ensures notebooks have the matplotlib inline backend
    configured for proper display in Jupyter environments.
    """

    def transform(
        self, cells: list[NotebookCell]
    ) -> tuple[list[NotebookCell], dict[str, int]]:
        """Add %matplotlib inline magic if not present.

        Args:
            cells: Notebook cells

        Returns:
            Tuple of (modified_cells, stats)
        """
        stats = {"magic_added": 0}

        # Check if already present
        if has_matplotlib_magic(cells):
            return cells.copy(), stats

        # Create new list to avoid mutating input
        result = cells.copy()

        # Insert magic at first code cell
        config_cell = create_matplotlib_config_cell()
        first_code_idx = find_first_code_cell_index(result)
        result.insert(first_code_idx, config_cell)

        stats["magic_added"] = 1
        return result, stats

    def name(self) -> str:
        """Return transformation name."""
        return "matplotlib_inline"

    def description(self) -> str:
        """Return transformation description."""
        return "Add %matplotlib inline magic for inline plotting"

    def should_apply(self, cells: list[NotebookCell]) -> bool:
        """Only apply if magic not already present."""
        return not has_matplotlib_magic(cells)
