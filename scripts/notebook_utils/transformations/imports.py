"""IPython.display import transformer."""

from ..cells import (
    create_ipython_display_import_cell,
    find_cell_with_pattern,
    has_ipython_display_import,
    uses_display,
)
from ..types import NotebookCell
from .base import NotebookTransformer


class IPythonDisplayImportTransformer(NotebookTransformer):
    """Adds IPython.display import when display() is used.

    This transformation ensures notebooks that use the display() function
    have the necessary import statement.
    """

    def transform(
        self, cells: list[NotebookCell]
    ) -> tuple[list[NotebookCell], dict[str, int]]:
        """Add IPython.display import if needed.

        Args:
            cells: Notebook cells

        Returns:
            Tuple of (modified_cells, stats)
        """
        stats = {"import_added": 0}

        # Check if import needed
        if not uses_display(cells) or has_ipython_display_import(cells):
            return cells.copy(), stats

        # Create new list to avoid mutating input
        result = cells.copy()

        # Insert import after %matplotlib inline if present
        matplotlib_idx = find_cell_with_pattern(result, "%matplotlib inline")
        if matplotlib_idx is not None:
            insert_idx = matplotlib_idx + 1
        else:
            # Fallback: add at beginning
            insert_idx = 0

        import_cell = create_ipython_display_import_cell()
        result.insert(insert_idx, import_cell)

        stats["import_added"] = 1
        return result, stats

    def name(self) -> str:
        """Return transformation name."""
        return "ipython_display_import"

    def description(self) -> str:
        """Return transformation description."""
        return "Add IPython.display import when display() is used"

    def should_apply(self, cells: list[NotebookCell]) -> bool:
        """Only apply if display() is used and import not present."""
        return uses_display(cells) and not has_ipython_display_import(cells)
