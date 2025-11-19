"""plt.show() replacement transformer."""

import re

from ..types import NotebookCell
from .base import NotebookTransformer


def find_figure_variable(source: list[str], show_line_idx: int) -> str:
    """Find the figure variable name by looking backwards from plt.show().

    Args:
        source: Source code lines
        show_line_idx: Index of line containing plt.show()

    Returns:
        Figure variable name or 'plt.gcf()' as fallback
    """
    # Look backwards up to 20 lines for figure assignment
    start_idx = max(0, show_line_idx - 20)
    for i in range(show_line_idx - 1, start_idx - 1, -1):
        line = source[i]

        # Match patterns like: fig = plt.figure() or fig, ax = plt.subplots()
        fig_match = re.search(r"(\w+)\s*[,=].*plt\.(figure|subplots)", line)
        if fig_match:
            return fig_match.group(1)

    # Default: use plt.gcf() to get current figure
    return "plt.gcf()"


def replace_plt_show(source: list[str]) -> tuple[list[str], int]:
    """Replace plt.show() with display pattern using context-aware logic.

    Args:
        source: Source code lines

    Returns:
        Tuple of (modified_source, num_replacements)
    """
    modified = []
    replacements = 0

    for i, line in enumerate(source):
        # Skip comments (lines that start with # after stripping whitespace)
        stripped = line.lstrip()
        if stripped.startswith("#"):
            modified.append(line)
            continue

        # Check if line contains plt.show()
        if "plt.show()" not in line:
            modified.append(line)
            continue

        # Check for plt.show() inside string literals (basic check)
        # Count quotes before plt.show() to detect if inside string
        before_show = line.split("plt.show()")[0]
        double_quotes = before_show.count('"')
        single_quotes = before_show.count("'")

        # If odd number of quotes, we're inside a string literal
        if double_quotes % 2 == 1 or single_quotes % 2 == 1:
            modified.append(line)
            continue

        # Use regex to match plt.show() as a standalone statement
        match = re.match(r"^(\s*)(.*)plt\.show\(\)(.*)$", line)

        if match:
            indent = match.group(1)
            before = match.group(2).strip()
            after = match.group(3).strip()

            # Only replace if it's a standalone call
            if before == "" and after in ["", "\n"]:
                # Find the figure variable name
                fig_var = find_figure_variable(source, i)

                # Replace with three-line pattern
                modified.append(f"{indent}plt.tight_layout()\n")
                modified.append(f"{indent}display({fig_var})\n")
                modified.append(f"{indent}plt.close({fig_var})\n")
                replacements += 1
            else:
                # Complex case - don't replace to avoid breaking code
                modified.append(line)
        else:
            modified.append(line)

    return modified, replacements


class PltShowReplacementTransformer(NotebookTransformer):
    """Replaces plt.show() calls with display/close pattern.

    This transformation improves notebook display behavior by:
    1. Adding plt.tight_layout() for better spacing
    2. Using display() for explicit rendering
    3. Closing figures with plt.close() to free memory
    """

    def transform(
        self, cells: list[NotebookCell]
    ) -> tuple[list[NotebookCell], dict[str, int]]:
        """Replace plt.show() in all code cells.

        Args:
            cells: Notebook cells

        Returns:
            Tuple of (modified_cells, stats)
        """
        stats = {"replacements": 0, "cells_modified": 0}

        result = []

        for cell in cells:
            if cell.get("cell_type") != "code":
                result.append(cell.copy())
                continue

            source = cell.get("source", [])
            if isinstance(source, str):
                source = [source]

            modified_source, num_replacements = replace_plt_show(source)

            if num_replacements > 0:
                # Create modified cell
                modified_cell = cell.copy()
                modified_cell["source"] = modified_source
                result.append(modified_cell)

                stats["replacements"] += num_replacements
                stats["cells_modified"] += 1
            else:
                result.append(cell.copy())

        return result, stats

    def name(self) -> str:
        """Return transformation name."""
        return "plt_show_replacement"

    def description(self) -> str:
        """Return transformation description."""
        return "Replace plt.show() with display/close pattern"

    def should_apply(self, cells: list[NotebookCell]) -> bool:
        """Always check cells for plt.show() calls."""
        # Could optimize by checking if any cell contains plt.show()
        # but transformation itself is fast enough
        return True
